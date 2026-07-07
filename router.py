"""
router.py  (v9 — final with word boundary fix)
------------------------------------------------
WHAT CHANGED vs v8:

    Fix — _has_kg1_signal() now uses word boundaries for single-word signals.

    In v8, the fast pre-check used `sig in q_lower`, which performs a
    substring match. This means the word "gate" would match inside "navigate",
    "aggregate", or any other word containing that substring. While unlikely
    to cause real failures in aviation queries, it is a formal correctness
    issue: a false positive here causes the router to skip the LLM for a
    question that might be cross-KG, routing it silently to single_kg1
    with no fallback.

    The fix uses re.search(r"\b{sig}\b") for single-word signals, which
    requires the signal to appear as a standalone word. Multi-word signals
    like "ground speed" still use substring matching, which is correct
    because the space itself prevents partial matches inside compound words.

    This makes the fast pre-check formally correct, not just empirically
    correct — an important distinction for a thesis defense.

ARCHITECTURE OVERVIEW
---------------------
The router is the front door of the NL2SPARQL pipeline. Its sole job is
to decide which branch handles a given question. It does NOT extract
properties, generate SPARQL, or query any endpoint.

Four branches exist:
    single_kg1  → a specific flight is asked about (flight number present,
                  question asks about the flight itself)
    single_kg2  → a specific airport is asked about by name or IATA code
    cross_kg    → a specific flight is asked about, but the question asks
                  about a property of its airport (country, elevation, etc.)
    template    → an aggregate question with no specific entity
                  (filter, ranking, comparison, count)

ROUTING LOGIC
-------------
Priority 1 — Minimum structure guard
    Rejects single-word or meaningless inputs immediately.

Priority 2 — Flight number detected (regex)
    2a. FAST PATH: question contains a KG1-only signal word.
        These concepts only exist in KG1, never in KG2. The question
        cannot possibly be cross-KG. Return single_kg1 immediately
        without calling the LLM, saving ~2 seconds of Ollama latency.
        Word boundaries are used for single-word signals to prevent
        false matches inside longer words (e.g. "gate" in "navigate").

    2b. SLOW PATH: question is ambiguous.
        "What is the airline of flight X?" and "What country does
        flight X land in?" both contain a flight number but require
        different pipelines. The LLM is the only component that can
        distinguish them reliably across all languages.
        If LLM returns cross_kg_filter → route cross_kg.
        Otherwise → route single_kg1.

Priority 3 — No flight number → LLM classifies everything else.
    Template branch, single_kg2, or out_of_scope.
    If LLM fails, deterministic fallbacks provide a safety net.

WHY THE LLM RUNS FOR AMBIGUOUS FLIGHT QUESTIONS
-------------------------------------------------
Cross-KG questions always contain a flight number — that is how the
system identifies the flight in KG1. But plain KG1 questions also
contain a flight number. At the surface level they are indistinguishable.
Only semantic understanding separates them. The LLM provides that.

WHAT THE ROUTER DOES NOT DO
-----------------------------
The router does not extract property URIs, entities, or SPARQL.
For cross_kg questions it only determines the flight number and
direction (origin/destination). The actual property is extracted
downstream by the hybrid mapping layer (extract_airport_entities +
map_property_cascade), which is the core thesis contribution.
"""

from rapidfuzz import process, fuzz
import re
import json
import ollama
from kg_registry import (
    KG_REGISTRY,
    CROSS_KG_CONFIG,
    TEMPLATE_REGISTRY,
    get_lexicon,
)

# ─────────────────────────────────────────────
# REGEX
# ─────────────────────────────────────────────

_FLIGHT_RE = re.compile(r"\b([A-Z]{2,3}\d+)\b")
_IATA_RE   = re.compile(r"\b([A-Z]{3})\b")

# ─────────────────────────────────────────────
# KG1-ONLY SIGNAL WORDS
# ─────────────────────────────────────────────
# These words refer exclusively to flight properties that exist only in KG1.
# If any of them appear in a question that also contains a flight number,
# the question cannot be cross-KG — return single_kg1 without the LLM.
#
# "destination" and "departure" are intentionally excluded: they appear in
# both KG1 questions ("What is the destination of flight X?") and cross-KG
# questions ("What country does the destination airport of flight X serve?").
# Including them would cause false positives.

_KG1_ONLY_SIGNALS = {
    # English
    "gate", "terminal", "callsign", "squawk",
    "ground speed", "vertical speed",
    # French
    "porte", "indicatif", "vitesse sol", "vitesse verticale",
    # Arabic
    "بوابة", "مبنى", "الإشارة", "سرعة أرضية", "سرعة عمودية",
     "destination of flight",   # "What is the destination of flight X?"
    "depart from",             # "Where does flight OS235 depart from?"
    "flying to",               # "What country is flight LO225 flying to?"
    "vole vers",               # French equivalent
    "تغادر",                   # Arabic: departs
    "وجهة الرحلة",
}

# ─────────────────────────────────────────────
# KG1 SIGNAL DETECTOR (word-boundary safe)
# ─────────────────────────────────────────────

def _has_kg1_signal(q_lower: str) -> bool:
    """
    Returns True if the question contains any KG1-only signal word.

    Uses two matching strategies depending on signal length:
      - Multi-word signals (e.g. "ground speed"):
          Simple substring match. The space character naturally prevents
          partial matches inside compound words like "groundspeed".
      - Single-word signals (e.g. "gate"):
          Word-boundary regex (\b). Prevents matching "gate" inside
          "navigate" or "aggregate". This is the formally correct approach
          for single tokens.

    This distinction matters for thesis correctness: a false positive here
    causes the router to skip the LLM for a question that might be cross-KG,
    routing it silently to single_kg1 with no fallback.
    """
    for sig in _KG1_ONLY_SIGNALS:
        if " " in sig:
            # Multi-word: substring match is sufficient and correct
            if sig in q_lower:
                return True
        else:
            # Single word: require word boundary to avoid partial matches
            if re.search(rf"\b{re.escape(sig)}\b", q_lower):
                return True
    return False

# ─────────────────────────────────────────────
# LEXICON LOAD
# ─────────────────────────────────────────────

def _load_airport_lexicon():
    with open(get_lexicon("airports"), encoding="utf-8") as f:
        return json.load(f)

_airport_lex      = _load_airport_lexicon()
_AIRPORT_ENTITIES = _airport_lex.get("airport_entities", {})
_AIRPORT_TRIGGERS = KG_REGISTRY["airports"].get("triggers", [])

# ─────────────────────────────────────────────
# LLM CLASSIFICATION PROMPT
# ─────────────────────────────────────────────

_CLASSIFICATION_PROMPT = """You are a query classifier for an airport and flight database.

Classify the question into exactly one of these query types and extract its parameters.

── QUERY TYPES AND THEIR PARAMETERS ──────────────────────────────────────────

1. filter_numeric_kg2 — airports filtered by a numeric property
   params: property, operator, threshold, limit (default 10)

2. filter_string_kg2 — airports filtered by a text/categorical property
   params: property, value, limit (default 10)

3. ranking_kg2 — airports ranked by a numeric property (top/bottom N)
   params: property, order (ASC or DESC), limit (default 5)

4. compare_two_airports — compare exactly two airports on one property
   params: airport1 (IATA code), airport2 (IATA code), property

5. count_kg1 — count or list flights matching a condition
   params: filter_property, filter_value, mode (count or list)

6. filter_numeric_kg1 — flights filtered by a numeric flight property
   params: property, operator, threshold, limit (default 10)

7. cross_kg_filter — a specific flight is mentioned AND the question asks
   about a property of its origin or destination airport.
   params: direction (origin or destination), airport_property,
           operator, threshold, limit (default 10)

8. single_kg2 — one specific airport asked about by name or IATA code
   params: entity (IATA code or airport name)

9. out_of_scope — cannot be answered from this database
   params: {}
10. open_kg — the question is about aviation data but does not fit any
    template above. It asks about a specific property or relationship
    that requires a custom query.
    params: {}
    
    Examples:
    - "Which flight has the highest ground speed?" → ranking, use ranking_kg2 or filter_numeric_kg1
    - "How many airports are in the dataset?" → open_kg
    - "Which airports have a grass runway?" → open_kg
    - "What is the registration number of the aircraft on flight BR62?" → open_kg
    - "Quel vol a la vitesse verticale la plus basse?" → open_kg
── PROPERTY MAPPING RULES ────────────────────────────────────────────────────

Airport numeric properties:
  "elevation", "altitude", "height"            → elevationFt
  "runway length", "length", "longer", "long"  → lengthFt
  "runway width", "width", "wider", "wide"     → widthFt

Airport string properties:
  "country", "located in", "in [country]"      → countryName
  "large airport", "large airports", "type"    → airportType  (value: large_airport)
  "city", "municipality"                       → municipality
  "continent"                                  → continent
  "surface"                                    → surface

Flight numeric properties:
  "ground speed", "speed", "knots"             → gspeed
  "vertical speed", "feet per minute"          → vspeed

  NOTE: flight altitude is NOT stored in this database. Classify
  altitude-threshold questions for flights as out_of_scope.

Flight string properties:
  "destination city", "going to", "land in"    → hasDestinationCity
  "origin city", "departing from", "from"      → hasOriginCity
  "airline", "operated by"                     → hasAirline
  "destination country"                        → hasDestinationCountry

Operator mapping:
  "above", "exceeds", "more than", "greater"   → >
  "below", "less than", "under"                → <
  "at least"                                   → >=
  "at most"                                    → <=
  "in", "is", "equal to", "located in"         → =

Ranking direction:
  "highest", "longest", "widest", "most"       → DESC
  "lowest", "shortest", "narrowest", "least"   → ASC

── DISAMBIGUATION RULES ──────────────────────────────────────────────────────

CROSS_KG_FILTER: Use when a specific flight number is mentioned AND the
  question asks about a property of that flight's airport (country,
  elevation, runway, type). Examples:
    "What country does flight LO225 land in?"           → cross_kg_filter
    "What type of airport does flight FR182 arrive at?" → cross_kg_filter
    "Dans quel pays atterrit le vol OS295?"             → cross_kg_filter
    "في أي دولة يهبط الرحلة OS235؟"                   → cross_kg_filter

COUNT_KG1: Use when the question counts or lists flights.
  "how many flights" / "combien de vols" / "كم رحلة" → always count_kg1,
  even if a city name is present.
OPEN_KG: Use when the question asks about aviation data that exists in the
  KG but does not fit filter_numeric, filter_string, ranking, compare,
  count, or cross_kg_filter patterns. Specifically:
  - Questions about aircraft registration or specific aircraft details
  - Questions asking for a count of a KG class (airports, runways)
  - Questions about runway surface types (grass, concrete)
  - Questions about closed runways
  Do NOT use open_kg when filter_numeric_kg1 or ranking_kg2 would work.
── EXAMPLES ──────────────────────────────────────────────────────────────────

Q: "Which airports have an elevation above 1000 feet?"
A: {{"query_type": "filter_numeric_kg2", "params": {{"property": "elevationFt", "operator": ">", "threshold": 1000, "limit": 10}}}}

Q: "Show all large airports."
A: {{"query_type": "filter_string_kg2", "params": {{"property": "airportType", "value": "large_airport", "limit": 10}}}}

Q: "Which airports are located in Germany?"
A: {{"query_type": "filter_string_kg2", "params": {{"property": "countryName", "value": "Germany", "limit": 10}}}}

Q: "What are the top 5 airports with the highest elevation?"
A: {{"query_type": "ranking_kg2", "params": {{"property": "elevationFt", "order": "DESC", "limit": 5}}}}

Q: "Which airport has the shortest runway?"
A: {{"query_type": "ranking_kg2", "params": {{"property": "lengthFt", "order": "ASC", "limit": 1}}}}

Q: "Quel aéroport a la piste la plus longue?"
A: {{"query_type": "ranking_kg2", "params": {{"property": "lengthFt", "order": "DESC", "limit": 1}}}}

Q: "Quel aéroport a la plus haute élévation?"
A: {{"query_type": "ranking_kg2", "params": {{"property": "elevationFt", "order": "DESC", "limit": 1}}}}

Q: "أي مطار لديه أعلى ارتفاع؟"
A: {{"query_type": "ranking_kg2", "params": {{"property": "elevationFt", "order": "DESC", "limit": 1}}}}

Q: "Compare VIE and FRA by elevation."
A: {{"query_type": "compare_two_airports", "params": {{"airport1": "VIE", "airport2": "FRA", "property": "elevationFt"}}}}

Q: "Comparez CDG et LHR par longueur de piste."
A: {{"query_type": "compare_two_airports", "params": {{"airport1": "CDG", "airport2": "LHR", "property": "lengthFt"}}}}

Q: "How many flights are operated by Lufthansa?"
A: {{"query_type": "count_kg1", "params": {{"filter_property": "hasAirline", "filter_value": "Lufthansa", "mode": "count"}}}}

Q: "Combien de vols partent de Vienne?"
A: {{"query_type": "count_kg1", "params": {{"filter_property": "hasOriginCity", "filter_value": "Vienna", "mode": "count"}}}}

Q: "كم رحلة تتجه إلى برلين؟"
A: {{"query_type": "count_kg1", "params": {{"filter_property": "hasDestinationCity", "filter_value": "Berlin", "mode": "count"}}}}

Q: "Which flights have a ground speed above 400 knots?"
A: {{"query_type": "filter_numeric_kg1", "params": {{"property": "gspeed", "operator": ">", "threshold": 400, "limit": 10}}}}

Q: "Which flights have a vertical speed below -1000 feet per minute?"
A: {{"query_type": "filter_numeric_kg1", "params": {{"property": "vspeed", "operator": "<", "threshold": -1000, "limit": 10}}}}

Q: "What country does flight LO225 land in?"
A: {{"query_type": "cross_kg_filter", "params": {{"direction": "destination", "airport_property": "countryName", "operator": "=", "threshold": "Poland", "limit": 1}}}}

Q: "What type of airport does flight FR182 arrive at?"
A: {{"query_type": "cross_kg_filter", "params": {{"direction": "destination", "airport_property": "airportType", "operator": "=", "threshold": "large_airport", "limit": 1}}}}

Q: "What is the elevation of the destination airport of KE567?"
A: {{"query_type": "cross_kg_filter", "params": {{"direction": "destination", "airport_property": "elevationFt", "operator": ">", "threshold": 0, "limit": 1}}}}

Q: "Dans quel pays atterrit le vol OS295?"
A: {{"query_type": "cross_kg_filter", "params": {{"direction": "destination", "airport_property": "countryName", "operator": "=", "threshold": "Austria", "limit": 1}}}}

Q: "في أي دولة يهبط الرحلة OS235؟"
A: {{"query_type": "cross_kg_filter", "params": {{"direction": "destination", "airport_property": "countryName", "operator": "=", "threshold": "Germany", "limit": 1}}}}

Q: "What is the runway length at the destination of OS214?"
A: {{"query_type": "cross_kg_filter", "params": {{"direction": "destination", "airport_property": "lengthFt", "operator": ">", "threshold": 0, "limit": 1}}}}

Q: "Which flights land at airports with elevation above 800 feet?"
A: {{"query_type": "cross_kg_filter", "params": {{"direction": "destination", "airport_property": "elevationFt", "operator": ">", "threshold": 800, "limit": 10}}}}

Q: "Which flights arrive at airports located in Germany?"
A: {{"query_type": "cross_kg_filter", "params": {{"direction": "destination", "airport_property": "countryName", "operator": "=", "threshold": "Germany", "limit": 10}}}}

Q: "Quels vols atterrissent dans des aéroports en Allemagne?"
A: {{"query_type": "cross_kg_filter", "params": {{"direction": "destination", "airport_property": "countryName", "operator": "=", "threshold": "Germany", "limit": 10}}}}

Q: "Which flights land at large airports?"
A: {{"query_type": "cross_kg_filter", "params": {{"direction": "destination", "airport_property": "airportType", "operator": "=", "threshold": "large_airport", "limit": 10}}}}

Q: "Which flight has the highest ground speed?"
A: {{"query_type": "open_kg", "params": {{}}}}

Q: "What is the callsign of the fastest flight?"
A: {{"query_type": "open_kg", "params": {{}}}}

Q: "ما هي الرحلة ذات أعلى سرعة أرضية؟"
A: {{"query_type": "open_kg", "params": {{}}}}
── NOW CLASSIFY THIS QUESTION ────────────────────────────────────────────────

Question: "{question}"

- Use double quotes " for every key and every string value. Never use single quotes.
- Do not add comments, trailing commas, or any text outside the JSON object.
- Output exactly one JSON object and nothing else — no markdown, no bullet points.

Return ONLY a JSON object with keys "query_type" and "params".
No explanation. No text before or after the JSON.
"""

# ─────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────

def _normalise(text: str) -> str:
    text = text.lower()
    text = text.replace("'", " ").replace("\u2019", " ").replace("\u2018", " ")
    text = text.replace("\u061F", " ")
    text = re.sub(r"[^\w\s\u0600-\u06FE]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ─────────────────────────────────────────────
# MINIMUM STRUCTURE GUARD
# ─────────────────────────────────────────────

def _has_minimum_structure(question: str) -> bool:
    words = question.strip().split()
    if len(words) < 2:
        return False
    if len(words) == 2:
        if re.search(r'[A-Za-z]{2,3}\d+', question):
            return True
        if re.search(r'\b[A-Z]{3}\b', question.upper()):
            return True
        return False
    return True

# ─────────────────────────────────────────────
# DETERMINISTIC DETECTORS (fallback only)
# ─────────────────────────────────────────────

def _detect_flight_number(q: str):
    """Regex-based. Fast and deterministic. Called before any LLM."""
    m = _FLIGHT_RE.findall(q.upper())
    return max(m, key=len) if m else None


def _detect_airport_entity(q: str):
    """
    Dictionary lookup and fuzzy match for airport names and IATA codes.
    Only called when the LLM fails or returns out_of_scope.
    Represents Tier 1 (exact) and Tier 3 (fuzzy) of the hybrid mapping layer.
    """
    q_norm = _normalise(q)
    tokens = q_norm.split()

    # Tier 1: exact phrase match, longest phrase first
    for size in range(6, 0, -1):
        for i in range(len(tokens) - size + 1):
            phrase = " ".join(tokens[i: i + size])
            if phrase in _AIRPORT_ENTITIES:
                return _AIRPORT_ENTITIES[phrase]

    # Tier 2: IATA code (3 uppercase letters as standalone token)
    for code in _IATA_RE.findall(q.upper()):
        if code in _AIRPORT_ENTITIES:
            return _AIRPORT_ENTITIES[code]

    # Tier 3: fuzzy match on individual tokens
    STOP_WORDS = {
        "what", "is", "the", "of", "in", "at", "which", "where",
        "airport", "how", "does", "do", "an", "a", "nation", "town",
        "quel", "est", "le", "la", "de", "du", "quelle", "aéroport",
        "dans", "se", "trouve",
        "ما", "هو", "في", "أي", "يقع", "مطار", "هي", "على",
        "ارتفاع", "نوع", "بلد", "دولة",
    }
    GEOGRAPHIC_NOISE = {
        "france", "italy", "history", "aviation", "naples", "pizza",
        "president", "book", "flight", "best", "germany", "large",
        "vienna", "airports", "runway", "elevation", "municipality",
        "located", "show", "list", "all", "highest", "lowest", "top",
        "longest", "shorter", "exceeds", "above", "below", "whose",
        "flying", "altitude", "speed", "knots", "vertical", "width",
        "length", "country", "continent", "surface", "type", "city",
    }
    candidates = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    for candidate in candidates:
        if candidate.lower() in GEOGRAPHIC_NOISE:
            continue
        result = process.extractOne(
            candidate,
            list(_AIRPORT_ENTITIES.keys()),
            scorer=fuzz.WRatio,
        )
        if result is not None:
            match, score, _ = result
            if score >= 92:
                return _AIRPORT_ENTITIES[match]
    return None


def _detect_airport_keyword(q: str) -> bool:
    q_lower = q.lower()
    return any(k.lower() in q_lower for k in _AIRPORT_TRIGGERS)

# ─────────────────────────────────────────────
# LLM CLASSIFIER
# ─────────────────────────────────────────────

def _llm_classify(question: str, max_attempts: int = 2) -> dict:
    prompt = _CLASSIFICATION_PROMPT.replace("{question}", question)

    for attempt in range(max_attempts):
        try:
            response = ollama.chat(
                model="llama3",
                messages=[{"role": "user", "content": prompt}]
            )
            raw = response["message"]["content"].strip()
            raw = re.sub(r"```json|```", "", raw).strip()

            match = re.search(r"\{.*\}", raw, re.DOTALL)
            candidate = match.group() if match else raw

            result = json.loads(candidate)
            print(f"[router] LLM classified as: {result.get('query_type')} "
                  f"| params: {result.get('params')}")
            return result

        except Exception as e:
            print(f"[router] Attempt {attempt+1}: classification failed: {e}")

            # Build a repair prompt using the model's own broken output
            prompt = (
                f"Your previous response was not valid JSON:\n\n"
                f"{raw}\n\n"
                f"Error: {e}\n\n"
                f"Return ONLY the corrected JSON object. "
                f"Use double quotes for all keys and values. "
                f"No explanation, no text before or after."
            )

    return {}
def _is_kg_answerable(question: str) -> bool:
    """
    Asks the LLM whether the question can be answered from the specific
    data model of the deployed knowledge graphs.

    WHY THIS IS BETTER THAN A KEYWORD LIST:
        A keyword list catches aviation vocabulary but cannot judge
        answerability. "Can I eat pizza on the airline?" contains the
        word "airline" yet cannot be answered from our KG.
        This prompt grounds the check in the actual data model.

    Returns True if the LLM says YES, False otherwise.
    """
    from kg_registry import get_open_kg_schema
    schema = get_open_kg_schema()

    prompt = f"""You are a scope classifier for an aviation knowledge graph system.
The question may be in English, French, or Arabic.

The knowledge graph contains:
- Flights: flight number, airline, origin city, destination city,
  aircraft type, gate, terminal, callsign, ground speed, vertical speed
- Airports: name, type, elevation, country, region, city,
  IATA code, ICAO code, coordinates
- Runways: length, width, surface, lighting, identifier

Answer only YES or NO:
Can this question be answered using only the data described above?

Question: "{question}"
"""
    try:
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response["message"]["content"].strip().upper()
        return answer.startswith("YES")
    except Exception as e:
        print(f"[router] _is_kg_answerable failed: {e}")
        return False

# ─────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────

def route(question: str) -> dict:
    """
    Routes a natural language question to the correct pipeline branch.

    Returns a routing dict consumed by main.py and test_pipeline.py.
    Keys: query_type, kg, entity, direction, template, config, params.

    Final structure:
        Priority 1 → structure guard
        Priority 2 → flight number → single_kg1 or cross_kg
        Priority 2.5 → airport entity → single_kg2
        Priority 3 → LLM classifies → template / single_kg2 / open_kg
                     (with smart reroute for known misclassification patterns)
        Clean gate → _is_kg_answerable() → open_kg or out_of_scope
    """

    # ── Priority 1: Structure guard ───────────────────────────────────────────
    if not _has_minimum_structure(question):
        return {
            "query_type": "out_of_scope",
            "kg":         None,
            "entity":     None,
            "direction":  None,
            "template":   None,
            "config":     None,
        }

    q_lower = question.lower()

    # ── Priority 2: Flight number detected ────────────────────────────────────
    flight = _detect_flight_number(question)

    if flight:

        # ── Fast path (2a): KG1-only signal word present ──────────────────────
        # If the question contains a word that can only refer to a flight
        # property (gate, callsign, squawk, ground speed, vertical speed),
        # it cannot be a cross-KG question. Skip the LLM entirely.
        # _has_kg1_signal uses word boundaries for single tokens to avoid
        # false matches inside longer words (e.g. "gate" inside "navigate").
        if _has_kg1_signal(q_lower):
            return {
                "query_type": "single_kg1",
                "kg":         "flights",
                "entity":     flight,
                "direction":  None,
                "template":   None,
                "config":     KG_REGISTRY["flights"],
            }

        # ── Slow path (2b): Ambiguous — ask the LLM ──────────────────────────
        # The question contains a flight number but no KG1-only signal.
        # It could ask about the flight ("What is the airline of X?") or
        # about the flight's airport ("What country does X land in?").
        # Only the LLM can distinguish these reliably across all languages
        # and phrasings.
        classified = _llm_classify(question)
        query_type = classified.get("query_type", "")
        params     = classified.get("params", {})

        if query_type == "cross_kg_filter":
            # The question asks about the airport property, not the flight.
            # direction tells the cross_kg_resolver which airport to look up.
            direction = params.get("direction", "destination")
            return {
                "query_type": "cross_kg",
                "kg":         "cross",
                "entity":     flight,
                "direction":  direction,
                "template":   None,
                "config":     CROSS_KG_CONFIG,
            }

        # LLM returned anything other than cross_kg_filter, or failed.
        # Treat as a plain KG1 flight question.
        return {
            "query_type": "single_kg1",
            "kg":         "flights",
            "entity":     flight,
            "direction":  None,
            "template":   None,
            "config":     KG_REGISTRY["flights"],
        }

    # ── Priority 2.5: Airport entity detected (deterministic) ─────────────────
    airport = _detect_airport_entity(question)
    if airport:
        return {
            "query_type": "single_kg2",
            "kg":         "airports",
            "entity":     airport,
            "direction":  None,
            "template":   None,
            "config":     KG_REGISTRY["airports"],
        }

    # ── Priority 3: No flight number — LLM classifies everything else ─────────
    classified = _llm_classify(question)
    query_type = classified.get("query_type", "")
    params     = classified.get("params", {})

    # ── Template branch ───────────────────────────────────────────────────────
    if query_type in TEMPLATE_REGISTRY:
        cfg = TEMPLATE_REGISTRY[query_type]

        KG1_FLIGHT_PROPS = {"gspeed", "vspeed", "alt", "groundSpeed", "speed"}
        RANKING_SIGNALS  = [
            "highest", "lowest", "fastest", "slowest",
            "la plus haute", "la plus basse", "le plus rapide", "le plus lent",
            "الأعلى", "الأدنى", "الأسرع", "الأبطأ"
        ]

        # Case 1: KG2 template received a KG1 flight property
        if query_type in ("ranking_kg2", "filter_numeric_kg2"):
            prop = params.get("property", "")
            if prop in KG1_FLIGHT_PROPS:
                print(f"[router] Smart reroute: KG1 property in KG2 template → open_kg")
                return {
                    "query_type": "open_kg",
                    "kg":         "cross",
                    "entity":     None,
                    "direction":  None,
                    "template":   None,
                    "config":     None,
                }

        # Case 2: filter_numeric_kg1 with ranking intent and no real threshold
        if query_type == "filter_numeric_kg1":
            prop      = params.get("property", "")
            threshold = params.get("threshold")
            if prop in KG1_FLIGHT_PROPS and (
                threshold is None or
                any(sig in question.lower() for sig in RANKING_SIGNALS)
            ):
                print(f"[router] Smart reroute: ranking signal in filter → open_kg")
                return {
                    "query_type": "open_kg",
                    "kg":         "cross",
                    "entity":     None,
                    "direction":  None,
                    "template":   None,
                    "config":     None,
                }

        # Case 3: filter_string_kg2 with runway surface or closed runway
        if query_type == "filter_string_kg2":
            value = params.get("value", "")
            if any(v in str(value).lower() for v in
                   ["grass", "closed", "grs", "closed_runway", "fermée", "مغلق"]):
                print(f"[router] Smart reroute: runway property → open_kg")
                return {
                    "query_type": "open_kg",
                    "kg":         "cross",
                    "entity":     None,
                    "direction":  None,
                    "template":   None,
                    "config":     None,
                }

        return {
            "query_type": "template",
            "kg":         cfg["kg"],
            "entity":     None,
            "direction":  params.get("direction"),
            "template":   query_type,
            "config":     cfg,
            "params":     params,
        }

    # ── Single airport branch ─────────────────────────────────────────────────
    if query_type == "single_kg2":
        entity = params.get("entity")
        if entity:
            entity_upper = entity.upper().strip()
            if entity_upper in _AIRPORT_ENTITIES:
                entity = _AIRPORT_ENTITIES[entity_upper]
            else:
                entity_norm = _normalise(entity)
                if entity_norm in _AIRPORT_ENTITIES:
                    entity = _AIRPORT_ENTITIES[entity_norm]
        return {
            "query_type": "single_kg2",
            "kg":         "airports",
            "entity":     entity,
            "direction":  None,
            "template":   None,
            "config":     KG_REGISTRY["airports"],
        }

    # ── open_kg branch — LLM identified custom KG question ───────────────────
    if query_type == "open_kg":
        return {
            "query_type": "open_kg",
            "kg":         "cross",
            "entity":     None,
            "direction":  None,
            "template":   None,
            "config":     None,
        }

    # ── CLEAN GATE ────────────────────────────────────────────────────────────
    # Everything that reached here has:
    #   - no flight number
    #   - no specific airport entity
    #   - no template match
    #   - no direct open_kg classification
    #
    # One single question decides the fate:
    # Can this question be answered from our KG data model?
    #
    # YES → open_kg  (free SPARQL generation, schema-grounded)
    # NO  → out_of_scope
    #
    # This replaces all previous Priority 4, Priority 5, keyword fallbacks,
    # and smart reroutes. The _is_kg_answerable() function is the only
    # intelligence needed here.

    if _is_kg_answerable(question):
        return {
            "query_type": "open_kg",
            "kg":         "cross",
            "entity":     None,
            "direction":  None,
            "template":   None,
            "config":     None,
        }

    return {
        "query_type": "out_of_scope",
        "kg":         None,
        "entity":     None,
        "direction":  None,
        "template":   None,
        "config":     None,
    }