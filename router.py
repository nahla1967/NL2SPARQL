"""
router.py  (v5 — LLM-based classification)
-------------------------------------------
DESIGN CHANGE vs v4:

    v4 used a lexicon of hand-crafted signal words to detect which template
    a question belongs to. Every new phrasing or synonym required editing
    the lexicon. This was fragile and did not scale.

    v5 replaces the signal-word matching with a single LLM call that
    classifies the question AND extracts its parameters in one step.
    The lexicon is no longer needed for routing — only the airport_entities
    dictionary is kept for deterministic IATA code lookup.

WHAT STAYS THE SAME:
    - Flight number detection (regex, deterministic, fast)
    - Airport entity detection (dictionary lookup + fuzzy match)
    - The output dict structure — identical to v4, so nothing else changes
    - CROSS_KG_CONFIG, KG_REGISTRY, TEMPLATE_REGISTRY usage

WHAT CHANGES:
    - _detect_template() is replaced by _llm_classify()
    - _detect_cross_kg_signal() is replaced by LLM classification
    - The lexicon's template_triggers section is no longer read
    - No more signal word lists to maintain

HOW IT WORKS:
    Priority 1: flight number regex → single_kg1
    Priority 2: LLM classification → template (covers all 7 template types
                including cross_kg_filter, which previously needed
                _detect_cross_kg_signal)
    Priority 3: airport entity lookup → single_kg2
    Priority 4: nothing matched → out_of_scope

    The LLM returns a JSON object with "query_type" and "params".
    If classification fails or returns out_of_scope, we fall through
    to deterministic entity detection as a safety net.
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
# LEXICON — only airport_entities is still used
# ─────────────────────────────────────────────

def _load_airport_lexicon():
    with open(get_lexicon("airports"), encoding="utf-8") as f:
        return json.load(f)

_airport_lex     = _load_airport_lexicon()
_AIRPORT_ENTITIES = _airport_lex.get("airport_entities", {})
_AIRPORT_TRIGGERS = KG_REGISTRY["airports"].get("triggers", [])

# ─────────────────────────────────────────────
# LLM CLASSIFICATION PROMPT
# ─────────────────────────────────────────────
# This single prompt replaces the entire signal-word system.
# The LLM classifies the question AND extracts parameters in one call.
# Adding support for a new phrasing requires zero code changes.

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

7. cross_kg_filter — flights filtered by a property of their origin or destination airport
   params: direction (origin or destination), airport_property, operator, threshold, limit (default 10)

8. single_kg2 — one specific airport asked about by name or IATA code
   params: entity (IATA code or airport name)

9. out_of_scope — the question cannot be answered from this database
   Examples: flight altitude questions (altitude data is not stored)
   params: {}

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
  "altitude" (for flights), "flying at"        → alt

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
  "in", "is", "equal to"                       → =

Ranking direction:
  "highest", "longest", "widest", "most"       → DESC
  "lowest", "shortest", "narrowest", "least"   → ASC

── IMPORTANT DISAMBIGUATION RULES ────────────────────────────────────────────

- If the question mentions both FLIGHTS and an AIRPORT PROPERTY (elevation,
  country, runway length, type), classify as cross_kg_filter.
- If the question asks about FLIGHT speed, altitude, or vertical speed with
  a numeric threshold, classify as filter_numeric_kg1.
- If the question asks about AIRPORT elevation/length/width with a threshold,
  classify as filter_numeric_kg2.
- "altitude" for FLIGHTS → property = "alt"
- "altitude" for AIRPORTS → property = "elevationFt"
- If two IATA codes are present and a property is mentioned, classify as
  compare_two_airports.

── EXAMPLES ──────────────────────────────────────────────────────────────────

Q: "Which airports have an elevation above 1000 feet?"
A: {{"query_type": "filter_numeric_kg2", "params": {{"property": "elevationFt", "operator": ">", "threshold": 1000, "limit": 10}}}}

Q: "Show all large airports."
A: {{"query_type": "filter_string_kg2", "params": {{"property": "airportType", "value": "large_airport", "limit": 10}}}}

Q: "Which airports are located in Germany?"
A: {{"query_type": "filter_string_kg2", "params": {{"property": "countryName", "value": "Germany", "limit": 10}}}}

Q: "Show airports whose municipality is Vienna."
A: {{"query_type": "filter_string_kg2", "params": {{"property": "municipality", "value": "Vienna", "limit": 10}}}}

Q: "What are the top 5 airports with the highest elevation?"
A: {{"query_type": "ranking_kg2", "params": {{"property": "elevationFt", "order": "DESC", "limit": 5}}}}

Q: "Which airport has the shortest runway?"
A: {{"query_type": "ranking_kg2", "params": {{"property": "lengthFt", "order": "ASC", "limit": 1}}}}

Q: "Compare VIE and FRA by elevation."
A: {{"query_type": "compare_two_airports", "params": {{"airport1": "VIE", "airport2": "FRA", "property": "elevationFt"}}}}

Q: "Compare LHR and MAD by runway width."
A: {{"query_type": "compare_two_airports", "params": {{"airport1": "LHR", "airport2": "MAD", "property": "widthFt"}}}}

Q: "How many flights are operated by Lufthansa?"
A: {{"query_type": "count_kg1", "params": {{"filter_property": "hasAirline", "filter_value": "Lufthansa", "mode": "count"}}}}

Q: "Which flights have a ground speed above 400 knots?"
A: {{"query_type": "filter_numeric_kg1", "params": {{"property": "gspeed", "operator": ">", "threshold": 400, "limit": 10}}}}

Q: "List flights with altitude above 30000 feet."
A: {{"query_type": "filter_numeric_kg1", "params": {{"property": "alt", "operator": ">", "threshold": 30000, "limit": 10}}}}

Q: "Which flights land at airports with elevation above 800 feet?"
A: {{"query_type": "cross_kg_filter", "params": {{"direction": "destination", "airport_property": "elevationFt", "operator": ">", "threshold": 800, "limit": 10}}}}

Q: "Which flights arrive at airports located in Germany?"
A: {{"query_type": "cross_kg_filter", "params": {{"direction": "destination", "airport_property": "countryName", "operator": "=", "threshold": "Germany", "limit": 10}}}}

Q: "Which flights land at large airports?"
A: {{"query_type": "cross_kg_filter", "params": {{"direction": "destination", "airport_property": "airportType", "operator": "=", "threshold": "large_airport", "limit": 10}}}}

── NOW CLASSIFY THIS QUESTION ────────────────────────────────────────────────

Question: "{question}"

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
# DETERMINISTIC DETECTORS (kept as safety net)
# ─────────────────────────────────────────────

def _detect_flight_number(q: str):
    """Regex-based flight number detection. Fast and reliable."""
    m = _FLIGHT_RE.findall(q.upper())
    return max(m, key=len) if m else None


def _detect_airport_entity(q: str):
    """
    Dictionary lookup + fuzzy match for airport names and IATA codes.
    This is deterministic and does not call the LLM.
    """
    q_norm = _normalise(q)
    tokens = q_norm.split()

    # Exact phrase match (longest first)
    for size in range(6, 0, -1):
        for i in range(len(tokens) - size + 1):
            phrase = " ".join(tokens[i: i + size])
            if phrase in _AIRPORT_ENTITIES:
                return _AIRPORT_ENTITIES[phrase]

    # IATA code match
    for code in _IATA_RE.findall(q.upper()):
        if code in _AIRPORT_ENTITIES:
            return _AIRPORT_ENTITIES[code]

    # Fuzzy match on remaining candidates
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

def _llm_classify(question: str) -> dict:
    """
    Calls the LLM once to classify the question and extract parameters.
    Returns a dict with "query_type" and "params", or {} on failure.

    This replaces the entire signal-word matching system from v4.
    The LLM handles all phrasing variations, synonyms, and languages
    naturally — no lexicon maintenance required.
    """
    prompt = _CLASSIFICATION_PROMPT.replace("{question}", question)
    try:
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response["message"]["content"].strip()

        # Strip markdown fences if present
        raw = re.sub(r"```json|```", "", raw).strip()

        # Extract the first {...} block even if the LLM added surrounding text.
        # re.DOTALL makes "." match newlines, so nested objects are captured.
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            print(f"[router] LLM returned no JSON: {repr(raw[:80])}")
            return {}

        result = json.loads(match.group())
        print(f"[router] LLM classified as: {result.get('query_type')} | params: {result.get('params')}")
        return result

    except Exception as e:
        print(f"[router] LLM classification failed: {e}")
        return {}

# ─────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────

def route(question: str) -> dict:
    """
    Routes a question to the correct handler.

    Priority order:
      1. Minimum structure guard  → out_of_scope
      2. Flight number regex      → single_kg1
      3. LLM classification       → template | single_kg2 | out_of_scope
      4. Airport entity fallback  → single_kg2  (if LLM fails)
      5. Airport keyword fallback → single_kg2  (last resort)
      6. Default                  → out_of_scope
    """

    # ── Guard ─────────────────────────────────────────────────────────────────
    if not _has_minimum_structure(question):
        return {
            "query_type": "out_of_scope",
            "kg": None, "entity": None,
            "direction": None, "template": None, "config": None,
        }

    # ── Priority 1: Flight number (regex — fast, no LLM needed) ───────────────
    # A flight number like "KE567" or "OS214" is unambiguous.
    # We keep this deterministic because flight numbers have a strict format.
    flight = _detect_flight_number(question)
    if flight:
        return {
            "query_type": "single_kg1",
            "kg":         "flights",
            "entity":     flight,
            "direction":  None,
            "template":   None,
            "config":     KG_REGISTRY["flights"],
        }

    # ── Priority 2: LLM classification ────────────────────────────────────────
    # The LLM handles all template types, cross-KG filters, and single airport
    # lookups. It is the core of the new routing logic.
    classified = _llm_classify(question)
    query_type = classified.get("query_type", "")
    params     = classified.get("params", {})

    # ── Template branch ───────────────────────────────────────────────────────
    if query_type in TEMPLATE_REGISTRY:
        cfg = TEMPLATE_REGISTRY[query_type]
        return {
            "query_type": "template",
            "kg":         cfg["kg"],
            "entity":     None,
            "direction":  params.get("direction"),   # used by cross_kg_filter
            "template":   query_type,
            "config":     cfg,
            "params":     params,                    # pass params downstream
        }

    # ── Single airport branch (LLM detected a specific airport) ───────────────
    if query_type == "single_kg2":
        entity = params.get("entity")
        # Resolve the entity string to an IATA code if possible
        if entity:
            entity_upper = entity.upper().strip()
            # Direct IATA lookup first
            if entity_upper in _AIRPORT_ENTITIES:
                entity = _AIRPORT_ENTITIES[entity_upper]
            else:
                # Try normalised name lookup
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

    # ── Priority 3: Airport entity fallback ───────────────────────────────────
    # If the LLM failed or returned out_of_scope, try deterministic detection.
    # This preserves backward compatibility with the v4 behavior.
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

    # ── Priority 4: Airport keyword fallback ──────────────────────────────────
    if _detect_airport_keyword(question):
        return {
            "query_type": "single_kg2",
            "kg":         "airports",
            "entity":     None,
            "direction":  None,
            "template":   None,
            "config":     KG_REGISTRY["airports"],
        }

    # ── Default ───────────────────────────────────────────────────────────────
    return {
        "query_type": "out_of_scope",
        "kg":         None,
        "entity":     None,
        "direction":  None,
        "template":   None,
        "config":     None,
    }