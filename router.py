"""
router.py  (v10 — merged: word boundary fix + ASK detection)
------------------------------------------------
WHAT CHANGED vs v9:

    Fix — Two duplicate `route()` definitions existed in v9 (an incomplete
    draft added while wiring up ASK detection, and the original complete
    pipeline). Python silently used only the second, meaning ASK-style
    questions ("Is BR62's callsign EVA062?") never reached the new
    detection logic at all — they fell through to ordinary flight-number
    routing instead. This version merges the two into a single route(),
    with ASK detection running as Priority 1.5, between the structure
    guard and flight-number detection.

    Also carried over from v9: _has_kg1_signal() uses word boundaries for
    single-word signals, so "gate" no longer matches inside "navigate" or
    "aggregate".

ARCHITECTURE OVERVIEW
---------------------
The router is the front door of the NL2SPARQL pipeline. Its sole job is
to decide which branch handles a given question. It does NOT extract
properties, generate SPARQL, or query any endpoint.

Branches:
    ask_query   → yes/no question about a known entity's property value
    single_kg1  → a specific flight is asked about (flight number present,
                  question asks about the flight itself)
    single_kg2  → a specific airport is asked about by name or IATA code
    single_kg3  → a specific university entity is asked about
    cross_kg    → a specific flight is asked about, but the question asks
                  about a property of its airport (country, elevation, etc.)
    template    → an aggregate question with no specific entity
                  (filter, ranking, comparison, count)
    open_kg     → answerable from the KG but doesn't fit any template
    out_of_scope → cannot be answered from this database

ROUTING LOGIC
-------------
Priority 1   — Minimum structure guard
    Rejects single-word or meaningless inputs immediately.

Priority 1.5 — ASK-style question + known entity (any KG)
    Questions starting with an ASK signal ("is", "are", "does", "was",
    "est-ce que", "هل", ...) that also name a known flight, airport, or
    university entity are routed straight to ask_query. These questions
    naturally mention two entity-shaped strings (the subject and the
    comparison value), so flight-number detection here uses the
    FIRST match, not the longest — see _detect_flight_number_first().

Priority 2   — Flight number detected (regex)
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

Priority 2.5 — Airport entity detected (deterministic)

Priority 2.7 — University entity detected (deterministic), unless the
    question has a count/list or filter signal, in which case it falls
    through to the LLM so it can reach count_kg3 / filter_string_kg3.

Priority 3   — No flight/airport/university entity match →
    LLM classifies everything else: template branch, single_kg2, open_kg,
    or out_of_scope, with smart reroutes for known misclassification
    patterns and a final clean-gate check via _is_kg_answerable().

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

WHAT CHANGED vs v10 (this revision — majority-vote gates)
-----------------------------------------------------------
    test_broken_rows.py caught real sampling variance in both
    _has_ask_signal() and _is_kg_answerable(): the identical question
    "Is BLQ located in France?" returned different answers across 5
    calls to the same prompt (4 False, 1 True). That's LLM sampling
    noise, not a logic bug — no amount of prompt-tuning fixes it
    reliably. Both gates now vote across 3 independent calls
    (_llm_yes_no_majority) and take the majority answer instead of
    trusting a single unconstrained sample. This triples the latency
    cost of these two specific checks, but they're cheap binary calls,
    and reliability matters more than shaving a couple seconds here.
"""

from rapidfuzz import process, fuzz
from template_resolver import KG2_NUMERIC_PROPS, KG2_STRING_PROPS
import re
import ast
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
# the question cannot be cross-KG - return single_kg1 without the LLM.
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
    # City of the FLIGHT itself (hasOriginCity / hasDestinationCity — a
    # KG1 literal, not the destination airport's own properties). Without
    # these, "What is the departure city of flight X?" falls through to
    # the LLM, where the cross_kg_filter few-shot examples (all about the
    # destination airport's country/elevation/type) dominate and the
    # question gets misrouted to cross_kg.
    "departure city", "origin city", "destination city",
    "ville de départ", "ville de destination", "ville d'origine",
    "مدينة مغادرة", "مدينة انطلاق", "مدينة وصول",
}

# Aircraft-registration questions mention a flight number (so Priority 2
# fires) but must be answered from open_kg, not single_kg1/kg2/cross_kg —
# the registration number lives on the Aircraft node, outside all three
# templated branches. Left to the LLM, this was misrouted a different way
# in every language (single_kg1 in en/fr, cross_kg in ar), because no
# few-shot example in the prompt covers this question type at all.
_OPEN_KG_SIGNALS = {
    "registration number", "registration no",
    "numéro d'immatriculation", "numero d'immatriculation", "immatriculation",
    "رقم التسجيل", "رقم تسجيل",
}

def _has_open_kg_signal(q_lower: str) -> bool:
    return any(sig in q_lower for sig in _OPEN_KG_SIGNALS)

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
          Word-boundary regex (\\b). Prevents matching "gate" inside
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

def _detect_airport_keyword(q: str) -> bool:
    q_lower = q.lower()
    return any(k.lower() in q_lower for k in _AIRPORT_TRIGGERS)

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

10. group_aggregate_kg1 — aggregate a numeric flight property, grouped by airline
    params: group_by ("airline"), property (gspeed or vspeed),
            function (AVG, SUM, MAX, or MIN)

11. group_aggregate_kg2 — aggregate a numeric airport property, grouped by
    country or continent
    params: group_by ("country" or "continent"), property
            (elevationFt, lengthFt, or widthFt), function (AVG, SUM, MAX, or MIN)

12. group_aggregate_kg3 — aggregate a COUNT of a relation (courses taught,
    courses taken), grouped by department
    params: group_by ("department"), property (teacherOf or takesCourse),
            function (AVG, MAX, or MIN — no SUM, since summing counts of
            counts is rarely a meaningful question)   
13. open_kg — the question is about aviation data but does not fit any
    template above. It asks about a specific property or relationship
    that requires a custom query.
    params: {}
    
    Examples:
    - "Which flight has the highest ground speed?" → ranking, use ranking_kg2 or filter_numeric_kg1
    - "How many airports are in the dataset?" → open_kg
    - "Which airports have a grass runway?" → open_kg
    - "What is the registration number of the aircraft on flight BR62?" → open_kg
    - "Quel vol a la vitesse verticale la plus basse?" → open_kg

14. count_kg3 — count or list university entities linked to a specific
    named entity (professor, student, department). Only applies when the
    question BOTH names a specific entity (e.g. "FullProfessor0",
    "Department0") AND asks "how many" / "list all", not a single fact.
    params: property, direction, mode    
15. filter_string_kg3 — list university people (professors/students)
    belonging to a specific NAMED department, filtered by relationship
    type. Use when the question asks "which/who" belongs to a department
    by name, not about one already-named person.
    params: property, value, limit

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
  "grass runway", "asphalt runway", "paved runway", "runway surface",
  "runway material", "what is the runway made of"   → surface
      (value: the surface type mentioned — "grass", "asphalt", "concrete",
       etc. Always use the property name "surface" exactly — never invent
       names like "hasGrassRunway", "pavementType", or "runwayMaterial".)

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

University properties (only when a LUBM entity name like "FullProfessor0",
"Department0", "GraduateStudent3" appears in the question):
  "teach", "courses taught"                    → property=teacherOf, direction=outgoing
  "take", "enrolled in", "courses taken"       → property=takesCourse, direction=outgoing
  "students", "members" (of a department)      → property=memberOf, direction=incoming
  "professors", "faculty", "staff" (of a dept) → property=worksFor, direction=incoming
  "departments" (of a university)              → property=subOrganizationOf, direction=incoming  
Group-by / aggregate signal words:
  "average", "mean"                            → function=AVG
  "total", "sum"                                → function=SUM
  "highest", "most", "maximum"                  → function=MAX
  "lowest", "least", "minimum"                  → function=MIN
  "per airline", "by airline", "for each airline" → group_by=airline
  "per country", "by country"                   → group_by=country
  "per continent", "by continent"               → group_by=continent
  "per department", "by department"             → group_by=department
  Group-by / aggregate signal words:
  "average", "mean"                                          → function=AVG
  "moyenne", "moyen"                                         → function=AVG
  "متوسط", "معدل"                                            → function=AVG

  "total", "sum"                                              → function=SUM
  "total", "somme"                                            → function=SUM
  "مجموع", "إجمالي"                                          → function=SUM

  "highest", "most", "maximum"                                → function=MAX
  "le plus élevé", "maximum", "le plus"                       → function=MAX
  "الأعلى", "الأكثر", "أقصى"                                  → function=MAX

  "lowest", "least", "minimum"                                → function=MIN
  "le plus bas", "minimum", "le moins"                        → function=MIN
  "الأدنى", "الأقل", "أدنى"                                   → function=MIN

  "per airline", "by airline", "for each airline"             → group_by=airline
  "par compagnie", "par compagnie aérienne"                   → group_by=airline
  "لكل شركة طيران", "حسب شركة الطيران"                        → group_by=airline

  "per country", "by country"                                 → group_by=country
  "par pays"                                                  → group_by=country
  "لكل دولة", "حسب الدولة", "بحسب الدولة"                     → group_by=country

  "per continent", "by continent"                             → group_by=continent
  "par continent"                                             → group_by=continent
  "لكل قارة", "حسب القارة"                                    → group_by=continent

  "per department", "by department"                           → group_by=department
  "par département"                                           → group_by=department
  "لكل قسم", "حسب القسم", "بحسب القسم"                        → group_by=department
  IMPORTANT: distinguish group_aggregate from ranking_kg2/filter_numeric_kg1.
  Ranking questions ask for the top/bottom N individual entities
  ("which airport has the highest elevation?" → ranking_kg2).
  Group-aggregate questions ask for a computed value PER CATEGORY
  ("what is the average elevation per country?" → group_aggregate_kg2).
  The word "per", "by", "for each", or "grouped by" is the strongest signal.
── DISAMBIGUATION RULES ──────────────────────────────────────────────────────

CROSS_KG_FILTER: Use when a specific flight number is mentioned AND the
  question asks about a property of that flight's airport (country,
  elevation, runway, type). Examples:
    "What country does flight LO225 land in?"           → cross_kg_filter
    "What type of airport does flight FR182 arrive at?" → cross_kg_filter
    "Dans quel pays atterrit le vol OS295?"             → cross_kg_filter
    "في أي دولة يهبط الرحلة OS235؟"                   → cross_kg_filter

COUNT_KG1: Use when the question counts or lists FLIGHTS specifically.
  "how many flights" / "combien de vols" / "كم رحلة" → always count_kg1,
  even if a city name is present.
  Do NOT use count_kg1 to count airports or runways — KG1 has no runway
  data at all. "How many runways are closed?" / "how many airports..."
  belong to open_kg, even though they also start with "how many".
OPEN_KG: Use when the question asks about aviation data that exists in the
  KG but does not fit filter_numeric, filter_string, ranking, compare,
  count, or cross_kg_filter patterns. Specifically:
  - Questions about aircraft registration or specific aircraft details
  - Questions asking for a count of a KG class (airports, runways)
  - Questions about runway surface types (grass, concrete)
  - Questions about closed runways
  Do NOT use open_kg when filter_numeric_kg1 or ranking_kg2 would work.
 COUNT_KG3: Use when the question names a specific university entity AND
  counts or lists something linked to it.
  "how many courses does X teach" / "combien de cours enseigne X" / "كم مادة يدرّس X"
  → always count_kg3, even though X is a specific entity — the count/list
  intent takes priority over single-entity lookup. 

  
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
Q: "How many flights arrive in Vienna?"
A: {{"query_type": "count_kg1", "params": {{"filter_property": "hasDestinationCity", "filter_value": "Vienna", "mode": "count"}}}}
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

Q: "What is the weather forecast for JFK tomorrow?"
A: {{"query_type": "out_of_scope", "params": {{}}}}

Q: "Am I allowed to bring a guitar on flight BR62?"
A: {{"query_type": "out_of_scope", "params": {{}}}}

Q: "Who invented the first commercial airplane?"
A: {{"query_type": "out_of_scope", "params": {{}}}}

Q: "Quel est le prix du billet pour le vol AF123?"
A: {{"query_type": "out_of_scope", "params": {{}}}}

Q: "هل تقدم شركة الطيران وجبات نباتية؟"
A: {{"query_type": "out_of_scope", "params": {{}}}}

Q: "How many courses does FullProfessor0 teach?"
A: {{"query_type": "count_kg3", "params": {{"property": "teacherOf", "direction": "outgoing", "mode": "count"}}}}

Q: "List the courses that GraduateStudent3 takes."
A: {{"query_type": "count_kg3", "params": {{"property": "takesCourse", "direction": "outgoing", "mode": "list"}}}}

Q: "How many students are in Department0?"
A: {{"query_type": "count_kg3", "params": {{"property": "memberOf", "direction": "incoming", "mode": "count"}}}}

Q: "Which professors work for Department3?"
A: {{"query_type": "filter_string_kg3", "params": {{"property": "worksFor", "value": "Department3", "limit": 10}}}}

Q: "List students who are members of Department1."
A: {{"query_type": "filter_string_kg3", "params": {{"property": "memberOf", "value": "Department1", "limit": 10}}}}
Q: "What is the average ground speed per airline?"
A: {{"query_type": "group_aggregate_kg1", "params": {{"group_by": "airline", "property": "gspeed", "function": "AVG"}}}}

Q: "What is the maximum elevation per country?"
A: {{"query_type": "group_aggregate_kg2", "params": {{"group_by": "country", "property": "elevationFt", "function": "MAX"}}}}

Q: "Quelle est la longueur de piste moyenne par pays?"
A: {{"query_type": "group_aggregate_kg2", "params": {{"group_by": "country", "property": "lengthFt", "function": "AVG"}}}}

Q: "Which department teaches the most courses on average per professor?"
A: {{"query_type": "group_aggregate_kg3", "params": {{"group_by": "department", "property": "teacherOf", "function": "AVG"}}}}

Q: "Which airports have a grass runway?"
A: {{"query_type": "filter_string_kg2", "params": {{"property": "surface", "value": "grass", "limit": 10}}}}

Q: "How many runways in the dataset are closed?"
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
    """Regex-based. Fast and deterministic. Called before any LLM.

    Returns the LONGEST match. Correct for ordinary questions, which
    only ever mention one flight number. Do NOT use this for ASK-style
    comparison questions ("Is BR62's callsign EVA062?") — those mention
    two entity-shaped strings and need _detect_flight_number_first()
    instead, since "longest wins" would pick the comparison value
    instead of the actual subject.
    """
    m = _FLIGHT_RE.findall(q.upper())
    return max(m, key=len) if m else None

def _detect_flight_number_first(q: str):
    """
    ASK-specific variant: returns the FIRST match, not the longest.
    ASK questions naturally mention two entity-shaped strings (the
    subject and the comparison value) — the subject always comes first.
    """
    m = _FLIGHT_RE.findall(q.upper())
    return m[0] if m else None

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

_UNIVERSITY_ENTITY_RE = re.compile(
    r'\b((?:[A-Z][a-zA-Z]*)?(?:Professor|Student|Course|Department|Group|'
    r'University|Lecturer|Publication)\d+)\b'
)

def _detect_university_entity(q: str):
    """
    Regex-based, like flight numbers. LUBM entity names are synthetic and
    fully regular (TypeName + digits), so no fuzzy matching is needed.
    Returns the matched name string (e.g. "FullProfessor0"), not a URI —
    the actual URI (with its department) is resolved later, by querying
    Fuseki for the entity whose ub:name equals this string.
    """
    m = _UNIVERSITY_ENTITY_RE.search(q)
    return m.group(1) if m else None

_COUNT_SIGNALS = ["how many", "combien de", "كم", "list all", "count"]
_FILTER_SIGNALS = ["which professors", "which students", "who works for",
                    "who is a member of", "list the professors", "list the students",
                    "quels professeurs", "quels étudiants", "أي أستاذ", "أي طالب"]

def _has_filter_signal(q: str) -> bool:
    q_lower = q.lower()
    return any(sig in q_lower for sig in _FILTER_SIGNALS)

def _has_count_signal(q: str) -> bool:
    """
    Word-boundary-safe, same pattern as _has_kg1_signal(). Plain
    substring matching let "count" (from _COUNT_SIGNALS) match inside
    "country" — e.g. "What country is CTA located in?" was silently
    treated as a count question, skipping Priority 2.5 entirely and
    falling through to the LLM, which misrouted it to cross_kg_filter.
    """
    q_lower = q.lower()
    for sig in _COUNT_SIGNALS:
        if " " in sig:
            if sig in q_lower:
                return True
        else:
            if re.search(rf"\b{re.escape(sig)}\b", q_lower):
                return True
    return False

_COMPARE_SIGNALS = ["compare", "comparer", "comparez", "vs", "versus", "قارن"]

def _has_compare_signal(q: str) -> bool:
    """Same word-boundary fix as _has_count_signal(), applied preventively —
    not yet observed causing a bug, but same substring-match risk."""
    q_lower = q.lower()
    for sig in _COMPARE_SIGNALS:
        if " " in sig:
            if sig in q_lower:
                return True
        else:
            if re.search(rf"\b{re.escape(sig)}\b", q_lower):
                return True
    return False


# ─────────────────────────────────────────────
# LLM MAJORITY-VOTE HELPER
# ─────────────────────────────────────────────

def _llm_yes_no_majority(prompt: str, k: int = 3) -> bool:
    """
    Calls ollama.chat() k times on the same yes/no prompt and returns the
    majority vote, instead of trusting a single unconstrained sample.

    WHY: test_broken_rows.py caught a real 4-1 split on the identical
    question "Is BLQ located in France?" — same prompt, different answers
    across calls. That's sampling variance, not a logic bug, so no amount
    of prompt-tuning fixes it reliably. A single yes/no LLM call is cheap
    enough that voting 3x is a small cost for removing coin-flip behavior
    on borderline phrasings.

    k=3 (odd, so no ties) is a deliberate minimum — enough to smooth out
    single-sample noise without tripling latency for every question in
    the pipeline (only _has_ask_signal and _is_kg_answerable use this).
    """
    votes = []
    for _ in range(k):
        try:
            response = ollama.chat(
                model="llama3",
                messages=[{"role": "user", "content": prompt}]
            )
            votes.append(response["message"]["content"].strip().upper().startswith("YES"))
        except Exception as e:
            print(f"[router] _llm_yes_no_majority call failed: {e}")
            votes.append(False)
    result = sum(votes) > len(votes) / 2
    if len(set(votes)) > 1:
        print(f"[router] _llm_yes_no_majority: split vote {votes} → {result}")
    return result


def _has_ask_signal(question: str) -> bool:
    """
    Asks the LLM whether this question is a yes/no CONFIRMATION question
    (asserting a specific value and asking to confirm it) versus an OPEN
    information-seeking question. This replaces surface-pattern matching
    (keyword lists, language-specific regexes for inversion, etc.) with
    semantic classification, so it generalizes to any language or
    phrasing — including constructions we have never seen — rather than
    requiring every new phrasing to be added by hand.

    Uses majority-vote sampling (_llm_yes_no_majority) rather than a
    single call — identical questions were observed to get different
    answers across runs (see test_broken_rows.py, "Is BLQ located in
    France?": 4/5 False, 1/5 True on a single-call basis).

    ARABIC FAST-PATH: "هل" is a dedicated Arabic question-particle that
    marks yes/no questions specifically (comparable to English question-
    inversion, but a single fixed word at a fixed position, so it's far
    more reliable than the surrounding grammar). LLM voting on Arabic
    text was observed to be unstable — the identical question sometimes
    routed True, sometimes False, across separate runs (see eval log,
    ask_query_003 ar). Since "هل" is a deterministic linguistic marker,
    we check for it first and skip the LLM vote entirely when present,
    removing that instability for the common case. This is a shortcut,
    not an override: if the question does NOT start with "هل", we still
    fall through to the existing LLM voting unchanged — so this only
    ever helps Arabic ask-style questions, and never touches English,
    French, or Arabic questions phrased without "هل".
    """
    if question.strip().startswith("هل"):
        print(f"[router] _has_ask_signal('{question[:40]}...') → True (fast-path: 'هل' prefix)")
        return True

    prompt = f"""Does this question ask to CONFIRM whether a specific
property already has a specific value (a yes/no question)? Or does it
ASK for information (what/which/how much)?

Examples:
Q: "Is BR62's callsign EVA062?" → YES
Q: "La porte du vol OS830 est-elle A17?" → YES
Q: "هل مطار فيينا يقع في النمسا؟" → YES
Q: "What is the callsign of BR62?" → NO
Q: "Quelle est la porte du vol OS830?" → NO
Q: "ما هو مطار الوصول؟" → NO
Q: "هل يقع مطار زيورخ في سويسرا؟" → YES
Q: "في أي دولة يقع مطار أثينا؟" → NO

Answer only YES or NO.

Question: "{question}"
"""
    try:
        result = _llm_yes_no_majority(prompt)
        print(f"[router] _has_ask_signal('{question[:40]}...') → {result}")
        return result
    except Exception as e:
        print(f"[router] _has_ask_signal LLM check failed: {e}")
        return False

# ─────────────────────────────────────────────
# LLM CLASSIFIER
# ─────────────────────────────────────────────

def _llm_classify(question: str, max_attempts: int = 2) -> dict:
    prompt = _CLASSIFICATION_PROMPT.replace("{question}", question)

    for attempt in range(max_attempts):

        raw = ""
        try:
            response = ollama.chat(
                model="llama3",
                messages=[{"role": "user", "content": prompt}]
            )
            raw = response["message"]["content"].strip()
            raw = re.sub(r"```json|```", "", raw).strip()

            match = re.search(r"\{.*\}", raw, re.DOTALL)
            candidate = match.group() if match else raw

            try:
                result = json.loads(candidate)
            except json.JSONDecodeError:
                # llama3 sometimes echoes Python-dict style (single quotes)
                # instead of strict JSON — more common for fr/ar prompts.
                # ast.literal_eval is a real parser, not a string
                # replace(), so a genuine apostrophe inside a value
                # (e.g. "l'aéroport") doesn't get mangled.
                result = ast.literal_eval(candidate)

            result["query_type"] = _normalize_query_type(result.get("query_type", ""), question)
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

    Uses majority-vote sampling (_llm_yes_no_majority) rather than a
    single call, for the same reliability reason as _has_ask_signal().

    Returns True if the majority of calls say YES, False otherwise.
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

The knowledge graph does NOT contain: weather, prices/tickets, passenger
policies (pets, baggage, check-in), history, news, opinions, safety
records, or anything not explicitly listed above.

EXAMPLES:
Q: "What is the weather forecast for JFK tomorrow?"     → NO (weather not in KG)
Q: "Am I allowed to bring a guitar on flight BR62?"     → NO (policy, not in KG)
Q: "Who invented the first commercial airplane?"        → NO (general knowledge, not in KG)
Q: "Can I bring a pet on flight FR947?"           → NO (policy, not in KG)
Q: "What is the history of the airline industry?" → NO (general knowledge, not in KG)
Q: "Quel temps fait-il à l'aéroport VIE?"          → NO
Q: "هل يمكنني اصطحاب حيوان أليف؟"                  → NO
Q: "What is the elevation of ZRH?"                → YES (elevation is in KG)
Q: "Is ZRH located in Switzerland?"               → YES (country is in KG)
Q: "هل يقع مطار زيورخ في سويسرا؟"                  → YES
Q: "في أي دولة يقع مطار أثينا؟" → NO
Answer only YES or NO:
Can this question be answered using only the data described above?

Question: "{question}"
"""
    try:
        result = _llm_yes_no_majority(prompt)
        print(f"[router] _is_kg_answerable('{question[:40]}...') → {result}")
        return result
    except Exception as e:
        print(f"[router] _is_kg_answerable failed: {e}")
        return False

# ─────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────
_VALID_QUERY_TYPES = set(TEMPLATE_REGISTRY.keys()) | {"single_kg2", "open_kg", "out_of_scope"}

def _normalize_query_type(query_type, question: str = "") -> str:
    """
    Corrects a query_type the LLM invented but that isn't actually
    registered. Two-stage:

    1. Family match — strip the trailing "_kgN" and look for the one
       real registered type sharing the same base name. Handles the
       dominant failure mode: right template family, wrong KG number
       ("filter_string_kg1" meant to be "_kg2"). Plain fuzzy matching
       can't do this reliably — "kg1"→"kg2" and "kg1"→"kg3" are both
       single-character edits, so the score ties and picks arbitrarily
       (this is what went wrong last run: kg1 → kg3 instead of kg2).

    2. If the family has multiple real KG variants (e.g. filter_string
       exists for both kg2 and kg3), disambiguate using which KG the
       question actually names — reusing the same entity/keyword
       detectors already used elsewhere in this file, not new logic.

    3. Fallback: plain fuzzy match, for genuinely unrelated typos that
       don't fit the "_kgN" pattern at all.
    """
    if not isinstance(query_type, str):
        # The LLM occasionally returns query_type as something other than
        # a plain string (e.g. a nested object). Treat as unclassified
        # rather than crash — "" falls through cleanly to the existing
        # clean gate later in route(), same as any other unrecognized type.
        print(f"[router] LLM returned non-string query_type "
              f"({type(query_type).__name__}); treating as unclassified")
        return ""
    if query_type in _VALID_QUERY_TYPES:
        return query_type

    base = re.sub(r"_kg\d+$", "", query_type)
    family = [t for t in _VALID_QUERY_TYPES
              if re.sub(r"_kg\d+$", "", t) == base and t != query_type]

    if len(family) == 1:
        corrected = family[0]
        print(f"[router] Correcting hallucinated query_type '{query_type}' "
              f"→ '{corrected}' (family match)")
        return corrected

    if len(family) > 1:
        if _detect_university_entity(question) or any(
            sig in question.lower() for sig in _FILTER_SIGNALS
        ):
            kg_guess = "university"
        elif _detect_airport_keyword(question) or _detect_airport_entity(question):
            kg_guess = "airports"
        else:
            kg_guess = None
        if kg_guess:
            for t in family:
                if TEMPLATE_REGISTRY[t]["kg"] == kg_guess:
                    print(f"[router] Correcting hallucinated query_type '{query_type}' "
                          f"→ '{t}' (kg={kg_guess})")
                    return t

    match = process.extractOne(query_type, list(_VALID_QUERY_TYPES), scorer=fuzz.WRatio)
    if match and match[1] >= 85:
        corrected, score, _ = match
        print(f"[router] Correcting hallucinated query_type '{query_type}' "
              f"→ '{corrected}' (score={score})")
        return corrected
    return query_type
    return query_type  # leave as-is — falls through to the clean gate, same as before
def route(question: str) -> dict:
    """
    Routes a natural language question to the correct pipeline branch.

    Returns a routing dict consumed by main.py and test_pipeline.py.
    Keys: query_type, kg, entity, direction, template, config, params.

    Final structure:
        Priority 1   → structure guard
        Priority 1.5 → ASK-style question + known entity → ask_query
        Priority 2   → flight number → single_kg1 or cross_kg
        Priority 2.5 → airport entity → single_kg2
        Priority 2.7 → university entity → single_kg3 (unless count/filter signal)
        Priority 3   → LLM classifies → template / single_kg2 / open_kg
                       (with smart reroute for known misclassification patterns)
        Clean gate   → _is_kg_answerable() → open_kg or out_of_scope
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

    # ── Priority 1.5: ASK-style question + known entity (any KG) ──────────────
    if _has_ask_signal(question):
        flight_entity      = _detect_flight_number_first(question)
        airport_entity     = _detect_airport_entity(question)
        university_entity  = _detect_university_entity(question)

        if flight_entity:
            return {
                "query_type": "ask_query", "kg": "flights",
                "entity": flight_entity, "direction": None, "template": "ask_query",
            }
        elif airport_entity:
            return {
                "query_type": "ask_query", "kg": "airports",
                "entity": airport_entity, "direction": None, "template": "ask_query",
            }
        elif university_entity:
            return {
                "query_type": "ask_query", "kg": "university",
                "entity": university_entity, "direction": None, "template": "ask_query",
            }

    # ── Priority 2: Flight number detected ────────────────────────────────────
    flight = _detect_flight_number(question)

    if flight:

        # ── Fast path (2a-pre): open_kg signal word present ────────────────────
        # Aircraft-registration questions mention a flight number but must
        # be answered from open_kg (registration lives on the Aircraft
        # node). Checked BEFORE the KG1-only check so it takes priority.
        if _has_open_kg_signal(q_lower):
            return {
                "query_type": "open_kg",
                "kg":         "cross",
                "entity":     None,
                "direction":  None,
                "template":   None,
                "config":     None,
            }

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
            direction = params.get("direction", "destination")
            return {"query_type": "cross_kg", "kg": "cross", "entity": flight,
                     "direction": direction, "template": None, "config": CROSS_KG_CONFIG}

        if query_type == "out_of_scope":          # NEW — respect it instead of forcing single_kg1
            return {"query_type": "out_of_scope", "kg": None, "entity": None,
                     "direction": None, "template": None, "config": None}

           # unchanged fallback for everything else

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
    print(f"[router] Priority 2.5 check: airport={airport!r}")
    if airport and not _has_compare_signal(question) and not _has_count_signal(question):
        # A known entity being present doesn't guarantee the QUESTION is
        # answerable about that entity — "What's the weather at VIE?"
        # also matches here. Reuse the same answerability gate Priority 3
        # already relies on, instead of a hardcoded keyword list that
        # could never cover every out-of-scope phrasing.
        if not _is_kg_answerable(question):
            return {
                "query_type": "out_of_scope",
                "kg":         None,
                "entity":     None,
                "direction":  None,
                "template":   None,
                "config":     None,
            }
        return {
            "query_type": "single_kg2",
            "kg":         "airports",
            "entity":     airport,
            "direction":  None,
            "template":   None,
            "config":     KG_REGISTRY["airports"],
        }

    # ── Priority 2.7: University entity detected (deterministic) ──────────────
    # Only short-circuits to single_kg3 for single-value lookups. Count/list
    # questions ("how many courses does X teach") fall through to the LLM
    # classifier so they can reach count_kg3 instead — same pattern already
    # used for count_kg1 (see COUNT_KG1 rule below: count signals win even
    # when a specific entity is named).
    university_entity = _detect_university_entity(question)
    if university_entity and not _has_count_signal(question) and not _has_filter_signal(question):
        return {
            "query_type": "single_kg3",
            "kg":         "university",
            "entity":     university_entity,
            "direction":  None,
            "template":   None,
            "config":     KG_REGISTRY["university"],
        }

    # ── Priority 3: No flight/airport/university match — LLM classifies ───────
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

        # Case 1.5: cross_kg_filter classified with no flight entity in the
        # question at all. Priority 2 already handles every question that
        # contains a flight number, so by the time execution reaches here
        # (Priority 3), there is no legitimate way for a question to need
        # cross_kg_filter — it can only be a misclassification. The LLM
        # reaches for it anyway when a question mentions an airport
        # property + threshold/limit that overlaps with cross_kg_filter's
        # own few-shot examples (e.g. "elevationFt > 800, limit 10").
        # Reroute to the matching airport-only template using whichever
        # property type the LLM actually extracted.
        if query_type == "cross_kg_filter":
            prop = params.get("airport_property", "")
            if prop in KG2_NUMERIC_PROPS:
                print(f"[router] Smart reroute: cross_kg_filter with no flight entity → filter_numeric_kg2")
                query_type = "filter_numeric_kg2"
                params = {"property": prop, "operator": params.get("operator", ">"),
                          "threshold": params.get("threshold"), "limit": params.get("limit", 10)}
                cfg = TEMPLATE_REGISTRY[query_type]
            elif prop in KG2_STRING_PROPS:
                print(f"[router] Smart reroute: cross_kg_filter with no flight entity → filter_string_kg2")
                query_type = "filter_string_kg2"
                params = {"property": prop, "value": params.get("threshold"), "limit": params.get("limit", 10)}
                cfg = TEMPLATE_REGISTRY[query_type]
            else:
                print(f"[router] Smart reroute: cross_kg_filter with no flight entity, unrecognised property → open_kg")
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
        # Case 4: ranking_kg2 received a categorical (non-numeric) property.
        # "Show all large airports" / "which airports are in Greece" ask for
        # a FILTER, not a ranking — the LLM sometimes reaches for ranking_kg2
        # anyway. If we can recover the target value, reroute to
        # filter_string_kg2 (same fix as Cases 1-3). If not, fall back to
        # open_kg rather than build a template with a missing required
        # field — same safety convention already used everywhere else here.
        if query_type == "ranking_kg2":
            prop = params.get("property", "")
            if prop not in KG2_NUMERIC_PROPS and prop in KG2_STRING_PROPS:
                value = params.get("value") or params.get("threshold")
                if value:
                    print(f"[router] Smart reroute: categorical property in ranking_kg2 → filter_string_kg2")
                    query_type = "filter_string_kg2"
                    params = {"property": prop, "value": value, "limit": params.get("limit", 10)}
                else:
                    print(f"[router] Smart reroute: categorical property in ranking_kg2, no value → open_kg")
                    return {
                        "query_type": "open_kg", "kg": "cross", "entity": None,
                        "direction": None, "template": None, "config": None,
                    }
        # Case 5: filter_string_kg2 received a NUMERIC property
        # (elevationFt/lengthFt/widthFt) — the reverse of Case 4. Observed
        # in real eval runs: a threshold phrase like "runway longer than
        # 10000 feet" sometimes gets filed under filter_string_kg2 even
        # though the property extraction itself is correct and numeric.
        # The LLM sometimes already extracts a separate "operator" key
        # despite the wrong top-level category (seen in the fr run); when
        # it doesn't, parse a leading >/</>=/<= off the "value" string
        # instead. Reroute deterministically rather than trust either
        # shape blindly.
        if query_type == "filter_string_kg2":
            prop = params.get("property", "")
            if prop in KG2_NUMERIC_PROPS:
                raw_value = str(params.get("value", ""))
                operator  = params.get("operator")
                threshold = None
                if operator in (">", "<", ">=", "<="):
                    threshold = raw_value
                else:
                    m = re.match(r"\s*(>=|<=|>|<)?\s*(-?\d+(?:\.\d+)?)", raw_value)
                    if m:
                        operator  = m.group(1) or ">"
                        threshold = m.group(2)
                if threshold is not None:
                    print(f"[router] Smart reroute: filter_string_kg2 with numeric property → filter_numeric_kg2")
                    query_type = "filter_numeric_kg2"
                    params = {"property": prop, "operator": operator,
                              "threshold": threshold, "limit": params.get("limit", 10)}
                    cfg = TEMPLATE_REGISTRY[query_type]
                else:
                    print(f"[router] Smart reroute: filter_string_kg2 with numeric property, "
                          f"no parseable threshold → open_kg")
                    return {
                        "query_type": "open_kg", "kg": "cross", "entity": None,
                        "direction": None, "template": None, "config": None,
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
    # ── out_of_scope branch — LLM already said no, trust it ───────────────────
    # Without this check, an explicit out_of_scope answer from _llm_classify()
    # falls through to the clean gate below, which asks a SECOND, independent
    # question and can overrule the first answer. Two classifiers voting,
    # only the second counted — this silently discarded correct
    # out_of_scope classifications.
    if query_type == "out_of_scope":
        return {
            "query_type": "out_of_scope",
            "kg":         None,
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