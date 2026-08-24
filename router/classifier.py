"""
LLM-based classification and query-type normalization for the NL2SPARQL router.
"""

import ast
import json
import re

import ollama
from rapidfuzz import process, fuzz

from kg_registry import TEMPLATE_REGISTRY, get_open_kg_schema
from .rules import _WH_WORDS
from router.detectors import  (
    _detect_flight_number_first,
    _detect_airport_entity,
    _detect_airport_keyword,
    _detect_university_entity,
    _has_filter_signal,
)

_VALID_QUERY_TYPES = set(TEMPLATE_REGISTRY.keys()) | {"single_kg2", "open_kg", "out_of_scope"}

# ── CLASSIFICATION PROMPT ────────────────────────────────────────
_CLASSIFICATION_PROMPT = """You are a query classifier for an airport and flight database.

Classify the question into exactly one of these query types and extract its parameters.

── QUERY TYPES AND THEIR PARAMETERS ──────────────────────────────────────────

1. filter_numeric_kg2 — airports filtered by a numeric property
   params: property, operator, threshold

2. filter_string_kg2 — airports filtered by a text/categorical property
   params: property, value

3. ranking_kg2 — airports ranked by a numeric property (top/bottom N)
   params: property, order (ASC or DESC), limit (default 5)

4. compare_two_airports — compare exactly two airports on one property
   params: airport1 (IATA code), airport2 (IATA code), property

5. count_kg1 — count or list flights matching a condition
   params: filter_property, filter_value, mode (count or list)

6. filter_numeric_kg1 — flights filtered by a numeric flight property
   params: property, operator, threshold

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
  "students", "members" (of a department)    → property=memberOf, direction=incoming
  "professors", "faculty", "staff" (of a dept) → property=worksFor, direction=incoming
  "departments" (of a university)             → property=subOrganizationOf, direction=incoming

Group-by / aggregate signal words:
  "average", "mean"                            → function=AVG
  "moyenne", "moyen"                           → function=AVG
  "متوسط", "معدل"                              → function=AVG
  "total", "sum"                               → function=SUM
  "total", "somme"                             → function=SUM
  "مجموع", "إجمالي"                            → function=SUM
  "highest", "most", "maximum"                 → function=MAX
  "le plus élevé", "maximum", "le plus"      → function=MAX
  "الأعلى", "الأكثر", "أقصى"                   → function=MAX
  "lowest", "least", "minimum"                 → function=MIN
  "le plus bas", "minimum", "le moins"         → function=MIN
  "الأدنى", "الأقل", "أدنى"                    → function=MIN
  "per airline", "by airline", "for each airline" → group_by=airline
  "par compagnie", "par compagnie aérienne"    → group_by=airline
  "لكل شركة طيران", "حسب شركة الطيران"         → group_by=airline
  "per country", "by country"                  → group_by=country
  "par pays"                                   → group_by=country
  "لكل دولة", "حسب الدولة", "بحسب الدولة"      → group_by=country
  "per continent", "by continent"              → group_by=continent
  "par continent"                              → group_by=continent
  "لكل قارة", "حسب القارة"                     → group_by=continent
  "per department", "by department"            → group_by=department
  "par département"                            → group_by=department
  "لكل قسم", "حسب القسم", "بحسب القسم"         → group_by=department

  IMPORTANT: distinguish group_aggregate from ranking_kg2/filter_numeric_kg1.
  Ranking questions ask for the top/bottom N individual entities.
  Group-aggregate questions ask for a computed value PER CATEGORY.
  The word "per", "by", "for each", or "grouped by" is the strongest signal.

── DISAMBIGUATION RULES ─────────────────────────────────────────────────────

CROSS_KG_FILTER: Use when a specific flight number is mentioned AND the
  question asks about a property of that flight's airport.
COUNT_KG1: Use when the question counts or lists FLIGHTS specifically.
  Do NOT use count_kg1 to count airports or runways.
OPEN_KG: Use when the question asks about aviation data that exists in the
  KG but does not fit the template patterns. Specifically:
  - aircraft registration, counts of KG classes, runway surface types,
    closed runways.
  Do NOT use open_kg when filter_numeric_kg1 or ranking_kg2 would work.
COUNT_KG3: Use when the question names a specific university entity AND
  counts or lists something linked to it.

── EXAMPLES ──────────────────────────────────────────────────────────────────

Q: "Which airports have an elevation above 1000 feet?"
A: {"query_type": "filter_numeric_kg2", "params": {"property": "elevationFt", "operator": ">", "threshold": 1000}}

Q: "Show all large airports."
A: {"query_type": "filter_string_kg2", "params": {"property": "airportType", "value": "large_airport"}}

Q: "Which airports are located in Germany?"
A: {"query_type": "filter_string_kg2", "params": {"property": "countryName", "value": "Germany"}}

Q: "What are the top 5 airports with the highest elevation?"
A: {"query_type": "ranking_kg2", "params": {"property": "elevationFt", "order": "DESC", "limit": 5}}

Q: "Which airport has the shortest runway?"
A: {"query_type": "ranking_kg2", "params": {"property": "lengthFt", "order": "ASC", "limit": 1}}

Q: "Quel aéroport a la piste la plus longue?"
A: {"query_type": "ranking_kg2", "params": {"property": "lengthFt", "order": "DESC", "limit": 1}}

Q: "Compare VIE and FRA by elevation."
A: {"query_type": "compare_two_airports", "params": {"airport1": "VIE", "airport2": "FRA", "property": "elevationFt"}}

Q: "How many flights are operated by Lufthansa?"
A: {"query_type": "count_kg1", "params": {"filter_property": "hasAirline", "filter_value": "Lufthansa", "mode": "count"}}

Q: "Which flights have a ground speed above 400 knots?"
A: {"query_type": "filter_numeric_kg1", "params": {"property": "gspeed", "operator": ">", "threshold": 400}}

Q: "What country does flight LO225 land in?"
A: {"query_type": "cross_kg_filter", "params": {"direction": "destination", "airport_property": "countryName", "operator": "=", "threshold": "Poland", "limit": 1}}

Q: "What is the elevation of the destination airport of KE567?"
A: {"query_type": "cross_kg_filter", "params": {"direction": "destination", "airport_property": "elevationFt", "operator": ">", "threshold": 0, "limit": 1}}

Q: "Which flight has the highest ground speed?"
A: {"query_type": "open_kg", "params": {}}

Q: "What is the weather forecast for JFK tomorrow?"
A: {"query_type": "out_of_scope", "params": {}}

Q: "How many courses does FullProfessor0 teach?"
A: {"query_type": "count_kg3", "params": {"property": "teacherOf", "direction": "outgoing", "mode": "count"}}

Q: "Which professors work for Department3?"
A: {"query_type": "filter_string_kg3", "params": {"property": "worksFor", "value": "Department3", "limit": 10}}

Q: "What is the average ground speed per airline?"
A: {"query_type": "group_aggregate_kg1", "params": {"group_by": "airline", "property": "gspeed", "function": "AVG"}}

Q: "Which department teaches the most courses on average per professor?"
A: {"query_type": "group_aggregate_kg3", "params": {"group_by": "department", "property": "teacherOf", "function": "AVG"}}

Q: "Which airports have a grass runway?"
A: {"query_type": "filter_string_kg2", "params": {"property": "surface", "value": "grass"}}

Q: "How many runways in the dataset are closed?"
A: {"query_type": "open_kg", "params": {}}

── NOW CLASSIFY THIS QUESTION ────────────────────────────────────────────────

Question: "{question}"

- Use double quotes " for every key and every string value. Never use single quotes.
- Do not add comments, trailing commas, or any text outside the JSON object.
- Output exactly one JSON object and nothing else — no markdown, no bullet points.

Return ONLY a JSON object with keys "query_type" and "params".
No explanation. No text before or after the JSON.
"""


# ── LLM MAJORITY-VOTE HELPER ─────────────────────────────────────
def _llm_yes_no_majority(prompt: str, k: int = 3) -> bool:
    votes = []
    for _ in range(k):
        try:
            response = ollama.chat(
                model="llama3",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0}
            )
            votes.append(response["message"]["content"].strip().upper().startswith("YES"))
        except Exception as e:
            print(f"[router] _llm_yes_no_majority call failed: {e}")
            votes.append(False)
    result = sum(votes) > len(votes) / 2
    if len(set(votes)) > 1:
        print(f"[router] _llm_yes_no_majority: split vote {votes} → {result}")
    return result


# ── ASK SIGNAL DETECTOR ──────────────────────────────────────────
def _has_ask_signal(question: str) -> bool:
    if question.strip().startswith("هل"):
        print(f"[router] _has_ask_signal('{question[:40]}...') → True (fast-path: 'هل' prefix)")
        return True

    q_stripped = question.strip().lower()
    if any(q_stripped.startswith(w) for w in _WH_WORDS):
        print(f"[router] _has_ask_signal('{question[:40]}...') → False (fast-path: WH-word opener)")
        return False

    _early_tokens = re.findall(r"\w+", q_stripped)[:3]
    if any(tok in _WH_WORDS for tok in _early_tokens):
        print(f"[router] _has_ask_signal('{question[:40]}...') → False (fast-path: WH-word in opening tokens)")
        return False

    prompt = f'''Does this question ask to CONFIRM whether a specific
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
Q: "Is BLQ located in France?" → YES
Q: "هل يقع مطار بولونيا في فرنسا؟" → YES
Q: "CDG est-il situé en Belgique?" → YES
Q: "في أي دولة يقع مطار أثينا؟" → NO
Q: "ما هي إشارة النداء للرحلة TK500؟" → NO

Answer only YES or NO.

Question: "{question}"
'''
    try:
        result = _llm_yes_no_majority(prompt)
        print(f"[router] _has_ask_signal('{question[:40]}...') → {result}")
        return result
    except Exception as e:
        print(f"[router] _has_ask_signal LLM check failed: {e}")
        return False


# ── KG ANSWERABILITY GATE ───────────────────────────────────────
def _is_kg_answerable(question: str) -> bool:
    schema = get_open_kg_schema()
    prompt = f'''You are a scope classifier for an aviation knowledge graph system.
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
Q: "What is the weather forecast for JFK tomorrow?"     → NO
Q: "Am I allowed to bring a guitar on flight BR62?"     → NO
Q: "Who invented the first commercial airplane?"        → NO
Q: "What is the elevation of ZRH?"                → YES
Q: "Is ZRH located in Switzerland?"               → YES
Q: "هل يقع مطار زيورخ في سويسرا؟"                  → YES
Q: "في أي دولة يقع مطار أثينا؟"                    → NO

Answer only YES or NO:
Can this question be answered using only the data described above?

Question: "{question}"
'''
    try:
        result = _llm_yes_no_majority(prompt)
        print(f"[router] _is_kg_answerable('{question[:40]}...') → {result}")
        return result
    except Exception as e:
        print(f"[router] _is_kg_answerable failed: {e}")
        return False


# ── JSON OBJECT EXTRACTOR ───────────────────────────────────────
def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    string_char = ""
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == string_char:
                in_string = False
            continue

        if ch in ('"', "'"):
            in_string = True
            string_char = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


# ── QUERY TYPE NORMALIZER ─────────────────────────────────────
def _normalize_query_type(query_type, question: str = "") -> str:
    if not isinstance(query_type, str):
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
        if _detect_university_entity(question) or _has_filter_signal(question):
            kg_guess = "university"
        elif _detect_airport_keyword(question) or _detect_airport_entity(question):
            kg_guess = "airports"
        else:
            kg_guess = None
        if kg_guess:
            from kg_registry import TEMPLATE_REGISTRY as _TR
            for t in family:
                if _TR[t]["kg"] == kg_guess:
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


# ── LLM CLASSIFIER ──────────────────────────────────────────────
def _llm_classify(question: str, max_attempts: int = 2) -> dict:
    prompt = _CLASSIFICATION_PROMPT.replace("{question}", question)

    _SMART_PUNCTUATION = str.maketrans({
        "\u201c": '"', "\u201d": '"',
        "\u2018": "'", "\u2019": "'",
    })

    for attempt in range(max_attempts):
        raw = ""
        try:
            response = ollama.chat(
                model="llama3",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0}
            )
            raw = response["message"]["content"].strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            raw = raw.translate(_SMART_PUNCTUATION)

            candidate = _extract_first_json_object(raw)
            if candidate is None:
                raise ValueError("no balanced JSON object found in response")

            try:
                result = json.loads(candidate)
            except json.JSONDecodeError:
                result = ast.literal_eval(candidate)

            if not isinstance(result, dict):
                raise ValueError(f"parsed object is not a dict: {type(result)}")

            result["query_type"] = _normalize_query_type(result.get("query_type", ""), question)
            print(f"[router] LLM classified as: {result.get('query_type')} "
                  f"| params: {result.get('params')}")
            return result

        except Exception as e:
            print(f"[router] Attempt {attempt+1}: classification failed: {e}")
            prompt = (
                f"Your previous response was not valid JSON:\n\n"
                f"{raw}\n\n"
                f"Error: {e}\n\n"
                f"Return ONLY the corrected JSON object. "
                f"Use double quotes for all keys and values. "
                f"No explanation, no text before or after."
            )

    return {}
