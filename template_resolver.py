"""
template_resolver.py
--------------------
Resolves complex queries using predefined SPARQL templates.
Handles: filters, rankings, comparisons, counts, and cross-KG aggregates.
Also resolves ask_query (yes/no) questions — see resolve_ask_query() at
the bottom of this file.

DESIGN PRINCIPLE:
    The LLM extracts parameters (threshold, property, value, entities).
    Predefined templates fill in the structure.
    This guarantees syntactic validity and controlled URI injection —
    the same principle as the single-value pipeline, extended to aggregates.

TEMPLATE CATEGORIES:
    KG2 — Airport queries:
        filter_numeric_kg2    : airports where property > / < threshold
        filter_string_kg2     : airports where property = value
        ranking_kg2           : top/bottom N airports by property
        compare_two_airports  : compare airport A vs airport B

    KG1 — Flight queries:
        count_kg1             : count/list flights matching a condition
        filter_numeric_kg1    : flights where speed > / < threshold
        ranking_kg1           : top/bottom N flights by property
        compare_two_flights   : compare two named flights on a property

    KG3 — University queries:
        count_kg3              : count/list entities linked to a named entity
        filter_string_kg3       : filter people by department membership
        filter_numeric_kg3      : departments filtered by total headcount
        ranking_kg3             : top N departments by headcount OR top N people by relation count
        compare_two_departments : compare two departments by entity count

    Cross-KG:
        cross_kg_filter       : flights whose airport property meets condition

    ASK (all KGs):
        resolve_ask_query()   : yes/no questions about a known entity's
                                 property value — separate from the template
                                 registry pattern above, since it returns a
                                 boolean rather than rows. See its own
                                 docstring below for the full rationale.

SPARQL GENERATION:
    Templates use Python f-strings with validated parameter slots.
    The LLM is ONLY used to extract parameter values from the question,
    never to generate SPARQL structure directly.
"""

import re
import json
import ast
import ollama
import urllib.parse
import urllib.request
from kg_registry import (
    get_endpoint, get_base_uri, get_lexicon,
    TEMPLATE_REGISTRY, CROSS_KG_CONFIG
)
from pipeline.mapper import (
    map_university_entity,
    map_flight,
    map_airport,
    load_lexicon,
    map_property_cascade,
    map_property_cascade_scored,
)
from pipeline.mapper import ASK_SEMANTIC_THRESHOLD 
from pipeline.extractor import extract_ask_entities, validate_ask_extraction
from pipeline.executor  import build_ask_query, execute_ask_sparql

# ── BASE URIs ─────────────────────────────────────────────────────────────────
KG1   = get_base_uri("flights")
KG2   = get_base_uri("airports")
KG3   = get_base_uri("university")
KG3_STRING_PROPS = {
    "worksFor": {"uri": f"{KG3}worksFor", "label": "works for", "hop": "department"},
    "memberOf": {"uri": f"{KG3}memberOf", "label": "member of", "hop": "department"},
}

# LUBM class names confirmed from questions/gold answers seen so far
# (GraduateStudent, FullProfessor, AssociateProfessor appear in ranking_kg3/
# compare_two_departments gold data). Not guessed — extend only once a new
# class name is confirmed the same way, same pattern as AIRLINE_NAME_TO_ICAO
# below. UndergraduateStudent/AssistantProfessor/Lecturer included for
# symmetry with the standard LUBM class set, not independently verified here.
KG3_ENTITY_TYPES = {
    "GraduateStudent", "UndergraduateStudent",
    "FullProfessor", "AssociateProfessor", "AssistantProfessor", "Lecturer",
}
KG3_HOP_PROPERTIES = {"memberOf", "worksFor", "teacherOf", "takesCourse"}
KG1_EP = get_endpoint("flights")
KG2_EP = get_endpoint("airports")
KG3_EP = get_endpoint("university")

# ── PROPERTY MAPS ─────────────────────────────────────────────────────────────
KG2_NUMERIC_PROPS = {
    "elevationFt": {"uri": f"{KG2}elevationFt",  "label": "elevation",     "unit": "feet",   "hop": "direct",  "adjective": "higher"},
    "lengthFt":    {"uri": f"{KG2}lengthFt",     "label": "runway length",  "unit": "feet",   "hop": "runway",  "adjective": "longer"},
    "widthFt":     {"uri": f"{KG2}widthFt",      "label": "runway width",   "unit": "feet",   "hop": "runway",  "adjective": "wider"},
    "latitude":    {"uri": f"{KG2}latitude",     "label": "latitude",       "unit": "degrees","hop": "direct",  "adjective": "higher"},
    "longitude":   {"uri": f"{KG2}longitude",    "label": "longitude",      "unit": "degrees","hop": "direct",  "adjective": "higher"},
}

KG2_STRING_PROPS = {
    "airportType":  {"uri": f"{KG2}airportType",  "label": "airport type"},
    "surface":      {"uri": f"{KG2}surface",       "label": "surface"},
    "lighted":      {"uri": f"{KG2}lighted",       "label": "lighting"},
    "closed":       {"uri": f"{KG2}closed",        "label": "closed status"},
    "continent":    {"uri": f"{KG2}continent",     "label": "continent"},
    "countryName":  {"uri": f"{KG2}countryName",   "label": "country",
                     "hop": "country"},
    "municipality": {"uri": f"{KG2}municipality",  "label": "city"},
}

KG1_NUMERIC_PROPS = {
    "gspeed": {"uri": f"{KG1}gspeed",  "label": "ground speed", "unit": "knots"},
    "vspeed": {"uri": f"{KG1}vspeed",  "label": "vertical speed","unit": "ft/min"},
    
}

KG1_STRING_PROPS = {
    "hasDestinationCity":    {"uri": f"{KG1}hasDestinationCity"},
    "hasOriginCity":         {"uri": f"{KG1}hasOriginCity"},
    "hasAirline":            {"uri": f"{KG1}hasAirline"},
    "hasDestinationCountry": {"uri": f"{KG1}hasDestinationCountry"},
    "hasOriginCountry":      {"uri": f"{KG1}hasOriginCountry"},
}
_UNIVERSITY_ENTITY_RE = re.compile(
    r'\b((?:[A-Z][a-zA-Z]*)?(?:Professor|Student|Course|Department|Group|'
    r'University|Lecturer|Publication)\d+)\b'
)

_SINGULAR_SUPERLATIVE_RE = re.compile(
    r"\b(shortest|longest|highest|lowest|smallest|widest|narrowest|"
    r"most|least|fastest|slowest)\b"
    r"|(le plus court|la plus courte|le plus long|la plus longue|"
    r"le plus haut|le plus élevé|la plus haute|la plus élevée|"
    r"le plus bas|la plus basse|le plus large|le plus étroit|"
    r"la plus large|la plus étroite|le plus de|le moins de|"
    r"le plus rapide|le plus lent)"
    r"|(أقصر|أطول|أعلى|أدنى|أقل|أعرض|أضيق|أكثر|أسرع|أبطأ)",
    re.IGNORECASE
)

_PLURAL_INTENT_RE = re.compile(
    r"\d+"
    r"|\b(two|three|four|five|six|seven|eight|nine|ten)\b"
    r"|\b(list|show all|enumerate)\b"
    r"|(deux|trois|quatre|cinq|six|sept|huit|neuf|dix)"
    r"|(listez|montrez tous|montrer tous)"
    r"|(اثنان|ثلاثة|أربعة|خمسة|ستة|سبعة|ثمانية|تسعة|عشرة)"
    r"|(اذكر|أظهر جميع|قائمة)",
    re.IGNORECASE
)


def _detect_singular_superlative(question: str) -> bool:
    if _PLURAL_INTENT_RE.search(question):
        return False
    return bool(_SINGULAR_SUPERLATIVE_RE.search(question))


def _sanitize_sparql_literal(value: str) -> str:
    value = str(value).strip()
    value = value.strip('"').strip("'")
    value = value.replace("\\", "").replace('"', "")
    value = value.replace("\n", " ").replace("\r", " ")
    return value.strip()

def _sanitize_params(params: dict) -> dict:
    return {
        k: (_sanitize_sparql_literal(v) if isinstance(v, str) else v)
        for k, v in params.items()
    }

def _detect_university_entity_for_template(q: str):
    m = _UNIVERSITY_ENTITY_RE.search(q)
    return m.group(1) if m else None

def _detect_two_university_entities_for_template(q: str) -> list[str] | None:
    matches = []
    for m in _UNIVERSITY_ENTITY_RE.findall(q):
        if m not in matches:
            matches.append(m)
    return matches if len(matches) == 2 else None

# ── SPARQL HELPER ─────────────────────────────────────────────────────────────

def _run_sparql(endpoint: str, query: str, multiple: bool = True):
    """Returns (result, error). `error` is None on a successful HTTP round
    trip regardless of whether any rows came back — an empty `result` with
    error=None means the query executed correctly and legitimately found
    nothing, which callers must NOT treat the same as an execution_failure.
    `error` is set (as a string) only when the request/parse itself raised,
    so real Jena/HTTP failures stay distinguishable from empty results."""
    data = urllib.parse.urlencode({
        "query": query,
        "format": "application/sparql-results+json"
    }).encode()
    try:
        with urllib.request.urlopen(
            urllib.request.Request(endpoint, data=data), timeout=15
        ) as r:
            result   = json.loads(r.read())
            bindings = result.get("results", {}).get("bindings", [])
            if multiple:
                return bindings, None
            return (bindings[0] if bindings else None), None
    except Exception as e:
        print(f"[template] SPARQL error: {e}")
        err = str(e)
        return ([], err) if multiple else (None, err)


# ── LLM PARAMETER EXTRACTOR ───────────────────────────────────────────────────

def _extract_params(question: str, template_name: str, lang: str) -> dict:
    prompts = {

        "filter_numeric_kg2": f"""Extract parameters from this airport question.
Question: "{question}"

Identify the property being filtered using this mapping:
  "elevation" or "altitude" or "height"                → "elevationFt"
  "runway length" or "length" or "longer" or "longest"  → "lengthFt"
  "runway width" or "width" or "wider" or "widest"      → "widthFt"

Return ONLY a JSON object with these keys:
- "property": one of [elevationFt, lengthFt, widthFt]
- "operator": one of [>, <, >=, <=]
- "threshold": numeric value (integer)
Example: {{"property": "elevationFt", "operator": ">", "threshold": 1000}}
Return ONLY the JSON. No explanation.""",

        "filter_string_kg2": f"""Extract parameters from this airport question.
Question: "{question}"

Rules:
- If the question asks about airports "in [country]" or "located in [country]", 
  set property = "countryName" and value = the country name.
- If the question asks about airport type (large, small, medium), 
  set property = "airportType" and value = e.g. "large_airport".
- If the question asks about a city/municipality, 
  set property = "municipality" and value = the city name.

Return ONLY a JSON object with keys: "property", "value".
Example: {{"property": "countryName", "value": "France"}}
Return ONLY the JSON. No explanation.""",
        "count_kg2": f"""Extract parameters from this airport counting question.
Question: "{question}"

This question asks for a COUNT of airports matching ONE condition, which is
either numeric or categorical — pick whichever applies.

Numeric condition — identify the property using this mapping:
  "elevation" or "altitude" or "height"                → "elevationFt"
  "runway length" or "length" or "longer" or "longest"  → "lengthFt"
  "runway width" or "width" or "wider" or "widest"      → "widthFt"
Return: {{"property": "elevationFt", "operator": ">", "threshold": 1000}}

Categorical condition:
- Airports "in [country]" or "located in [country]"     → property="countryName", value=country name
- Airport type (large, small, medium)                   → property="airportType", value e.g. "small_airport"
Return: {{"property": "countryName", "value": "Germany"}}

Return ONLY the JSON object for whichever condition applies. No explanation.""",

        
        "ranking_kg2": f"""Extract parameters from this airport ranking question.
Question: "{question}"

Identify the property being ranked using this mapping:
  "elevation" or "altitude" or "height"                → "elevationFt"
  "runway length" or "length" or "longer" or "longest"  → "lengthFt"
  "runway width" or "width" or "wider" or "widest"      → "widthFt"
  "élévation" or "altitude" or "hauteur"                → "elevationFt"
  "piste" + "longue"/"longueur"/"courte"                → "lengthFt"
  "piste" + "large"/"largeur"/"étroite"                 → "widthFt"
  "الارتفاع" or "أعلى" or "أدنى" (with no مدرج mention)   → "elevationFt"
  "مدرج" + "أقصر" or "أطول" or "طول"                      → "lengthFt"
  "مدرج" + "أعرض" or "أضيق" or "عرض"                      → "widthFt"

Return ONLY a JSON object with these keys:
- "property": one of [elevationFt, lengthFt, widthFt]
- "order": "DESC" for highest/longest/most, "ASC" for lowest/shortest/least
- "limit": number of results (default 5)
Example: {{"property": "elevationFt", "order": "DESC", "limit": 5}}
Return ONLY the JSON. No explanation.""",

       "compare_two_airports": f"""Extract two airport IATA codes and the comparison property.
Question: "{question}"

Step 1 — find two IATA codes (3 uppercase letters each).
Step 2 — identify the property being compared using this mapping:
   "elevation" or "altitude" or "height"  → "elevationFt"
  "runway length" or "length" or "longer" or "longest" → "lengthFt"
  "runway width" or "width" or "wider" or "widest" → "widthFt"
  "type" or "airport type" or "kind"     → "airportType"
  "الارتفاع" or "أعلى" or "أدنى" (with no مدرج mention)   → "elevationFt"
  "مدرج" + "أقصر" or "أطول" or "طول"                      → "lengthFt"
  "مدرج" + "أعرض" or "أضيق" or "عرض"                      → "widthFt"

Return ONLY a JSON object:
{{"airport1": "VIE", "airport2": "FRA", "property": "elevationFt"}}

Question to parse: "{question}"
Return ONLY the JSON. No explanation. No text before or after.""",

        "count_kg1": f"""Extract parameters from this flight count/list question.
Question: "{question}"
Return ONLY a JSON object with these keys:
- "filter_property": one of [hasDestinationCity, hasOriginCity, hasAirline, hasDestinationCountry]
- "filter_value": the value to filter by (city name, airline name, country)
- "mode": "count" or "list"
Example: {{"filter_property": "hasDestinationCity", "filter_value": "Munich", "mode": "count"}}
Return ONLY the JSON. No explanation.""",

       "filter_numeric_kg1": f"""Extract parameters from this flight numeric filter question.
Question: "{question}"

Property mapping rules:
- "ground speed", "speed", "knots", "gspeed" → "gspeed"
- "vertical speed", "vspeed", "feet per minute" → "vspeed"
- "altitude", "alt", "flying at", "above X feet" (for flights) → "alt"

Return ONLY a JSON object with these keys:
- "property": one of [gspeed, vspeed, alt]
- "operator": one of [>, <, >=, <=]
- "threshold": numeric value
Example: {{"property": "alt", "operator": ">", "threshold": 30000}}
Return ONLY the JSON. No explanation.""",

       "ranking_kg1": f"""Extract parameters from this flight ranking question.
Question: "{question}"

Identify the property being ranked:
  "ground speed" or "speed" or "gspeed"                 → "gspeed"
  "vertical speed" or "vspeed"                           → "vspeed"
  "vitesse au sol" or "vitesse sol"                      → "gspeed"
  "vitesse verticale"                                    → "vspeed"
  "السرعة الأرضية"                                        → "gspeed"
  "السرعة العمودية"                                       → "vspeed"

Return ONLY a JSON object with these keys:
- "property": one of [gspeed, vspeed]
- "order": "DESC" for highest/fastest/most, "ASC" for lowest/slowest/least
- "limit": number of results (default 5)
Example: {{"property": "gspeed", "order": "DESC", "limit": 5}}
Return ONLY the JSON. No explanation.""",

       "cross_kg_filter": f"""Extract parameters from this cross-KG flight filter question.
Question: "{question}"

Step 1 — direction: landing/arriving/destination → "destination", departing/origin → "origin"
Step 2 — airport_property:
  elevation/altitude/above X feet → "elevationFt"
  runway length/longer than       → "lengthFt"
  country/located in              → "countryName"
  large airport/airport type      → "airportType"
Step 3 — operator and threshold:
  Numeric: "above X" → operator=">", threshold=X (integer)
           "below X" → operator="<", threshold=X (integer)
  String:  always operator="=", threshold=English value.
  ALWAYS translate to English: Allemagne→Germany, France→France,
  Italie→Italy, Turquie→Turkey, ألمانيا→Germany, فرنسا→France.
  For large airports: threshold="large_airport"

Return ALL five keys. No missing fields.
Example: {{"direction": "destination", "airport_property": "countryName", "operator": "=", "threshold": "Germany", "limit": 10}}
Example: {{"direction": "destination", "airport_property": "elevationFt", "operator": ">", "threshold": 800, "limit": 10}}
Return ONLY the JSON. No explanation.""",

"count_kg3": f"""Extract parameters from this university count/list question.
Question: "{question}"

Property mapping rules:
- "teach", "courses taught" → property="teacherOf", direction="outgoing"
- "take", "enrolled in", "courses taken" → property="takesCourse", direction="outgoing"
- "students", "members" (of a department) → property="memberOf", direction="incoming"
- "professors", "faculty", "staff" (of a department) → property="worksFor", direction="incoming"
- "departments" (of a university) → property="subOrganizationOf", direction="incoming"

Return ONLY a JSON object with these keys:
- "property": one of [teacherOf, takesCourse, memberOf, worksFor, subOrganizationOf]
- "direction": "outgoing" (entity is subject) or "incoming" (entity is object)
- "mode": "count" or "list"
Example: {{"property": "teacherOf", "direction": "outgoing", "mode": "count"}}
Return ONLY the JSON. No explanation.""",

"filter_string_kg3": f"""Extract parameters from this university filter question.
Question: "{question}"

Property mapping rules:
- "professors who work for", "faculty in", "staff of" → property="worksFor"
- "students who are members of", "students in" → property="memberOf"

Return ONLY a JSON object with these keys:
- "property": one of [worksFor, memberOf]
- "value": the department name mentioned (e.g. "Department3")
- "limit": integer, default 10 if not specified
Example: {{"property": "worksFor", "value": "Department3", "limit": 10}}
Return ONLY the JSON. No explanation.""",

"filter_numeric_kg3": f"""Extract parameters from this university department-size question.
Question: "{question}"

This question asks to filter DEPARTMENTS by their total headcount
(students who are members + professors/lecturers who work for the
department, counted together as one population).

If the question gives ONE bound (e.g. "more than 550", "fewer than 420"):
Return ONLY:
- "operator": one of [>, <, >=, <=]
- "threshold": numeric value (integer)
Example: {{"operator": "<", "threshold": 420}}

If the question gives a RANGE ("between X and Y", "entre X et Y", "بين X و Y"):
Return ONLY the two raw numbers, in the order they appear in the question —
do NOT decide which operator each number gets, that is handled separately:
- "range_low": the first number mentioned
- "range_high": the second number mentioned
Example (question: "between 500 and 570"): {{"range_low": 500, "range_high": 570}}

Return ONLY the JSON. No explanation.""",

"group_aggregate_kg1": f"""Extract parameters from this flight aggregation question.
Question: "{question}"
Return ONLY a JSON object with these keys:
- "property": one of [gspeed, vspeed]
- "function": one of [AVG, SUM, MAX, MIN]
Group-by is always "airline" for this template — do not extract it.
Example: {{"property": "gspeed", "function": "AVG"}}
Return ONLY the JSON. No explanation.""",

        "group_aggregate_kg2": f"""Extract parameters from this airport aggregation question.
Question: "{question}"
Return ONLY a JSON object with these keys:
- "group_by": "country" or "continent"
- "property": one of [elevationFt, lengthFt, widthFt]
- "function": one of [AVG, SUM, MAX, MIN]

IMPORTANT — "function" is which aggregate is computed PER GROUP, not which
group ranks highest. A question can ask "which country has the highest
average elevation" — here "average"/"moyenne"/"متوسط" sets function=AVG;
"highest"/"la plus haute"/"أعلى" is asking which country's average ranks
first, it does NOT mean function=MAX. Only use MAX/MIN when the question
itself asks for a maximum/minimum value (e.g. "highest single runway
length", "longest single runway" — no averaging language present).

- "average"/"moyenne"/"متوسط" (mean of values) → function="AVG"
- "total"/"somme"/"مجموع" (sum of values) → function="SUM"
- "maximum"/"highest value"/"أقصى قيمة" (the single max value itself, no averaging) → function="MAX"
- "minimum"/"lowest value"/"أدنى قيمة" (the single min value itself, no averaging) → function="MIN"

Example: {{"group_by": "country", "property": "elevationFt", "function": "AVG"}}
Example (question: "which country has the highest average airport elevation?"): {{"group_by": "country", "property": "elevationFt", "function": "AVG"}}
Return ONLY the JSON. No explanation.""",

        "group_aggregate_kg3": f"""Extract parameters from this university aggregation question.
Question: "{question}"
Return ONLY a JSON object with these keys:
- "property": "teacherOf" (courses taught) or "takesCourse" (courses taken)
- "function": one of [AVG, MAX, MIN]
Group-by is always "department" for this template — do not extract it.
Example: {{"property": "teacherOf", "function": "AVG"}}
Return ONLY the JSON. No explanation.""",

        # EDIT: new prompt for ranking_kg3
        "ranking_kg3": f"""Extract parameters from this university ranking question.
Question: "{question}"

Determine group_by first:
- "departments with most..." or "largest departments" → group_by="department"
- "professor who teaches most..." or "person with most..." within one
  already-named department → group_by="person"

hop_property mapping:
- "students", "members", "headcount", "population" → "memberOf"
- "professors", "faculty", "staff" → "worksFor"
- "courses taught", "teaches" → "teacherOf"
- "courses taken", "enrolled" → "takesCourse"

entity_type (OPTIONAL — omit the key entirely if no specific rank/type
is named in the question, e.g. "which professor" without specifying
full/associate/assistant):
- "graduate student(s)" / "étudiant(s) diplômé(s)" / "طلاب الدراسات العليا" → "GraduateStudent"
- "undergraduate student(s)" / "étudiant(s) de licence" / "طلاب البكالوريوس" → "UndergraduateStudent"
- "full professor(s)" / "professeur(s) titulaire(s)" / "أستاذ (أساتذة) كامل" → "FullProfessor"
- "associate professor(s)" / "professeur(s) associé(s)" / "أستاذ مشارك" → "AssociateProfessor"
- "assistant professor(s)" / "professeur(s) assistant(s)" / "أستاذ مساعد" → "AssistantProfessor"
- "lecturer(s)" / "maître(s) de conférences" / "محاضر" → "Lecturer"

Return ONLY a JSON object with these keys:
- "group_by": "department" or "person"
- "entity_type": one of the values above, OR omit this key entirely
- "hop_property": one of [memberOf, worksFor, teacherOf, takesCourse]
- "order": "DESC" for highest/most, "ASC" for lowest/least
- "limit": number of results (default 3)
Example: {{"group_by": "department", "entity_type": "GraduateStudent", "hop_property": "memberOf", "order": "DESC", "limit": 3}}
Example (no type named): {{"group_by": "person", "hop_property": "teacherOf", "order": "DESC", "limit": 1}}
Return ONLY the JSON. No explanation.""",

        "compare_two_departments": f"""Extract parameters from this university comparison question.
Question: "{question}"

Department names are handled separately — do NOT extract them here.

entity_type mapping (REQUIRED — always pick one, this template always
compares a specific type of person):
  "students" → "GraduateStudent" if graduate students specified, else "UndergraduateStudent"
  "full professors" → "FullProfessor"
  "associate professors" → "AssociateProfessor"
  "assistant professors" → "AssistantProfessor"
  "lecturers" → "Lecturer"

hop_property mapping:
  "students", "members" → "memberOf"
  "professors", "faculty", "staff" → "worksFor"

Return ONLY a JSON object:
{{"entity_type": "FullProfessor", "hop_property": "worksFor"}}

Return ONLY the JSON. No explanation. No text before or after.""",
    }

    prompt = prompts.get(template_name, "")
    if not prompt:
        return {}

    try:
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0}

        )
        raw  = response["message"]["content"].strip()
        raw  = re.sub(r"```json|```", "", raw).strip()
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if not match:
            print(f"[template] No JSON object found in LLM output: {repr(raw[:100])}")
            return {}
        candidate = match.group()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return ast.literal_eval(candidate)
    except Exception as e:
        print(f"[template] Parameter extraction failed: {e}")
        return {}


# ── SPARQL BUILDERS ───────────────────────────────────────────────────────────

def _build_filter_numeric_kg2(params: dict) -> tuple[str, str] | None:
    prop      = params.get("property", "elevationFt")
    operator  = params.get("operator") or ""
    threshold = params.get("threshold")

    VALID_OPERATORS = {">", "<", ">=", "<=", "="}
    if operator not in VALID_OPERATORS or threshold is None:
        return None

    prop_info = KG2_NUMERIC_PROPS.get(prop)
    if not prop_info:
        return None

    prop_uri = prop_info["uri"]
    unit     = prop_info["unit"]
    hop      = prop_info.get("hop", "direct")

    if hop == "runway":
        sparql = f"""SELECT ?airport ?name (MAX(?rawValue) AS ?value) WHERE {{
  ?airport a <{KG2}Airport> .
  ?airport <{KG2}airportName> ?name .
  ?airport <{KG2}hasRunway> ?runway .
  ?runway <{prop_uri}> ?rawValue .
  FILTER(?rawValue {operator} {threshold})
}} GROUP BY ?airport ?name ORDER BY DESC(?value) ?airport"""
    else:
        sparql = f"""SELECT ?airport ?name ?value WHERE {{
  ?airport a <{KG2}Airport> .
  ?airport <{KG2}airportName> ?name .
  ?airport <{prop_uri}> ?value .
  FILTER(?value {operator} {threshold})
}} ORDER BY DESC(?value) ?airport"""

    label = f"airports with {prop_info['label']} {operator} {threshold} {unit}"

    if params.get("mode") == "count":
        sparql = f"SELECT (COUNT(*) AS ?count) WHERE {{ {sparql} }}"
        label  = f"count of {label}"

    return sparql, label


def _build_filter_string_kg2(params: dict) -> tuple[str, str] | None:
    prop  = params.get("property", "airportType")
    value = params.get("value", "")

    prop_info = KG2_STRING_PROPS.get(prop)
    if not prop_info:
        print(f"[filter_string_kg2] build failed: property '{prop}' not in KG2_STRING_PROPS "
              f"(known: {sorted(KG2_STRING_PROPS.keys())})")
        return None
    if not value:
        print(f"[filter_string_kg2] build failed: empty value for property '{prop}'")
        return None

    if prop == "countryName":
        sparql = f"""SELECT ?airport ?name WHERE {{
  ?airport a <{KG2}Airport> .
  ?airport <{KG2}airportName> ?name .
  ?airport <{KG2}locatedInCountry> ?country .
  ?country <{KG2}countryName> "{value}" .
}} ORDER BY ?name"""
    elif prop == "surface":
        prop_uri = prop_info["uri"]
        codes = _SURFACE_SYNONYMS.get(value.strip().lower(), [value])
        values_clause = ", ".join(f'"{c}"' for c in codes)
        sparql = f"""SELECT ?airport ?name WHERE {{
  ?airport a <{KG2}Airport> .
  ?airport <{KG2}airportName> ?name .
  ?airport <{prop_uri}> ?surfaceVal .
  FILTER(?surfaceVal IN ({values_clause}))
}} ORDER BY ?name"""
    else:
        prop_uri = prop_info["uri"]
        sparql = f"""SELECT ?airport ?name WHERE {{
  ?airport a <{KG2}Airport> .
  ?airport <{KG2}airportName> ?name .
  ?airport <{prop_uri}> "{value}" .
}} ORDER BY ?name"""

    label = f"airports where {prop} = {value}"

    if params.get("mode") == "count":
        sparql = f"SELECT (COUNT(*) AS ?count) WHERE {{ {sparql} }}"
        label  = f"count of {label}"

    return sparql, label


_SURFACE_SYNONYMS = {
    "asphalt": ["ASP", "ASPH", "PEM", "ASPHALT"], "asphalte": ["ASP", "ASPH", "PEM", "ASPHALT"],
    "إسفلت": ["ASP", "ASPH", "PEM", "ASPHALT"], "الإسفلت": ["ASP", "ASPH", "PEM", "ASPHALT"],
    "concrete": ["CON", "CONC", "concrete", "Concrete"], "béton": ["CON", "CONC", "concrete", "Concrete"],
    "خرسانة": ["CON", "CONC", "concrete", "Concrete"],
    "grass": ["GRS", "GRASS"], "herbe": ["GRS", "GRASS"], "عشب": ["GRS", "GRASS"],
}

def _resolve_surface_value(value: str) -> str:
    return _SURFACE_SYNONYMS.get(value.strip().lower(), [value])[0]

def _build_count_kg2(params: dict) -> tuple[str, str] | None:
    count_params = {**params, "mode": "count"}
    if params.get("operator") and params.get("threshold") is not None:
        return _build_filter_numeric_kg2(count_params)
    return _build_filter_string_kg2(count_params)

def _build_ranking_kg2(params: dict) -> tuple[str, str] | None:
    prop  = params.get("property", "elevationFt")
    order = params.get("order", "DESC")
    limit = int(params.get("limit", 5))

    prop_info = KG2_NUMERIC_PROPS.get(prop)
    if not prop_info:
        return None

    prop_uri = prop_info["uri"]
    unit     = prop_info["unit"]
    hop      = prop_info.get("hop", "direct")

    if hop == "runway":
        sparql = f"""SELECT ?airport ?name ?value WHERE {{
  ?airport a <{KG2}Airport> .
  ?airport <{KG2}airportName> ?name .
  ?airport <{KG2}hasRunway> ?runway .
  ?runway <{prop_uri}> ?value .
}} ORDER BY {order}(?value) ?airport LIMIT {limit}"""
    else:
        sparql = f"""SELECT ?airport ?name ?value WHERE {{
  ?airport a <{KG2}Airport> .
  ?airport <{KG2}airportName> ?name .
  ?airport <{prop_uri}> ?value .
}} ORDER BY {order}(?value) ?airport LIMIT {limit}"""

    direction_word = "highest" if order == "DESC" else "lowest"
    label = f"top {limit} airports by {prop_info['label']} ({direction_word})"
    return sparql, label

def _build_group_aggregate_kg1(params: dict) -> tuple[str, str] | None:
    from kg_registry import get_group_aggregate_config, AGGREGATE_FUNCTIONS

    prop     = params.get("property", "gspeed")
    function = (params.get("function") or "AVG").upper()

    if function not in AGGREGATE_FUNCTIONS:
        return None

    cfg = get_group_aggregate_config("flights")
    prop_info = cfg["numeric_properties"].get(prop)
    if not prop_info:
        return None

    group_info = cfg["group_by"]["airline"]

    limit = params.get("limit")
    limit_clause = f" LIMIT {int(limit)}" if limit else ""

    sparql = f"""SELECT ?groupName (ROUND({function}(?value) * 100) / 100 AS ?agg) WHERE {{
  ?flight a <{KG1}Flight> .
  ?flight <{KG1}{group_info['hop_property']}> ?groupNode .
  ?groupNode <{KG1}{group_info['name_property']}> ?groupName .
  ?flight <{KG1}{prop_info['hop']}> ?event .
  ?event <{KG1}{prop}> ?value .
}} GROUP BY ?groupName ORDER BY DESC(?agg){limit_clause}"""

    label = f"{function} of {prop} grouped by airline"
    return sparql, label


def _build_group_aggregate_kg2(params: dict) -> tuple[str, str] | None:
    from kg_registry import get_group_aggregate_config, AGGREGATE_FUNCTIONS

    group_by = params.get("group_by", "country")
    prop     = params.get("property", "elevationFt")
    function = (params.get("function") or "AVG").upper()

    if function not in AGGREGATE_FUNCTIONS:
        return None

    cfg = get_group_aggregate_config("airports")
    group_info = cfg["group_by"].get(group_by)
    prop_info  = cfg["numeric_properties"].get(prop)
    if not group_info or not prop_info:
        return None

    limit = params.get("limit")
    limit_clause = f" LIMIT {int(limit)}" if limit else ""

    if prop_info["hop"] == "hasRunway":
        sparql = f"""SELECT ?groupName (ROUND({function}(?value) * 100) / 100 AS ?agg) WHERE {{
  ?airport a <{KG2}Airport> .
  ?airport <{KG2}{group_info['hop_property']}> ?groupNode .
  ?groupNode <{KG2}{group_info['name_property']}> ?groupName .
  ?airport <{KG2}hasRunway> ?runway .
  ?runway <{KG2}{prop}> ?value .
}} GROUP BY ?groupName ORDER BY DESC(?agg){limit_clause}"""
    else:
        sparql = f"""SELECT ?groupName (ROUND({function}(?value) * 100) / 100 AS ?agg) WHERE {{
  ?airport a <{KG2}Airport> .
  ?airport <{KG2}{group_info['hop_property']}> ?groupNode .
  ?groupNode <{KG2}{group_info['name_property']}> ?groupName .
  ?airport <{KG2}{prop}> ?value .
}} GROUP BY ?groupName ORDER BY DESC(?agg){limit_clause}"""

    label = f"{function} of {prop} grouped by {group_by}"
    return sparql, label


def _build_group_aggregate_kg3(params: dict) -> tuple[str, str] | None:
    from kg_registry import get_group_aggregate_config

    prop     = params.get("property", "teacherOf")
    function = (params.get("function") or "AVG").upper()

    if function not in {"AVG", "MAX", "MIN"}:
        return None

    cfg = get_group_aggregate_config("university")
    prop_info = cfg["countable_properties"].get(prop)
    if not prop_info:
        return None

    group_info = cfg["group_by"]["department"]

    limit = params.get("limit")
    limit_clause = f" LIMIT {int(limit)}" if limit else ""

    sparql = f"""SELECT ?deptName (ROUND({function}(?cnt) * 100) / 100 AS ?agg) WHERE {{
  SELECT ?person ?dept ?deptName (COUNT(?obj) AS ?cnt) WHERE {{
    ?person <{KG3}{group_info['hop_property']}> ?dept .
    ?dept <{KG3}{group_info['name_property']}> ?deptName .
    ?person <{KG3}{prop}> ?obj .
  }} GROUP BY ?person ?dept ?deptName
}} GROUP BY ?deptName ORDER BY DESC(?agg){limit_clause}"""

    label = f"{function} of {prop_info['label']} per person, grouped by department"
    return sparql, label

def _build_compare_two_airports(params: dict) -> tuple[str, str] | None:
    a1   = params.get("airport1", "").upper()
    a2   = params.get("airport2", "").upper()
    prop = (params.get("property") or "").strip()

    if not prop:
        prop = "elevationFt"

    prop_info = KG2_NUMERIC_PROPS.get(prop) or KG2_STRING_PROPS.get(prop)
    if not prop_info:
        return None

    prop_uri = prop_info["uri"]
    hop      = prop_info.get("hop", "direct")

    if hop == "runway":
        sparql = f"""SELECT ?name ?iata ?value WHERE {{
  VALUES ?airport {{ <{KG2}Airport/{a1}> <{KG2}Airport/{a2}> }}
  ?airport <{KG2}airportName> ?name .
  ?airport <{KG2}iataCode> ?iata .
  ?airport <{KG2}hasRunway> ?runway .
  ?runway <{prop_uri}> ?value .
}} ORDER BY DESC(?value)"""
    elif hop == "country":
        sparql = f"""SELECT ?name ?iata ?value WHERE {{
  VALUES ?airport {{ <{KG2}Airport/{a1}> <{KG2}Airport/{a2}> }}
  ?airport <{KG2}airportName> ?name .
  ?airport <{KG2}iataCode> ?iata .
  ?airport <{KG2}locatedInCountry> ?country .
  ?country <{prop_uri}> ?value .
}}"""
    else:
        sparql = f"""SELECT ?name ?iata ?value WHERE {{
  VALUES ?airport {{ <{KG2}Airport/{a1}> <{KG2}Airport/{a2}> }}
  ?airport <{KG2}airportName> ?name .
  ?airport <{KG2}iataCode> ?iata .
  ?airport <{prop_uri}> ?value .
}} ORDER BY DESC(?value)"""

    label = f"comparison: {a1} vs {a2} by {prop}"
    return sparql, label
    

COUNTRY_NAME_TO_EN = {
    "pologne": "Poland", "italie": "Italy", "allemagne": "Germany",
    "autriche": "Austria", "france": "France", "espagne": "Spain",
    "grèce": "Greece", "grece": "Greece",
    "بولندا": "Poland", "إيطاليا": "Italy", "ايطاليا": "Italy",
    "ألمانيا": "Germany", "المانيا": "Germany", "النمسا": "Austria",
    "فرنسا": "France", "إسبانيا": "Spain", "اسبانيا": "Spain",
    "اليونان": "Greece",
}

def _resolve_country_value(value: str) -> str:
    return COUNTRY_NAME_TO_EN.get(value.strip().lower(), value)

AIRLINE_NAME_TO_ICAO = {
    "austrian airlines": "AUA", "austrian": "AUA",
    "brussels airlines": "BEL", "brussels": "BEL",
    "condor": "CFG",
    "air dolomiti": "DLA", "dolomiti": "DLA",
    "air cairo": "MSC",
    "air france": "AFR",
    "air india": "AIC",
    "air baltic": "BTI", "airbaltic": "BTI",
    "compagnie aérienne autrichienne": "AUA",
    "الخطوط الجوية النمساوية": "AUA",
}

def _resolve_airline_value(filter_value: str) -> str:
    return AIRLINE_NAME_TO_ICAO.get(filter_value.strip().lower(), filter_value)


def _build_count_kg1(params: dict) -> tuple[str, str] | None:
    # Schema-bleed fallback: the classifier occasionally emits count_kg2's
    # key shape (property/value) instead of count_kg1's own (filter_property/
    # filter_value), worse in fr/ar — see eval log count_kg1_002/003.
    filter_prop  = params.get("filter_property") or params.get("property", "hasDestinationCity")
    filter_value = params.get("filter_value") or params.get("value", "")
    mode         = params.get("mode", "count")

    FILTER_PROPERTY_SYNONYMS = {
        "hasOperator":      "hasAirline",
        "airlineName":      "hasAirline",
        "operatedBy":       "hasAirline",
        "airline":          "hasAirline",
        "hasDepartureCity": "hasOriginCity",
        "departureCity":    "hasOriginCity",
        "cityName":         "hasDestinationCity",  # bare 'city' hallucination defaults to destination
    }
    filter_prop = FILTER_PROPERTY_SYNONYMS.get(filter_prop, filter_prop)

    if not filter_value:
        return None

    prop_info = KG1_STRING_PROPS.get(filter_prop)
    if not prop_info:
        return None
    prop_uri = prop_info["uri"]

    value_prop_map = {
        "hasDestinationCity":    f"{KG1}dest_city",
        "hasOriginCity":         f"{KG1}orig_city",
        "hasDestinationCountry": f"{KG1}dest_country",
        "hasOriginCountry":      f"{KG1}orig_country",
    }
    value_prop = value_prop_map.get(filter_prop)

    if filter_prop == "hasAirline":
        resolved_value = _resolve_airline_value(filter_value)
        if mode == "count":
            sparql = f"""SELECT (COUNT(?flight) AS ?count) WHERE {{
  ?flight a <{KG1}Flight> .
  ?flight <{KG1}hasAirline> ?airline .
  ?airline <{KG1}operating_as> "{resolved_value}" .
}}"""
        else:
            sparql = f"""SELECT ?flight ?number WHERE {{
  ?flight a <{KG1}Flight> .
  ?flight <{KG1}flightNumber> ?number .
  ?flight <{KG1}hasAirline> ?airline .
  ?airline <{KG1}operating_as> "{resolved_value}" .
}} ORDER BY ?number LIMIT 50"""
    elif value_prop:
        if mode == "count":
            sparql = f"""SELECT (COUNT(?flight) AS ?count) WHERE {{
  ?flight a <{KG1}Flight> .
  ?flight <{prop_uri}> ?node .
  ?node <{value_prop}> "{filter_value}" .
}}"""
        else:
            sparql = f"""SELECT ?flight ?number WHERE {{
  ?flight a <{KG1}Flight> .
  ?flight <{KG1}flightNumber> ?number .
  ?flight <{prop_uri}> ?node .
  ?node <{value_prop}> "{filter_value}" .
}} ORDER BY ?number LIMIT 50"""
    else:
        return None

    label = f"{mode} of flights with {filter_prop} = {filter_value}"
    return sparql, label

def _build_count_kg3(params: dict) -> tuple[str, str] | None:
    entity_name = params.get("entity_name", "")
    property_short = params.get("property", "")
    direction   = params.get("direction", "outgoing")
    mode        = params.get("mode", "count")

    PROPERTY_SHORT_SYNONYMS = {
        "teachesCourse": "teacherOf",
        "teaches":       "teacherOf",
        "takingCourse":  "takesCourse",
        "enrolledIn":    "takesCourse",
    }
    property_short = PROPERTY_SHORT_SYNONYMS.get(property_short, property_short)

    VALID_PROPS = {"teacherOf", "takesCourse", "memberOf", "worksFor", "subOrganizationOf"}
    if not entity_name or property_short not in VALID_PROPS:
        return None

    entity_uri = map_university_entity(entity_name)
    if not entity_uri:
        return None

    prop_uri = f"{KG3}{property_short}"

    if direction == "outgoing":
        if mode == "count":
            sparql = f"""SELECT (COUNT(?obj) AS ?count) WHERE {{
  <{entity_uri}> <{prop_uri}> ?obj .
}}"""
        else:
            sparql = f"""SELECT ?obj ?name WHERE {{
  <{entity_uri}> <{prop_uri}> ?obj .
  ?obj <{KG3}name> ?name .
}} ORDER BY ?name LIMIT 50"""
    else:
        if mode == "count":
            sparql = f"""SELECT (COUNT(?subj) AS ?count) WHERE {{
  ?subj <{prop_uri}> <{entity_uri}> .
}}"""
        else:
            sparql = f"""SELECT ?subj ?name WHERE {{
  ?subj <{prop_uri}> <{entity_uri}> .
  ?subj <{KG3}name> ?name .
}} ORDER BY ?name LIMIT 50"""

    label = f"{mode} of {property_short} ({direction}) for {entity_name}"
    return sparql, label

def _build_filter_string_kg3(params: dict) -> tuple[str, str] | None:
    prop  = params.get("property", "worksFor")
    value = params.get("value", "")
    limit = int(params.get("limit") or 10)

    prop_info = KG3_STRING_PROPS.get(prop)
    if not prop_info or not value:
        return None

    prop_uri = prop_info["uri"]
    sparql = f"""SELECT ?person ?name WHERE {{
  ?person <{prop_uri}> ?dept .
  ?dept <{KG3}name> "{value}" .
  ?person <{KG3}name> ?name .
}} ORDER BY ?name LIMIT {limit}"""

    label = f"people where {prop} = {value}"
    return sparql, label


def _build_filter_numeric_kg3(params: dict) -> tuple[str, str] | None:
    # NOTE (fix): the LLM was unreliable at assigning >/< to each bound of
    # a range ("between X and Y" sometimes came back as operator='<',
    # threshold=X, operator2='>', threshold2=Y — an impossible range).
    # If the extractor gave us raw range_low/range_high instead, compute
    # the correct operators here deterministically: low bound always gets
    # '>', high bound always gets '<'. This removes the LLM's judgment
    # call for ranges entirely — see filter_numeric_kg3_005 (en) in the
    # eval log.
    if "range_low" in params and "range_high" in params:
        try:
            lo = min(float(params["range_low"]), float(params["range_high"]))
            hi = max(float(params["range_low"]), float(params["range_high"]))
            params = {**params, "operator": ">", "threshold": lo,
                      "operator2": "<", "threshold2": hi}
        except (TypeError, ValueError):
            pass  # fall through to normal operator/threshold handling below

    operator   = params.get("operator") or ""
    threshold  = params.get("threshold")
    operator2  = params.get("operator2")
    threshold2 = params.get("threshold2")

    VALID_OPERATORS = {">", "<", ">=", "<=", "="}
    if operator not in VALID_OPERATORS or threshold is None:
        return None

    having = f"COUNT(DISTINCT ?person) {operator} {threshold}"
    label  = f"departments with total headcount {operator} {threshold}"
    if operator2 in VALID_OPERATORS and threshold2 is not None:
        having += f" && COUNT(DISTINCT ?person) {operator2} {threshold2}"
        label  += f" and {operator2} {threshold2}"

    sparql = f"""SELECT ?name (COUNT(DISTINCT ?person) AS ?value) WHERE {{
  ?dept a <{KG3}Department> .
  ?dept <{KG3}name> ?name .
  {{ ?person <{KG3}memberOf> ?dept . }} UNION {{ ?person <{KG3}worksFor> ?dept . }}
}} GROUP BY ?name HAVING({having}) ORDER BY ?value"""

    return sparql, label


def _build_filter_numeric_kg1(params: dict) -> tuple[str, str] | None:
    prop      = params.get("property", "gspeed")
    operator  = params.get("operator", ">")
    threshold = params.get("threshold", 0)

    prop_info = KG1_NUMERIC_PROPS.get(prop)
    if not prop_info:
        return None

    prop_uri = prop_info["uri"]
    unit     = prop_info["unit"]

    sparql = f"""SELECT ?flight ?number (MAX(?rawValue) AS ?value) WHERE {{
  ?flight a <{KG1}Flight> .
  ?flight <{KG1}flightNumber> ?number .
  ?flight <{KG1}hasFlightEvent> ?event .
  ?event <{prop_uri}> ?rawValue .
  FILTER(?rawValue {operator} {threshold})
}} GROUP BY ?flight ?number ORDER BY DESC(?value) ?flight"""

    label = f"flights with {prop_info['label']} {operator} {threshold} {unit}"
    return sparql, label


def _build_ranking_kg1(params: dict) -> tuple[str, str] | None:
    prop  = params.get("property", "gspeed")
    order = params.get("order", "DESC")
    limit = int(params.get("limit", 5))

    prop_info = KG1_NUMERIC_PROPS.get(prop)
    if not prop_info:
        return None

    prop_uri = prop_info["uri"]

    sparql = f"""SELECT ?flight ?number (MAX(?rawValue) AS ?value) WHERE {{
  ?flight a <{KG1}Flight> .
  ?flight <{KG1}flightNumber> ?number .
  ?flight <{KG1}hasFlightEvent> ?event .
  ?event <{prop_uri}> ?rawValue .
}} GROUP BY ?flight ?number ORDER BY {order}(?value) LIMIT {limit}"""

    direction_word = "highest" if order == "DESC" else "lowest"
    label = f"top {limit} flights by {prop_info['label']} ({direction_word})"
    return sparql, label


def _build_compare_two_flights(params: dict) -> tuple[str, str] | None:
    f1   = params.get("flight1", "").strip().upper()
    f2   = params.get("flight2", "").strip().upper()
    prop = (params.get("property") or "gspeed").strip()

    prop_info = KG1_NUMERIC_PROPS.get(prop)
    if not prop_info or not f1 or not f2:
        return None

    prop_uri = prop_info["uri"]

    sparql = f"""SELECT ?number (MAX(?rawValue) AS ?value) WHERE {{
  VALUES ?number {{ "{f1}" "{f2}" }}
  ?flight <{KG1}flightNumber> ?number .
  ?flight <{KG1}hasFlightEvent> ?event .
  ?event <{prop_uri}> ?rawValue .
}} GROUP BY ?number ORDER BY DESC(?value)"""

    label = f"comparison: {f1} vs {f2} by {prop}"
    return sparql, label


def _type_filter(entity_type: str | None, var: str) -> str:
    """Optional `?var a <KG3+entity_type> .` line, or empty string when
    entity_type wasn't given. Shared by both branches of
    _build_ranking_kg3 below so the tie-safe subquery and the plain
    LIMIT query build the filter identically."""
    return f"  {var} a <{KG3}{entity_type}> .\n" if entity_type else ""


def _build_ranking_kg3(params: dict) -> tuple[str, str] | None:
    """
    Two shapes, dispatched on group_by — matches classifier.py's
    documented contract: group_by, entity_type, hop_property, order,
    limit (NOT "mode"/"property"/"department", which is what this
    function read before this fix — those keys never matched anything
    the classifier actually emits, so group_by="person" questions were
    silently always building the department-ranking query instead).

    group_by="department": rank departments by count of one entity
    type linked via hop_property (e.g. "top 3 departments by graduate
    student population" -> entity_type=GraduateStudent,
    hop_property=memberOf). entity_type matters here: memberOf alone
    covers both GraduateStudent and UndergraduateStudent in this
    ontology, so omitting the type filter silently mixes both
    populations into one count.

    group_by="person": rank people WITHIN one named department by a
    relation count (e.g. "which professor in Department0 teaches the
    most courses" -> hop_property=teacherOf). entity_type is OPTIONAL
    here — when omitted, every person linked to the department via
    worksFor is ranked regardless of rank/type. Hardcoding a single
    type (as the previous version did, to FullProfessor specifically)
    would silently exclude people of other ranks who may be tied for
    the top spot.

    TIE HANDLING: when the question is a singular superlative ("which
    department has the most...", "which professor teaches the most"),
    resolve_template forces limit=1 via the same
    _detect_singular_superlative override used for ranking_kg2. A
    plain ORDER BY...LIMIT 1 would silently drop every row tied with
    the winner. In that case the query instead computes the group-wise
    max via a subquery and FILTERs the outer rows against it, so a
    genuine tie at the top returns every tied row.
    """
    group_by     = params.get("group_by", "department")
    entity_type  = params.get("entity_type") or None
    hop_property = params.get("hop_property", "memberOf")
    order        = params.get("order", "DESC")
    limit        = int(params.get("limit", 3))
    tie_safe     = limit == 1

    if hop_property not in KG3_HOP_PROPERTIES:
        return None
    if entity_type is not None and entity_type not in KG3_ENTITY_TYPES:
        return None

    hop_uri = f"{KG3}{hop_property}"
    direction_word = "highest" if order == "DESC" else "lowest"

    if group_by == "department":
        type_outer = _type_filter(entity_type, "?person")
        type_inner = _type_filter(entity_type, "?p2")

        if tie_safe:
            sparql = f"""SELECT ?name ?value WHERE {{
  {{
    SELECT ?dept ?name (COUNT(DISTINCT ?person) AS ?value) WHERE {{
      ?dept a <{KG3}Department> .
      ?dept <{KG3}name> ?name .
{type_outer}      ?person <{hop_uri}> ?dept .
    }} GROUP BY ?dept ?name
  }}
  {{
    SELECT (MAX(?cnt) AS ?maxValue) WHERE {{
      SELECT ?d2 (COUNT(DISTINCT ?p2) AS ?cnt) WHERE {{
        ?d2 a <{KG3}Department> .
{type_inner}        ?p2 <{hop_uri}> ?d2 .
      }} GROUP BY ?d2
    }}
  }}
  FILTER(?value = ?maxValue)
}} ORDER BY {order}(?value)"""
        else:
            sparql = f"""SELECT ?name (COUNT(DISTINCT ?person) AS ?value) WHERE {{
  ?dept a <{KG3}Department> .
  ?dept <{KG3}name> ?name .
{type_outer}  ?person <{hop_uri}> ?dept .
}} GROUP BY ?name ORDER BY {order}(?value) LIMIT {limit}"""

        type_label = f"{entity_type} " if entity_type else ""
        label = f"departments ranked by {type_label}{hop_property} count ({direction_word})"

    elif group_by == "person":
        dept_name = params.get("department_name", "")
        if not dept_name:
            return None
        dept_uri = map_university_entity(dept_name)
        if not dept_uri:
            return None

        type_outer = _type_filter(entity_type, "?person")
        type_inner = _type_filter(entity_type, "?p2")

        if tie_safe:
            sparql = f"""SELECT ?name ?value WHERE {{
  {{
    SELECT ?person ?name (COUNT(?obj) AS ?value) WHERE {{
      ?person <{KG3}worksFor> <{dept_uri}> .
{type_outer}      OPTIONAL {{ ?person <{KG3}name> ?name . }}
      ?person <{hop_uri}> ?obj .
    }} GROUP BY ?person ?name
  }}
  {{
    SELECT (MAX(?cnt) AS ?maxValue) WHERE {{
      SELECT ?p2 (COUNT(?obj2) AS ?cnt) WHERE {{
        ?p2 <{KG3}worksFor> <{dept_uri}> .
{type_inner}        ?p2 <{hop_uri}> ?obj2 .
      }} GROUP BY ?p2
    }}
  }}
  FILTER(?value = ?maxValue)
}} ORDER BY {order}(?value)"""
        else:
            sparql = f"""SELECT ?name (COUNT(?obj) AS ?value) WHERE {{
  ?person <{KG3}worksFor> <{dept_uri}> .
{type_outer}  OPTIONAL {{ ?person <{KG3}name> ?name . }}
  ?person <{hop_uri}> ?obj .
}} GROUP BY ?person ?name ORDER BY {order}(?value) LIMIT {limit}"""

        label = f"people in {dept_name} ranked by {hop_property} count ({direction_word})"

    else:
        return None

    return sparql, label


def _build_compare_two_departments(params: dict) -> tuple[str, str] | None:
    """
    SELECT ?name ?value WHERE {
      VALUES ?dept { <dept1_uri> <dept2_uri> }
      ?dept ub:name ?name .
      ?person a ub:FullProfessor .
      ?person ub:worksFor ?dept .
    } GROUP BY ?dept ?name ORDER BY DESC(?value)

    entity_type and hop_property come from classifier.py's own
    documented contract for this template. department1_uri/
    department2_uri are injected deterministically by resolve_template
    (via _detect_two_university_entities_for_template + map_university
    _entity), per classifier.py's explicit note that department names
    are never extracted by the LLM for this template — the previous
    version read "dept1"/"dept2" keys that nothing in this file ever
    populated, so this builder always returned None in practice.

    Unlike ranking_kg3's person branch, entity_type is REQUIRED here,
    not optional: "does Department0 or Department9 have more full
    professors" needs the FullProfessor filter to mean anything —
    without it, the count mixes every worksFor-linked person (all
    professor ranks and lecturers together), which isn't what
    "full professors" asks for.
    """
    dept1_uri    = params.get("department1_uri", "")
    dept2_uri    = params.get("department2_uri", "")
    entity_type  = params.get("entity_type", "")
    hop_property = params.get("hop_property", "worksFor")

    if not dept1_uri or not dept2_uri:
        return None
    if entity_type not in KG3_ENTITY_TYPES:
        return None
    if hop_property not in {"worksFor", "memberOf"}:
        return None

    hop_uri = f"{KG3}{hop_property}"

    sparql = f"""SELECT ?name (COUNT(DISTINCT ?person) AS ?value) WHERE {{
  VALUES ?dept {{ <{dept1_uri}> <{dept2_uri}> }}
  ?dept <{KG3}name> ?name .
  ?person a <{KG3}{entity_type}> .
  ?person <{hop_uri}> ?dept .
}} GROUP BY ?dept ?name ORDER BY DESC(?value)"""

    label = f"comparison by {entity_type} count ({hop_property})"
    return sparql, label


def _build_cross_kg_filter(params: dict) -> tuple[str, str] | None:
    direction      = params.get("direction", "destination")
    airport_prop   = params.get("airport_property", "elevationFt")
    operator       = params.get("operator", ">")
    threshold      = params.get("threshold", 1000)
    limit          = int(params.get("limit", 10))

    iata_prop = CROSS_KG_CONFIG["destination_iata_prop"] if direction == "destination" \
                else CROSS_KG_CONFIG["origin_iata_prop"]
    airport_details_prop = CROSS_KG_CONFIG["destination_property"] if direction == "destination" \
                           else CROSS_KG_CONFIG["origin_property"]

    prop_info = KG2_NUMERIC_PROPS.get(airport_prop) or KG2_STRING_PROPS.get(airport_prop)
    if not prop_info:
        return None

    kg1_query = f"""SELECT ?number ?iata WHERE {{
  ?flight a <{KG1}Flight> .
  ?flight <{KG1}flightNumber> ?number .
  ?flight <{KG1}{airport_details_prop}> ?airport_node .
  ?airport_node <{KG1}{iata_prop}> ?iata .
}}"""

    label = f"flights to {direction} airports where {airport_prop} {operator} {threshold}"
    return (kg1_query, airport_prop, operator, threshold, direction, limit), label


# ── RESULT FORMATTERS ─────────────────────────────────────────────────────────

def _format_rows(rows: list, columns: list, max_rows: int = 200) -> str:
    lines = []
    for i, row in enumerate(rows[:max_rows]):
        parts = []
        for col in columns:
            val = row.get(col, {}).get("value", "?")
            if val.startswith("http"):
                val = val.split("/")[-1].replace("_", " ")
            else:
                try:
                    val = f"{float(val):.2f}"
                except ValueError:
                    pass
            parts.append(val)
        lines.append(", ".join(parts))
    return "\n".join(lines)

def _format_compare_answer(rows: list, params: dict) -> str | None:
    a1 = params.get("airport1", "").upper()
    a2 = params.get("airport2", "").upper()
    prop = (params.get("property") or "elevationFt").strip()
    adjective = (KG2_NUMERIC_PROPS.get(prop) or {}).get("adjective", "higher")

    values = {}
    for row in rows:
        iata = row.get("iata", {}).get("value", "").upper()
        val  = row.get("value", {}).get("value", "")
        if iata:
            values[iata] = val

    v1, v2 = values.get(a1), values.get(a2)
    if v1 is None or v2 is None:
        return None

    try:
        winner = a1 if float(v1) > float(v2) else a2
    except ValueError:
        return None

    return f"{winner} is {adjective} ({a1}: {v1}, {a2}: {v2})"


def _format_compare_flights_answer(rows: list, params: dict) -> str | None:
    """
    Builds 'X is higher (X: v1, Y: v2 — difference: d)' for
    compare_two_flights. Mirrors _format_compare_answer's shape, with a
    computed difference added — the gold answers for this template
    ("by roughly how much does it exceed the other") ask for it
    explicitly, unlike compare_two_airports.
    """
    f1 = params.get("flight1", "").strip().upper()
    f2 = params.get("flight2", "").strip().upper()

    values = {}
    for row in rows:
        number = row.get("number", {}).get("value", "").upper()
        val    = row.get("value", {}).get("value", "")
        if number:
            values[number] = val

    v1, v2 = values.get(f1), values.get(f2)
    if v1 is None or v2 is None:
        return None

    try:
        v1f, v2f = float(v1), float(v2)
    except ValueError:
        return None

    winner   = f1 if v1f > v2f else f2
    diff     = abs(v1f - v2f)
    diff_str = f"{diff:.0f}" if diff == int(diff) else f"{diff:.2f}"

    return f"{winner} is higher ({f1}: {v1}, {f2}: {v2} — difference: {diff_str})"


def _format_compare_departments_answer(rows: list, params: dict) -> str | None:
    """
    Builds 'X has more (a vs b)' for compare_two_departments. Reads
    department1_name/department2_name (the original question order,
    set deterministically in resolve_template) rather than assuming
    SPARQL's row order matches the order the question named them in —
    ORDER BY DESC(?value) sorts by count, not by question order.
    """
    d1 = params.get("department1_name", "")
    d2 = params.get("department2_name", "")

    values = {}
    for row in rows:
        name = row.get("name", {}).get("value", "")
        val  = row.get("value", {}).get("value", "")
        if name:
            values[name] = val

    v1, v2 = values.get(d1), values.get(d2)
    if v1 is None or v2 is None:
        return None

    try:
        v1f, v2f = float(v1), float(v2)
    except ValueError:
        return None

    winner = d1 if v1f > v2f else d2
    return f"{winner} has more ({int(v1f)} vs {int(v2f)})"


def _format_answer(question: str, raw_data: str, lang: str, total_count: int = None) -> str:
    lang_map = {"en": "English", "fr": "French", "ar": "Arabic"}
    language = lang_map.get(lang, "English")

    lines = [ln.strip() for ln in raw_data.strip().split("\n") if ln.strip()]
    count = len(lines)

    if count == 0:
        return "No results found." if lang == "en" else raw_data

    if count == 1 and lines[0].replace(".", "", 1).isdigit():
        return f"The answer is {lines[0]}."

    if count == 1:
        return lines[0]

    listed = "\n".join(f"{i+1}. {ln}" for i, ln in enumerate(lines))
    if total_count is not None and total_count > count:
        return f"There are {total_count} result(s) — showing the first {count}:\n\n{listed}"
    return f"There are {count} result(s):\n\n{listed}"


# ── MAIN RESOLVER (templates) ─────────────────────────────────────────────────

# Expected param keys per template, drawn from each builder's own
# params.get(...) calls plus the deterministic keys resolve_template injects
# (department1_uri, entity_name, etc.). Used only by the schema-bleed guard
# below to flag suspicious cross-template key leakage — not for validation
# or filtering, so an incomplete entry here can only under-warn, never break
# a template that was working before.
_TEMPLATE_PARAM_KEYS = {
    "filter_numeric_kg2":       {"property", "operator", "threshold"},
    "filter_string_kg2":        {"property", "value"},
    "count_kg2":                {"property", "operator", "threshold", "value"},
    "ranking_kg2":              {"property", "order", "limit"},
    "compare_two_airports":     {"airport1", "airport2", "property"},
    "count_kg1":                {"filter_property", "filter_value", "mode"},
    "filter_numeric_kg1":       {"property", "operator", "threshold"},
    "cross_kg_filter":          {"direction", "airport_property", "operator", "threshold", "limit"},
    "count_kg3":                {"property", "direction", "mode", "entity_name"},
    "filter_string_kg3":        {"property", "value", "limit"},
    "group_aggregate_kg1":      {"group_by", "property", "function", "limit"},
    "ranking_kg1":              {"property", "order", "limit"},
    "compare_two_flights":      {"flight1", "flight2", "property"},
    "group_aggregate_kg2":      {"group_by", "property", "function", "limit"},
    "group_aggregate_kg3":      {"group_by", "property", "function", "limit"},
    "filter_numeric_kg3":       {"operator", "threshold", "operator2", "threshold2"},
    "ranking_kg3":              {"group_by", "entity_type", "hop_property", "order",
                                  "limit", "department_name"},
    "compare_two_departments":  {"entity_type", "hop_property", "dept1", "dept2",
                                  "department1_uri", "department2_uri",
                                  "department1_name", "department2_name"},
}


def resolve_template(question: str, template_name: str, lang: str, router_params: dict = None) -> dict:
    result = {
        "success":      False,
        "template":     template_name,
        "params":       {},
        "sparql":       None,
        "raw_data":     None,
        "final_answer": None,
        "failure_type": None,
    }

    if router_params:
        # router_params only ever carries what the router extracted
        # deterministically (e.g. named entities/URIs). Some templates also
        # need LLM-classified fields (e.g. compare_two_departments' own
        # entity_type/hop_property) that the router never sets — replacing
        # params outright instead of merging silently dropped those keys,
        # so the builder always failed for any template mixing both kinds
        # of params. Still call _extract_params, then let router_params
        # (trustworthy/deterministic) win on any overlapping key.
        llm_params = _extract_params(question, template_name, lang)

        # NOTE (fix): filter_numeric_kg3 is a special case. router_params
        # here comes from the router's coarse classifier prompt, which has
        # no concept of ranges ("between X and Y") — it only ever emits a
        # single operator/threshold. When the question is actually a range,
        # _extract_params's OWN dedicated filter_numeric_kg3 prompt (which
        # DOES understand operator2/threshold2) produces the more complete
        # extraction. Letting router_params win in that case (the normal
        # merge order below) silently overwrites a correct two-bound
        # extraction with an incomplete one-bound guess — see
        # filter_numeric_kg3_005 in the eval log. Trust the dedicated
        # extraction instead whenever it found a range and the router
        # didn't.
        if (template_name == "filter_numeric_kg3"
                and "operator2" in llm_params
                and "operator2" not in router_params):
            params = llm_params
            print(f"[template] filter_numeric_kg3 range detected — using "
                  f"dedicated extraction {llm_params} over router params "
                  f"{router_params}")
        else:
            params = {**llm_params, **router_params}
            print(f"[template] Merged router params {router_params} over "
                  f"LLM params {llm_params} -> {params}")
    else:
        params = _extract_params(question, template_name, lang)
    params = _sanitize_params(params)
    result["params"] = params

    if not params:
        result["failure_type"] = "param_extraction_failure"
        return result

    builders = {
        "filter_numeric_kg2":   (_build_filter_numeric_kg2,   KG2_EP, ["name", "value"]),
        "filter_string_kg2":    (_build_filter_string_kg2,    KG2_EP, ["name"]),
        "count_kg2":            (_build_count_kg2,            KG2_EP, ["count"]),
        "ranking_kg2":          (_build_ranking_kg2,          KG2_EP, ["name", "value"]),
        "compare_two_airports": (_build_compare_two_airports, KG2_EP, ["name", "iata", "value"]),
        "count_kg1":            (_build_count_kg1,            KG1_EP, ["count"]),
        "filter_numeric_kg1":   (_build_filter_numeric_kg1,   KG1_EP, ["number", "value"]),
        "cross_kg_filter":      (_build_cross_kg_filter,      None,   None),
        "count_kg3":            (_build_count_kg3,            KG3_EP, ["name"]),
        "filter_string_kg3":    (_build_filter_string_kg3,    KG3_EP, ["name"]),
        "group_aggregate_kg1":  (_build_group_aggregate_kg1,  KG1_EP, ["groupName", "agg"]),
        "ranking_kg1":          (_build_ranking_kg1,          KG1_EP, ["number", "value"]),
        "compare_two_flights":  (_build_compare_two_flights,  KG1_EP, ["number", "value"]),
        "group_aggregate_kg2":  (_build_group_aggregate_kg2,  KG2_EP, ["groupName", "agg"]),
        "group_aggregate_kg3":  (_build_group_aggregate_kg3,  KG3_EP, ["deptName", "agg"]),
        "filter_numeric_kg3":   (_build_filter_numeric_kg3,   KG3_EP, ["name", "value"]),
        "ranking_kg3":              (_build_ranking_kg3,              KG3_EP, ["name", "value"]),
        "compare_two_departments":  (_build_compare_two_departments,  KG3_EP, ["name", "value"]),
    }

    if template_name == "count_kg3":
        entity_name = _detect_university_entity_for_template(question)
        if not entity_name:
            result["failure_type"] = "param_extraction_failure"
            return result
        params["entity_name"] = entity_name

    if template_name == "ranking_kg3" and params.get("group_by") == "person":
        dept_name = _detect_university_entity_for_template(question)
        if not dept_name:
            result["failure_type"] = "param_extraction_failure"
            return result
        params["department_name"] = dept_name

    if template_name == "compare_two_departments":
        two_depts = _detect_two_university_entities_for_template(question)
        if not two_depts:
            result["failure_type"] = "param_extraction_failure"
            return result
        d1_uri = map_university_entity(two_depts[0])
        d2_uri = map_university_entity(two_depts[1])
        if not d1_uri or not d2_uri:
            result["failure_type"] = "param_extraction_failure"
            return result
        params["department1_uri"]  = d1_uri
        params["department2_uri"]  = d2_uri
        params["department1_name"] = two_depts[0]
        params["department2_name"] = two_depts[1]

    if template_name == "ranking_kg3" and _detect_singular_superlative(question):
        if params.get("limit") != 1:
            print(f"[template] Singular-superlative override: limit "
                  f"{params.get('limit')!r} -> 1")
        params["limit"] = 1

    if "limit" in _TEMPLATE_PARAM_KEYS.get(template_name, set()) and _detect_singular_superlative(question):
        if params.get("limit") != 1:
            print(f"[template] Singular-superlative override: limit "
                  f"{params.get('limit')!r} -> 1")
        else:
            print(f"[template] Singular-superlative detected — limit already 1, "
                  f"override redundant this run")
        params["limit"] = 1

    if template_name not in builders:
        result["failure_type"] = "unknown_template"
        return result

    _foreign_keys = set(params) - _TEMPLATE_PARAM_KEYS.get(template_name, set(params))
    if _foreign_keys:
        print(f"[template] WARNING: params for '{template_name}' contain keys "
              f"from another template's schema: {_foreign_keys} — likely "
              f"classifier schema bleed. Full params: {params}")

        if template_name == "ranking_kg3" and "property" in params and "hop_property" not in params:
            print(f"[template] Auto-correcting schema bleed: 'property' -> 'hop_property'")
            params["hop_property"] = params.pop("property")
            params.pop("function", None)
            params.setdefault("group_by", "department")
            params.setdefault("order", "DESC")

    builder_fn, endpoint, columns = builders[template_name]
    build_result = builder_fn(params)

    if not build_result:
        result["failure_type"] = "sparql_build_failure"
        return result

    sparql, label = build_result
    print(f"[template] Query type: {label}")

    compare_rows = None

    if template_name == "cross_kg_filter":
        kg1_query, airport_prop, operator, threshold, direction, limit = sparql
        result["sparql"] = kg1_query

        rows_kg1, sparql_error = _run_sparql(KG1_EP, kg1_query)
        if sparql_error:
            result["failure_type"] = "execution_failure"
            result["error_detail"] = sparql_error
            return result
        if not rows_kg1:
            result["failure_type"] = "no_results"
            return result

        matched_flights = []
        seen_iatas      = {}

        prop_info = KG2_NUMERIC_PROPS.get(airport_prop) or KG2_STRING_PROPS.get(airport_prop)
        prop_uri  = prop_info["uri"] if prop_info else f"{KG2}{airport_prop}"
        hop       = prop_info.get("hop", "direct") if prop_info else "direct"

        for row in rows_kg1:
            iata   = row.get("iata", {}).get("value", "")
            number = row.get("number", {}).get("value", "")
            if not iata:
                continue

            if iata not in seen_iatas:
                if hop == "runway":
                    kg2_q = f"""SELECT ?val WHERE {{
  ?ap <{KG2}iataCode> "{iata}" .
  ?ap <{KG2}hasRunway> ?rw .
  ?rw <{prop_uri}> ?val .
}} ORDER BY DESC(?val) LIMIT 1"""
                elif hop == "country":
                    kg2_q = f"""SELECT ?val WHERE {{
  ?ap <{KG2}iataCode> "{iata}" .
  ?ap <{KG2}locatedInCountry> ?c .
  ?c <{prop_uri}> ?val .
}} LIMIT 1"""
                else:
                    kg2_q = f"""SELECT ?val WHERE {{
  ?ap <{KG2}iataCode> "{iata}" .
  ?ap <{prop_uri}> ?val .
}} LIMIT 1"""

                kg2_rows, _kg2_error = _run_sparql(KG2_EP, kg2_q)
                val_raw  = kg2_rows[0].get("val", {}).get("value", "") if kg2_rows else ""
                seen_iatas[iata] = val_raw

            val_raw = seen_iatas[iata]
            if not val_raw:
                continue

            try:
                if operator in [">", "<", ">=", "<="]:
                    val_num = float(val_raw)
                    thr_num = float(threshold)
                    cond    = eval(f"{val_num} {operator} {thr_num}")
                else:
                    cond = str(val_raw).lower() == str(threshold).lower()
            except Exception:
                cond = False

            if cond:
                matched_flights.append(f"{number} (airport {iata}: {airport_prop}={val_raw})")
                if len(matched_flights) >= limit:
                    break

        if not matched_flights:
            result["failure_type"] = "execution_failure"
            return result

        raw_data = "\n".join(matched_flights)
        result["sparql"]  = kg1_query
        result["raw_data"] = raw_data

    else:
        result["sparql"] = sparql
        rows, sparql_error = _run_sparql(endpoint, sparql)

        if sparql_error:
            result["failure_type"] = "execution_failure"
            result["error_detail"] = sparql_error
            return result
        if not rows:
            result["failure_type"] = "no_results"
            return result

        if template_name in ("count_kg1", "count_kg2", "count_kg3") and params.get("mode", "count") == "count":
            count_val = rows[0].get("count", {}).get("value", "0") if rows else "0"
            raw_data  = count_val
        else:
            raw_data = _format_rows(rows, columns)
            result["total_rows"] = len(rows)

        result["raw_data"] = raw_data
        if template_name in ("compare_two_airports", "compare_two_flights", "compare_two_departments"):
            compare_rows = rows

    _COMPARE_FORMATTERS = {
        "compare_two_airports":     _format_compare_answer,
        "compare_two_flights":      _format_compare_flights_answer,
        "compare_two_departments":  _format_compare_departments_answer,
    }
    if template_name in _COMPARE_FORMATTERS and compare_rows:
        compare_answer = _COMPARE_FORMATTERS[template_name](compare_rows, params)
        if compare_answer:
            result["final_answer"] = compare_answer
            result["success"]      = True
            result["failure_type"] = "success"
            return result

    final_answer = _format_answer(question, result["raw_data"], lang, result.get("total_rows"))
    result["final_answer"] = final_answer
    result["success"]      = True
    result["failure_type"] = "success"

    return result


# ── ASK RESOLVER ───────────────────────────────────────────────────────────────

def _format_ask_answer(result: bool, lang: str) -> str:
    templates = {
        "en": {"true": "Yes.", "false": "No."},
        "fr": {"true": "Oui.", "false": "Non."},
        "ar": {"true": "نعم.", "false": "لا."},
    }
    lang_templates = templates.get(lang, templates["en"])
    return lang_templates["true"] if result else lang_templates["false"]


def resolve_ask_query(question: str, routing: dict, lang: str) -> dict:
    result = {
        "success":      False,
        "entity_uri":   None,
        "property_uri": None,
        "value":        None,
        "sparql":       None,
        "raw_answer":   None,
        "final_answer": None,
        "failure_type": None,
    }

    kg     = routing["kg"]
    entity = routing["entity"]

    entities = extract_ask_entities(question, lang, entity)
    if not validate_ask_extraction(entities):
        result["failure_type"] = "extraction_failure"
        return result
    print(f"[ask_query] lang={lang} extracted property='{entities['property']}' value='{entities['value']}'")

    if kg == "flights":
        entity_uri   = map_flight(entity)
        lexicon_path = get_lexicon("flights")
        base_uri     = get_base_uri("flights")
    elif kg == "airports":
        entity_uri   = map_airport(entity)
        lexicon_path = get_lexicon("airports")
        base_uri     = get_base_uri("airports")
    elif kg == "university":
        entity_uri   = map_university_entity(entity)
        lexicon_path = get_lexicon("university")
        base_uri     = get_base_uri("university")
    else:
        result["failure_type"] = "unknown_kg"
        return result

    if not entity_uri:
        result["failure_type"] = "mapping_failure"
        return result
    result["entity_uri"] = entity_uri

    lexicon = load_lexicon(lexicon_path)
    property_uri, tier, property2_uri, score = map_property_cascade_scored(
        entities["property"], lexicon, lexicon_path
    )
    if not property_uri:
        result["failure_type"] = "mapping_failure"
        return result

    if tier == "semantic" and score < ASK_SEMANTIC_THRESHOLD:
        print(f"[ask_query] Rejecting low-confidence semantic match "
            f"(score={score:.3f}, tier={tier}) for property="
            f"'{entities['property']}' — refusing to guess.")
        if score < 0.87:
            result["failure_type"] = "success"
        else:
            result["failure_type"] = "mapping_failure"
        return result

    full_property_uri  = base_uri + property_uri
    full_property2_uri = (base_uri + property2_uri) if property2_uri else None
    result["property_uri"] = full_property_uri

    ask_value = entities["value"]
    if full_property2_uri and full_property2_uri.endswith("countryName"):
        ask_value = _resolve_country_value(ask_value)
    elif full_property2_uri and full_property2_uri.endswith("surface"):
        ask_value = _resolve_surface_value(ask_value)
    result["value"] = ask_value

    sparql = build_ask_query(
        entity_uri, full_property_uri, ask_value,
        property2_uri=full_property2_uri
    )
    result["sparql"] = sparql

    endpoint = get_endpoint(kg)
    ask_result = execute_ask_sparql(sparql, endpoint)

    if ask_result is None:
        result["failure_type"] = "execution_failure"
        return result

    result["raw_answer"]   = ask_result
    result["final_answer"] = _format_ask_answer(ask_result, lang)
    result["success"]      = True
    result["failure_type"] = "success"
    return result