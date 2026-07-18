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

    KG3 — University queries:
        count_kg3              : count/list entities linked to a named entity
        filter_string_kg3      : filter people by department membership

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
)
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
KG1_EP = get_endpoint("flights")
KG2_EP = get_endpoint("airports")
KG3_EP = get_endpoint("university")

# ── PROPERTY MAPS ─────────────────────────────────────────────────────────────
# Maps short property names to full URIs + metadata.
# Used to validate extracted parameters before injecting into templates.

KG2_NUMERIC_PROPS = {
    "elevationFt": {"uri": f"{KG2}elevationFt",  "label": "elevation",     "unit": "feet",   "hop": "direct"},
    "lengthFt":    {"uri": f"{KG2}lengthFt",     "label": "runway length",  "unit": "feet",   "hop": "runway"},
    "widthFt":     {"uri": f"{KG2}widthFt",      "label": "runway width",   "unit": "feet",   "hop": "runway"},
    "latitude":    {"uri": f"{KG2}latitude",     "label": "latitude",       "unit": "degrees","hop": "direct"},
    "longitude":   {"uri": f"{KG2}longitude",    "label": "longitude",      "unit": "degrees","hop": "direct"},
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

def _detect_university_entity_for_template(q: str):
    m = _UNIVERSITY_ENTITY_RE.search(q)
    return m.group(1) if m else None
# ── SPARQL HELPER ─────────────────────────────────────────────────────────────

def _run_sparql(endpoint: str, query: str, multiple: bool = True):
    """Execute SPARQL and return list of binding dicts."""
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
                return bindings
            return bindings[0] if bindings else None
    except Exception as e:
        print(f"[template] SPARQL error: {e}")
        return [] if multiple else None


# ── LLM PARAMETER EXTRACTOR ───────────────────────────────────────────────────

def _extract_params(question: str, template_name: str, lang: str) -> dict:
    """
    Uses the LLM to extract structured parameters from the question.
    The LLM extracts VALUES only — never SPARQL structure.

    Returns a dict with extracted parameters or empty values on failure.
    """
    prompts = {

        "filter_numeric_kg2": f"""Extract parameters from this airport question.
Question: "{question}"
Return ONLY a JSON object with these keys:
- "property": one of [elevationFt, lengthFt, widthFt]
- "operator": one of [>, <, >=, <=]
- "threshold": numeric value (integer)
- "limit": number of results to return (default 10)
Example: {{"property": "elevationFt", "operator": ">", "threshold": 1000, "limit": 10}}
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

Return ONLY a JSON object with keys: "property", "value", "limit" (default 10).
Example: {{"property": "countryName", "value": "France", "limit": 10}}
Return ONLY the JSON. No explanation.""",

        "ranking_kg2": f"""Extract parameters from this airport ranking question.
Question: "{question}"
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
- "limit": number of results (default 10)
Example: {{"property": "alt", "operator": ">", "threshold": 30000, "limit": 10}}
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
Example: {{"group_by": "country", "property": "elevationFt", "function": "AVG"}}
Return ONLY the JSON. No explanation.""",

        "group_aggregate_kg3": f"""Extract parameters from this university aggregation question.
Question: "{question}"
Return ONLY a JSON object with these keys:
- "property": "teacherOf" (courses taught) or "takesCourse" (courses taken)
- "function": one of [AVG, MAX, MIN]
Group-by is always "department" for this template — do not extract it.
Example: {{"property": "teacherOf", "function": "AVG"}}
Return ONLY the JSON. No explanation.""",
    }

    prompt = prompts.get(template_name, "")
    if not prompt:
        return {}

    try:
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )
        raw  = response["message"]["content"].strip()
        # Strip markdown code fences if present
        raw  = re.sub(r"```json|```", "", raw).strip()
        # Find the first {...} block even if the LLM added surrounding text
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if not match:
            print(f"[template] No JSON object found in LLM output: {repr(raw[:100])}")
            return {}
        return json.loads(match.group())                # ← replaces json.loads(raw)
    except Exception as e:
        print(f"[template] Parameter extraction failed: {e}")
        return {}


# ── SPARQL BUILDERS ───────────────────────────────────────────────────────────

def _build_filter_numeric_kg2(params: dict) -> tuple[str, str] | None:
    """
    SELECT ?airport ?name ?value WHERE {
      ?airport a ao:Airport .
      ?airport ao:airportName ?name .
      ?airport ao:elevationFt ?value .
      FILTER(?value > 1000)
    } ORDER BY DESC(?value) LIMIT 10
    """
    prop      = params.get("property", "elevationFt")
    operator  = params.get("operator") or ""
    threshold = params.get("threshold")
    limit     = int(params.get("limit") or 10)

    VALID_OPERATORS = {">", "<", ">=", "<=", "="}
    if operator not in VALID_OPERATORS or threshold is None:
        return None   # triggers sparql_build_failure instead of HTTP 400

    VALID_OPERATORS = {">", "<", ">=", "<=", "="}
    if operator not in VALID_OPERATORS or threshold is None:
        return None   # triggers sparql_build_failure instead of HTTP 400

    # Runway properties need a hop through hasRunway
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
  FILTER(?value {operator} {threshold})
}} ORDER BY DESC(?value) ?airport LIMIT {limit}"""
    else:
        sparql = f"""SELECT ?airport ?name ?value WHERE {{
  ?airport a <{KG2}Airport> .
  ?airport <{KG2}airportName> ?name .
  ?airport <{prop_uri}> ?value .
  FILTER(?value {operator} {threshold})
}} ORDER BY DESC(?value) ?airport LIMIT {limit}"""

    label = f"airports with {prop_info['label']} {operator} {threshold} {unit}"
    return sparql, label


def _build_filter_string_kg2(params: dict) -> tuple[str, str] | None:
    """
    SELECT ?airport ?name WHERE {
      ?airport a ao:Airport .
      ?airport ao:airportName ?name .
      ?airport ao:airportType "large_airport" .
    } LIMIT 10
    """
    prop  = params.get("property", "airportType")
    value = params.get("value", "")
    limit = int(params.get("limit") or 10)

    prop_info = KG2_STRING_PROPS.get(prop)
    if not prop_info or not value:
        return None

    # Country requires a hop through locatedInCountry
    if prop == "countryName":
        sparql = f"""SELECT ?airport ?name WHERE {{
  ?airport a <{KG2}Airport> .
  ?airport <{KG2}airportName> ?name .
  ?airport <{KG2}locatedInCountry> ?country .
  ?country <{KG2}countryName> "{value}" .
}} ORDER BY ?name LIMIT {limit}"""
    else:
        prop_uri = prop_info["uri"]
        sparql = f"""SELECT ?airport ?name WHERE {{
  ?airport a <{KG2}Airport> .
  ?airport <{KG2}airportName> ?name .
  ?airport <{prop_uri}> "{value}" .
}} ORDER BY ?name LIMIT {limit}"""

    label = f"airports where {prop} = {value}"
    return sparql, label


def _build_ranking_kg2(params: dict) -> tuple[str, str] | None:
    """
    SELECT ?airport ?name ?value WHERE {
      ?airport a ao:Airport .
      ?airport ao:airportName ?name .
      ?airport ao:elevationFt ?value .
    } ORDER BY DESC(?value) LIMIT 5
    """
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
    """
    SELECT ?groupName (AVG(?value) AS ?agg) WHERE {
      ?flight a fo:Flight .
      ?flight fo:hasAirline ?airline .
      ?airline fo:operating_as ?groupName .
      ?flight fo:hasFlightEvent ?event .
      ?event fo:gspeed ?value .
    } GROUP BY ?groupName ORDER BY DESC(?agg)
    """
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

    sparql = f"""SELECT ?groupName (ROUND({function}(?value) * 100) / 100 AS ?agg) WHERE {{
  ?flight a <{KG1}Flight> .
  ?flight <{KG1}{group_info['hop_property']}> ?groupNode .
  ?groupNode <{KG1}{group_info['name_property']}> ?groupName .
  ?flight <{KG1}{prop_info['hop']}> ?event .
  ?event <{KG1}{prop}> ?value .
}} GROUP BY ?groupName ORDER BY DESC(?agg)"""

    label = f"{function} of {prop} grouped by airline"
    return sparql, label


def _build_group_aggregate_kg2(params: dict) -> tuple[str, str] | None:
    """
    Direct property (elevationFt):
      SELECT ?groupName (AVG(?value) AS ?agg) WHERE {
        ?airport a ao:Airport .
        ?airport ao:locatedInCountry ?c .
        ?c ao:countryName ?groupName .
        ?airport ao:elevationFt ?value .
      } GROUP BY ?groupName

    Runway property (lengthFt/widthFt): adds the hasRunway hop.
    """
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

    if prop_info["hop"] == "hasRunway":
        sparql = f"""SELECT ?groupName (ROUND({function}(?value) * 100) / 100 AS ?agg) WHERE {{
  ?airport a <{KG2}Airport> .
  ?airport <{KG2}{group_info['hop_property']}> ?groupNode .
  ?groupNode <{KG2}{group_info['name_property']}> ?groupName .
  ?airport <{KG2}hasRunway> ?runway .
  ?runway <{KG2}{prop}> ?value .
}} GROUP BY ?groupName ORDER BY DESC(?agg)"""
    else:
        sparql = f"""SELECT ?groupName (ROUND({function}(?value) * 100) / 100 AS ?agg) WHERE {{
  ?airport a <{KG2}Airport> .
  ?airport <{KG2}{group_info['hop_property']}> ?groupNode .
  ?groupNode <{KG2}{group_info['name_property']}> ?groupName .
  ?airport <{KG2}{prop}> ?value .
}} GROUP BY ?groupName ORDER BY DESC(?agg)"""

    label = f"{function} of {prop} grouped by {group_by}"
    return sparql, label


def _build_group_aggregate_kg3(params: dict) -> tuple[str, str] | None:
    """
    Nested subquery — see design note. Inner query counts the relation
    per (person, department) pair; outer query aggregates those counts
    per department.

      SELECT ?deptName (AVG(?cnt) AS ?agg) WHERE {
        SELECT ?dept ?deptName (COUNT(?obj) AS ?cnt) WHERE {
          ?person ub:worksFor ?dept .
          ?dept ub:name ?deptName .
          ?person ub:teacherOf ?obj .
        } GROUP BY ?dept ?deptName
      } GROUP BY ?deptName
    """
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

    sparql = f"""SELECT ?deptName (ROUND({function}(?cnt) * 100) / 100 AS ?agg) WHERE {{
  SELECT ?person ?dept ?deptName (COUNT(?obj) AS ?cnt) WHERE {{
    ?person <{KG3}{group_info['hop_property']}> ?dept .
    ?dept <{KG3}{group_info['name_property']}> ?deptName .
    ?person <{KG3}{prop}> ?obj .
  }} GROUP BY ?person ?dept ?deptName
}} GROUP BY ?deptName ORDER BY DESC(?agg)"""

    label = f"{function} of {prop_info['label']} per person, grouped by department"
    return sparql, label
def _build_compare_two_airports(params: dict) -> tuple[str, str] | None:
    """
    SELECT ?name ?value WHERE {
      VALUES ?airport { ao:Airport/VIE ao:Airport/FRA }
      ?airport ao:airportName ?name .
      ?airport ao:elevationFt ?value .
    } ORDER BY DESC(?value)
    """
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
        sparql = f"""SELECT ?name ?value WHERE {{
  VALUES ?airport {{ <{KG2}Airport/{a1}> <{KG2}Airport/{a2}> }}
  ?airport <{KG2}airportName> ?name .
  ?airport <{KG2}hasRunway> ?runway .
  ?runway <{prop_uri}> ?value .
}} ORDER BY DESC(?value)"""
    elif hop == "country":
        sparql = f"""SELECT ?name ?value WHERE {{
  VALUES ?airport {{ <{KG2}Airport/{a1}> <{KG2}Airport/{a2}> }}
  ?airport <{KG2}airportName> ?name .
  ?airport <{KG2}locatedInCountry> ?country .
  ?country <{prop_uri}> ?value .
}}"""
    else:
        sparql = f"""SELECT ?name ?value WHERE {{
  VALUES ?airport {{ <{KG2}Airport/{a1}> <{KG2}Airport/{a2}> }}
  ?airport <{KG2}airportName> ?name .
  ?airport <{prop_uri}> ?value .
}} ORDER BY DESC(?value)"""

    label = f"comparison: {a1} vs {a2} by {prop}"
    return sparql, label
    

# ── AIRLINE NAME → ICAO CODE LOOKUP ────────────────────────────────────────
# The KG1 Airline nodes only store ICAO 3-letter codes (operating_as /
# painted_as), never a human-readable name. The LLM extraction prompt for
# count_kg1 asks for a natural-language "airline name", so without this
# lookup, filter_value never matches anything in the graph (silent 0 result,
# not a query error — this is why it passed sparql_valid=True in eval runs).
#
# Verified against independent public sources (Wikipedia List of airline
# codes + airline-code lookup sites), NOT guessed from memory.
# The KG's full set of ICAO codes present in the current 369-flight dataset
# is 31 codes; only the ones below are confirmed. The rest are left
# unmapped on purpose rather than guessed — extend this dict once verified.
AIRLINE_NAME_TO_ICAO = {
    # English
    "austrian airlines": "AUA", "austrian": "AUA",
    "brussels airlines": "BEL", "brussels": "BEL",
    "condor": "CFG",
    "air dolomiti": "DLA", "dolomiti": "DLA",
    "air cairo": "MSC",
    "air france": "AFR",
    "air india": "AIC",
    "air baltic": "BTI", "airbaltic": "BTI",
    # French
    "compagnie aérienne autrichienne": "AUA",
    # Arabic
    "الخطوط الجوية النمساوية": "AUA",
}
# NOTE: AZG, BRX, CTN, EVA, EWL, FCM, FIN, FSF, KAL, LGL, LOT, MAE, MAY,
# OAW, PEV, PGT, RYS, SXS, THY, TKJ, TVF, WMT are the other ICAO codes
# present in flight_ontology-materialized.ttl. Add them here once you've
# verified each against an authoritative source (e.g. ICAO Doc 8585) —
# deliberately left out rather than filled in from an unverified guess.


def _resolve_airline_value(filter_value: str) -> str:
    """Map a natural-language airline name to its KG ICAO code if known;
    otherwise pass the raw value through unchanged (preserves prior
    behavior for values that already are ICAO codes, e.g. from list/API
    input rather than an LLM-extracted name)."""
    return AIRLINE_NAME_TO_ICAO.get(filter_value.strip().lower(), filter_value)


def _build_count_kg1(params: dict) -> tuple[str, str] | None:
    """
    SELECT (COUNT(?flight) AS ?count) WHERE {
      ?flight a fo:Flight .
      ?flight fo:hasDestinationCity ?city .
      ?city fo:dest_city "Munich" .
    }
    -- OR for list mode --
    SELECT ?flight ?number WHERE {
      ?flight a fo:Flight .
      ?flight fo:flightNumber ?number .
      ?flight fo:hasDestinationCity ?city .
      ?city fo:dest_city "Munich" .
    } ORDER BY ?number
    """
    filter_prop  = params.get("filter_property", "hasDestinationCity")
    filter_value = params.get("filter_value", "")
    mode         = params.get("mode", "count")

    if not filter_value:
        return None

    prop_info = KG1_STRING_PROPS.get(filter_prop)
    if not prop_info:
        return None

    prop_uri = prop_info["uri"]

    # Determine the value property based on the filter property
    value_prop_map = {
        "hasDestinationCity":    f"{KG1}dest_city",
        "hasOriginCity":         f"{KG1}orig_city",
        "hasDestinationCountry": f"{KG1}dest_country",
        "hasOriginCountry":      f"{KG1}orig_country",
    }
    value_prop = value_prop_map.get(filter_prop)

    if filter_prop == "hasAirline":
        # Airline requires operating_as lookup — resolve name → ICAO code first
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
    """
    Two shapes depending on direction:

    OUTGOING (entity is subject, e.g. "how many courses does X teach"):
        SELECT (COUNT(?obj) AS ?count) WHERE {
          <entity_uri> ub:teacherOf ?obj .
        }

    INCOMING (entity is object, e.g. "how many students are in department X"):
        SELECT (COUNT(?subj) AS ?count) WHERE {
          ?subj ub:memberOf <entity_uri> .
        }

    List mode adds ?name via ub:name and orders by it, mirroring count_kg1's
    list-mode pattern.
    """
    entity_name = params.get("entity_name", "")
    property_short = params.get("property", "")
    direction   = params.get("direction", "outgoing")
    mode        = params.get("mode", "count")

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
    else:  # incoming
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
    """
    SELECT ?person ?name WHERE {
      ?person ub:worksFor ?dept .
      ?dept ub:name "Department3" .
      ?person ub:name ?name .
    } ORDER BY ?name LIMIT 10
    """
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

def _build_filter_numeric_kg1(params: dict) -> tuple[str, str] | None:
    """
    SELECT ?flight ?number ?value WHERE {
      ?flight a fo:Flight .
      ?flight fo:flightNumber ?number .
      ?flight fo:hasFlightEvent ?event .
      ?event fo:gspeed ?value .
      FILTER(?value > 400)
    } ORDER BY DESC(?value) LIMIT 10
    """
    prop      = params.get("property", "gspeed")
    operator  = params.get("operator", ">")
    threshold = params.get("threshold", 0)
    limit     = int(params.get("limit", 10))

    prop_info = KG1_NUMERIC_PROPS.get(prop)
    if not prop_info:
        return None

    prop_uri = prop_info["uri"]
    unit     = prop_info["unit"]

    sparql = f"""SELECT ?flight ?number ?value WHERE {{
  ?flight a <{KG1}Flight> .
  ?flight <{KG1}flightNumber> ?number .
  ?flight <{KG1}hasFlightEvent> ?event .
  ?event <{prop_uri}> ?value .
  FILTER(?value {operator} {threshold})
}} ORDER BY DESC(?value) ?flight LIMIT {limit}"""

    label = f"flights with {prop_info['label']} {operator} {threshold} {unit}"
    return sparql, label


def _build_cross_kg_filter(params: dict) -> tuple[str, str] | None:
    """
    Two-step query:
    Step 1 (KG1): get flights + their destination IATA codes
    Step 2 (KG2): filter airports by property condition
    Step 3: intersect — return only flights whose airport matches
    """
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

    # Step 1: get all (flight_number, iata) pairs from KG1
    kg1_query = f"""SELECT ?number ?iata WHERE {{
  ?flight a <{KG1}Flight> .
  ?flight <{KG1}flightNumber> ?number .
  ?flight <{KG1}{airport_details_prop}> ?airport_node .
  ?airport_node <{KG1}{iata_prop}> ?iata .
}}"""

    label = f"flights to {direction} airports where {airport_prop} {operator} {threshold}"
    return (kg1_query, airport_prop, operator, threshold, direction, limit), label


# ── RESULT FORMATTERS ─────────────────────────────────────────────────────────

def _format_rows(rows: list, columns: list, max_rows: int = 20) -> str:
    """Converts SPARQL result rows to a readable string."""
    lines = []
    for i, row in enumerate(rows[:max_rows]):
        parts = []
        for col in columns:
            val = row.get(col, {}).get("value", "?")
            # Clean URIs
            if val.startswith("http"):
                val = val.split("/")[-1].replace("_", " ")
            else:
                try:
                    val = f"{float(val):.2f}"
                except ValueError:
                    pass  # not a number — leave it as-is (a name, code, etc.)
            parts.append(val)
        lines.append(", ".join(parts))
    return "\n".join(lines)

def _format_answer(question: str, raw_data: str, lang: str, total_count: int = None) -> str:

    """
    Formats the raw result as a natural language answer.
    Counting and numbering are computed in Python — never left to the
    LLM to restate — after the duplicate-numbering bug found in KG3
    testing (e.g. two entries both labeled '9.').
    """
    lang_map = {"en": "English", "fr": "French", "ar": "Arabic"}
    language = lang_map.get(lang, "English")

    lines = [ln.strip() for ln in raw_data.strip().split("\n") if ln.strip()]
    count = len(lines)

    if count == 0:
        return "No results found." if lang == "en" else raw_data

    if count == 1 and lines[0].replace(".", "", 1).isdigit():
        # A bare number — e.g. a count_kg1 / count_kg3 result.
        return f"The answer is {lines[0]}."

    if count == 1:
        return lines[0]

    listed = "\n".join(f"{i+1}. {ln}" for i, ln in enumerate(lines))
    if total_count is not None and total_count > count:
        return f"There are {total_count} result(s) — showing the first {count}:\n\n{listed}"
    return f"There are {count} result(s):\n\n{listed}"


# ── MAIN RESOLVER (templates) ─────────────────────────────────────────────────

def resolve_template(question: str, template_name: str, lang: str, router_params: dict = None) -> dict:
    """
    Main entry point for template resolution.

    Args:
        question      : original user question
        template_name : one of the registered template types
        lang          : detected language (en/fr/ar)

    Returns:
        {
            success      : bool
            template     : template name
            params       : extracted parameters
            sparql       : generated SPARQL (or description for cross-KG)
            raw_data     : formatted result rows
            final_answer : natural language answer
            failure_type : success | param_extraction_failure |
                           sparql_build_failure | execution_failure
        }
    """
    result = {
        "success":      False,
        "template":     template_name,
        "params":       {},
        "sparql":       None,
        "raw_data":     None,
        "final_answer": None,
        "failure_type": None,
    }

    # ── Step 1: extract parameters ────────────────────────────────────────────
    
    if router_params:
        params = router_params
        print(f"[template] Reusing router params (skipping re-extraction): {params}")
    else:
        params = _extract_params(question, template_name, lang)
    result["params"] = params

    if not params:
        result["failure_type"] = "param_extraction_failure"
        return result

    

    # ── Step 2: build SPARQL ──────────────────────────────────────────────────
    builders = {
        "filter_numeric_kg2":   (_build_filter_numeric_kg2,   KG2_EP, ["name", "value"]),
        "filter_string_kg2":    (_build_filter_string_kg2,    KG2_EP, ["name"]),
        "ranking_kg2":          (_build_ranking_kg2,          KG2_EP, ["name", "value"]),
        "compare_two_airports": (_build_compare_two_airports, KG2_EP, ["name", "value"]),
        "count_kg1":            (_build_count_kg1,            KG1_EP, ["count"]),
        "filter_numeric_kg1":   (_build_filter_numeric_kg1,   KG1_EP, ["number", "value"]),
        "cross_kg_filter":      (_build_cross_kg_filter,      None,   None),
        "count_kg3": (_build_count_kg3, KG3_EP, ["name"]),
        "filter_string_kg3": (_build_filter_string_kg3, KG3_EP, ["name"]),
        "group_aggregate_kg1": (_build_group_aggregate_kg1, KG1_EP, ["groupName", "agg"]),
        "group_aggregate_kg2": (_build_group_aggregate_kg2, KG2_EP, ["groupName", "agg"]),
        "group_aggregate_kg3": (_build_group_aggregate_kg3, KG3_EP, ["deptName", "agg"]),
    }
    # KG3 templates need the entity name, detected deterministically —
    # same regex the router uses, not extracted by the LLM (avoids the
    # unreliability we saw with LLM-based entity extraction elsewhere).
    if template_name == "count_kg3":
        entity_name = _detect_university_entity_for_template(question)
        if not entity_name:
            result["failure_type"] = "param_extraction_failure"
            return result
        params["entity_name"] = entity_name
    if template_name not in builders:
        result["failure_type"] = "unknown_template"
        return result

    builder_fn, endpoint, columns = builders[template_name]
    build_result = builder_fn(params)

    if not build_result:
        result["failure_type"] = "sparql_build_failure"
        return result

    sparql, label = build_result
    print(f"[template] Query type: {label}")

    # ── Step 3: execute ───────────────────────────────────────────────────────

    # Special handling for cross-KG filter (two-step execution)
    if template_name == "cross_kg_filter":
        kg1_query, airport_prop, operator, threshold, direction, limit = sparql
        result["sparql"] = kg1_query

        # Get all flight+IATA pairs from KG1
        rows_kg1 = _run_sparql(KG1_EP, kg1_query)
        if not rows_kg1:
            result["failure_type"] = "execution_failure"
            return result

        # For each IATA, check KG2 property condition
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

            # Cache KG2 lookups
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

                kg2_rows = _run_sparql(KG2_EP, kg2_q)
                val_raw  = kg2_rows[0].get("val", {}).get("value", "") if kg2_rows else ""
                seen_iatas[iata] = val_raw

            val_raw = seen_iatas[iata]
            if not val_raw:
                continue

            # Evaluate condition
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
        # Standard single-endpoint execution
        result["sparql"] = sparql
        rows = _run_sparql(endpoint, sparql)

        if not rows:
            result["failure_type"] = "execution_failure"
            return result

        # Handle count queries specially
        if template_name in ("count_kg1", "count_kg3") and params.get("mode", "count") == "count":
            count_val = rows[0].get("count", {}).get("value", "0") if rows else "0"
            raw_data  = count_val
        else:
            raw_data = _format_rows(rows, columns)
            result["total_rows"] = len(rows)

        result["raw_data"] = raw_data

    # ── Step 4: format answer ─────────────────────────────────────────────────
    final_answer = _format_answer(question, result["raw_data"], lang, result.get("total_rows"))
    result["final_answer"] = final_answer
    result["success"]      = True
    result["failure_type"] = "success"

    return result


# ── ASK RESOLVER ───────────────────────────────────────────────────────────────

def _format_ask_answer(result: bool, lang: str) -> str:
    """
    Template-based yes/no formatting — deterministic by design, not
    LLM-generated. An ASK result is binary; there is no natural-language
    ambiguity for an LLM to resolve, only risk of it contradicting the
    boolean it was given. Same reasoning already applied to
    format_answer_list's count/listing logic.
    """
    templates = {
        "en": {"true": "Yes.", "false": "No."},
        "fr": {"true": "Oui.", "false": "Non."},
        "ar": {"true": "نعم.", "false": "لا."},
    }
    lang_templates = templates.get(lang, templates["en"])
    return lang_templates["true"] if result else lang_templates["false"]


def resolve_ask_query(question: str, routing: dict, lang: str) -> dict:
    """
    Resolves an ask_query routing decision into a boolean SPARQL ASK
    answer. Mirrors the single_kg1/kg2/kg3 pipelines (extract → map →
    build → execute), but dispatches entity resolution across all three
    KGs based on routing["kg"], since ASK questions can target any of
    them (see router.py Priority 1.5).

    Kept as its own function rather than folded into resolve_template()/
    TEMPLATE_REGISTRY, because ASK questions:
      - resolve a single named entity (like single_kg1/kg2/kg3), not a
        filtered set (like the template branch)
      - return a boolean, not rows
      - need a value comparison (FILTER), not just property retrieval
    Forcing this into the template builder pattern would require either
    a fake "builder" that doesn't build SELECT-shaped SPARQL, or a
    special case inside resolve_template() that breaks its "always
    returns rows" assumption.

    Returns:
        {
            success      : bool
            entity_uri   : resolved subject URI, or None
            property_uri : resolved (first-hop) property URI, or None
            value        : the asserted value from the question
            sparql       : the generated ASK query
            raw_answer   : True / False / None
            final_answer : "Yes." / "No." in the detected language
            failure_type : success | extraction_failure | mapping_failure |
                           execution_failure | unknown_kg
        }
    """
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

    # ── Step 1: extract property + asserted value ─────────────────────────────
    entities = extract_ask_entities(question, lang, entity)
    if not validate_ask_extraction(entities):
        result["failure_type"] = "extraction_failure"
        return result
    print(f"[ask_query] lang={lang} extracted property='{entities['property']}' value='{entities['value']}'")
    # ── Step 2: resolve entity URI + lexicon path per KG ───────────────────────
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

    # ── Step 3: map property (direct or two-hop) ────────────────────────────────
    lexicon = load_lexicon(lexicon_path)
    property_uri, tier, property2_uri = map_property_cascade(
        entities["property"], lexicon, lexicon_path
    )
    if not property_uri:
        result["failure_type"] = "mapping_failure"
        return result

    full_property_uri  = base_uri + property_uri
    full_property2_uri = (base_uri + property2_uri) if property2_uri else None
    result["property_uri"] = full_property_uri
    result["value"]        = entities["value"]

    # ── Step 4: build SPARQL ASK ─────────────────────────────────────────────
    sparql = build_ask_query(
        entity_uri, full_property_uri, entities["value"],
        property2_uri=full_property2_uri
    )
    result["sparql"] = sparql

    # ── Step 5: execute ───────────────────────────────────────────────────────
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