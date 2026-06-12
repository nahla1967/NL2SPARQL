"""
template_resolver.py
--------------------
Resolves complex queries using predefined SPARQL templates.
Handles: filters, rankings, comparisons, counts, and cross-KG aggregates.

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

    Cross-KG:
        cross_kg_filter       : flights whose airport property meets condition

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

# ── BASE URIs ─────────────────────────────────────────────────────────────────
KG1   = get_base_uri("flights")
KG2   = get_base_uri("airports")
KG1_EP = get_endpoint("flights")
KG2_EP = get_endpoint("airports")

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
    "alt":    {"uri": f"{KG1}alt",     "label": "altitude",       "unit": "feet"},
    "altitude":  {"uri": f"{KG1}alt",     "label": "altitude",       "unit": "feet"}, 
}

KG1_STRING_PROPS = {
    "hasDestinationCity":    {"uri": f"{KG1}hasDestinationCity"},
    "hasOriginCity":         {"uri": f"{KG1}hasOriginCity"},
    "hasAirline":            {"uri": f"{KG1}hasAirline"},
    "hasDestinationCountry": {"uri": f"{KG1}hasDestinationCountry"},
    "hasOriginCountry":      {"uri": f"{KG1}hasOriginCountry"},
}

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

Airport property mapping rules:
- "elevation", "altitude", "above X feet" → airport_property = "elevationFt"
- "runway length", "runway longer than" → airport_property = "lengthFt"
- "country", "in Germany", "in France" → airport_property = "countryName"
- "large airport", "large airports", "airport type" → airport_property = "airportType"
- "continent" → airport_property = "continent"

For string comparisons (country, type), set operator = "=" and threshold = the value.
For "large airports", set threshold = "large_airport".

Return ONLY a JSON object with keys:
- "direction": "destination" or "origin"
- "airport_property": one of [elevationFt, lengthFt, countryName, airportType, continent]
- "operator": one of [>, <, >=, <=, =]
- "threshold": the filter value (number or string)
- "limit": number of results (default 10)
Example: {{"direction": "destination", "airport_property": "airportType", "operator": "=", "threshold": "large_airport", "limit": 10}}
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
    operator  = params.get("operator", ">")
    threshold = params.get("threshold", 0)
    limit     = int(params.get("limit", 10))

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
}} ORDER BY DESC(?value) LIMIT {limit}"""
    else:
        sparql = f"""SELECT ?airport ?name ?value WHERE {{
  ?airport a <{KG2}Airport> .
  ?airport <{KG2}airportName> ?name .
  ?airport <{prop_uri}> ?value .
  FILTER(?value {operator} {threshold})
}} ORDER BY DESC(?value) LIMIT {limit}"""

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
    limit = int(params.get("limit", 10))

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
}} ORDER BY {order}(?value) LIMIT {limit}"""
    else:
        sparql = f"""SELECT ?airport ?name ?value WHERE {{
  ?airport a <{KG2}Airport> .
  ?airport <{KG2}airportName> ?name .
  ?airport <{prop_uri}> ?value .
}} ORDER BY {order}(?value) LIMIT {limit}"""

    direction_word = "highest" if order == "DESC" else "lowest"
    label = f"top {limit} airports by {prop_info['label']} ({direction_word})"
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
        # Airline requires operating_as lookup
        if mode == "count":
            sparql = f"""SELECT (COUNT(?flight) AS ?count) WHERE {{
  ?flight a <{KG1}Flight> .
  ?flight <{KG1}hasAirline> ?airline .
  ?airline <{KG1}operating_as> "{filter_value}" .
}}"""
        else:
            sparql = f"""SELECT ?flight ?number WHERE {{
  ?flight a <{KG1}Flight> .
  ?flight <{KG1}flightNumber> ?number .
  ?flight <{KG1}hasAirline> ?airline .
  ?airline <{KG1}operating_as> "{filter_value}" .
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
}} ORDER BY DESC(?value) LIMIT {limit}"""

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
            parts.append(val)
        lines.append(", ".join(parts))
    return "\n".join(lines)


def _format_answer(question: str, raw_data: str, lang: str) -> str:
    """Uses LLM to format the raw result as a natural language answer."""
    lang_map = {"en": "English", "fr": "French", "ar": "Arabic"}
    language = lang_map.get(lang, "English")

    prompt = f"""You are an answer formatter.

The user asked in {language}: "{question}"
The database returned these results:
{raw_data}

Write a clear, natural answer in {language} only.
If it is a list, present it as a numbered list.
If it is a count, state it as a simple sentence.
If it is a comparison, state which is higher/lower/better.
Return only the answer. Do not explain your reasoning."""

    try:
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"[format_error] {raw_data}"


# ── MAIN RESOLVER ─────────────────────────────────────────────────────────────

def resolve_template(question: str, template_name: str, lang: str) -> dict:
    """
    Main entry point for template resolution.

    Args:
        question      : original user question
        template_name : one of the 7 template types
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
    params = _extract_params(question, template_name, lang)
    result["params"] = params

    if not params:
        result["failure_type"] = "param_extraction_failure"
        return result

    print(f"[template] Extracted params: {params}")

    # ── Step 2: build SPARQL ──────────────────────────────────────────────────
    builders = {
        "filter_numeric_kg2":   (_build_filter_numeric_kg2,   KG2_EP, ["name", "value"]),
        "filter_string_kg2":    (_build_filter_string_kg2,    KG2_EP, ["name"]),
        "ranking_kg2":          (_build_ranking_kg2,          KG2_EP, ["name", "value"]),
        "compare_two_airports": (_build_compare_two_airports, KG2_EP, ["name", "value"]),
        "count_kg1":            (_build_count_kg1,            KG1_EP, ["count"]),
        "filter_numeric_kg1":   (_build_filter_numeric_kg1,   KG1_EP, ["number", "value"]),
        "cross_kg_filter":      (_build_cross_kg_filter,      None,   None),
    }

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
        if template_name == "count_kg1" and params.get("mode", "count") == "count":
            count_val = rows[0].get("count", {}).get("value", "0") if rows else "0"
            raw_data  = count_val
        else:
            raw_data = _format_rows(rows, columns)

        result["raw_data"] = raw_data

    # ── Step 4: format answer ─────────────────────────────────────────────────
    final_answer = _format_answer(question, result["raw_data"], lang)
    result["final_answer"] = final_answer
    result["success"]      = True
    result["failure_type"] = "success"

    return result