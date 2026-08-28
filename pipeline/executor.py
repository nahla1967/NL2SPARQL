"""
executor.py  (v3 — GPT fixes applied)
--------------------------------------
CHANGES vs v2:
    Fix 1: URI pattern matching uses "Country/" not "/Country/"
            for robustness across serialization variants.

    Fix 2: name_props = [] initialized at top of resolve_entity()
            to prevent UnboundLocalError on unknown URI patterns.

    Fix 3: execute_sparql() gains multiple=False parameter.
            When multiple=True, returns list of all binding values
            instead of just the first. Required for template queries.

    No logic changes to any other function.
"""

import json
import urllib.parse
import urllib.request

import ollama
from datetime import datetime
from rdflib.plugins.sparql import prepareQuery
from kg_registry import get_base_uri, get_endpoint

# ── ENDPOINTS — read from registry (single source of truth) ──────────────────
# Never hardcode URLs here. If Fuseki port changes, update kg_registry.py only.
KG1_URL = get_endpoint("flights")
KG2_URL = get_endpoint("airports")

# ── LOOKUP TABLES ─────────────────────────────────────────────────────────────
AIRLINE_CODES = {
    "AFR": "Air France", "AIC": "Air India", "AUA": "Austrian Airlines",
    "AZG": "Azerbaijan Airlines", "BEL": "Brussels Airlines",
    "BRX": "Braathens Regional Airways", "BTI": "Air Baltic",
    "CFG": "Condor", "CTN": "Croatia Airlines", "DLA": "Air Dolomiti",
    "EVA": "EVA Air", "EWL": "Eurowings", "FCM": "Air Belgium",
    "FIN": "Finnair", "FSF": "FLY7 Finland", "KAL": "Korean Air",
    "LDA": "Lauda Air", "LGL": "Luxair", "LOT": "LOT Polish Airlines",
    "MAE": "Mali Air", "MAY": "Malta Air", "MSC": "MSC Air Cargo",
    "OAW": "Helvetic Airways", "PEV": "People's Viennaline",
    "PGT": "Pegasus Airlines", "RYS": "Ryanair Sun", "SXS": "SunExpress",
    "THY": "Turkish Airlines", "TKJ": "Turkish Airlines Charter",
    "TVF": "Transavia France", "WMT": "Wizz Air Malta"
}

COUNTRY_CODES = {
    "AT": "Austria", "DE": "Germany", "FR": "France", "TR": "Turkey",
    "GB": "United Kingdom", "TW": "Taiwan", "JP": "Japan", "IN": "India",
    "LV": "Latvia", "PL": "Poland", "BE": "Belgium", "LU": "Luxembourg",
    "BG": "Bulgaria", "HR": "Croatia", "CY": "Cyprus", "GR": "Greece",
    "IT": "Italy", "ES": "Spain", "SE": "Sweden", "DK": "Denmark",
    "FI": "Finland", "AZ": "Azerbaijan", "TH": "Thailand", "MT": "Malta",
    "RS": "Serbia", "AL": "Albania", "US": "United States",
    "EG": "Egypt", "IQ": "Iraq", "RO": "Romania", "CH": "Switzerland",
}
# Inverse mapping: translated surface names → canonical KG code
# Used by build_ask_query() to normalize values extracted from non-English
# questions before injecting them into the SPARQL FILTER.
_SURFACE_NAME_TO_CODE = {

    "asphalt":    "ASP", "asphalte":    "ASP", "en asphalte":   "ASP",
    "concrete":   "CON", "béton":       "CON", "beton":         "CON",
    "grass":      "GRS", "herbe":       "GRS",
    "gravel":     "GRV", "gravier":     "GRV",
    "bitumen":    "BIT", "bitume":      "BIT",
    "clay":       "CLA", "argile":      "CLA",
    "sand":       "SAN", "sable":       "SAN",
    "water":      "WAT", "eau":         "WAT",
}
SURFACE_CODES = {
    "ASP":  "Asphalt", "ASPH": "Asphalt", "CON": "Concrete",
    "GRS":  "Grass",   "GRV":  "Gravel",  "PEM": "Asphalt",
    "CONC": "Concrete","BIT":  "Bitumen", "CLA": "Clay",
    "SAN":  "Sand",    "WAT":  "Water",
}

# ── SPARQL VALIDATION ─────────────────────────────────────────────────────────
def validate_sparql(query: str) -> bool:
    try:
        prepareQuery(query)
        return True
    except Exception:
        return False

# ── URI HELPERS ───────────────────────────────────────────────────────────────
def clean_uri(value: str) -> str:
    if value.startswith("http"):
        return value.split("/")[-1].replace("_", " ")
    return urllib.parse.unquote(value).replace("_", " ")

# ── KG1 ENTITY RESOLUTION ────────────────────────────────────────────────────
_entity_cache: dict[str, str] = {}

def resolve_entity(uri: str) -> str:
    """
    Resolves a KG1 URI to a human-readable value.
    Completely unchanged from v1.

    Fix applied (GPT point 4):
        name_props = [] initialized at the top so the final
        for-loop never raises UnboundLocalError on unknown URI patterns.
    """
    if not uri.startswith("http"):
        return uri
    if uri in _entity_cache:
        return _entity_cache[uri]

    base      = "http://www.semanticweb.org/ontologies/flight_ontology#"
    url       = KG1_URL
    result    = None
    name_props = []    # Fix 4: always defined — prevents UnboundLocalError

    if "Route/" in uri:
        separator = "#Route/" if "#Route/" in uri else "/Route/"
        fragment  = uri.split(separator, 1)[-1]
        result    = urllib.parse.unquote(fragment).replace("_", " ").strip()

    elif "TimeInstant/" in uri:
        query = (f"SELECT ?eta ?date WHERE {{"
                 f" OPTIONAL {{ <{uri}> <{base}eta> ?eta . }}"
                 f" OPTIONAL {{ <{uri}> <{base}date> ?date . }} }} LIMIT 1")
        data = urllib.parse.urlencode(
            {"query": query, "format": "application/sparql-results+json"}
        ).encode()
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, data=data)
            ) as r:
                bindings = json.loads(r.read())["results"]["bindings"]
                if bindings:
                    eta  = bindings[0].get("eta",  {}).get("value")
                    date = bindings[0].get("date", {}).get("value")
                    if eta:
                        dt     = datetime.strptime(eta, "%Y-%m-%dT%H:%M:%SZ")
                        result = dt.strftime("%d %B %Y at %H:%M UTC")
                    elif date:
                        result = date
        except Exception as e:
            print(f"[resolve_entity] TimeInstant error: {e}")
        if result is None:
            result = clean_uri(uri)

    elif "Airline/" in uri:
        query = f"SELECT ?value WHERE {{ <{uri}> <{base}operating_as> ?value . }} LIMIT 1"
        data  = urllib.parse.urlencode(
            {"query": query, "format": "application/sparql-results+json"}
        ).encode()
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, data=data)
            ) as r:
                bindings = json.loads(r.read())["results"]["bindings"]
                if bindings:
                    code   = bindings[0][list(bindings[0].keys())[0]]["value"]
                    result = AIRLINE_CODES.get(code, code)
        except Exception as e:
            print(f"[resolve_entity] Airline error: {e}")
        if result is None:
            result = clean_uri(uri)

    elif "Location/" in uri:
        query = (f"SELECT ?lat ?long ?alt WHERE {{"
                 f" <{uri}> <{base}lat> ?lat ."
                 f" <{uri}> <{base}long> ?long ."
                 f" <{uri}> <{base}alt> ?alt . }} LIMIT 1")
        data = urllib.parse.urlencode(
            {"query": query, "format": "application/sparql-results+json"}
        ).encode()
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, data=data)
            ) as r:
                bindings = json.loads(r.read())["results"]["bindings"]
                if bindings:
                    lat  = bindings[0].get("lat",  {}).get("value", "?")
                    lon  = bindings[0].get("long", {}).get("value", "?")
                    alt  = bindings[0].get("alt",  {}).get("value", "?")
                    result = f"lat: {lat}, long: {lon}, alt: {alt}"
        except Exception as e:
            print(f"[resolve_entity] Location error: {e}")
        if result is None:
            result = clean_uri(uri)

    elif "FlightEvent/" in uri:
        query = (f"SELECT ?gspeed ?vspeed WHERE {{"
                 f" <{uri}> <{base}gspeed> ?gspeed ."
                 f" <{uri}> <{base}vspeed> ?vspeed . }} LIMIT 1")
        data = urllib.parse.urlencode(
            {"query": query, "format": "application/sparql-results+json"}
        ).encode()
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, data=data)
            ) as r:
                bindings = json.loads(r.read())["results"]["bindings"]
                if bindings:
                    gs     = bindings[0].get("gspeed", {}).get("value", "?")
                    vs     = bindings[0].get("vspeed", {}).get("value", "?")
                    result = f"ground speed: {gs} kt, vertical speed: {vs} ft/min"
        except Exception as e:
            print(f"[resolve_entity] FlightEvent error: {e}")
        if result is None:
            result = clean_uri(uri)

    elif "Airport/" in uri:
        query = (f"SELECT ?orig_iata ?dest_iata WHERE {{"
                 f" OPTIONAL {{ <{uri}> <{base}orig_iata> ?orig_iata . }}"
                 f" OPTIONAL {{ <{uri}> <{base}dest_iata> ?dest_iata . }} }} LIMIT 1")
        data = urllib.parse.urlencode(
            {"query": query, "format": "application/sparql-results+json"}
        ).encode()
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, data=data)
            ) as r:
                bindings = json.loads(r.read())["results"]["bindings"]
                if bindings:
                    orig   = bindings[0].get("orig_iata", {}).get("value", "?")
                    dest   = bindings[0].get("dest_iata", {}).get("value", "?")
                    result = f"origin: {orig}, destination: {dest}"
        except Exception as e:
            print(f"[resolve_entity] Airport error: {e}")
        if result is None:
            result = clean_uri(uri)

    elif "City/" in uri:
        name_props = ["orig_city"]

    elif "Aircraft/" in uri:
        name_props = ["type"]

    else:
        result = clean_uri(uri)

    # name_props fallback loop — safe because name_props = [] by default
    if result is None:
        for name_prop in name_props:
            query = (f"SELECT ?value WHERE {{"
                     f" <{uri}> <{base}{name_prop}> ?value . }} LIMIT 1")
            data  = urllib.parse.urlencode(
                {"query": query, "format": "application/sparql-results+json"}
            ).encode()
            try:
                with urllib.request.urlopen(
                    urllib.request.Request(url, data=data)
                ) as r:
                    bindings = json.loads(r.read())["results"]["bindings"]
                    if bindings:
                        result = bindings[0][list(bindings[0].keys())[0]]["value"]
                        break
            except Exception as e:
                print(f"[resolve_entity] {name_prop} error: {e}")
        if result is None:
            result = clean_uri(uri)

    _entity_cache[uri] = result
    return result

# ── KG2 ENTITY RESOLUTION ─────────────────────────────────────────────────────
_kg2_entity_cache: dict[str, str] = {}

def resolve_entity_kg2(uri: str) -> str:
    """
    Resolves a KG2 URI to a human-readable value.

    Fix applied (GPT point 2):
        URI checks use "Country/" not "/Country/" for robustness
        across serialization variants (#Country/, /Country/, etc.)
    """
    if not uri.startswith("http"):
        return uri
    if uri in _kg2_entity_cache:
        return _kg2_entity_cache[uri]

    base   = "http://www.semanticweb.org/ontologies/airport_ontology#"
    result = None

    if "Country/" in uri:       # Fix 2: no leading slash
        query = f"SELECT ?name WHERE {{ <{uri}> <{base}countryName> ?name . }} LIMIT 1"
        data  = urllib.parse.urlencode(
            {"query": query, "format": "application/sparql-results+json"}
        ).encode()
        try:
            with urllib.request.urlopen(
                urllib.request.Request(KG2_URL, data=data)
            ) as r:
                bindings = json.loads(r.read())["results"]["bindings"]
                if bindings:
                    result = bindings[0]["name"]["value"]
        except Exception as e:
            print(f"[resolve_entity_kg2] Country error: {e}")

    elif "Region/" in uri:      # Fix 2
        query = f"SELECT ?name WHERE {{ <{uri}> <{base}regionName> ?name . }} LIMIT 1"
        data  = urllib.parse.urlencode(
            {"query": query, "format": "application/sparql-results+json"}
        ).encode()
        try:
            with urllib.request.urlopen(
                urllib.request.Request(KG2_URL, data=data)
            ) as r:
                bindings = json.loads(r.read())["results"]["bindings"]
                if bindings:
                    result = bindings[0]["name"]["value"]
        except Exception as e:
            print(f"[resolve_entity_kg2] Region error: {e}")

    elif "Runway/" in uri:      # Fix 2
        query = f"""SELECT ?length ?surface ?lighted WHERE {{
  OPTIONAL {{ <{uri}> <{base}lengthFt> ?length . }}
  OPTIONAL {{ <{uri}> <{base}surface>  ?surface . }}
  OPTIONAL {{ <{uri}> <{base}lighted>  ?lighted . }}
}} LIMIT 1"""
        data = urllib.parse.urlencode(
            {"query": query, "format": "application/sparql-results+json"}
        ).encode()
        try:
            with urllib.request.urlopen(
                urllib.request.Request(KG2_URL, data=data)
            ) as r:
                bindings = json.loads(r.read())["results"]["bindings"]
                if bindings:
                    length  = bindings[0].get("length",  {}).get("value", "?")
                    surface = bindings[0].get("surface", {}).get("value", "?")
                    surface = SURFACE_CODES.get(surface, surface)
                    lighted = bindings[0].get("lighted", {}).get("value", "?")
                    result  = f"{length} ft, {surface}, lighted: {lighted}"
        except Exception as e:
            print(f"[resolve_entity_kg2] Runway error: {e}")

    if result is None:
        result = clean_uri(uri)

    _kg2_entity_cache[uri] = result
    return result

# ── SPARQL EXECUTION ──────────────────────────────────────────────────────────

def execute_sparql(
    sparql_query: str,
    endpoint:     str  = KG1_URL,
    multiple:     bool = False,
) -> dict:
    """
    Executes a SPARQL SELECT query.

    Returns a dict, always: {"value": ..., "error": ...}
    Exactly one of the two keys is non-None.
        - On success: "value" holds the result (str, list, or None if
          the query executed fine but matched zero rows), "error" is None.
        - On failure: "value" is None, "error" holds a message describing
          what went wrong.

    Args:
        sparql_query : the SPARQL query string
        endpoint     : SPARQL endpoint URL (defaults to KG1)
        multiple     : if True, returns list of all result values
                       if False (default), returns only the first value
    """
    data = urllib.parse.urlencode({
        "query":  sparql_query,
        "format": "application/sparql-results+json"
    }).encode()
    req = urllib.request.Request(endpoint, data=data)

    try:
        with urllib.request.urlopen(req) as response:
            raw_response = response.read()
            result = json.loads(raw_response)
            if "results" not in result:
                return {"value": None, "error": "malformed response: no 'results' key"}
            bindings = result["results"]["bindings"]
            if not bindings:
                return {"value": None, "error": None}

            def _resolve_binding(binding: dict) -> str | None:
                if not binding:
                    return None
                
                first_key = list(binding.keys())[0]
                raw       = binding[first_key]["value"]
                if "airport_ontology" in raw:
                    resolved = resolve_entity_kg2(raw)
                elif raw.startswith("http"):
                    resolved = resolve_entity(raw)
                else:
                    resolved = raw
                if resolved and len(resolved) == 2 and resolved.isupper():
                    resolved = COUNTRY_CODES.get(resolved, resolved)
                if resolved and resolved.upper() in SURFACE_CODES:
                    resolved = SURFACE_CODES[resolved.upper()]
                return resolved

            if multiple:
                resolved = [_resolve_binding(b) for b in bindings]
                return {"value": [r for r in resolved if r is not None], "error": None}
            else:
                return {"value": _resolve_binding(bindings[0]), "error": None}

    except urllib.error.URLError as e:
        return {"value": None, "error": f"URLError: {e}"}
    except Exception as e:
        return {"value": None, "error": f"{type(e).__name__}: {e}"}
    
def execute_ask_sparql(query: str, endpoint: str) -> bool | None:
    """
    Executes a SPARQL ASK query and returns True/False.
    Separate from execute_sparql() because Fuseki's response shape for
    ASK is fundamentally different: {"boolean": true/false}, not
    {"results": {"bindings": [...]}}.
    """
    data = urllib.parse.urlencode({
        "query":  query,
        "format": "application/sparql-results+json"
    }).encode()
    req = urllib.request.Request(endpoint, data=data)
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read())
            print(f"[debug] ASK raw response: {result}")
            return result.get("boolean")
    except urllib.error.URLError as e:
        print(f"[execute_ask_sparql] Fuseki unreachable: {e}")
    except Exception as e:
        print(f"[execute_ask_sparql] Unexpected error: {e}")
    return None
def build_ask_query(
    entity_uri:    str,
    property_uri:  str,
    value:         str,
    property2_uri: str | None = None,
) -> str:
    """
    Builds a SPARQL ASK query checking whether an entity has a specific
    property value — direct (one-hop) or via an intermediate node (two-hop).
    """
    # Normalize surface values before injecting into SPARQL
    # (handles translations like "en asphalte" → "ASP")
    lookup = value.lower().strip().strip('"')
    if lookup in _SURFACE_NAME_TO_CODE:
        value = _SURFACE_NAME_TO_CODE[lookup]
        print(f"[build_ask_query] Normalized '{lookup}' → '{value}'")

    safe_value = value.replace('"', '\\"')

    if property2_uri:
        return f"""ASK {{
  <{entity_uri}> <{property_uri}> ?intermediate .
  ?intermediate <{property2_uri}> ?value .
  BIND(REPLACE(REPLACE(STR(?value), "^.*[/#]", ""), "_", " ") AS ?local_value)
  FILTER(?local_value = "{safe_value}")
}}"""

    return f"""ASK {{
  <{entity_uri}> <{property_uri}> ?value .
  BIND(REPLACE(REPLACE(STR(?value), "^.*[/#]", ""), "_", " ") AS ?local_value)
  FILTER(?local_value = "{safe_value}")
}}"""
# ── ANSWER FORMATTING ─────────────────────────────────────────────────────────
def format_answer(question: str, raw_value: str, lang: str) -> str:
    lang_map      = {"en": "English", "fr": "French", "ar": "Arabic"}
    language_name = lang_map.get(lang, "English")

    prompt = f"""You are an answer formatter.

The user asked this question in {language_name}: "{question}"
The answer from the database is: "{raw_value}"

Write a short, natural sentence answering the question, using ONLY the
value given above. Do not use any outside knowledge about what the
entities in the question might mean.

If the database value does not fully answer what was asked (for example,
the question asks about two things but only one value was given), say
that you only have partial information, and state only what the value
actually shows. Do not invent missing data.

You MUST write the answer in {language_name} only.
Do not translate. Do not switch language.
Return only the sentence."""

    try:
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0}
        )
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"[format_answer] Ollama error: {e}")
        return f"[format_error] {raw_value}"

def format_answer_list(question: str, values: list, lang: str) -> str:
    """
    Formats a list of values into a natural language answer.
    Count and listing are built in Python — never left to the LLM to restate.
    """
    lang_map      = {"en": "English", "fr": "French", "ar": "Arabic"}
    language_name = lang_map.get(lang, "English")
    count         = len(values)

    # Built in plain Python — guaranteed correct, no matter what the LLM does.
    listed_items = "\n".join(f"{i+1}. {v}" for i, v in enumerate(values))
    intro = {
        "en": f"There are {count} result(s):",
        "fr": f"Il y a {count} résultat(s) :",
        "ar": f"يوجد {count} نتيجة/نتائج:",
    }.get(lang, f"There are {count} result(s):")

    return f"{intro}\n\n{listed_items}"