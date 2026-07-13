"""
cross_kg_resolver.py
--------------------
Resolves questions that require data from both KG1 (flights) and KG2 (airports).

PATTERN:
    "What country is the destination airport of flight OS295?"

    Step 1 → KG1:  OS295 + direction  →  IATA code  (e.g. "MUC")
    Step 2 → KG2:  "MUC"             →  Airport URI (ao:Airport/MUC)
    Step 3 → KG2:  Airport URI + property  →  answer  (e.g. "Germany")

GENERIC INTERFACE:
    resolve_cross_kg(flight_uri, direction, property_uri, property_short, property2_uri=None)

    All cross-KG questions reduce to these parameters.
    The function handles the two-step SPARQL pipeline internally.

PROPERTY NAMES:
    All endpoint URLs and property names are read from kg_registry.py.
    If the ontology changes, only kg_registry.py needs updating.
"""

import json
import urllib.parse
import urllib.request
from kg_registry import CROSS_KG_CONFIG, get_base_uri


# ── SPARQL HELPER ─────────────────────────────────────────────────────────────

def _sparql_query(endpoint: str, query: str) -> list[dict]:
    """
    Executes a SPARQL SELECT query against an endpoint.
    Returns a list of binding dicts, or empty list on failure.
    """
    data = urllib.parse.urlencode({
        "query":  query,
        "format": "application/sparql-results+json"
    }).encode()
    req = urllib.request.Request(endpoint, data=data)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result   = json.loads(resp.read())
            return result.get("results", {}).get("bindings", [])
    except Exception as e:
        print(f"[cross_kg_resolver] SPARQL error on {endpoint}: {e}")
        return []


# ── STEP 1: GET IATA FROM KG1 ─────────────────────────────────────────────────

def _get_iata_from_kg1(flight_uri: str, direction: str) -> str | None:
    """
    Queries KG1 to get the IATA code of the origin or destination airport.

    direction: 'origin'      → uses orig_iata property
               'destination' → uses dest_iata property

    Returns the IATA string (e.g. 'MUC'), or None if not found.
    """
    kg1_base    = CROSS_KG_CONFIG["kg1_base"]
    endpoint    = CROSS_KG_CONFIG["kg1_endpoint"]
    airport_prop = CROSS_KG_CONFIG["origin_property"] if direction == "origin" \
                   else CROSS_KG_CONFIG["destination_property"]
    iata_prop   = CROSS_KG_CONFIG["origin_iata_prop"] if direction == "origin" \
                  else CROSS_KG_CONFIG["destination_iata_prop"]

    query = f"""
SELECT ?iata WHERE {{
  <{flight_uri}> <{kg1_base}{airport_prop}> ?airport_node .
  ?airport_node  <{kg1_base}{iata_prop}>    ?iata .
}}
LIMIT 1
"""
    bindings = _sparql_query(endpoint, query)
    if bindings:
        return bindings[0].get("iata", {}).get("value")
    return None


# ── STEP 2: GET AIRPORT URI FROM KG2 ─────────────────────────────────────────

def _get_airport_uri_from_kg2(iata: str) -> str | None:
    """
    Queries KG2 to find the Airport URI for a given IATA code.

    Example: "MUC" → ao:Airport/MUC URI
    Returns the full URI string, or None if not found.
    """
    kg2_base   = CROSS_KG_CONFIG["kg2_base"]
    endpoint   = CROSS_KG_CONFIG["kg2_endpoint"]
    iata_prop  = CROSS_KG_CONFIG["kg2_iata_property"]

    query = f"""
SELECT ?airport WHERE {{
  ?airport <{kg2_base}{iata_prop}> "{iata}" .
}}
LIMIT 1
"""
    bindings = _sparql_query(endpoint, query)
    if bindings:
        return bindings[0].get("airport", {}).get("value")
    return None


# ── STEP 3: GET PROPERTY VALUE FROM KG2 ───────────────────────────────────────

def _get_airport_property(airport_uri: str, property_uri: str, property2_uri: str | None = None) -> str | None:
    endpoint = CROSS_KG_CONFIG["kg2_endpoint"]
    kg2_base = CROSS_KG_CONFIG["kg2_base"]

    if property2_uri:
        query_two_hop = f"""
SELECT ?value WHERE {{
  <{airport_uri}> <{property_uri}> ?node .
  ?node <{property2_uri}> ?value .
}}
LIMIT 1
"""
        bindings = _sparql_query(endpoint, query_two_hop)
        if bindings:
            raw = bindings[0].get("value", {}).get("value", "")
            if raw.startswith("http"):
                return _resolve_uri_label(raw, endpoint, kg2_base)
            return raw
        return None

    # --- Attempt 2: one-hop through object property ---
    # Tries: Airport → objectProp → Node → dataProps
    # This handles locatedInCountry → countryName, locatedInRegion → regionName
    query_hop = f"""
SELECT ?label WHERE {{
  <{airport_uri}> <{property_uri}> ?node .
  ?node ?labelProp ?label .
  FILTER(isLiteral(?label))
}}
LIMIT 1
"""
    bindings = _sparql_query(endpoint, query_hop)
    if bindings:
        return bindings[0].get("label", {}).get("value")

    return None


def _resolve_uri_label(uri: str, endpoint: str, base: str) -> str | None:
    """
    When a property returns a URI (e.g. ao:Country/DE), resolves it to
    a human-readable label by fetching the first literal property of that node.

    Priority: countryName > regionName > any literal property.
    """
    preferred = ["countryName", "regionName", "airportName", "regionCode", "isoCode"]

    for prop in preferred:
        query = f"""
SELECT ?value WHERE {{
  <{uri}> <{base}{prop}> ?value .
}}
LIMIT 1
"""
        bindings = _sparql_query(endpoint, query)
        if bindings:
            return bindings[0].get("value", {}).get("value")

    # Fallback: any literal property
    query_any = f"""
SELECT ?value WHERE {{
  <{uri}> ?p ?value .
  FILTER(isLiteral(?value))
}}
LIMIT 1
"""
    bindings = _sparql_query(endpoint, query_any)
    if bindings:
        return bindings[0].get("value", {}).get("value")

    # Last resort: extract from URI fragment
    return uri.split("/")[-1].replace("_", " ")


# ── MAIN RESOLVER ─────────────────────────────────────────────────────────────

def resolve_cross_kg(
    flight_uri: str,
    direction: str,
    property_uri: str,
    property_short: str,
    property2_uri: str | None = None,
) -> dict:
    """
    Generic cross-KG resolver. Bridges KG1 and KG2 via the IATA code.

    Args:
        flight_uri      : full KG1 flight URI
        direction       : 'origin' or 'destination'
        property_uri    : full KG2 property URI (e.g. ao:elevationFt full URI)
        property_short  : short name for logging (e.g. 'elevationFt')
        property2_uri   : optional second-hop KG2 property URI, for two-hop
                           lookups (e.g. locatedInCountry → countryName)

    Returns a dict:
    {
        success      : bool
        iata         : str | None   — the bridging IATA code
        airport_uri  : str | None   — the KG2 airport URI
        raw_value    : str | None   — the resolved property value
        failure_type : str | None   — step that failed if any
    }
    """
    result = {
        "success":      False,
        "iata":         None,
        "airport_uri":  None,
        "raw_value":    None,
        "failure_type": None,
    }

    # Step 1 — KG1: flight → IATA
    iata = _get_iata_from_kg1(flight_uri, direction)
    if not iata:
        print(f"[cross_kg] Step 1 failed: no IATA for {flight_uri} ({direction})")
        result["failure_type"] = "kg1_iata_not_found"
        return result

    result["iata"] = iata
    print(f"[cross_kg] Step 1 ✓ IATA = {iata}")

    # Step 2 — KG2: IATA → Airport URI
    airport_uri = _get_airport_uri_from_kg2(iata)
    if not airport_uri:
        print(f"[cross_kg] Step 2 failed: no KG2 airport for IATA '{iata}'")
        result["failure_type"] = "kg2_airport_not_found"
        return result

    result["airport_uri"] = airport_uri
    print(f"[cross_kg] Step 2 ✓ Airport URI = {airport_uri}")

    # Step 3 — KG2: Airport URI → property value
    raw_value = _get_airport_property(airport_uri, property_uri, property2_uri)
    if not raw_value:
        print(f"[cross_kg] Step 3 failed: property '{property_short}' not found")
        result["failure_type"] = "kg2_property_not_found"
        return result

    result["raw_value"]    = raw_value
    result["success"]      = True
    result["failure_type"] = "success"
    print(f"[cross_kg] Step 3 ✓ Value = {raw_value}")

    return result