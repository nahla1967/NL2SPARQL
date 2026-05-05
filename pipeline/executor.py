import json
import urllib.parse
import urllib.request
import ollama
from datetime import datetime
from rdflib.plugins.sparql import prepareQuery

# ── LOOKUP TABLES ─────────────────────────────────────────────────────────────

# Expands ICAO 3-letter airline codes stored in the KG to full airline names.
# Without this, the LLM formatter receives raw codes like "THY" and may
# hallucinate incorrect expansions.
AIRLINE_CODES = {
    "AFR": "Air France",
    "AIC": "Air India",
    "AUA": "Austrian Airlines",
    "AZG": "Azerbaijan Airlines",
    "BEL": "Brussels Airlines",
    "BRX": "Braathens Regional Airways",
    "BTI": "Air Baltic",
    "CFG": "Condor",
    "CTN": "Croatia Airlines",
    "DLA": "Air Dolomiti",
    "EVA": "EVA Air",
    "EWL": "Eurowings",
    "FCM": "Air Belgium",
    "FIN": "Finnair",
    "FSF": "FLY7 Finland",
    "KAL": "Korean Air",
    "LDA": "Lauda Air",
    "LGL": "Luxair",
    "LOT": "LOT Polish Airlines",
    "MAE": "Mali Air",
    "MAY": "Malta Air",
    "MSC": "MSC Air Cargo",
    "OAW": "Helvetic Airways",
    "PEV": "People's Viennaline",
    "PGT": "Pegasus Airlines",
    "RYS": "Ryanair Sun",
    "SXS": "SunExpress",
    "THY": "Turkish Airlines",
    "TKJ": "Turkish Airlines Charter",
    "TVF": "Transavia France",
    "WMT": "Wizz Air Malta"
}

# Expands ISO 3166-1 alpha-2 country codes stored in the KG to full country names.
# Without this, the LLM formatter receives raw codes like "AT" and may
# hallucinate incorrect expansions (e.g. "DE" → "Denmark" instead of "Germany").
COUNTRY_CODES = {
    "AT": "Austria",
    "DE": "Germany",
    "FR": "France",
    "TR": "Turkey",
    "GB": "United Kingdom",
    "TW": "Taiwan",
    "JP": "Japan",
    "IN": "India",
    "LV": "Latvia",
    "PL": "Poland",
    "BE": "Belgium",
    "LU": "Luxembourg",
    "BG": "Bulgaria",
    "HR": "Croatia",
    "CY": "Cyprus",
    "GR": "Greece",
    "IT": "Italy",
    "ES": "Spain",
    "SE": "Sweden",
    "DK": "Denmark",
    "FI": "Finland",
    "AZ": "Azerbaijan",
    "TH": "Thailand",
    "MT": "Malta",
    "RS": "Serbia",
    "AL": "Albania"
}


# ── SPARQL VALIDATION ─────────────────────────────────────────────────────────

def validate_sparql(query):
    """
    Validates SPARQL syntax using rdflib's parser.
    Returns True only if the query is syntactically correct.
    """
    try:
        prepareQuery(query)
        return True
    except Exception:
        return False


# ── URI AND VALUE RESOLUTION ──────────────────────────────────────────────────

def clean_uri(value):
    """
    Converts a raw URI or encoded literal into a readable string.
    Used as a last-resort fallback when no specific resolver applies.
    """
    if value.startswith("http"):
        return value.split("/")[-1].replace("_", " ")
    return urllib.parse.unquote(value).replace("_", " ")


def resolve_entity(uri):
    """
    Resolves a KG URI to a human-readable value.

    Branch order matters here.
    Route is handled first with pure string parsing — no HTTP call needed —
    because Route URIs encode their label directly in the fragment
    (e.g. .../Route/Vienna to Bangkok). Attempting an HTTP lookup on a URI
    that contains a literal space would cause urllib to raise a ValueError,
    so we extract the label before any network code runs.

    The remaining branches follow in specificity order:
    - TimeInstant → two-hop SPARQL query (Flight → TimeInstant → eta)
    - Airline     → one-hop SPARQL query + AIRLINE_CODES expansion
    - Aircraft    → one-hop SPARQL query for type label
    - City        → one-hop SPARQL query for orig_city label
    - Other       → clean_uri fallback
    """
    if not uri.startswith("http"):
        return uri

    base = "http://www.semanticweb.org/ontologies/flight_ontology#"
    url = "http://localhost:3030/flights/sparql"

    # ── Route: pure string extraction, no HTTP ────────────────────────────────
    # Route URIs store the human-readable label directly after /Route/.
    # The label may contain spaces or special characters that are NOT
    # percent-encoded in this KG (e.g. "Vienna to Bangkok" not "Vienna%20to%20Bangkok").
    # urllib would fail on such a URI, so we extract the label with a plain
    # string split before any network call is attempted.
    if "/Route/" in uri or "#Route/" in uri:
       separator = "#Route/" if "#Route/" in uri else "/Route/"
       fragment = uri.split(separator, 1)[-1]
       return urllib.parse.unquote(fragment).replace("_", " ").strip()

    # ── TimeInstant: two-hop SPARQL ───────────────────────────────────────────
    # Arrival time is stored as:
    #   Flight → hasTimeInstant → TimeInstant → eta → datetime string
    # The generator retrieves the TimeInstant URI in one hop.
    # This branch performs the second hop to get the actual datetime value.
    if "/TimeInstant/" in uri or "#TimeInstant/" in uri:
        query = f"""
SELECT ?value WHERE {{
  <{uri}> <{base}eta> ?value .
}}
LIMIT 1
"""
        data = urllib.parse.urlencode({
            "query": query,
            "format": "application/sparql-results+json"
        }).encode()
        req = urllib.request.Request(url, data=data)
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read())
                bindings = result["results"]["bindings"]
                if bindings:
                    raw_time = bindings[0][list(bindings[0].keys())[0]]["value"]
                    dt = datetime.strptime(raw_time, "%Y-%m-%dT%H:%M:%SZ")
                    return dt.strftime("%d %B %Y at %H:%M UTC")
        except Exception:
            pass
        return clean_uri(uri)

    # ── Airline: ICAO code lookup + expansion ─────────────────────────────────
    # Airlines store an ICAO code in operating_as, not a readable name.
    # We retrieve the code then expand it via AIRLINE_CODES.
    if "/Airline/" in uri or "#Airline/" in uri:
        query = f"""
SELECT ?value WHERE {{
  <{uri}> <{base}operating_as> ?value .
}}
LIMIT 1
"""
        data = urllib.parse.urlencode({
            "query": query,
            "format": "application/sparql-results+json"
        }).encode()
        req = urllib.request.Request(url, data=data)
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read())
                bindings = result["results"]["bindings"]
                if bindings:
                    code = bindings[0][list(bindings[0].keys())[0]]["value"]
                    return AIRLINE_CODES.get(code, code)
        except Exception:
            pass
        return clean_uri(uri)

    # ── City and Aircraft: generic label lookup ───────────────────────────────
    
    if "/City/" in uri or "#City/" in uri:
        name_props = ["orig_city"]
    elif "/Aircraft/" in uri or "#Aircraft/" in uri:
        name_props = ["type"]
   
    else:
        return clean_uri(uri)

    for name_prop in name_props:
        query = f"""
SELECT ?value WHERE {{
  <{uri}> <{base}{name_prop}> ?value .
}}
LIMIT 1
"""
        data = urllib.parse.urlencode({
            "query": query,
            "format": "application/sparql-results+json"
        }).encode()
        req = urllib.request.Request(url, data=data)
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read())
                bindings = result["results"]["bindings"]
                if bindings:
                    first_key = list(bindings[0].keys())[0]
                    return bindings[0][first_key]["value"]
        except Exception as e:
            print(f"[resolve_entity] Inner query failed: {e}")

    return clean_uri(uri)


# ── SPARQL EXECUTION ──────────────────────────────────────────────────────────

def execute_sparql(sparql_query):
    """
    Executes a SPARQL SELECT query against the local Fuseki endpoint.
    Resolves the returned value through resolve_entity, then expands
    any 2-letter ISO country code to its full country name.
    Returns None if the query fails or returns no results.
    """
    url = "http://localhost:3030/flights/sparql"
    data = urllib.parse.urlencode({
        "query": sparql_query,
        "format": "application/sparql-results+json"
    }).encode()
    req = urllib.request.Request(url, data=data)
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read())
            if "results" not in result:
                return None
            bindings = result["results"]["bindings"]
            if bindings:
                first_key = list(bindings[0].keys())[0]
                raw_value = resolve_entity(bindings[0][first_key]["value"])

                # Expand ISO country codes to full names.
                # Prevents the LLM formatter from hallucinating wrong expansions.
                if raw_value and len(raw_value) == 2 and raw_value.isupper():
                    raw_value = COUNTRY_CODES.get(raw_value, raw_value)

                return raw_value

    except urllib.error.URLError as e:
        print(f"[execute_sparql] Fuseki unreachable: {e}")
    except Exception as e:
        print(f"[execute_sparql] Unexpected error: {e}")

    return None


# ── ANSWER FORMATTING ─────────────────────────────────────────────────────────

def format_answer(question, raw_value, lang):
    """
    Uses the LLM to reformulate the raw KG answer into a natural language
    sentence in the same language as the user's original question.
    Falls back to a structured error string if Ollama is unavailable,
    so the log entry remains useful for evaluation.
    """
    lang_map = {
        "en": "English",
        "fr": "French",
        "ar": "Arabic"
    }
    language_name = lang_map.get(lang, "English")

    prompt = f"""You are an answer formatter.

The user asked this question in {language_name}: "{question}"
The answer from the database is: "{raw_value}"

Write a short, natural sentence answering the question.
You MUST write the answer in {language_name} only.
Do not translate. Do not switch language.
Return only the sentence."""

    try:
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"[format_answer] Ollama error: {e}")
        return f"[format_error] {raw_value}"