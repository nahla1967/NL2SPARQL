import json
import urllib.parse
import urllib.request
import ollama
from rdflib.plugins.sparql import prepareQuery

# Country code lookup table.
# Used to resolve ISO 3166-1 alpha-2 codes returned by the KG
# into readable country names before passing them to the answer formatter.
# Without this, the LLM may hallucinate incorrect expansions (e.g. "DE" → "Denmark").
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


def resolve_country_code(value):
    """
    Resolves a 2-letter ISO country code to its full name.
    Returns the original value unchanged if not found in the table.
    """
    return COUNTRY_CODES.get(value.strip().upper(), value)


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


def clean_uri(value):
    """
    Converts a raw URI into a readable string as a last-resort fallback.
    Strips the namespace and replaces underscores with spaces.
    """
    if value.startswith("http"):
        return value.split("/")[-1].replace("_", " ")
    return urllib.parse.unquote(value).replace("_", " ")


def resolve_entity(uri):
    """
    Resolves a KG URI to a human-readable value by querying Fuseki
    for a label property, depending on the entity type.

    Handles: City, Airline, Aircraft, Route.
    Falls back to clean_uri for unrecognised entity types.
    """
    if not uri.startswith("http"):
        return uri

    base = "http://www.semanticweb.org/ontologies/flight_ontology#"
    url = "http://localhost:3030/flights/sparql"

    if "/City/" in uri:
        name_props = ["orig_city"]
    elif "/Airline/" in uri:
        name_props = ["operating_as"]
        elif "Aircraft/" in uri:
    elif "/Aircraft/" in uri:
        name_props = ["type"]
    elif "/Route/" in uri:
        route_name = uri.split("/Route/")[-1]
        return urllib.parse.unquote(route_name).replace("_", " ")
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
        except Exception:
            pass

    return clean_uri(uri)


def execute_sparql(sparql_query):
    """
    Executes a SPARQL SELECT query against the local Fuseki endpoint.
    Resolves the returned value through resolve_entity and resolve_country_code.
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

                # Resolve 2-letter ISO country codes to full country names.
                # Prevents the LLM formatter from hallucinating incorrect expansions.
                if raw_value and len(raw_value) == 2 and raw_value.isupper():
                    raw_value = resolve_country_code(raw_value)

                return raw_value

    except urllib.error.URLError as e:
        print(f"[execute_sparql] Fuseki unreachable: {e}")
    except Exception as e:
        print(f"[execute_sparql] Unexpected error: {e}")

    return None


def format_answer(question, raw_value, lang):
    """
    Uses the LLM to reformulate the raw KG answer into a natural language
    sentence in the same language as the user's original question.
    Falls back to a structured error string if Ollama is unavailable.
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