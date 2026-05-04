import json
import urllib.parse
import urllib.request
import ollama
from rdflib.plugins.sparql import prepareQuery  # real SPARQL parser

def validate_sparql(query):
    """
    Validates SPARQL syntax using rdflib's parser.
    Returns True only if the query is syntactically correct.
    The previous implementation only checked for keyword presence,
    which is not sufficient for a valid evaluation metric.
    """
    try:
        prepareQuery(query)
        return True
    except Exception:
        return False
def clean_uri(value):
    if value.startswith("http"):
        return value.split("/")[-1].replace("_", " ")
    return value

def resolve_entity(uri):
    # If plain text already — return directly
    if not uri.startswith("http"):
        return uri

    base = "http://www.semanticweb.org/ontologies/flight_ontology#"
    url = "http://localhost:3030/flights/sparql"

    # Try each known name property one by one
    for name_prop in ["operating_as", "dest_city", "orig_city", "type"]:
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

    # If no name property found — extract readable part from URI
    return clean_uri(uri)

def execute_sparql(sparql_query):
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
                return resolve_entity(bindings[0][first_key]["value"])
    except urllib.error.URLError as e:
        print(f"[execute_sparql] Fuseki unreachable: {e}")
    except Exception as e:
        print(f"[execute_sparql] Unexpected error: {e}")
    return None
def format_answer(question, raw_value, lang):
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

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"].strip()