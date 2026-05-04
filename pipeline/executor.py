import json
import urllib.parse
import urllib.request
import ollama
from rdflib.plugins.sparql import prepareQuery

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
    if not uri.startswith("http"):
        return uri

    base = "http://www.semanticweb.org/ontologies/flight_ontology#"
    url = "http://localhost:3030/flights/sparql"

    if "/City/" in uri:
        name_props = ["orig_city", "dest_city"]
    elif "/Airline/" in uri:
        name_props = ["operating_as"]
    elif "/Aircraft/" in uri:
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
        except Exception:
            pass

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

    # Problem 4 — The previous implementation had no error handling here.
    # This is the last step of the pipeline, which means a crash here would
    # discard all previous work (entity extraction, URI mapping, SPARQL execution)
    # and produce no log entry for that test case.
    #
    # The fix wraps the Ollama call in a try/except block. If the LLM service
    # is unavailable or returns an unexpected response, the function returns a
    # structured fallback string instead of raising an exception.
    #
    # The fallback includes the raw KG value so the run is still partially useful
    # for evaluation: you can still score Execution Accuracy and SPARQL Validity,
    # even if the natural language formatting step failed.
    try:
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"[format_answer] Ollama error: {e}")
        # Return the raw value so the log entry is still informative.
        # The prefix makes it easy to identify formatting failures
        # when reviewing logs.jsonl during evaluation.
        return f"[format_error] {raw_value}"