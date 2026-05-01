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
            bindings = result["results"]["bindings"]
            if bindings:
                first_key = list(bindings[0].keys())[0]
                return clean_uri(bindings[0][first_key]["value"])
    except urllib.error.URLError as e:
        print(f"[execute_sparql] Fuseki unreachable: {e}")
    except Exception as e:
        print(f"[execute_sparql] Unexpected error: {e}")
    return None

def format_answer(question, raw_value, lang):
    prompt = f"""
The user asked: "{question}"
The answer from the database is: "{raw_value}"
Write a natural, short answer in the same language as the question.
Only return the sentence. No explanation.
"""
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"].strip()