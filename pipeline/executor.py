import json
import urllib.parse
import urllib.request
import ollama

def validate_sparql(query):
    return "SELECT" in query and "WHERE" in query

def execute_sparql(sparql_query):
    url = "http://localhost:3030/flights/sparql"
    data = urllib.parse.urlencode({
        "query": sparql_query,
        "format": "application/sparql-results+json"
    }).encode()
    req = urllib.request.Request(url, data=data)
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read())
        bindings = result["results"]["bindings"]
        if bindings:
            first_key = list(bindings[0].keys())[0]
            return bindings[0][first_key]["value"]
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