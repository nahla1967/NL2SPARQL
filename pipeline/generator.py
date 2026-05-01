import ollama

BASE = "http://www.semanticweb.org/ontologies/flight_ontology#"
def extract_sparql(text):
    start = text.find("SELECT")
    if start != -1:
        return text[start:].strip()
    return text.strip()

def inject_and_generate(flight_uri, property_short, user_question, strategy="zero-shot"):
    property_uri = BASE + property_short

    if strategy == "zero-shot":
        prompt = f"""Return ONLY this SPARQL query, exactly as written:

SELECT ?value
WHERE {{
  <{flight_uri}> <{property_uri}> ?value .
}}

Do not change anything. Do not add anything."""

    elif strategy == "few-shot":
        prompt = f"""You generate SPARQL queries for a flight knowledge graph.

Examples:

Question: What airline operates BR62?
SPARQL:
SELECT ?value
WHERE {{
  <{BASE}Flight/flight_3a28bc61> <{BASE}hasAirline> ?value .
}}

Question: What is the departure city of AF1739?
SPARQL:
SELECT ?value
WHERE {{
  <{BASE}Flight/flight_3a3a6d0c> <{BASE}hasOriginCity> ?value .
}}

Question: What aircraft is used for flight 7L280?
SPARQL:
SELECT ?value
WHERE {{
  <{BASE}Flight/flight_3a363e0e> <{BASE}hasAircraft> ?value .
}}

Now generate ONLY the SPARQL query for this:
Question: {user_question}
SPARQL:
SELECT ?value
WHERE {{
  <{flight_uri}> <{property_uri}> ?value .
}}"""

    elif strategy == "cot":
        prompt = f"""You generate SPARQL queries for a flight knowledge graph.
Think step by step:
1. The flight URI is: <{flight_uri}>
2. The property URI is: <{property_uri}>
3. The query must SELECT ?value WHERE the flight has that property.

Now return ONLY this SPARQL query:

SELECT ?value
WHERE {{
  <{flight_uri}> <{property_uri}> ?value .
}}

Do not add explanation. Do not add markdown."""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    return extract_sparql(response["message"]["content"])