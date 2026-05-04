import ollama

BASE = "http://www.semanticweb.org/ontologies/flight_ontology#"

def extract_sparql(text):
    start = text.find("SELECT")
    if start != -1:
        query = text[start:].strip()
        open_braces = query.count("{")
        close_braces = query.count("}")
        deficit = open_braces - close_braces
        if deficit > 0:      # ← INSIDE the if block
            query += "\n}" * deficit
        return query
    return text.strip()

def inject_and_generate(flight_uri, property_short, user_question, strategy="zero-shot"):
    property_uri = BASE + property_short

    if strategy == "zero-shot":
        prompt = f"""You are a SPARQL query generator for a flight knowledge graph.

The flight subject URI is: <{flight_uri}>
The property URI is: <{property_uri}>

Write a valid SPARQL SELECT query that retrieves the value of that property for that flight.
The query must start with SELECT ?value WHERE {{
Use full URIs with angle brackets. Do not use PREFIX declarations.
Return only the SPARQL query. No explanation. No markdown."""

    elif strategy == "few-shot":
        prompt = f"""You are a SPARQL query generator for a flight knowledge graph.

Here are examples of correct SPARQL queries:

Question: What airline operates flight BR62?
Flight URI: <{BASE}Flight/flight_3a28bc61>
Property URI: <{BASE}hasAirline>
SPARQL:
SELECT ?value WHERE {{
  <{BASE}Flight/flight_3a28bc61> <{BASE}hasAirline> ?value .
}}

Question: What is the departure city of flight AF1739?
Flight URI: <{BASE}Flight/flight_3a3a6d0c>
Property URI: <{BASE}hasOriginCity>
SPARQL:
SELECT ?value WHERE {{
  <{BASE}Flight/flight_3a3a6d0c> <{BASE}hasOriginCity> ?value .
}}

Now write a SPARQL query for:
Flight URI: <{flight_uri}>
Property URI: <{property_uri}>

The query must start with SELECT ?value WHERE {{
Use full URIs with angle brackets. Do not use PREFIX declarations.
Return only the SPARQL query. No explanation. No markdown."""

    elif strategy == "cot":
        prompt = f"""You are a SPARQL query generator for a flight knowledge graph.
Think step by step:

Step 1 — Identify the subject.
The flight we are asking about is: <{flight_uri}>

Step 2 — Identify the property.
The property being asked about is: <{property_uri}>

Step 3 — Identify what to retrieve.
We want to retrieve an unknown value — call it ?value.

Step 4 — Build the WHERE clause.
Connect the subject to the property to get the value:
<{flight_uri}> <{property_uri}> ?value .

Step 5 — Write the full query.
The query must start with SELECT ?value WHERE {{
Use full URIs with angle brackets. Do not use PREFIX declarations.
Return only the SPARQL query. No explanation. No markdown."""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    return extract_sparql(response["message"]["content"])