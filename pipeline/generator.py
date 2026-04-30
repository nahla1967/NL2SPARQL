import ollama

def inject_and_generate(flight_uri, property_short, user_question, strategy="zero-shot"):
    base = "http://www.semanticweb.org/ontologies/flight_ontology#"
    property_uri = base + property_short

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
Question: Where does OS235 depart from?
SPARQL:
SELECT ?value
WHERE {{
  <http://www.semanticweb.org/ontologies/flight_ontology#Flight/flight_3a33107d> <http://www.semanticweb.org/ontologies/flight_ontology#hasOriginCity> ?value .
}}

Question: What airline operates BR62?
SPARQL:
SELECT ?value
WHERE {{
  <http://www.semanticweb.org/ontologies/flight_ontology#Flight/flight_example> <http://www.semanticweb.org/ontologies/flight_ontology#hasAirline> ?value .
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
    return response["message"]["content"].strip()