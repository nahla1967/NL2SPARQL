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

def inject_and_generate(flight_uri, property_short, user_question, strategy="zero-shot", property2_uri=None):
    property_uri = BASE + property_short
    is_two_hop = property2_uri is not None
    property2_full = BASE + property2_uri if is_two_hop else None

    if strategy == "zero-shot":
        if is_two_hop:
            prompt = f"""You are a SPARQL query generator for a flight knowledge graph.

            The flight subject URI is: <{flight_uri}>
            The first property URI is: <{property_uri}>
            The second property URI is: <{property2_full}>

            Write a valid SPARQL SELECT query that retrieves the value using two hops.
            The query must follow this exact structure:
            SELECT ?value WHERE {{
            <flight_uri> <property1_uri> ?intermediate .
            ?intermediate <property2_uri> ?value .
            }}

            Use full URIs with angle brackets. Do not use PREFIX declarations.
            Return only the SPARQL query. No explanation. No markdown."""

        else:
            prompt = f"""You are a SPARQL query generator for a flight knowledge graph.

            The flight subject URI is: <{flight_uri}>
            The property URI is: <{property_uri}>

            Write a valid SPARQL SELECT query that retrieves the value of that property for that flight.
            The query must start with SELECT ?value WHERE {{
            Use full URIs with angle brackets. Do not use PREFIX declarations.
            Return only the SPARQL query. No explanation. No markdown."""

    elif strategy == "few-shot":
        if is_two_hop:
            prompt = f"""You are a SPARQL query generator for a flight knowledge graph.

            Here is an example of a correct two-hop SPARQL query:

            Question: What type of aircraft is used on flight BR62?
            Flight URI: <{BASE}Flight/flight_3a28bc61>
            First property URI: <{BASE}hasAircraft>
            Second property URI: <{BASE}type>
            SPARQL:
            SELECT ?value WHERE {{
            <{BASE}Flight/flight_3a28bc61> <{BASE}hasAircraft> ?intermediate .
            ?intermediate <{BASE}type> ?value .
            }}

            Now write a two-hop SPARQL query for:
            Flight URI: <{flight_uri}>
            First property URI: <{property_uri}>
            Second property URI: <{property2_full}>

            The query must follow the same structure as the example.
            Use full URIs with angle brackets. Do not use PREFIX declarations.
            Return only the SPARQL query. No explanation. No markdown."""

        else:
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
        if is_two_hop:
            prompt = f"""You are a SPARQL query generator for a flight knowledge graph.
            Think step by step:

            Step 1 — Identify the subject.
            The flight we are asking about is: <{flight_uri}>

            Step 2 — Identify the first property.
            The first property leads to an intermediate node: <{property_uri}>

            Step 3 — Identify the second property.
            The second property retrieves the final value from the intermediate node: <{property2_full}>

            Step 4 — Build the WHERE clause.
            First hop: <{flight_uri}> <{property_uri}> ?intermediate .
            Second hop: ?intermediate <{property2_full}> ?value .

            Step 5 — Write the full query.
            The query must start with SELECT ?value WHERE {{
            Use full URIs with angle brackets. Do not use PREFIX declarations.
            Return only the SPARQL query. No explanation. No markdown."""

        else:
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