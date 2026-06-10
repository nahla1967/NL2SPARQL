import ollama

# Used ONLY for the static few-shot examples below.
# Never applied to real query URIs.
FLIGHT_BASE = "http://www.semanticweb.org/ontologies/flight_ontology#"

def extract_sparql(text):
    start = text.find("SELECT")
    if start != -1:
        query = text[start:].strip()
        open_braces  = query.count("{")
        close_braces = query.count("}")
        deficit = open_braces - close_braces
        if deficit > 0:
            query += "\n}" * deficit
        return query
    return text.strip()

# ── SIGNATURE CHANGE ─────────────────────────────────────────────────────────
# property_uri and property2_uri are now FULL URIs passed by the caller.
# The generator no longer prepends any BASE — it has no way to know which
# ontology it is working with, so the caller must resolve that.
# ─────────────────────────────────────────────────────────────────────────────
def inject_and_generate(entity_uri, property_uri, user_question,
                        strategy="zero-shot", property2_uri=None):

    is_two_hop    = property2_uri is not None

    if strategy == "zero-shot":
        if is_two_hop:
            prompt = f"""You are a SPARQL query generator for a knowledge graph.

            The subject URI is: <{entity_uri}>
            The first property URI is: <{property_uri}>
            The second property URI is: <{property2_uri}>

            Write a valid SPARQL SELECT query that retrieves the value using two hops.
            The query must follow this exact structure:
            SELECT ?value WHERE {{
            <subject_uri> <property1_uri> ?intermediate .
            ?intermediate <property2_uri> ?value .
            }}

            Use full URIs with angle brackets. Do not use PREFIX declarations.
            Return only the SPARQL query. No explanation. No markdown."""

        else:
            prompt = f"""You are a SPARQL query generator for a knowledge graph.

            The subject URI is: <{entity_uri}>
            The property URI is: <{property_uri}>

            Write a valid SPARQL SELECT query that retrieves the value of that property.
            The query must start with SELECT ?value WHERE {{
            Use full URIs with angle brackets. Do not use PREFIX declarations.
            Return only the SPARQL query. No explanation. No markdown."""

    elif strategy == "few-shot":
        if is_two_hop:
            prompt = f"""You are a SPARQL query generator for a knowledge graph.

            Here is an example of a correct two-hop SPARQL query:

            Question: What type of aircraft is used on flight BR62?
            Subject URI: <{FLIGHT_BASE}Flight/flight_3a28bc61>
            First property URI: <{FLIGHT_BASE}hasAircraft>
            Second property URI: <{FLIGHT_BASE}type>
            SPARQL:
            SELECT ?value WHERE {{
            <{FLIGHT_BASE}Flight/flight_3a28bc61> <{FLIGHT_BASE}hasAircraft> ?intermediate .
            ?intermediate <{FLIGHT_BASE}type> ?value .
            }}

            Now write a two-hop SPARQL query for:
            Subject URI: <{entity_uri}>
            First property URI: <{property_uri}>
            Second property URI: <{property2_uri}>

            The query must follow the same structure as the example.
            Use full URIs with angle brackets. Do not use PREFIX declarations.
            Return only the SPARQL query. No explanation. No markdown."""

        else:
            prompt = f"""You are a SPARQL query generator for a knowledge graph.

            Here are examples of correct SPARQL queries:

            Question: What airline operates flight BR62?
            Subject URI: <{FLIGHT_BASE}Flight/flight_3a28bc61>
            Property URI: <{FLIGHT_BASE}hasAirline>
            SPARQL:
            SELECT ?value WHERE {{
            <{FLIGHT_BASE}Flight/flight_3a28bc61> <{FLIGHT_BASE}hasAirline> ?value .
            }}

            Question: What is the departure city of flight AF1739?
            Subject URI: <{FLIGHT_BASE}Flight/flight_3a3a6d0c>
            Property URI: <{FLIGHT_BASE}hasOriginCity>
            SPARQL:
            SELECT ?value WHERE {{
            <{FLIGHT_BASE}Flight/flight_3a3a6d0c> <{FLIGHT_BASE}hasOriginCity> ?value .
            }}

            Now write a SPARQL query for:
            Subject URI: <{entity_uri}>
            Property URI: <{property_uri}>

            The query must start with SELECT ?value WHERE {{
            Use full URIs with angle brackets. Do not use PREFIX declarations.
            Return only the SPARQL query. No explanation. No markdown."""

    elif strategy == "cot":
        if is_two_hop:
            prompt = f"""You are a SPARQL query generator for a knowledge graph.
            Think step by step:

            Step 1 — Identify the subject.
            The entity we are asking about is: <{entity_uri}>

            Step 2 — Identify the first property.
            The first property leads to an intermediate node: <{property_uri}>

            Step 3 — Identify the second property.
            The second property retrieves the final value: <{property2_uri}>

            Step 4 — Build the WHERE clause.
            First hop:  <{entity_uri}> <{property_uri}> ?intermediate .
            Second hop: ?intermediate <{property2_uri}> ?value .

            Step 5 — Write the full query.
            The query must start with SELECT ?value WHERE {{
            Use full URIs with angle brackets. Do not use PREFIX declarations.
            Return only the SPARQL query. No explanation. No markdown."""

        else:
            prompt = f"""You are a SPARQL query generator for a knowledge graph.
            Think step by step:

            Step 1 — Identify the subject.
            The entity we are asking about is: <{entity_uri}>

            Step 2 — Identify the property.
            The property being asked about is: <{property_uri}>

            Step 3 — Identify what to retrieve.
            We want to retrieve an unknown value — call it ?value.

            Step 4 — Build the WHERE clause.
            Connect the subject to the property to get the value:
            <{entity_uri}> <{property_uri}> ?value .

            Step 5 — Write the full query.
            The query must start with SELECT ?value WHERE {{
            Use full URIs with angle brackets. Do not use PREFIX declarations.
            Return only the SPARQL query. No explanation. No markdown."""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    return extract_sparql(response["message"]["content"])