"""
generator.py  (v3 — reversed-triple fix)
------------------------------------------
WHAT CHANGED vs v2:
    Added fix_reversed_triple() inside extract_sparql().

    WHY THIS IS NEEDED:
        For Arabic questions, the LLM sometimes generates triples in
        the wrong order:
            <subject> ?value <property> .    ← WRONG
        instead of:
            <subject> <property> ?value .    ← CORRECT

        This is a known LLM behavior: Arabic reads right-to-left, and
        the model occasionally mirrors that order into SPARQL.

        The fix is a regex that detects this exact pattern and flips it.
        No LLM call needed — pure string correction.

    This fix is applied BEFORE returning from extract_sparql(), so it
    protects all three strategies (zero-shot, few-shot, cot) equally.

    No other changes. The KG-agnostic design from v2 is preserved.

WHAT CHANGED (this revision — removed hardcoded BASE)
-----------------------------------------------------------
    FLIGHT_BASE was a private, hardcoded constant used only in this
    file's few-shot examples — inconsistent with the rest of the
    pipeline, which always resolves base URIs through
    kg_registry.get_base_uri(). Replaced with a call to that shared
    function at module load, so this file has no ontology knowledge of
    its own and stays in sync with kg_registry.py if the base URI ever
    changes.
"""

import re
import difflib
import ollama
from kg_registry import get_base_uri, get_open_kg_schema

_KNOWN_OPEN_KG_PROPERTIES = set(
    re.findall(r'^\s*(\w+)\s*(?:\(|→)', get_open_kg_schema(), re.MULTILINE)
)

# Used ONLY for the static few-shot examples below. Resolved from the
# shared registry instead of hardcoded, so this file has a single
# source of truth for the flight ontology's base URI, same as every
# other module in the pipeline.
FLIGHT_BASE = get_base_uri("flights")


# ── REVERSED TRIPLE FIX ───────────────────────────────────────────────────────

def fix_reversed_triple(sparql: str) -> str:
    """
    Detects and corrects reversed SPARQL triples.

    The LLM occasionally generates (especially for Arabic questions):
        <subject_uri> ?value <property_uri> .   ← WRONG ORDER

    This function flips them to the correct form:
        <subject_uri> <property_uri> ?value .   ← CORRECT

    The pattern: a full URI, then a variable, then another full URI,
    ending with a dot — unambiguously a reversed triple.

    Works for all variable names (?value, ?v, ?result, etc.).
    Does not modify two-hop triples (?intermediate is a variable, not URI).
    """
    # Matches: <uri> ?var <uri> .
    pattern = re.compile(
        r'(<[^>]+>)\s+\?(\w+)\s+(<[^>]+>)\s*\.',
        re.MULTILINE
    )

    def flip(m):
        subject      = m.group(1)
        var_name     = m.group(2)
        property_uri = m.group(3)
        print(f"[generator] Reversed triple detected — auto-correcting")
        return f"{subject} {property_uri} ?{var_name} ."

    return pattern.sub(flip, sparql)


# ── MISPLACED AGGREGATE FIX ───────────────────────────────────────────────────

def fix_misplaced_aggregate(sparql: str) -> str:
    """
    Detects and corrects a SPARQL aggregate expression placed AFTER the
    closing brace instead of inside the SELECT clause — observed on
    Arabic COUNT questions:

        SELECT ?count WHERE { ... } COUNT(?runway) AS ?count LIMIT 1   ← WRONG

    instead of:

        SELECT (COUNT(?runway) AS ?count) WHERE { ... } LIMIT 1        ← CORRECT

    Same bug family as fix_reversed_triple() above — the model moves a
    clause out of position specifically for Arabic output, likely the
    same right-to-left artifact. Pure string correction, no LLM call.
    """
    pattern = re.compile(
        r'SELECT\s+\?\w+\s+WHERE\s*(\{.*?\})\s*'
        r'(COUNT|SUM|MAX|MIN|AVG)\s*\(\s*\?(\w+)\s*\)\s+AS\s+\?(\w+)\s*'
        r'(LIMIT\s+\d+)?',
        re.IGNORECASE | re.DOTALL
    )

    def fix(m):
        body, func, agg_target, out_var = m.groups()
        return f"SELECT ({func.upper()}(?{agg_target}) AS ?{out_var}) WHERE {{ {body} }}"
    return pattern.sub(fix, sparql)


def fix_aggregate_in_filter(sparql: str) -> str:
    """
    Detects an aggregate expression used inside FILTER() instead of the
    SELECT clause — a fourth shape in the misplaced-aggregate bug family,
    observed on an Arabic COUNT question:

        SELECT ?count WHERE { ... FILTER(COUNT(?runway) = ?count) }  ← WRONG
        (invalid SPARQL: aggregates cannot appear inside FILTER without
        GROUP BY — Fuseki rejects this with HTTP 400)

    instead of:

        SELECT (COUNT(?runway) AS ?count) WHERE { ... }              ← CORRECT

    Same bug family as fix_misplaced_aggregate() / fix_aggregate_inside_braces()
    — the model moves the aggregate out of the SELECT clause. The backreference
    \\1 ensures the variable used inside FILTER matches the one declared in
    SELECT, so this only fires on the exact malformed shape, not on
    legitimate FILTER usage elsewhere.
    """
    pattern = re.compile(
        r'SELECT\s+\?(\w+)\s+WHERE\s*\{\s*(.*?)\s*'
        r'FILTER\s*\(\s*(COUNT|SUM|MAX|MIN|AVG)\s*\(\s*\?(\w+)\s*\)\s*=\s*\?\1\s*\)\s*\}',
        re.IGNORECASE | re.DOTALL
    )

    def fix(m):
        out_var, body, func, agg_target = m.groups()
        print(f"[generator] Aggregate-in-FILTER detected — auto-correcting")
        return f"SELECT ({func.upper()}(?{agg_target}) AS ?{out_var}) WHERE {{ {body} }}"

    return pattern.sub(fix, sparql)


# ── UNKNOWN PREDICATE VALIDATOR (open_kg branch only) ─────────────────────────

def fix_unknown_predicate(sparql: str) -> str:
    """
    Validates every predicate local name in a generated query against
    _KNOWN_OPEN_KG_PROPERTIES and auto-corrects near-miss hallucinations —
    observed on an Arabic run that generated "closes" instead of "closed".

    WHY THIS IS GENERAL rather than a one-word patch: the open_kg branch
    (generate_open_kg_sparql) is the one generation path that does NOT go
    through the Hybrid Mapping Layer (mapper.py) — the LLM writes raw
    predicate URIs directly from the schema text, with nothing checking
    them before execution. This validator closes that gap for ANY
    near-miss property name, not just "closed", so it also protects
    against future hallucinations like "iataCod" or "municipalty".

    A correction is applied only when:
      - the predicate's local name is not already a known property, AND
      - exactly one known property is within similarity of the fuzzy-match
        threshold (cutoff=0.75).
    If zero or multiple candidates match, the predicate is left untouched
    rather than guessed at — an ambiguous case is safer to fail loudly
    (execution error) than to silently substitute the wrong property.

    Class names in triples like "?x a <...#Runway>" are skipped (checked
    via the leading uppercase letter), since this validator only concerns
    itself with predicate positions, not rdf:type objects.
    """
    pattern = re.compile(r'<([^#>]+#)(\w+)>')

    def fix(m):
        base, local_name = m.group(1), m.group(2)
        if local_name in _KNOWN_OPEN_KG_PROPERTIES or local_name[:1].isupper():
            return m.group(0)
        candidates = difflib.get_close_matches(
            local_name, _KNOWN_OPEN_KG_PROPERTIES, n=2, cutoff=0.75
        )
        if len(candidates) == 1:
            print(f"[generator] Unknown predicate '{local_name}' auto-corrected to '{candidates[0]}'")
            return f"<{base}{candidates[0]}>"
        return m.group(0)

    return pattern.sub(fix, sparql)


# ── SPARQL EXTRACTOR ──────────────────────────────────────────────────────────

def fix_aggregate_inside_braces(sparql: str) -> str:
    """
    Detects an aggregate expression left as a bare statement inside the
    WHERE braces instead of wrapped in the SELECT clause:
        SELECT ?count WHERE { ... COUNT(?runway) AS ?count }   ← WRONG
    Same bug family as fix_misplaced_aggregate() — one more shape of it.
    """
    pattern = re.compile(
        r'SELECT\s+\?\w+\s+WHERE\s*\{\s*(.*?)\s*'
        r'(COUNT|SUM|MAX|MIN|AVG)\s*\(\s*\?(\w+)\s*\)\s+AS\s+\?(\w+)\s*\}',
        re.IGNORECASE | re.DOTALL
    )
    def fix(m):
        body, func, agg_target, out_var = m.groups()
        return f"SELECT ({func.upper()}(?{agg_target}) AS ?{out_var}) WHERE {{ {body} }}"
    return pattern.sub(fix, sparql)
# ── SPARQL EXTRACTOR ──────────────────────────────────────────────────────────

def extract_sparql(text: str) -> str:
    """
    Extracts the SPARQL SELECT query from the LLM response.

    1. Finds the SELECT keyword and strips everything before it.
    2. Auto-closes unclosed braces (handles truncated responses).
    3. Fixes reversed triples (handles Arabic LLM output quirk).
    """
    start = text.find("SELECT")
    if start != -1:
        query        = text[start:].strip()
        open_braces  = query.count("{")
        close_braces = query.count("}")
        deficit      = open_braces - close_braces
        if deficit > 0:
            query += "\n}" * deficit

        # Fix reversed triples before returning
        query = fix_reversed_triple(query)
        query = fix_misplaced_aggregate(query)
        query = fix_aggregate_inside_braces(query)
        query = fix_aggregate_in_filter(query)
        return query

    return text.strip()


# ── MAIN GENERATOR ────────────────────────────────────────────────────────────

def inject_and_generate(
    entity_uri:    str,
    property_uri:  str,
    user_question: str,
    strategy:      str = "zero-shot",
    property2_uri: str | None = None,
) -> str:
    """
    Generates a SPARQL SELECT query using an LLM.

    Args:
        entity_uri    : full URI of the subject (flight or airport)
        property_uri  : full URI of the first property
        user_question : original user question (for context)
        strategy      : zero-shot | few-shot | cot
        property2_uri : full URI of second property (two-hop queries only)

    Returns:
        A SPARQL SELECT query string starting with SELECT ?value WHERE {
    """
    is_two_hop = property2_uri is not None

    # ── ZERO-SHOT ──────────────────────────────────────────────────────────────
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

IMPORTANT: The triple order must always be: subject property object.
Never write: subject ?variable property.

Use full URIs with angle brackets. Do not use PREFIX declarations.
Return only the SPARQL query. No explanation. No markdown."""

        else:
            prompt = f"""You are a SPARQL query generator for a knowledge graph.

The subject URI is: <{entity_uri}>
The property URI is: <{property_uri}>

Write a valid SPARQL SELECT query that retrieves the value of that property.
The query must start with SELECT ?value WHERE {{

IMPORTANT: The triple order must always be: subject property object.
Write: <{entity_uri}> <{property_uri}> ?value .
Never write: <{entity_uri}> ?value <{property_uri}> .

Use full URIs with angle brackets. Do not use PREFIX declarations.
Return only the SPARQL query. No explanation. No markdown."""

    # ── FEW-SHOT ───────────────────────────────────────────────────────────────
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
IMPORTANT: Triple order is always subject → property → object (never reversed).
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
IMPORTANT: Triple order is always subject → property → object (never reversed).
Use full URIs with angle brackets. Do not use PREFIX declarations.
Return only the SPARQL query. No explanation. No markdown."""

    # ── CHAIN-OF-THOUGHT ───────────────────────────────────────────────────────
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
IMPORTANT: Triple order is always: subject property object.
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
IMPORTANT: Triple order is always: subject property object.
Correct:   <{entity_uri}> <{property_uri}> ?value .
WRONG:     <{entity_uri}> ?value <{property_uri}> .

Step 5 — Write the full query.
The query must start with SELECT ?value WHERE {{
Use full URIs with angle brackets. Do not use PREFIX declarations.
Return only the SPARQL query. No explanation. No markdown."""

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. "
                         f"Expected: zero-shot, few-shot, cot")

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0}
    )
    return extract_sparql(response["message"]["content"])

# ── OPEN KG SPARQL GENERATOR ──────────────────────────────────────────────────

FLIGHT_ONTOLOGY_NS  = "flight_ontology"
AIRPORT_ONTOLOGY_NS = "airport_ontology"
UNIVERSITY_ONTOLOGY_NS = "univ-bench.owl"

KG1_ENDPOINT = "http://localhost:3030/flights/sparql"
KG2_ENDPOINT = "http://localhost:3030/airports/sparql"
KG3_ENDPOINT = "http://localhost:3030/university/sparql"

def generate_open_kg_sparql(
    question: str,
    lang: str,
    schema: str,
) -> tuple[str, str]:
    """
    Generates a free SPARQL query for unanticipated aviation questions
    and determines which endpoint to execute it against.

    Unlike inject_and_generate(), this function does NOT receive pre-resolved
    URIs. Instead, it injects the full ontology schema into the prompt and
    asks the LLM to generate a query constrained to what actually exists.

    This is the open_kg branch — Branch F in the thesis.

    ENDPOINT DETECTION:
        After generation, the query is inspected for namespace markers.
        This is more reliable than asking the LLM to declare the endpoint,
        because namespace URIs are structurally present in any valid query
        that uses the schema — they cannot be omitted.

        Rules:
          - flight_ontology namespace only   → KG1 (flights endpoint)
          - airport_ontology namespace only  → KG2 (airports endpoint)
          - both namespaces present          → KG2 (cross-KG joins are
                                               handled separately; airport
                                               data is the outer query)
          - neither namespace detected       → KG2 (safe default)

    Args:
        question : the original user question
        lang     : detected language (en / fr / ar)
        schema   : the OPEN_KG_SCHEMA string from kg_registry.py

    Returns:
        A tuple (sparql: str, endpoint: str).
        sparql   : the generated SPARQL SELECT query, or "" on failure.
        endpoint : the full URL of the target Fuseki endpoint.
    """
    prompt = f"""You are a SPARQL expert for an aviation knowledge graph system.

The knowledge graphs have the following structure:
{schema}

The user asked: "{question}"

Write a valid SPARQL SELECT query that answers this question using ONLY
the classes and properties described above.

STRICT RULES:
- Use full URIs with angle brackets. Never use PREFIX declarations.
- The query must start with SELECT
- Use ?value as the main result variable when retrieving a single value.
- If the question requires data from both KGs, write a query for KG2 only
  (airports endpoint) since cross-KG joins are handled separately.
- Triple order is always: subject property object.
  CORRECT:   <uri> <property> ?value .
  WRONG:     <uri> ?value <property> .
- Do not invent properties that are not listed above.
- Only include the triple patterns strictly needed to answer the question — do not add extra properties the question didn't ask about
- If multiple results are possible, add LIMIT 10.

EXAMPLES of correct queries:

Q: "Which airports have a grass runway?"
SELECT ?name WHERE {{
  ?airport a <http://www.semanticweb.org/ontologies/airport_ontology#Airport> .
  ?airport <http://www.semanticweb.org/ontologies/airport_ontology#airportName> ?name .
  ?airport <http://www.semanticweb.org/ontologies/airport_ontology#hasRunway> ?runway .
  ?runway <http://www.semanticweb.org/ontologies/airport_ontology#surface> ?surface .
  FILTER(CONTAINS(LCASE(?surface), "gr"))
}} LIMIT 10

Q: "How many airports are in the dataset?"
SELECT (COUNT(?airport) AS ?count) WHERE {{
  ?airport a <http://www.semanticweb.org/ontologies/airport_ontology#Airport> .
}}


Q: "Which flight has the highest ground speed?"
SELECT ?number ?value WHERE {{
  ?flight a <http://www.semanticweb.org/ontologies/flight_ontology#Flight> .
  ?flight <http://www.semanticweb.org/ontologies/flight_ontology#flightNumber> ?number .
  ?flight <http://www.semanticweb.org/ontologies/flight_ontology#hasFlightEvent> ?event .
  ?event <http://www.semanticweb.org/ontologies/flight_ontology#gspeed> ?value .
}} ORDER BY DESC(?value) LIMIT 1
Q: "What is the registration number of flight OS235?"
SELECT ?value WHERE {{
  ?flight a <http://www.semanticweb.org/ontologies/flight_ontology#Flight> .
  ?flight <http://www.semanticweb.org/ontologies/flight_ontology#flightNumber> "OS235" .
  ?flight <http://www.semanticweb.org/ontologies/flight_ontology#hasAircraft> ?aircraft .
  ?aircraft <http://www.semanticweb.org/ontologies/flight_ontology#reg> ?value .
}}


Q: "How many airports in the dataset have a grass runway?"
SELECT (COUNT(DISTINCT ?airport) AS ?count) WHERE {{
  ?airport a <http://www.semanticweb.org/ontologies/airport_ontology#Airport> .
  ?airport <http://www.semanticweb.org/ontologies/airport_ontology#hasRunway> ?runway .
  ?runway <http://www.semanticweb.org/ontologies/airport_ontology#surface> ?surface .
  FILTER(CONTAINS(LCASE(?surface), "gr"))
}}
Q: "كم عدد المطارات التي لديها مدرج إسفلتي؟"
SELECT (COUNT(DISTINCT ?airport) AS ?count) WHERE {{
  ?airport a <http://www.semanticweb.org/ontologies/airport_ontology#Airport> .
  ?airport <http://www.semanticweb.org/ontologies/airport_ontology#hasRunway> ?runway .
  ?runway <http://www.semanticweb.org/ontologies/airport_ontology#surface> ?surface .
  FILTER(CONTAINS(LCASE(?surface), "asp"))
}}

Q: "How many runways in the dataset are lighted?"
SELECT (COUNT(?runway) AS ?count) WHERE {{
  ?runway a <http://www.semanticweb.org/ontologies/airport_ontology#Runway> .
  ?runway <http://www.semanticweb.org/ontologies/airport_ontology#lighted> true .
}}

Return ONLY the SPARQL query. No explanation. No markdown."""

    try:
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0}
        )
        sparql = extract_sparql(response["message"]["content"])
        print(f"[DEBUG raw sparql]\n{sparql}\n")
        select_vars = re.findall(r'SELECT\s+(.*?)\s+WHERE', sparql, re.IGNORECASE)
        if select_vars:
            requested = re.findall(r'\?(\w+)', select_vars[0])
            where_body = sparql[sparql.find("WHERE"):]
            for var in requested:
                if sparql.count(f"?{var}") < 2:  # appears in SELECT but nowhere else
                    print(f"[generator] SELECT variable ?{var} never bound in WHERE — discarding query")
                    sparql = ""
                    break
        sparql = fix_unknown_predicate(sparql)

        # ── Endpoint detection from namespace markers ──────────────────────────

        # ── Endpoint detection from namespace markers ──────────────────────────
        # The generated query will contain full URIs from whichever ontology
        # it targets. We inspect those URIs to determine the correct endpoint
        # rather than guessing or trying both sequentially.
        has_kg1 = FLIGHT_ONTOLOGY_NS     in sparql
        has_kg2 = AIRPORT_ONTOLOGY_NS    in sparql
        has_kg3 = UNIVERSITY_ONTOLOGY_NS in sparql

        if has_kg3 and not has_kg1 and not has_kg2:
            endpoint = KG3_ENDPOINT
            print(f"[generator] open_kg → KG3 (university endpoint)")
        elif has_kg1 and not has_kg2:
            endpoint = KG1_ENDPOINT
            print(f"[generator] open_kg → KG1 (flights endpoint)")
        elif has_kg2:
            endpoint = KG2_ENDPOINT
            print(f"[generator] open_kg → KG2 (airports endpoint)")
        else:
            endpoint = KG2_ENDPOINT
            print(f"[generator] open_kg → no namespace detected, defaulting to KG2")

        return sparql, endpoint

    except Exception as e:
        print(f"[generator] generate_open_kg_sparql failed: {e}")
        return "", KG2_ENDPOINT