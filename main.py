"""
main.py  (v2 — multi-KG)
------------------------
Entry point for the NL2SPARQL system.

WHAT CHANGED vs v1:
    Added registry-driven routing. The router decides which branch
    to execute before any other component runs.

    Three branches:
        single_kg1  → existing KG1 pipeline (zero changes inside)
        single_kg2  → new KG2 airport branch
        cross_kg    → two-step KG1 → IATA → KG2 resolver
        template    → predefined SPARQL templates (filter/rank/compare)

    The language detector and SPARQL generator are completely unchanged.

THESIS CLAIM:
    The core pipeline required no structural modification.
    Extractor, mapper, and executor each received fewer than 5 lines
    of change to accept a configuration object from the registry.
"""

import json

# ── PIPELINE COMPONENTS ───────────────────────────────────────────────────────
from pipeline.language  import detect_language
from router    import route
from pipeline.extractor import (
    extract_entities,
    validate_extraction,
    is_flight_question,
    extract_airport_entities,
    validate_airport_extraction,
    extract_university_entities,
    validate_university_extraction,
)
from pipeline.mapper import (
    get_university_entity_type,
    load_lexicon,
    map_property_cascade,
    map_flight,
    map_airport,
    map_university_entity,
)
from pipeline.generator import inject_and_generate, generate_open_kg_sparql
from pipeline.executor  import (
    validate_sparql,
    execute_sparql,
    format_answer,
    format_answer_list,
)
from cross_kg_resolver import resolve_cross_kg
from template_resolver import resolve_template
from kg_registry import get_base_uri, get_endpoint, get_lexicon

# ── TEST CONFIGURATION ────────────────────────────────────────────────────────
question  = "What university is Department5 part of?"
condition = "zero-shot"   # zero-shot | few-shot | cot

# ── STEP 0: LANGUAGE DETECTION ────────────────────────────────────────────────
lang = detect_language(question)
print(f"[0] Language : {lang}")

# ── STEP 1: ROUTING ───────────────────────────────────────────────────────────
routing = route(question)
query_type = routing["query_type"]
print(f"[1] Route    : {query_type} | entity={routing['entity']} | direction={routing['direction']}")

# Base log object — shared across all branches
log = {
    "condition":    condition,
    "language":     lang,
    "question":     question,
    "query_type":   query_type,
    "kg":           routing["kg"],
    "entity":       routing["entity"],
    "direction":    routing["direction"],
    "template":     routing["template"],
    "mapping_layer": None,
    "property_uri": None,
    "sparql":       None,
    "sparql_valid": False,
    "raw_answer":   None,
    "final_answer": None,
    "failure_type": None,
}

# ─────────────────────────────────────────────────────────────────────────────
# BRANCH A — OUT OF SCOPE
# ─────────────────────────────────────────────────────────────────────────────
if query_type == "out_of_scope":
    print("Out of scope.")
    log["failure_type"] = "out_of_scope"
    with open("logs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")
    print("Logged.")
    exit()

# ─────────────────────────────────────────────────────────────────────────────
# BRANCH B — SINGLE KG1
# ─────────────────────────────────────────────────────────────────────────────
elif query_type == "single_kg1":

    # Step 1: extract
    entities = extract_entities(question, lang)
    print(f"[2] Entities : {entities}")

    if not validate_extraction(entities) or not is_flight_question(entities):
        print("Extraction failed.")
        log["failure_type"] = "extraction_failure"
        log["extraction_failure_reason"] = entities.get("reason", "validation_failed")
        with open("logs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")
        exit()

    # Step 2: map
    lexicon = load_lexicon(get_lexicon("flights"))
    property_uri, mapping_layer, property2_uri = map_property_cascade(
        entities["property"], lexicon, get_lexicon("flights")
    )
    flight_uri = map_flight(entities["entity"])

    log.update({
        "flight_uri":    flight_uri,
        "property_uri":  property_uri,
        "property2_uri": property2_uri,
        "mapping_layer": mapping_layer,
    })

    if not flight_uri or not property_uri:
        print("Mapping failed.")
        log["failure_type"] = "mapping_failure"
        with open("logs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")
        exit()

    # Step 3: generate
    BASE = get_base_uri("flights")
    full_prop_uri  = BASE + property_uri
    full_prop2_uri = (BASE + property2_uri) if property2_uri else None

    sparql_query = inject_and_generate(
        flight_uri, full_prop_uri, question,
        strategy=condition, property2_uri=full_prop2_uri
    )
    log["sparql"] = sparql_query
    print(f"\nGenerated SPARQL:\n{sparql_query}")

    # Step 4: validate
    is_valid = (
        validate_sparql(sparql_query)
        and sparql_query.strip().startswith("SELECT")
        and "PREFIX" not in sparql_query
        and full_prop_uri in sparql_query
    )
    if full_prop2_uri:
        if full_prop2_uri not in sparql_query:
            is_valid = False
    log["sparql_valid"] = is_valid

    # Step 5: execute
    if is_valid:
        raw = execute_sparql(sparql_query, endpoint=get_endpoint("flights"))
        log["raw_answer"] = raw
        print(f"\nRaw answer: {raw}")
        if raw:
            answer = format_answer(question, raw, lang)
            log["final_answer"]  = answer
            log["failure_type"]  = "success"
            print(f"Final answer: {answer}")
        else:
            log["failure_type"] = "execution_failure"
    else:
        print("Invalid SPARQL.")
        log["failure_type"] = "generation_failure"

# ─────────────────────────────────────────────────────────────────────────────
# BRANCH C — SINGLE KG2
# ─────────────────────────────────────────────────────────────────────────────
elif query_type == "single_kg2":

    # Step 1: extract airport property phrase
    entities = extract_airport_entities(question, lang, routing["entity"])
    print(f"[2] Entities : {entities}")

    if not validate_airport_extraction(entities):
        print("Airport extraction failed.")
        log["failure_type"] = "extraction_failure"
        with open("logs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")
        exit()

    # Step 2: map property and airport URI
    lexicon_path = get_lexicon("airports")
    lexicon      = load_lexicon(lexicon_path)

    property_uri, mapping_layer, property2_uri = map_property_cascade(
        entities["property"], lexicon, lexicon_path
    )
    airport_uri = map_airport(entities["entity"]) if entities["entity"] else None

    log.update({
        "airport_uri":   airport_uri,
        "property_uri":  property_uri,
        "property2_uri": property2_uri,
        "mapping_layer": mapping_layer,
    })

    if not airport_uri or not property_uri:
        print("Mapping failed.")
        log["failure_type"] = "mapping_failure"
        with open("logs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")
        exit()

    # Step 3: generate
    BASE = get_base_uri("airports")
    full_prop_uri  = BASE + property_uri
    full_prop2_uri = (BASE + property2_uri) if property2_uri else None

    sparql_query = inject_and_generate(
        airport_uri, full_prop_uri, question,
        strategy=condition, property2_uri=full_prop2_uri
    )
    log["sparql"] = sparql_query
    print(f"\nGenerated SPARQL:\n{sparql_query}")

    # Step 4: validate
    is_valid = (
        validate_sparql(sparql_query)
        and sparql_query.strip().startswith("SELECT")
        and "PREFIX" not in sparql_query
        and full_prop_uri in sparql_query
    )
    log["sparql_valid"] = is_valid

    # Step 5: execute against KG2
    if is_valid:
        raw = execute_sparql(sparql_query, endpoint=get_endpoint("airports"))
        log["raw_answer"] = raw
        print(f"\nRaw answer: {raw}")
        if raw:
            answer = format_answer(question, raw, lang)
            log["final_answer"]  = answer
            log["failure_type"]  = "success"
            print(f"Final answer: {answer}")
        else:
            log["failure_type"] = "execution_failure"
    else:
        print("Invalid SPARQL.")
        log["failure_type"] = "generation_failure"

# ─────────────────────────────────────────────────────────────────────────────
# BRANCH D — CROSS KG
# ─────────────────────────────────────────────────────────────────────────────
elif query_type == "cross_kg":

    flight_number = routing["entity"]
    direction     = routing["direction"]

    # Step 1: resolve flight URI from KG1
    flight_uri = map_flight(flight_number)
    if not flight_uri:
        print(f"Flight {flight_number} not found in KG1.")
        log["failure_type"] = "mapping_failure"
        with open("logs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")
        exit()

    # Step 2: extract the airport property phrase
    entities = extract_airport_entities(question, lang, iata_from_router=None)
    print(f"[2] Entities : {entities}")

    # Step 3: map airport property
    lexicon_path = get_lexicon("airports")
    lexicon      = load_lexicon(lexicon_path)
    property_uri, mapping_layer, property2_uri = map_property_cascade(
        entities["property"], lexicon, lexicon_path
    )

    log.update({
        "flight_uri":    flight_uri,
        "property_uri":  property_uri,
        "mapping_layer": mapping_layer,
        "direction":     direction,
    })

    if not property_uri:
        print("Property mapping failed.")
        log["failure_type"] = "mapping_failure"
        with open("logs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")
        exit()

    # Step 4: build full property URI for KG2
    BASE          = get_base_uri("airports")
    full_prop_uri = BASE + property_uri

    # Step 5: cross-KG resolution (3-step bridge)
    result = resolve_cross_kg(
        flight_uri     = flight_uri,
        direction      = direction,
        property_uri   = full_prop_uri,
        property_short = property_uri,
    )

    log.update({
        "iata":        result.get("iata"),
        "airport_uri": result.get("airport_uri"),
        "raw_answer":  result.get("raw_value"),
        "failure_type": result.get("failure_type"),
    })

    if result["success"]:
        raw    = result["raw_value"]
        answer = format_answer(question, raw, lang)
        log["final_answer"] = answer
        log["failure_type"] = "success"
        print(f"\nRaw answer : {raw}")
        print(f"Final answer: {answer}")
    else:
        print(f"Cross-KG resolution failed: {result['failure_type']}")

# ─────────────────────────────────────────────────────────────────────────────
# BRANCH E — SINGLE KG3 (UNIVERSITY)
# ─────────────────────────────────────────────────────────────────────────────
elif query_type == "single_kg3":

    # Step 1: extract property phrase (entity already resolved by router)
    entities = extract_university_entities(question, lang, routing["entity"])
    print(f"[2] Entities : {entities}")

    if not validate_university_extraction(entities):
        print("University extraction failed.")
        log["failure_type"] = "extraction_failure"
        with open("logs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")
        exit()

    # Step 2: map property and resolve entity URI
    lexicon_path = get_lexicon("university")
    lexicon      = load_lexicon(lexicon_path)

    property_uri, mapping_layer, property2_uri = map_property_cascade(
        entities["property"], lexicon, lexicon_path
    )
    entity_uri = map_university_entity(entities["entity"]) if entities["entity"] else None
    # Disambiguate "part of" style phrases: memberOf (person -> dept)
    # vs subOrganizationOf (dept -> university) resolve to the same text
    # but need different properties depending on the entity's real type.
    if entity_uri and property_uri in ("memberOf", "subOrganizationOf"):
        entity_type = get_university_entity_type(entity_uri)
        if entity_type == "Department" and property_uri == "memberOf":
            property_uri = "subOrganizationOf"
            print(f"[disambiguation] Department entity — corrected memberOf → subOrganizationOf")
        elif entity_type != "Department" and property_uri == "subOrganizationOf":
            property_uri = "memberOf"
            print(f"[disambiguation] Non-department entity — corrected subOrganizationOf → memberOf")
    log.update({
        "entity_uri":    entity_uri,
        "property_uri":  property_uri,
        "property2_uri": property2_uri,
        "mapping_layer": mapping_layer,
    })

    if not entity_uri or not property_uri:
        print("Mapping failed.")
        log["failure_type"] = "mapping_failure"
        with open("logs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")
        exit()

    # Step 3: generate
    BASE = get_base_uri("university")
    full_prop_uri  = BASE + property_uri
    full_prop2_uri = (BASE + property2_uri) if property2_uri else None

    sparql_query = inject_and_generate(
        entity_uri, full_prop_uri, question,
        strategy=condition, property2_uri=full_prop2_uri
    )
    log["sparql"] = sparql_query
    print(f"\nGenerated SPARQL:\n{sparql_query}")

    # Step 4: validate
    is_valid = (
        validate_sparql(sparql_query)
        and sparql_query.strip().startswith("SELECT")
        and "PREFIX" not in sparql_query
        and full_prop_uri in sparql_query
    )
    log["sparql_valid"] = is_valid

    # Step 5: execute against KG3 (multiple=True — university properties
    # like teacherOf/takesCourse are naturally one-to-many)
    if is_valid:
        raw = execute_sparql(sparql_query, endpoint=get_endpoint("university"), multiple=True)
        log["raw_answer"] = raw
        print(f"\nRaw answer: {raw}")
        if raw:
            answer = format_answer_list(question, raw, lang)
            log["final_answer"]  = answer
            log["failure_type"]  = "success"
            print(f"Final answer: {answer}")
        else:
            log["failure_type"] = "execution_failure"
    else:
        print("Invalid SPARQL.")
        log["failure_type"] = "generation_failure"

# ─────────────────────────────────────────────────────────────────────────────
# BRANCH E — TEMPLATE (filter / ranking / comparison / count / group-aggregate)
# ─────────────────────────────────────────────────────────────────────────────
elif query_type == "template":

    template_name = routing["template"]
    print(f"[2] Template : {template_name}")

    # Delegate entirely to the template resolver.
    # It handles: param extraction (LLM), SPARQL building, execution,
    # and natural-language formatting — all in one call.
    result = resolve_template(question, template_name, lang)

    # Enrich the shared log with template-specific fields so the
    # evaluation JSONL stays consistent across all branches.
    log.update({
        "template_params": result.get("params"),
        "sparql":          result.get("sparql"),
        "raw_answer":      result.get("raw_data"),
        "final_answer":    result.get("final_answer"),
        "failure_type":    result.get("failure_type"),   # success | param_extraction_failure
                                                          # | sparql_build_failure | execution_failure
        "sparql_valid":    result.get("success", False),
    })

    if result["success"]:
        print(f"\nFinal answer: {result['final_answer']}")
    else:
        # Print a meaningful diagnostic rather than a silent failure.
        print(f"Template resolution failed: {result['failure_type']}")
        print(f"  Template   : {template_name}")
        print(f"  Params     : {result.get('params')}")
        print(f"  SPARQL     : {result.get('sparql')}")

# ─────────────────────────────────────────────────────────────────────────────
# BRANCH F — OPEN KG
# ─────────────────────────────────────────────────────────────────────────────
elif query_type == "open_kg":

    print("[Branch F] open_kg — LLM-generated SPARQL from schema")

    from kg_registry import get_open_kg_schema
    schema = get_open_kg_schema()

    # Step 1: generate SPARQL and detect target endpoint from namespace markers.
    # generate_open_kg_sparql returns a tuple (sparql, endpoint).
    # The endpoint is determined by inspecting which ontology namespace
    # appears in the generated query:
    #   flight_ontology   → KG1 (flights endpoint)
    #   airport_ontology  → KG2 (airports endpoint)
    # This replaces the previous blind sequential fallback (try KG1, then KG2
    # if empty), which was unreliable and architecturally incorrect.
    sparql_query, endpoint = generate_open_kg_sparql(question, lang, schema)

    log["sparql"] = sparql_query
    log["kg"]     = "kg1" if "flights" in endpoint else "kg2"
    print(f"\nGenerated SPARQL:\n{sparql_query}")
    print(f"[Branch F] Target endpoint: {endpoint}")

    if not sparql_query or not sparql_query.strip().startswith("SELECT"):
        log["failure_type"] = "generation_failure"
    else:
        # Step 2: validate syntax
        is_valid = validate_sparql(sparql_query)
        log["sparql_valid"] = is_valid

        if not is_valid:
            log["failure_type"] = "generation_failure"
        else:
            # Step 3: execute against the detected endpoint only — no fallback.
            # If the query returns empty, the failure is logged as
            # execution_failure rather than silently retrying the wrong KG.
            raw = execute_sparql(sparql_query, endpoint=endpoint, multiple=True)
            log["raw_answer"] = raw
            print(f"\nRaw answer: {raw}")

            if raw:
                answer = format_answer_list(question, raw, lang)
                log["final_answer"] = answer
                log["failure_type"] = "success"
                print(f"Final answer: {answer}")

# ─────────────────────────────────────────────────────────────────────────────
# BRANCH — ASK_QUERY (yes/no questions about a known entity)
# ─────────────────────────────────────────────────────────────────────────────
elif query_type == "ask_query":

    from template_resolver import resolve_ask_query
    result = resolve_ask_query(question, routing, lang)

    log.update({
        "entity_uri":   result.get("entity_uri"),
        "property_uri": result.get("property_uri"),
        "sparql":       result.get("sparql"),
        "raw_answer":   result.get("raw_answer"),
        "final_answer": result.get("final_answer"),
        "failure_type": result.get("failure_type"),
        "sparql_valid": result.get("success", False),
    })

    if result["success"]:
        print(f"\nRaw answer  : {result['raw_answer']}")
        print(f"Final answer: {result['final_answer']}")
    else:
        print(f"ASK resolution failed: {result['failure_type']}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: SAVE LOG
# ─────────────────────────────────────────────────────────────────────────────
with open("logs.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(log, ensure_ascii=False) + "\n")

print("\nLogged.")