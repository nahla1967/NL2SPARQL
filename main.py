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
)
from pipeline.mapper import (
    load_lexicon,
    map_property_cascade,
    map_flight,
    map_airport,
)
from pipeline.generator import inject_and_generate
from pipeline.executor  import (
    validate_sparql,
    execute_sparql,
    format_answer,
)
from cross_kg_resolver import resolve_cross_kg
from template_resolver import resolve_template
from kg_registry import get_base_uri, get_endpoint, get_lexicon
from pipeline.generator import inject_and_generate, generate_open_kg_sparql
# ── TEST CONFIGURATION ────────────────────────────────────────────────────────
question  = "ما طول مدرج مطار VIE؟"
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
# BRANCH E — TEMPLATE (filter / ranking / comparison / count)
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

    # Step 1: generate SPARQL from schema
    sparql_query = generate_open_kg_sparql(question, lang, schema)
    log["sparql"] = sparql_query
    print(f"\nGenerated SPARQL:\n{sparql_query}")

    if not sparql_query or not sparql_query.strip().startswith("SELECT"):
        log["failure_type"] = "generation_failure"
    else:
        # Step 2: validate syntax
        is_valid = validate_sparql(sparql_query)
        log["sparql_valid"] = is_valid

        if not is_valid:
            log["failure_type"] = "generation_failure"
        else:
            # Step 3: try KG1 first, then KG2 if empty
            raw = execute_sparql(sparql_query, endpoint=get_endpoint("flights"))
            if not raw:
                raw = execute_sparql(sparql_query, endpoint=get_endpoint("airports"))

            log["raw_answer"] = raw
            print(f"\nRaw answer: {raw}")

            if raw:
                answer = format_answer(question, raw, lang)
                log["final_answer"] = answer
                log["failure_type"] = "success"
                print(f"Final answer: {answer}")
            else:
                log["failure_type"] = "execution_failure"
# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: SAVE LOG
# ─────────────────────────────────────────────────────────────────────────────
with open("logs.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(log, ensure_ascii=False) + "\n")

print("\nLogged.")