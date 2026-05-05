import json

# Pipeline components
from pipeline.language import detect_language
from pipeline.extractor import (
    extract_entities,
    validate_extraction,
    is_flight_question
)
from pipeline.mapper import (
    load_lexicon,
    map_property,
    map_property_with_embeddings,
    map_flight
)
from pipeline.generator import inject_and_generate
from pipeline.executor import (
    validate_sparql,
    execute_sparql,
    format_answer
)

# ── TEST CONFIGURATION ────────────────────────────────────
# You manually change these for experiments
question =  "من هو طيار الرحلة BR62؟"
condition = "cot"  # options: "zero-shot", "few-shot", "cot"

# ── STEP 0: LANGUAGE DETECTION ────────────────────────────
# Why: the system must support 3 languages → behavior depends on language
lang = detect_language(question)

# ── STEP 1: ENTITY EXTRACTION ─────────────────────────────
# Why: the system must separate:
# - the flight (entity)
# - what is being asked (property)
entities = extract_entities(question, lang)

# ── VALIDATION: EXTRACTION ────────────────────────────────
# Why: prevent garbage from propagating into the pipeline
if not validate_extraction(entities) or not is_flight_question(entities):
    print("Out of scope or extraction failed.")

    log = {
        "condition": condition,
        "language": lang,
        "question": question,
        "entities": entities,
        "failure_type": "extraction_failure",
        "extraction_failure_reason": entities.get("reason", "validation_failed"),
        "final_answer": "out_of_scope"
    }

    with open("logs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")

    print("Logged.")
    exit()

# ── STEP 2: MAPPING ───────────────────────────────────────
# Why: LLMs cannot be trusted to guess URIs → must resolve BEFORE generation

lexicon = load_lexicon()

# First attempt: exact match (high precision)
property_uri = map_property(entities["property"], lexicon)

# Fallback: embeddings (semantic match)
if property_uri is None:
    property_uri = map_property_with_embeddings(
        entities["property"], lexicon
    )

# Flight mapping via KG lookup
flight_uri = map_flight(entities["entity"])

# ── INITIAL LOG OBJECT ────────────────────────────────────
log = {
    "condition": condition,
    "language": lang,
    "question": question,
    "entities": entities,
    "flight_uri": flight_uri,
    "property_uri": property_uri,
    "sparql": None,
    "sparql_valid": False,
    "raw_answer": None,
    "final_answer": None,
    "failure_type": None
}

# ── VALIDATION: MAPPING ───────────────────────────────────
# Why: if mapping fails, generation becomes meaningless
if not flight_uri or not property_uri:
    print("Mapping failed.")

    log["failure_type"] = "mapping_failure"

    with open("logs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")

    print("Logged.")
    exit()

# ── STEP 3: SPARQL GENERATION ─────────────────────────────
# Why: LLM is used ONLY for structure, not for knowledge
sparql_query = inject_and_generate(
    flight_uri,
    property_uri,
    question,
    strategy=condition
)

log["sparql"] = sparql_query

print("\nGenerated SPARQL:")
print(sparql_query)

# ── STEP 4: VALIDATION (CRITICAL FOR THESIS) ──────────────
# Why: prove that generation is controlled and correct

# 1. Syntax validation
is_valid = validate_sparql(sparql_query)

# 2. Enforce SELECT-only queries
if not sparql_query.strip().startswith("SELECT"):
    is_valid = False

# 3. Disallow PREFIX (constraint from design)
if "PREFIX" in sparql_query:
    is_valid = False

# 4. CRITICAL: enforce correct property usage
# Why: proves knowledge injection is respected
if property_uri not in sparql_query:
    is_valid = False

log["sparql_valid"] = is_valid

# ── STEP 5: EXECUTION ─────────────────────────────────────
if is_valid:
    raw = execute_sparql(sparql_query)
    log["raw_answer"] = raw

    print("\nRaw answer from KG:", raw)

    if raw:
        answer = format_answer(question, raw, lang)
        log["final_answer"] = answer
        log["failure_type"] = "success"

        print("Final answer:", answer)
    else:
        # Query valid but no result
        log["failure_type"] = "execution_failure"

else:
    print("Invalid SPARQL — logging failure")
    log["failure_type"] = "generation_failure"

# ── STEP 6: SAVE LOG ──────────────────────────────────────
# Why: logs are the basis of your 12 experiments evaluation
with open("logs.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(log, ensure_ascii=False) + "\n")

print("\nLogged.")