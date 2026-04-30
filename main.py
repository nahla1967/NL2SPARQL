import json
from pipeline.language import detect_language
from pipeline.extractor import extract_entities, validate_extraction, is_flight_question
from pipeline.mapper import load_lexicon, map_property, map_property_with_embeddings, map_flight
from pipeline.generator import inject_and_generate
from pipeline.executor import validate_sparql, execute_sparql, format_answer

# ── TEST ─────────────────────────────────────────────────
question = "Where does flight OS235 depart from?"
condition = "zero-shot"  # change to: "few-shot" or "cot"

lang = detect_language(question)
entities = extract_entities(question, lang)

# Step 1 — validate extraction
if not validate_extraction(entities) or not is_flight_question(entities):
    print("Out of scope or extraction failed.")
    log = {
        "condition": condition,
        "language": lang,
        "question": question,
        "entities": entities,
        "final_answer": "out_of_scope"
    }
    with open("logs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")
    print("Logged.")
    exit()

# Step 2 — mapping
lexicon = load_lexicon()
property_uri = map_property(entities["property"], lexicon)
if property_uri is None:
    property_uri = map_property_with_embeddings(entities["property"], lexicon)

flight_uri = map_flight(entities["entity"])

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
    "final_answer": None
}

# Step 3 — SPARQL generation and execution
if flight_uri and property_uri:
    sparql_query = inject_and_generate(
        flight_uri, property_uri, question, strategy=condition
    )
    log["sparql"] = sparql_query
    log["sparql_valid"] = validate_sparql(sparql_query)
    print("\nGenerated SPARQL:")
    print(sparql_query)

    if log["sparql_valid"]:
        raw = execute_sparql(sparql_query)
        log["raw_answer"] = raw
        print("\nRaw answer from KG:", raw)
        if raw:
            answer = format_answer(question, raw, lang)
            log["final_answer"] = answer
            print("Final answer:", answer)
    else:
        print("Invalid SPARQL — logging failure")
else:
    print("ERROR: missing flight or property URI")

# Step 4 — save log
with open("logs.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(log, ensure_ascii=False) + "\n")
print("\nLogged.")