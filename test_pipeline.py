import json
from pipeline.language import detect_language
from pipeline.extractor import extract_entities, validate_extraction, is_flight_question
from pipeline.mapper import load_lexicon, map_property, map_property_with_embeddings, map_flight
from pipeline.generator import inject_and_generate
from pipeline.executor import validate_sparql, execute_sparql, format_answer

TEST_CASES_LEXICON = [
    # English — properties not yet covered in your original suite
      # [CODE] properties — valid pipeline output, but answer is an internal ID
    {"question": "Who is the pilot of flight TK1887?",             "condition": "zero-shot", "expected_property": "hasPilot",          "expected_layer": "lexicon"},
    {"question": "Qui est l'hôtesse de l'air du vol OS235?",       "condition": "few-shot",  "expected_property": "hasFlightAttendant","expected_layer": "lexicon"},
    {"question": "من هو مضيف الرحلة AF1739؟",                     "condition": "cot",       "expected_property": "hasFlightAttendant","expected_layer": "lexicon"},

    # Out-of-scope questions — system should log extraction_failure
    {"question": "What is the capital of France?",                 "condition": "zero-shot", "expected_property": None,                "expected_layer": "rejected"},
    {"question": "Book me a ticket to Vienna.",                    "condition": "few-shot",  "expected_property": None,                "expected_layer": "rejected"},
    {"question": "احجز لي تذكرة إلى باريس.",                      "condition": "zero-shot", "expected_property": None,                "expected_layer": "rejected"},

    # Mixed-language stress tests (flight number in one language, question in another)
    {"question": "Where is flight TK1887 going? وين تروح؟",       "condition": "cot",       "expected_property": "hasDestinationCity", "expected_layer": "lexicon"},
]
lexicon = load_lexicon()
passed = 0
failed = 0

for i, case in enumerate(TEST_CASES_LEXICON):
    question  = case["question"]
    condition = case["condition"]
    expected  = case["expected_property"]

    print(f"\n── Test {i+1} ──────────────────────────────")
    print(f"Q: {question}")

    lang     = detect_language(question)
    entities = extract_entities(question, lang)

    if not validate_extraction(entities) or not is_flight_question(entities):
        print(f"❌ FAILED — extraction failed: {entities}")
        failed += 1
        continue

    extracted_property = entities.get("property", "")
    property_uri = map_property(extracted_property, lexicon)
    if property_uri is None:
        property_uri = map_property_with_embeddings(extracted_property, lexicon)

    flight_uri = map_flight(entities["entity"])

    if not flight_uri or not property_uri:
        print(f"❌ FAILED — mapping failed | property='{extracted_property}' uri={property_uri} flight={flight_uri}")
        failed += 1
        continue

    sparql = inject_and_generate(flight_uri, property_uri, question, strategy=condition)
    is_valid = validate_sparql(sparql) and sparql.strip().startswith("SELECT") and property_uri in sparql

    raw = execute_sparql(sparql) if is_valid else None

    if raw:
        print(f"✅ PASSED | property='{extracted_property}' | answer='{raw}'")
        passed += 1
    else:
        print(f"❌ FAILED | property='{extracted_property}' | sparql_valid={is_valid} | raw={raw}")
        failed += 1

print(f"\n══ RESULTS: {passed} passed, {failed} failed out of {len(TEST_CASES_LEXICON)} tests ══")