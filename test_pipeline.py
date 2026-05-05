import json
from pipeline.language import detect_language
from pipeline.extractor import extract_entities, validate_extraction, is_flight_question
from pipeline.mapper import load_lexicon, map_property, map_property_with_embeddings, map_flight
from pipeline.generator import inject_and_generate
from pipeline.executor import validate_sparql, execute_sparql, format_answer

TEST_CASES_LEXICON = [
    # English — properties not yet covered in your original suite
    {"question": "What is the runway of flight TK1887?",           "condition": "zero-shot", "expected_property": "runway",           "expected_layer": "lexicon"},
    {"question": "What is the terminal of flight OS235?",          "condition": "few-shot",  "expected_property": "terminal",          "expected_layer": "lexicon"},
    {"question": "What is the callsign of flight BR62?",           "condition": "cot",       "expected_property": "callsign",          "expected_layer": "lexicon"},
    {"question": "What is the weather condition of flight AF1739?","condition": "zero-shot", "expected_property": "weather",           "expected_layer": "lexicon"},
    {"question": "What is the departure country of flight OS235?", "condition": "few-shot",  "expected_property": "departure country",  "expected_layer": "lexicon"},
    {"question": "What is the arrival country of flight TK1887?",  "condition": "cot",       "expected_property": "arrival country",    "expected_layer": "lexicon"},
    {"question": "What is the route of flight BR62?",              "condition": "zero-shot", "expected_property": "route",             "expected_layer": "lexicon"},

    # French — key properties in French surface forms
    {"question": "Quelle est la météo du vol TK1887?",             "condition": "few-shot",  "expected_property": "météo",             "expected_layer": "lexicon"},
    {"question": "Quel est le terminal du vol OS235?",             "condition": "zero-shot", "expected_property": "terminal",          "expected_layer": "lexicon"},
    {"question": "Quel est le pays de départ du vol AF1739?",      "condition": "cot",       "expected_property": "pays de départ",    "expected_layer": "lexicon"},
    {"question": "Quel est l'avion utilisé pour le vol BR62?",     "condition": "few-shot",  "expected_property": "avion",             "expected_layer": "lexicon"},

    # Arabic — key properties in Arabic surface forms
    {"question": "ما هو مسار الرحلة OS235؟",                      "condition": "cot",       "expected_property": "المسار",            "expected_layer": "lexicon"},
    {"question": "ما هي حالة الطقس في الرحلة TK1887؟",           "condition": "zero-shot", "expected_property": "الطقس",             "expected_layer": "lexicon"},
    {"question": "ما هي الصالة الخاصة بالرحلة AF1739؟",          "condition": "few-shot",  "expected_property": "الصالة",            "expected_layer": "lexicon"},
    {"question": "ما هو بلد المغادرة للرحلة BR62؟",               "condition": "cot",       "expected_property": "بلد المغادرة",      "expected_layer": "lexicon"},
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