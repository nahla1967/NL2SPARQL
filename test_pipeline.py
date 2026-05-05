import json
from pipeline.language import detect_language
from pipeline.extractor import extract_entities, validate_extraction, is_flight_question
from pipeline.mapper import (
    load_lexicon,
    map_property,
    map_property_fuzzy,
    map_property_with_embeddings,
    map_flight
)
from pipeline.generator import inject_and_generate
from pipeline.executor import validate_sparql, execute_sparql, format_answer

lexicon = load_lexicon()

# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — MAPPING UNIT TESTS
# ══════════════════════════════════════════════════════════════════════════════

MAPPING_TESTS = [
    ("flying to",                  "hasDestinationCity",  "should hit exact"),
    ("piste d envol",              "hasRunway",           "should hit fuzzy"),
    ("tarmac strip",               "hasRunway",           "should hit semantic"),
    ("meteorological conditions",  "hasWeatherCondition", "should hit semantic"),
    ("boarding door",              "hasGate",             "should hit semantic"),
    ("من يتولى قيادة",             "hasPilot",            "should hit semantic"),
]

print("══ PART 1 — MAPPING UNIT TESTS ══════════════════════════════════════════")

map_passed = 0
map_failed = 0

for text, expected, note in MAPPING_TESTS:
    uri  = map_property(text, lexicon)
    tier = "exact"

    if uri is None:
        uri  = map_property_fuzzy(text, lexicon)
        tier = "fuzzy" if uri else None

    if uri is None:
        uri  = map_property_with_embeddings(text, lexicon)
        tier = "semantic" if uri else "none"

    prop  = uri if uri else "None"
    match = "✅" if (expected in prop) else "❌"

    print(f"  {match} [{tier:8}] '{text}' → {prop.split('#')[-1] if uri else 'None'} ({note})")

    if expected in prop:
        map_passed += 1
    else:
        map_failed += 1

print(f"\n  Results: {map_passed} passed, {map_failed} failed out of {len(MAPPING_TESTS)} mapping tests")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — STRESS TESTS — REAL USER BEHAVIOUR
# ══════════════════════════════════════════════════════════════════════════════

TEST_CASES = [
    {"question": "OS235 depart from where??",                   "condition": "zero-shot"},
    {"question": "quel est la companie du vol TK1887",          "condition": "few-shot"},
    {"question": "the wether of flight BR62 ?",                 "condition": "cot"},
    {"question": "AF1739 quel terminal",                        "condition": "zero-shot"},
    {"question": "من فضلك ما هي بوابة الرحلة OS235؟",         "condition": "few-shot"},
    {"question": "TK1887 runway",                               "condition": "cot"},
    {"question": "l avion de vol BR62 cest quoi",               "condition": "zero-shot"},
    {"question": "what is the arival time of flight AF1739",    "condition": "few-shot"},
    {"question": "OS235 وين تروح؟",                             "condition": "cot"},
    {"question": "give me the route for TK1887 please",         "condition": "zero-shot"},
]

print("\n\n══ PART 2 — STRESS TESTS — REAL USER BEHAVIOUR ═════════════════════════")

for i, case in enumerate(TEST_CASES):
    question  = case["question"]
    condition = case["condition"]

    print(f"\n── Test {i+1} | {condition}")
    print(f"   Q: {question}")

    lang     = detect_language(question)
    entities = extract_entities(question, lang)

    if not validate_extraction(entities) or not is_flight_question(entities):
        print(f"   ❌ REJECTED — {entities.get('reason', 'validation_failed')}")
        continue

    property_uri  = None
    mapping_layer = None

    property_uri = map_property(entities["property"], lexicon)
    if property_uri:
        mapping_layer = "exact"

    if property_uri is None:
        property_uri = map_property_fuzzy(entities["property"], lexicon)
        if property_uri:
            mapping_layer = "fuzzy"

    if property_uri is None:
        property_uri = map_property_with_embeddings(entities["property"], lexicon)
        if property_uri:
            mapping_layer = "semantic"

    flight_uri = map_flight(entities["entity"])

    if not flight_uri or not property_uri:
        print(f"   ❌ MAPPING FAILED — extracted='{entities.get('property')}' uri={property_uri}")
        continue

    sparql   = inject_and_generate(flight_uri, property_uri, question, strategy=condition)
    is_valid = (
        validate_sparql(sparql)
        and sparql.strip().startswith("SELECT")
        and "PREFIX" not in sparql
        and property_uri in sparql
    )
    raw = execute_sparql(sparql) if is_valid else None

    prop_name = property_uri.split('#')[-1] if property_uri else "None"

    if raw:
        print(f"   ✅ lang={lang} | tier={mapping_layer} | property={prop_name} | answer={raw}")
    else:
        print(f"   ⚠️  lang={lang} | tier={mapping_layer} | property={prop_name} | SPARQL valid={is_valid} | no answer")