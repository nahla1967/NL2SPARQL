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
# Purpose: test the mapping layer IN ISOLATION, bypassing the extractor.
# Why: the LLM extractor normalises surface text before mapping runs,
#      making it impossible to trigger fuzzy/semantic tiers via full pipeline.
#      These tests feed raw unusual strings directly to the mapper to verify
#      each tier activates correctly on its own.
# ══════════════════════════════════════════════════════════════════════════════

MAPPING_TESTS = [
    # (input_text,                  expected_property,     note)
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
# PART 2 — FULL PIPELINE INTEGRATION TESTS
# Purpose: test the complete pipeline end-to-end across all combinations of:
#   - language    : English, French, Arabic
#   - strategy    : zero-shot, few-shot, cot
#   - mapping tier: exact, fuzzy, semantic (as resolved by full pipeline)
#   - rejection   : out-of-scope questions that must never reach mapping
# ══════════════════════════════════════════════════════════════════════════════

TEST_CASES = [

    # ── TIER 1 : EXACT LEXICON MATCH ─────────────────────────────────────────
    # Standard canonical phrasings — LLM extractor maps them directly.
    {
        "question":          "Where does flight OS235 depart from?",
        "condition":         "zero-shot",
        "expected_property": "hasOriginCity",
        "expected_tier":     "exact",
        "language":          "en"
    },
    {
        "question":          "Quelle est la ville de départ du vol AF1739?",
        "condition":         "few-shot",
        "expected_property": "hasOriginCity",
        "expected_tier":     "exact",
        "language":          "fr"
    },
    {
        "question":          "ما هو مسار رحلة TK1887؟",
        "condition":         "cot",
        "expected_property": "hasRoute",
        "expected_tier":     "exact",
        "language":          "ar"
    },

    # ── TIER 2 : FUZZY MATCH ─────────────────────────────────────────────────
    # Near-canonical phrasings — LLM extractor may or may not normalise.
    # If extractor normalises, exact tier catches it (logged as ⚠️ not ❌).
    {
        "question":          "What's the departure town of flight BR62?",
        "condition":         "zero-shot",
        "expected_property": "hasOriginCity",
        "expected_tier":     "fuzzy",
        "language":          "en"
    },
    {
        "question":          "Vers quelle destination part le vol OS235?",
        "condition":         "cot",
        "expected_property": "hasDestinationCity",
        "expected_tier":     "fuzzy",
        "language":          "fr"
    },
    {
        "question":          "ما هي مدينة هبوط رحلة AF1739؟",
        "condition":         "few-shot",
        "expected_property": "hasDestinationCity",
        "expected_tier":     "fuzzy",
        "language":          "ar"
    },

    # ── TIER 3 : SEMANTIC EMBEDDINGS ─────────────────────────────────────────
    # Genuine paraphrases — designed to require embeddings.
    # LLaMA3 extractor normalises most of these (logged as ⚠️).
    # See PART 1 for isolated semantic tier verification.
    {
        "question":          "Which airport does flight TK1887 take off from?",
        "condition":         "few-shot",
        "expected_property": "hasOriginCity",
        "expected_tier":     "semantic",
        "language":          "en"
    },
    {
        "question":          "Quelle société exploite le vol BR62?",
        "condition":         "zero-shot",
        "expected_property": "hasAirline",
        "expected_tier":     "semantic",
        "language":          "fr"
    },
    {
        "question":          "ما هي الجهة المشغّلة لرحلة OS235؟",
        "condition":         "cot",
        "expected_property": "hasAirline",
        "expected_tier":     "semantic",
        "language":          "ar"
    },

    # ── REJECTION : OUT-OF-SCOPE ──────────────────────────────────────────────
    # Must be stopped at extraction. Reaching mapping = extractor bug.
    {
        "question":          "What is the capital of France?",
        "condition":         "zero-shot",
        "expected_property": None,
        "expected_tier":     "rejected",
        "language":          "en"
    },

    # ── NON-LEXICON : SEMANTIC STRESS TESTS ───────────────────────────────────
    # These questions use phrasings that do not exist in the lexicon at all.
    # The extractor will return the unusual phrase verbatim.
    # Exact and fuzzy tiers will both miss.
    # The embedding model must carry the resolution alone.
    # Purpose: stress-test the semantic tier in isolation across all 3 languages.

    # English
    {
        "question":          "What tarmac does flight OS235 use?",
        "condition":         "zero-shot",
        "expected_property": "hasRunway",
        "expected_tier":     "semantic",
        "language":          "en"
    },
    {
        "question":          "Which firm is operating flight BR62?",
        "condition":         "few-shot",
        "expected_property": "hasAirline",
        "expected_tier":     "semantic",
        "language":          "en"
    },
    {
        "question":          "What are the sky conditions for flight TK1887?",
        "condition":         "cot",
        "expected_property": "hasWeatherCondition",
        "expected_tier":     "semantic",
        "language":          "en"
    },
    {
        "question":          "Which concourse is used by flight AF1739?",
        "condition":         "zero-shot",
        "expected_property": "hasTerminal",
        "expected_tier":     "semantic",
        "language":          "en"
    },

    # French
    {
        "question":          "Quelle est la bande d'atterrissage du vol OS235?",
        "condition":         "few-shot",
        "expected_property": "hasRunway",
        "expected_tier":     "semantic",
        "language":          "fr"
    },
    {
        "question":          "Vers quel pays se dirige le vol TK1887?",
        "condition":         "cot",
        "expected_property": "hasDestinationCountry",
        "expected_tier":     "semantic",
        "language":          "fr"
    },
    {
        "question":          "Quel aéronef est utilisé pour le vol BR62?",
        "condition":         "zero-shot",
        "expected_property": "hasAircraft",
        "expected_tier":     "semantic",
        "language":          "fr"
    },

    # Arabic
    {
        "question":          "من يتولى إدارة رحلة AF1739؟",
        "condition":         "few-shot",
        "expected_property": "hasPilot",
        "expected_tier":     "semantic",
        "language":          "ar"
    },
    {
        "question":          "ما الطراز المستخدم في رحلة OS235؟",
        "condition":         "cot",
        "expected_property": "hasAircraft",
        "expected_tier":     "semantic",
        "language":          "ar"
    },
    {
        "question":          "ما هي أحوال الجو خلال رحلة BR62؟",
        "condition":         "zero-shot",
        "expected_property": "hasWeatherCondition",
        "expected_tier":     "semantic",
        "language":          "ar"
    },
]

print("\n\n══ PART 2 — FULL PIPELINE INTEGRATION TESTS ════════════════════════════")

passed = 0
failed = 0

for i, case in enumerate(TEST_CASES):
    question          = case["question"]
    condition         = case["condition"]
    expected_property = case["expected_property"]
    expected_tier     = case["expected_tier"]

    print(f"\n── Test {i+1} [{case['language'].upper()} | {condition} | expected tier: {expected_tier}]")
    print(f"   Q: {question}")

    # Step 0: language detection
    lang = detect_language(question)

    # Step 1: extraction
    entities = extract_entities(question, lang)

    if not validate_extraction(entities) or not is_flight_question(entities):
        if expected_tier == "rejected":
            print(f"   ✅ PASSED — correctly rejected at extraction")
            passed += 1
        else:
            print(f"   ❌ FAILED — unexpected extraction failure: {entities}")
            failed += 1
        continue

    if expected_tier == "rejected":
        print(f"   ❌ FAILED — should have been rejected but extraction succeeded: {entities}")
        failed += 1
        continue

    # Step 2: three-tier mapping
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
        print(f"   ❌ FAILED — mapping failed | property='{entities.get('property')}' uri={property_uri} flight={flight_uri}")
        failed += 1
        continue

    # Step 3: SPARQL generation and execution
    sparql   = inject_and_generate(flight_uri, property_uri, question, strategy=condition)
    is_valid = (
        validate_sparql(sparql)
        and sparql.strip().startswith("SELECT")
        and "PREFIX" not in sparql
        and property_uri in sparql
    )
    raw = execute_sparql(sparql) if is_valid else None

    # Result
    tier_ok     = (mapping_layer == expected_tier)
    property_ok = (property_uri.endswith(expected_property) if expected_property else False)
    answer_ok   = (raw is not None)

    status    = "✅ PASSED" if (property_ok and answer_ok) else "❌ FAILED"
    tier_flag = "✅" if tier_ok else "⚠️ "

    if property_ok and answer_ok:
        passed += 1
    else:
        failed += 1

    print(f"   {status}")
    print(f"   property  : {property_uri.split('#')[-1]} (expected {expected_property})")
    print(f"   tier      : {tier_flag} {mapping_layer} (expected {expected_tier})")
    print(f"   answer    : {raw}")

print(f"\n══ RESULTS: {passed} passed, {failed} failed out of {len(TEST_CASES)} pipeline tests ══")