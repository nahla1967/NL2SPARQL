import json
from pipeline.language   import detect_language
from pipeline.extractor  import extract_entities, validate_extraction, is_flight_question
from pipeline.mapper     import load_lexicon, map_property_cascade, map_flight
from pipeline.generator  import inject_and_generate
from pipeline.executor   import validate_sparql, execute_sparql, format_answer

# ══════════════════════════════════════════════════════════════════════════════
# TEST SUITE — 20 REALISTIC USER QUESTIONS
# ══════════════════════════════════════════════════════════════════════════════
#
# Coverage:
#   Languages   : English (8), French (6), Arabic (6)
#   Strategies  : zero-shot (7), few-shot (7), cot (6)
#   Error types : typos (3), out-of-scope (2), missing flight number (2),
#                 unknown flight (1), mixed language (2), clean (10)
#
# Expected outcome is documented per case so results are self-explanatory.
# ══════════════════════════════════════════════════════════════════════════════

TEST_CASES = [

    # ── TWO-HOP — ENGLISH ─────────────────────────────────────────────────────
    {
        "id": 1,
        "question":  "What is the aircraft type of flight OS295?",
        "condition": "zero-shot",
        "expected":  "success",
        "note":      "EN — two-hop: hasAircraft → type"
    },
    {
        "id": 2,
        "question":  "What is the aircraft type of flight OS295?",
        "condition": "few-shot",
        "expected":  "success",
        "note":      "EN — two-hop few-shot: hasAircraft → type"
    },
    {
        "id": 3,
        "question":  "What is the aircraft type of flight OS295?",
        "condition": "cot",
        "expected":  "success",
        "note":      "EN — two-hop CoT: hasAircraft → type"
    },
    {
        "id": 4,
        "question":  "What is the ground speed of flight OS295?",
        "condition": "zero-shot",
        "expected":  "success",
        "note":      "EN — two-hop: hasFlightEvent → gspeed"
    },
    {
        "id": 5,
        "question":  "What is the vertical speed of flight OS295?",
        "condition": "zero-shot",
        "expected":  "success",
        "note":      "EN — two-hop: hasFlightEvent → vspeed"
    },
    {
        "id": 6,
        "question":  "What is the registration of the aircraft of flight OS295?",
        "condition": "zero-shot",
        "expected":  "success",
        "note":      "EN — two-hop: hasAircraft → reg"
    },
    {
        "id": 7,
        "question":  "What is the transponder code of flight OS295?",
        "condition": "zero-shot",
        "expected":  "success",
        "note":      "EN — two-hop: hasAircraft → TransponderCode"
    },
    {
        "id": 8,
        "question":  "What livery does the airline of flight OS295 use?",
        "condition": "zero-shot",
        "expected":  "success",
        "note":      "EN — two-hop: hasAirline → painted_as"
    },

    # ── TWO-HOP — FRENCH ──────────────────────────────────────────────────────
    {
        "id": 9,
        "question":  "Quel est le type d'appareil du vol OS295?",
        "condition": "zero-shot",
        "expected":  "success",
        "note":      "FR — two-hop: hasAircraft → type"
    },
    {
        "id": 10,
        "question":  "Quelle est la vitesse au sol du vol OS295?",
        "condition": "few-shot",
        "expected":  "success",
        "note":      "FR — two-hop: hasFlightEvent → gspeed"
    },

    # ── TWO-HOP — ARABIC ──────────────────────────────────────────────────────
    {
        "id": 11,
        "question":  "ما نوع طائرة الرحلة OS295؟",
        "condition": "zero-shot",
        "expected":  "success",
        "note":      "AR — two-hop: hasAircraft → type"
    },
    {
        "id": 12,
        "question":  "ما هي سرعة الرحلة OS295؟",
        "condition": "cot",
        "expected":  "success",
        "note":      "AR — two-hop: hasFlightEvent → gspeed"
    },

    # ── SINGLE-HOP — REGRESSION CHECK ─────────────────────────────────────────
    {
        "id": 13,
        "question":  "What is the departure city of flight OS295?",
        "condition": "zero-shot",
        "expected":  "success",
        "note":      "EN — single-hop regression: hasOriginCity"
    },
    {
        "id": 14,
        "question":  "What is the airline of flight OS295?",
        "condition": "few-shot",
        "expected":  "success",
        "note":      "EN — single-hop regression: hasAirline"
    },
    {
        "id": 15,
        "question":  "Quelle est la ville de départ du vol OS295?",
        "condition": "zero-shot",
        "expected":  "success",
        "note":      "FR — single-hop regression: hasOriginCity"
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_test(case, lexicon):
    question  = case["question"]
    condition = case["condition"]
    expected  = case["expected"]

    result = {
        "id":            case["id"],
        "question":      question,
        "condition":     condition,
        "expected":      expected,
        "note":          case["note"],
        "language":      None,
        "entities":      None,
        "mapping_layer": None,
        "property_uri":  None,
        "flight_uri":    None,
        "sparql":        None,
        "sparql_valid":  False,
        "raw_answer":    None,
        "final_answer":  None,
        "failure_type":  None,
        "verdict":       None,   # PASS / FAIL / EXPECTED_FAIL
    }

    # ── STEP 0: language detection ─────────────────────────────────────────────
    lang = detect_language(question)
    result["language"] = lang

    # ── STEP 1: entity extraction ──────────────────────────────────────────────
    entities = extract_entities(question, lang)
    result["entities"] = entities

    if not validate_extraction(entities) or not is_flight_question(entities):
        result["failure_type"] = "extraction_failure"
        result["verdict"] = "PASS" if expected == "extraction_failure" else "FAIL"
        return result

    # ── STEP 2: mapping ────────────────────────────────────────────────────────
    property_uri, mapping_layer , property2_uri = map_property_cascade(entities["property"], lexicon)
    flight_uri = map_flight(entities["entity"])

    result["property_uri"]  = property_uri
    result["flight_uri"]    = flight_uri
    result["mapping_layer"] = mapping_layer

    if not flight_uri or not property_uri:
        result["failure_type"] = "mapping_failure"
        result["verdict"] = "PASS" if expected == "mapping_failure" else "FAIL"
        return result

    # ── STEP 3: SPARQL generation ──────────────────────────────────────────────
    sparql = inject_and_generate(flight_uri, property_uri, question, strategy=condition , property2_uri=property2_uri)
    result["sparql"] = sparql

    # ── STEP 4: validation ─────────────────────────────────────────────────────
    is_valid = (
        validate_sparql(sparql)
        and sparql.strip().startswith("SELECT")
        and "PREFIX" not in sparql
        and property_uri in sparql
    )
    result["sparql_valid"] = is_valid

    if not is_valid:
        result["failure_type"] = "generation_failure"
        result["verdict"] = "PASS" if expected == "generation_failure" else "FAIL"
        return result

    # ── STEP 5: execution ──────────────────────────────────────────────────────
    raw = execute_sparql(sparql)
    result["raw_answer"] = raw

    if raw:
        answer = format_answer(question, raw, lang)
        result["final_answer"]  = answer
        result["failure_type"]  = "success"
        result["verdict"]       = "PASS" if expected == "success" else "FAIL"
    else:
        result["failure_type"] = "execution_failure"
        result["verdict"] = "PASS" if expected == "execution_failure" else "FAIL"

    return result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    lexicon = load_lexicon()

    passed       = 0
    failed       = 0
    all_results  = []

    print("=" * 72)
    print("  NL2SPARQL — FULL USER STRESS TEST  (20 questions)")
    print("=" * 72)

    for case in TEST_CASES:
        print(f"\n── Test {case['id']:02d} | {case['condition']:9} | {case['note']}")
        print(f"   Q  : {case['question']}")

        result = run_test(case, lexicon)
        all_results.append(result)

        lang    = result["language"]
        tier    = result["mapping_layer"] or "—"
        outcome = result["failure_type"]  or "—"
        verdict = result["verdict"]

        prop = "—"
        if result["property_uri"]:
            prop = result["property_uri"].split("#")[-1]

        if verdict == "PASS" and outcome == "success":
            print(f"   ✅ PASS  | lang={lang} tier={tier} prop={prop}")
            print(f"   ↳ {result['final_answer']}")
            passed += 1

        elif verdict == "PASS":
            # expected failure — correctly rejected
            print(f"   ✅ PASS  | correctly rejected → {outcome}")
            passed += 1

        else:
            print(f"   ❌ FAIL  | expected={case['expected']} got={outcome}")
            print(f"            | lang={lang} tier={tier} prop={prop}")
            if result["sparql"]:
                print(f"            | SPARQL valid={result['sparql_valid']}")
            failed += 1

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    total = len(TEST_CASES)
    print("\n" + "=" * 72)
    print(f"  RESULTS : {passed}/{total} passed   {failed}/{total} failed")

    # breakdown by language
    for lg in ["en", "fr", "ar"]:
        group   = [r for r in all_results if r["language"] == lg]
        p       = sum(1 for r in group if r["verdict"] == "PASS")
        print(f"  {lg.upper()}      : {p}/{len(group)} passed")

    # breakdown by tier
    print()
    for tier in ["pre-norm", "exact", "fuzzy", "semantic"]:
        group = [r for r in all_results if r["mapping_layer"] == tier]
        if group:
            p = sum(1 for r in group if r["verdict"] == "PASS")
            print(f"  Tier [{tier:8}] : {p}/{len(group)} passed")

    print("=" * 72)

    # ── SAVE RESULTS ──────────────────────────────────────────────────────────
    with open("test_results.jsonl", "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n  Results saved to test_results.jsonl")