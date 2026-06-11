"""
test_pipeline.py
----------------
Tests the three working branches of the NL2SPARQL multi-KG pipeline.

HOW TO RUN:
    1. Make sure Fuseki is running with BOTH datasets:
          http://localhost:3030/flights/sparql   (KG1)
          http://localhost:3030/airports/sparql  (KG2)
    2. Make sure Ollama is running:
          ollama serve
    3. From your project root (NL2SPARQL/):
          python test_pipeline.py

WHAT IT TESTS:
    Branch B — single_kg1  : 5 questions (EN/FR/AR) against KG1
    Branch C — single_kg2  : 5 questions (EN/FR/AR) against KG2
    Branch D — cross_kg    : 5 questions (EN/FR/AR) bridging KG1 → KG2

OUTPUT:
    Prints a result table per branch.
    Saves all results to test_results.jsonl.
    Prints a final summary with pass/fail counts.
"""

import json
import sys
import time

from pipeline.language  import detect_language
from router             import route
from pipeline.extractor import (
    extract_entities, validate_extraction, is_flight_question,
    extract_airport_entities, validate_airport_extraction,
)
from pipeline.mapper import (
    load_lexicon, map_property_cascade, map_flight, map_airport,
)
from pipeline.generator import inject_and_generate
from pipeline.executor  import (
    validate_sparql, execute_sparql, format_answer,
)
from cross_kg_resolver  import resolve_cross_kg
from kg_registry        import get_base_uri, get_endpoint, get_lexicon

# ── TEST CASES ────────────────────────────────────────────────────────────────

BRANCH_TESTS = [

# ─────────────────────────────────────────────────────
# KG1 — FLIGHT QUERIES (15 tests)
# Covers: exact match, multilingual, short form, fuzzy
# ─────────────────────────────────────────────────────

# English — exact
("What is the departure city of flight OS295?",          "single_kg1", "en-exact"),
("Which airline operates flight BR62?",                  "single_kg1", "en-exact"),
("What is the destination of flight FR707?",             "single_kg1", "en-exact"),
("What aircraft is used for flight KE567?",              "single_kg1", "en-exact"),
("What is the arrival time of flight OS631?",            "single_kg1", "en-exact"),

# French — exact
("Quelle est la ville de départ du vol OS52?",           "single_kg1", "fr-exact"),
("Quel est l'aéroport d'arrivée du vol TO4388?",         "single_kg1", "fr-exact"),
("Quelle compagnie opère le vol FR6889?",                "single_kg1", "fr-exact"),

# Arabic — exact
("ما هي مدينة المغادرة للرحلة OS295؟",                  "single_kg1", "ar-exact"),
("إلى أين تتجه الرحلة FR164؟",                          "single_kg1", "ar-exact"),
("من يشغّل الرحلة DE1866؟",                             "single_kg1", "ar-exact"),

# Short / degraded form
("departure city OS631",                                 "single_kg1", "en-short"),
("airline FR9005",                                       "single_kg1", "en-short"),
("destination of BR62",                                  "single_kg1", "en-short"),
("vol OS295 ville départ",                               "single_kg1", "fr-short"),


# ─────────────────────────────────────────────────────
# KG2 — AIRPORT QUERIES (15 tests)
# Covers: exact, IATA, multilingual, fuzzy/typo
# ─────────────────────────────────────────────────────

# English — exact entity + exact property
("What is the elevation of Vienna airport?",             "single_kg2", "en-exact"),
("What country is Munich airport in?",                   "single_kg2", "en-exact"),
("What city is London Heathrow located in?",             "single_kg2", "en-exact"),
("What type of airport is Frankfurt airport?",           "single_kg2", "en-exact"),
("What country is Paris Charles de Gaulle airport in?",  "single_kg2", "en-exact"),

# IATA code as entity
("What is the elevation of MUC?",                        "single_kg2", "en-iata"),
("What country is VIE in?",                              "single_kg2", "en-iata"),
("City of LHR airport",                                  "single_kg2", "en-iata"),

# French
("Quelle est l'élévation de l'aéroport de Munich?",     "single_kg2", "fr-exact"),
("Dans quelle ville se trouve l'aéroport de Munich?",   "single_kg2", "fr-exact"),
("Quel est le pays de l'aéroport de Vienne?",           "single_kg2", "fr-exact"),

# Arabic
("ما هو ارتفاع مطار فيينا؟",                            "single_kg2", "ar-exact"),
("في أي دولة يقع مطار ميونخ؟",                          "single_kg2", "ar-exact"),
("ما هو نوع مطار فرانكفورت؟",                           "single_kg2", "ar-exact"),

# Typo / fuzzy
("elevtion of viena airport",                            "single_kg2", "en-typo"),


# ─────────────────────────────────────────────────────
# CROSS KG — FLIGHT + AIRPORT (10 tests)
# KG1 gives IATA → KG2 answers airport property
# Covers: origin, destination, EN/FR/AR
# ─────────────────────────────────────────────────────

# English — destination airport property
("What country is the destination airport of flight OS295?",     "cross_kg", "en-dest"),
("What is the elevation of the arrival airport of flight BR62?", "cross_kg", "en-dest"),
("What city is the destination airport of flight FR707?",        "cross_kg", "en-dest"),
("What type of airport does flight KE567 land at?",              "cross_kg", "en-dest"),

# English — origin airport property
("What country is the departure airport of flight OS631?",       "cross_kg", "en-orig"),
("What is the elevation of the origin airport of flight OS52?",  "cross_kg", "en-orig"),

# French
("Dans quel pays se trouve l'aéroport de destination du vol OS295?",  "cross_kg", "fr-dest"),
("Quelle est l'élévation de l'aéroport de départ du vol BR62?",       "cross_kg", "fr-orig"),

# Arabic
("ما هي دولة مطار الوصول للرحلة FR707؟",                "cross_kg", "ar-dest"),
("ما هو ارتفاع مطار المغادرة للرحلة OS295؟",            "cross_kg", "ar-orig"),


# ─────────────────────────────────────────────────────
# EMBEDDING / SEMANTIC FALLBACK (5 tests)
# Property phrases not in lexicon — should reach tier 3
# ─────────────────────────────────────────────────────

("How high above sea level is Vienna airport?",          "single_kg2", "en-semantic"),
("What nation is Frankfurt airport located in?",         "single_kg2", "en-semantic"),
("Which town does Munich airport serve?",                "single_kg2", "en-semantic"),
("What kind of facility is London Heathrow?",            "single_kg2", "en-semantic"),
("What is the ground speed of flight OS295?",            "single_kg1", "en-semantic"),


# ─────────────────────────────────────────────────────
# EDGE CASES / ROBUSTNESS (5 tests)
# ─────────────────────────────────────────────────────

("Who won the FIFA World Cup?",                          "out_of_scope", "noise"),
("What is the weather in Paris?",                        "out_of_scope", "noise"),
("elevation",                                            "out_of_scope", "noise"),
("flight",                                               "out_of_scope", "noise"),
("OS295 VIE elevation country",                          "single_kg2",   "ambiguous"),

]

ALL_TESTS = [
    ("FULL PIPELINE TEST", BRANCH_TESTS),
]


ALL_TESTS = [
  #  ("BRANCH B — single_kg1", BRANCH_B_TESTS),
    ("BRANCH C — single_kg2", BRANCH_TESTS),
   # ("BRANCH D — cross_kg",   BRANCH_D_TESTS),
]

STRATEGY = "zero-shot"

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def ok(s):   return f"{GREEN}{s}{RESET}"
def err(s):  return f"{RED}{s}{RESET}"
def warn(s): return f"{YELLOW}{s}{RESET}"

# ── CORE RUNNER ───────────────────────────────────────────────────────────────

def run_single_test(question: str, expected_type: str) -> dict:
    result = {
        "question":      question,
        "expected_type": expected_type,
        "detected_lang": None,
        "query_type":    None,
        "routing_ok":    False,
        "entity":        None,
        "property_uri":  None,
        "mapping_layer": None,
        "sparql":        None,
        "sparql_valid":  False,
        "raw_answer":    None,
        "final_answer":  None,
        "failure_type":  "not_run",
        "duration_s":    0,
    }

    t0 = time.time()

    try:
        lang = detect_language(question)
        result["detected_lang"] = lang

        routing    = route(question)
        query_type = routing["query_type"]
        result["query_type"] = query_type
        result["entity"]     = routing["entity"]
        result["routing_ok"] = (query_type == expected_type)

        if query_type == "out_of_scope":
            if expected_type == "out_of_scope":
                result["failure_type"] = "success"
            else:
                result["failure_type"] = "out_of_scope"
            return result

        # ── BRANCH B ──────────────────────────────────────────────────────────
        if query_type == "single_kg1":
            entities = extract_entities(question, lang)
            if not validate_extraction(entities) or not is_flight_question(entities):
                result["failure_type"] = "extraction_failure"
                return result

            lexicon = load_lexicon(get_lexicon("flights"))
            prop_uri, layer, prop2 = map_property_cascade(
                entities["property"], lexicon, get_lexicon("flights")
            )
            flight_uri = map_flight(entities["entity"])
            result["property_uri"]  = prop_uri
            result["mapping_layer"] = layer

            if not flight_uri or not prop_uri:
                result["failure_type"] = "mapping_failure"
                return result

            BASE = get_base_uri("flights")
            full_prop_uri  = BASE + prop_uri
            full_prop2_uri = (BASE + prop2) if prop2 else None

            sparql = inject_and_generate(
                flight_uri, full_prop_uri, question,
                strategy=STRATEGY, property2_uri=full_prop2_uri
            )
            result["sparql"] = sparql

            is_valid = (
                validate_sparql(sparql)
                and sparql.strip().startswith("SELECT")
                and "PREFIX" not in sparql
                and full_prop_uri in sparql
            )
            if full_prop2_uri:
                if full_prop2_uri not in sparql:
                    is_valid = False
            result["sparql_valid"] = is_valid

            if is_valid:
                raw = execute_sparql(sparql, endpoint=get_endpoint("flights"))
                result["raw_answer"] = raw
                if raw:
                    result["final_answer"] = format_answer(question, raw, lang)
                    result["failure_type"] = "success"
                else:
                    result["failure_type"] = "execution_failure"
            else:
                result["failure_type"] = "generation_failure"

        # ── BRANCH C ──────────────────────────────────────────────────────────
       # ── BRANCH C ──────────────────────────────────────────────────────────
        elif query_type == "single_kg2":

            entities = extract_airport_entities(
                question,
                lang,
                routing["entity"]
            )

            print("\n[DEBUG] Question:", question)
            print("[DEBUG] Extracted entities:", entities)

            if not validate_airport_extraction(entities):
                print("[DEBUG] AIRPORT EXTRACTION FAILED")
                result["failure_type"] = "extraction_failure"
                return result

            lexicon_path = get_lexicon("airports")
            lexicon = load_lexicon(lexicon_path)

            prop_uri, layer, prop2 = map_property_cascade(
                entities["property"],
                lexicon,
                lexicon_path
            )

            print("[DEBUG] Property:", entities["property"])
            print("[DEBUG] Mapped property:", prop_uri)

            airport_uri = (
                map_airport(entities["entity"])
                if entities["entity"]
                else None
            )

            print("[DEBUG] Airport entity:", entities["entity"])
            print("[DEBUG] Airport URI:", airport_uri)

            result["property_uri"] = prop_uri
            result["mapping_layer"] = layer

            if not airport_uri or not prop_uri:
                print("[DEBUG] MAPPING FAILURE")
                print("[DEBUG] airport_uri =", airport_uri)
                print("[DEBUG] prop_uri    =", prop_uri)

                result["failure_type"] = "mapping_failure"
                return result

            BASE = get_base_uri("airports")
            full_prop_uri = BASE + prop_uri
            full_prop2_uri = (BASE + prop2) if prop2 else None

            sparql = inject_and_generate(
                airport_uri,
                full_prop_uri,
                question,
                strategy=STRATEGY,
                property2_uri=full_prop2_uri
            )

            result["sparql"] = sparql

            is_valid = (
                validate_sparql(sparql)
                and sparql.strip().startswith("SELECT")
                and "PREFIX" not in sparql
                and full_prop_uri in sparql
            )

            result["sparql_valid"] = is_valid

            if is_valid:
                raw = execute_sparql(
                    sparql,
                    endpoint=get_endpoint("airports")
                )

                result["raw_answer"] = raw

                if raw:
                    result["final_answer"] = format_answer(
                        question,
                        raw,
                        lang
                    )
                    result["failure_type"] = "success"
                else:
                    result["failure_type"] = "execution_failure"

            else:
                result["failure_type"] = "generation_failure"

        # ── BRANCH D ──────────────────────────────────────────────────────────
        elif query_type == "cross_kg":
            flight_number = routing["entity"]
            direction     = routing["direction"]
            flight_uri    = map_flight(flight_number)

            if not flight_uri:
                result["failure_type"] = "mapping_failure"
                return result

            entities = extract_airport_entities(question, lang, iata_from_router=None)
            lexicon_path = get_lexicon("airports")
            lexicon      = load_lexicon(lexicon_path)
            prop_uri, layer, prop2 = map_property_cascade(
                entities["property"], lexicon, lexicon_path
            )
            result["property_uri"]  = prop_uri
            result["mapping_layer"] = layer

            if not prop_uri:
                result["failure_type"] = "mapping_failure"
                return result

            BASE          = get_base_uri("airports")
            full_prop_uri = BASE + prop_uri

            cross_result = resolve_cross_kg(
                flight_uri     = flight_uri,
                direction      = direction,
                property_uri   = full_prop_uri,
                property_short = prop_uri,
            )
            result["raw_answer"] = cross_result.get("raw_value")

            if cross_result["success"]:
                raw = cross_result["raw_value"]
                result["final_answer"] = format_answer(question, raw, lang)
                result["failure_type"] = "success"
            else:
                result["failure_type"] = cross_result.get("failure_type", "cross_kg_failure")

        elif query_type == "template":
            result["failure_type"] = "template_not_implemented"

    except Exception as e:
        result["failure_type"] = f"exception: {str(e)}"

    result["duration_s"] = round(time.time() - t0, 2)
    return result


# ── DISPLAY ───────────────────────────────────────────────────────────────────

def print_result(i: int, question: str, expected: str, r: dict):
    routing_mark = ok("✓") if r["routing_ok"] else err("✗")
    success_mark = ok("✓") if r["failure_type"] == "success" else err("✗")

    q_short = question[:55] + "…" if len(question) > 55 else question
    print(f"  [{i+1}] {q_short}")
    print(f"       lang={r['detected_lang']}  route={routing_mark} {r['query_type']}  "
          f"map={r['mapping_layer'] or '-'}  valid={r['sparql_valid']}  "
          f"end2end={success_mark}  ({r['duration_s']}s)")
    if r["final_answer"]:
        print(f"       → {r['final_answer']}")
    elif r["failure_type"] != "success":
        print(f"       {warn('⚠')} failure: {r['failure_type']}")
    print()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    all_results  = []
    total        = 0
    total_routed = 0
    total_passed = 0

    log_file = open("test_results.jsonl", "w", encoding="utf-8")

    for branch_name, cases in ALL_TESTS:
        print(f"\n{'─'*60}")
        print(f"  {branch_name}")
        print(f"{'─'*60}")

        branch_passed = 0
        branch_routed = 0

        for i, (question, expected_type, note) in enumerate(cases):
            print(f"\n  Running test {i+1}/{len(cases)}...")
            r = run_single_test(question, expected_type)
            r["note"]   = note
            r["branch"] = branch_name
            all_results.append(r)
            log_file.write(json.dumps(r, ensure_ascii=False) + "\n")
            print_result(i, question, expected_type, r)

            if r["routing_ok"]:                    branch_routed += 1
            if r["failure_type"] == "success":     branch_passed += 1

        total        += len(cases)
        total_routed += branch_routed
        total_passed += branch_passed

        print(f"  Branch result: routing {branch_routed}/{len(cases)}  "
              f"end-to-end {branch_passed}/{len(cases)}")

    log_file.close()

    print(f"\n{'═'*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'═'*60}")
    print(f"  Total questions : {total}")
    print(f"  Routing correct : {total_routed}/{total}  "
          f"({round(total_routed/total*100)}%)")
    print(f"  End-to-end pass : {total_passed}/{total}  "
          f"({round(total_passed/total*100)}%)")
    print(f"\n  Full results saved to: test_results.jsonl")
    print(f"{'═'*60}\n")

    failures = [r for r in all_results if r["failure_type"] != "success"]
    if failures:
        print("  Failure breakdown:")
        from collections import Counter
        counts = Counter(r["failure_type"] for r in failures)
        for ft, n in counts.most_common():
            print(f"    {ft:30s} × {n}")
        print()


if __name__ == "__main__":
    main()