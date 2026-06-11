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

NEW_BATCH = [
 
# ═══════════════════════════════════════════════════════════════════
# GROUP 1 — KG1 EXACT MATCH  (10 questions)
# Tests: flight number extraction, exact lexicon hits, EN/FR/AR
# Expected: all single_kg1, all success
# ═══════════════════════════════════════════════════════════════════
 
# English
("What is the gate of flight OS529?",                    "single_kg1", "kg1-en-exact"),
("What terminal does flight TK1847 use?",                "single_kg1", "kg1-en-exact"),
("What is the callsign of flight LO283?",                "single_kg1", "kg1-en-exact"),
("What weather conditions affect flight FR9005?",        "single_kg1", "kg1-en-exact"),
("Which runway does flight OS631 use?",                  "single_kg1", "kg1-en-exact"),
 
# French — different flights from batch 1
("Quelle est la porte du vol OS529?",                    "single_kg1", "kg1-fr-exact"),
("Quel est le terminal du vol TK1847?",                  "single_kg1", "kg1-fr-exact"),
("Quelle est la compagnie aérienne du vol LO283?",       "single_kg1", "kg1-fr-exact"),
 
# Arabic — different flights from batch 1
("ما هي بوابة الصعود للرحلة OS529؟",                     "single_kg1", "kg1-ar-exact"),
("ما هو المدرج الذي تستخدمه الرحلة FR707؟",              "single_kg1", "kg1-ar-exact"),
 
 
# ═══════════════════════════════════════════════════════════════════
# GROUP 2 — KG2 EXACT MATCH  (10 questions)
# Tests: airport entity detection, exact lexicon hits, EN/FR/AR
# Expected: all single_kg2, all success
# ═══════════════════════════════════════════════════════════════════
 
# English — different airports from batch 1
("What is the elevation of Istanbul airport?",           "single_kg2", "kg2-en-exact"),
("What country is Brussels airport in?",                 "single_kg2", "kg2-en-exact"),
("What is the IATA code of Stockholm airport?",          "single_kg2", "kg2-en-exact"),
("What city does Copenhagen airport serve?",             "single_kg2", "kg2-en-exact"),
("What is the ICAO code of Rome airport?",               "single_kg2", "kg2-en-exact"),
 
# French — different airports
("Quel est le type de l'aéroport de Bruxelles?",        "single_kg2", "kg2-fr-exact"),
("Dans quelle ville se trouve l'aéroport de Rome?",     "single_kg2", "kg2-fr-exact"),
("Quel est le pays de l'aéroport de Stockholm?",        "single_kg2", "kg2-fr-exact"),
 
# Arabic — different airports
("ما هو ارتفاع مطار إسطنبول؟",                          "single_kg2", "kg2-ar-exact"),
("في أي دولة يقع مطار بروكسل؟",                         "single_kg2", "kg2-ar-exact"),
 
 
# ═══════════════════════════════════════════════════════════════════
# GROUP 3 — NEW SYNONYM ENTRIES  (5 questions)
# Tests: the new lexicon entries added in lexicon_airports_patch.json
# nation → country, town/serve → municipality, kind → airportType
# Expected: all single_kg2, all success
# ═══════════════════════════════════════════════════════════════════
 
("What nation is London Heathrow airport in?",           "single_kg2", "kg2-synonym-nation"),
("Which town does Frankfurt airport serve?",             "single_kg2", "kg2-synonym-town"),
("What kind of facility is Munich airport?",             "single_kg2", "kg2-synonym-kind"),
("What nation is Vienna airport located in?",            "single_kg2", "kg2-synonym-nation"),
("Which city does Warsaw airport serve?",                "single_kg2", "kg2-synonym-serve"),
 
 
# ═══════════════════════════════════════════════════════════════════
# GROUP 4 — CROSS-KG  (10 questions)
# Tests: KG1 flight → IATA → KG2 airport property
# Tests the flexible cross-KG signal detection (Fix 2)
# Expected: all cross_kg, all success
# ═══════════════════════════════════════════════════════════════════
 
# English — destination
("What country is the destination airport of flight LO283?",     "cross_kg", "xkg-en-dest"),
("What is the elevation of the landing airport of flight TK1847?","cross_kg", "xkg-en-dest"),
("What type of airport does flight FR9005 land at?",              "cross_kg", "xkg-en-dest"),
("What city is the arrival airport of flight OS529 in?",          "cross_kg", "xkg-en-dest"),
 
# English — origin
("What country is the departure airport of flight LO283?",        "cross_kg", "xkg-en-orig"),
("What is the elevation of the origin airport of flight TK1847?", "cross_kg", "xkg-en-orig"),
 
# French
("Dans quel pays se trouve l'aéroport de destination du vol LO283?", "cross_kg", "xkg-fr-dest"),
("Quelle est l'élévation de l'aéroport de départ du vol OS529?",     "cross_kg", "xkg-fr-orig"),
 
# Arabic
("ما هي دولة مطار الوصول للرحلة TK1847؟",                "cross_kg", "xkg-ar-dest"),
("ما هو ارتفاع مطار المغادرة للرحلة LO283؟",             "cross_kg", "xkg-ar-orig"),
 
 
# ═══════════════════════════════════════════════════════════════════
# GROUP 5 — FUZZY / TYPO TOLERANCE  (5 questions)
# Tests: the fuzzy tier of map_property_cascade
# Deliberate typos and spelling variants
# Expected: single_kg1 or single_kg2, success
# ═══════════════════════════════════════════════════════════════════
 
("What is the elevtion of Viena airport?",               "single_kg2", "fuzzy-typo"),
("What cuntry is Frankfort airport in?",                 "single_kg2", "fuzzy-typo"),
("Which airlin operates flight FR707?",                  "single_kg1", "fuzzy-typo"),
("What is the departur city of flight OS295?",           "single_kg1", "fuzzy-typo"),
("destiantion of flight BR62",                           "single_kg1", "fuzzy-short-typo"),
 
 
# ═══════════════════════════════════════════════════════════════════
# GROUP 6 — SEMANTIC FALLBACK  (5 questions)
# Tests: embedding tier — phrases NOT in lexicon, must reach tier 3
# These are genuine paraphrases a real user might type
# Expected: success (semantic mapping to correct property)
# ═══════════════════════════════════════════════════════════════════
 
("How far above the ground is Vienna airport?",          "single_kg2", "semantic-elevation"),
("What is the official designation of Munich airport?",  "single_kg2", "semantic-airportType"),
("Where is Berlin Brandenburg airport physically located?","single_kg2","semantic-municipality"),
("How fast was flight OS295 flying?",                    "single_kg1", "semantic-speed"),
("Which carrier is responsible for flight DE1866?",      "single_kg1", "semantic-airline"),
 
 
# ═══════════════════════════════════════════════════════════════════
# GROUP 7 — OUT-OF-SCOPE / NOISE  (5 questions)
# Tests: the minimum structure guard (Fix 4) + general robustness
# None of these should produce a flight or airport answer
# Expected: all out_of_scope
# ═══════════════════════════════════════════════════════════════════
 
("Who is the president of France?",                      "out_of_scope", "noise-general"),
("How do I book a flight?",                              "out_of_scope", "noise-general"),
("runway",                                               "out_of_scope", "noise-single-word"),
("What is the best pizza in Naples?",                    "out_of_scope", "noise-location-trap"),
("Tell me about the history of aviation",                "out_of_scope", "noise-general"),
 
 
# ═══════════════════════════════════════════════════════════════════
# GROUP 8 — SHORT FORM QUERIES  (5 questions)
# Tests: 2-3 word inputs that ARE valid (contain flight number or entity)
# These were previously failing or untested
# Expected: success
# ═══════════════════════════════════════════════════════════════════
 
("gate OS529",                                           "single_kg1", "short-2words"),
("terminal TK1847",                                      "single_kg1", "short-2words"),
("elevation IST",                                        "single_kg2", "short-2words"),
("country VIE airport",                                  "single_kg2", "short-3words"),
("airline DE1866",                                       "single_kg1", "short-2words"),
 
 
# ═══════════════════════════════════════════════════════════════════
# GROUP 9 — TWO-HOP QUERIES  (5 questions)
# Tests: property2_uri injection — requires intermediate node
# Examples: aircraft type, country via locatedInCountry → countryName
# Expected: all success with two-hop SPARQL
# ═══════════════════════════════════════════════════════════════════
 
("What type of aircraft is used on flight OS295?",       "single_kg1", "two-hop-aircraft-type"),
("What is the aircraft registration of flight BR62?",    "single_kg1", "two-hop-aircraft-reg"),
("What is the ground speed of flight TK1847?",           "single_kg1", "two-hop-speed"),
("What continent is Vienna airport on?",                 "single_kg2", "two-hop-continent"),
("What region is Frankfurt airport in?",                 "single_kg2", "two-hop-region"),
 
]
 

ALL_TESTS = [
    ("FULL PIPELINE TEST", NEW_BATCH),
]


ALL_TESTS = [
  #  ("BRANCH B — single_kg1", BRANCH_B_TESTS),
    ("BRANCH C — single_kg2", NEW_BATCH),
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