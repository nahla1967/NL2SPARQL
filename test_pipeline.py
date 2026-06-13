"""
test_pipeline.py  (v2 — full branch coverage)
----------------------------------------------
WHAT CHANGED vs v1:

    The original run_single_test() returned immediately with
    wrong_route:single_kg1 (or single_kg2 / cross_kg) whenever the
    router did not return "template". This was correct when the test
    file only tested Branch E. Now that the suite covers all branches,
    each route must be followed through its own pipeline.

    run_single_test() now dispatches to four handlers:
        _run_single_kg1()   — Branch B: flight number + property
        _run_single_kg2()   — Branch C: airport name/IATA + property
        _run_cross_kg()     — Branch D: flight number + airport property
        _run_template()     — Branch E: filter / rank / compare / count

    The expected_type field in each test case determines what the router
    SHOULD return. If the router returns something different, routing_ok
    is False and the test is counted as a routing failure. If the router
    is correct, the full pipeline runs and end2end is True only if the
    pipeline also succeeds.

HOW TO RUN:
    1. Fuseki running:
          http://localhost:3030/flights/sparql
          http://localhost:3030/airports/sparql
    2. Ollama running:  ollama serve
    3. From project root (NL2SPARQL/):
          python test_pipeline.py
"""

import json
import time

from pipeline.language  import detect_language
from router             import route
from template_resolver  import resolve_template

# KG1 pipeline components
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
from cross_kg_resolver  import resolve_cross_kg
from kg_registry        import get_base_uri, get_endpoint, get_lexicon

# ── TEST CASES ────────────────────────────────────────────────────────────────

FULL_SYSTEM_TESTS = [

    # ══ BRANCH B — single_kg1 ═════════════════════════════════════════════════
    ("Where does flight OS235 depart from?",              "single_kg1", "kg1_en_exact"),
    ("What is the destination of flight KE567?",          "single_kg1", "kg1_en_exact"),
    ("Which airline operates flight FR182?",              "single_kg1", "kg1_en_exact"),
    ("What is the callsign of flight AI180?",             "single_kg1", "kg1_en_exact"),
    ("What aircraft is used on flight BR62?",             "single_kg1", "kg1_en_exact"),
    ("What country is flight LO225 flying to?",           "single_kg1", "kg1_en_fuzzy"),
    ("What is the ground speed of flight FR6889?",        "single_kg1", "kg1_en_fuzzy"),
    ("D'où part le vol OS295?",                           "single_kg1", "kg1_fr_exact"),
    ("Quelle est la destination du vol LG8854?",          "single_kg1", "kg1_fr_exact"),
    ("Quelle compagnie opère le vol FR947?",              "single_kg1", "kg1_fr_fuzzy"),
    ("من أين تغادر الرحلة OS235؟",                        "single_kg1", "kg1_ar_exact"),
    ("ما وجهة الرحلة KE567؟",                             "single_kg1", "kg1_ar_exact"),
    ("أي شركة تشغّل الرحلة FR182؟",                       "single_kg1", "kg1_ar_fuzzy"),

    # ══ BRANCH C — single_kg2 ═════════════════════════════════════════════════
    ("What is the elevation of VIE?",                     "single_kg2", "kg2_en_iata"),
    ("What country is ZRH in?",                           "single_kg2", "kg2_en_iata"),
    ("What type of airport is SOF?",                      "single_kg2", "kg2_en_iata"),
    ("What city does MUC serve?",                         "single_kg2", "kg2_en_iata"),
    ("What is the elevation of Vienna airport?",          "single_kg2", "kg2_en_name"),
    ("How long is the runway at London Heathrow?",        "single_kg2", "kg2_en_name"),
    ("What is the ICAO code of Charles de Gaulle?",       "single_kg2", "kg2_en_name"),
    ("Quelle est l'élévation de l'aéroport de Vienne?",  "single_kg2", "kg2_fr_name"),
    ("Dans quel pays se trouve l'aéroport de Munich?",   "single_kg2", "kg2_fr_name"),
    ("Quel type d'aéroport est SOF?",                     "single_kg2", "kg2_fr_iata"),
    ("ما ارتفاع مطار فيينا؟",                             "single_kg2", "kg2_ar_name"),
    ("ما طول مدرج مطار VIE؟",                             "single_kg2", "kg2_ar_iata"),
    ("ما نوع مطار ZRH؟",                                  "single_kg2", "kg2_ar_iata"),

    # ══ BRANCH D — cross_kg ═══════════════════════════════════════════════════
    ("What country is the destination airport of flight OS235?",   "cross_kg", "crosskg_en"),
    ("What is the elevation of the destination airport of KE567?", "cross_kg", "crosskg_en"),
    ("What country does flight LO225 land in?",                    "cross_kg", "crosskg_en"),
    ("What type of airport does flight FR182 arrive at?",          "cross_kg", "crosskg_en"),
    ("Dans quel pays atterrit le vol OS295?",                       "cross_kg", "crosskg_fr"),
    ("في أي دولة يهبط الرحلة OS235؟",                             "cross_kg", "crosskg_ar"),

    # ══ BRANCH E — template ═══════════════════════════════════════════════════

    # filter_numeric_kg2
    ("Which airports have an elevation above 1000 feet?",          "template", "filter_numeric_kg2"),
    ("List airports with runways shorter than 7000 feet.",         "template", "filter_numeric_kg2"),
    ("Quels aéroports ont une élévation supérieure à 500 pieds?",  "template", "filter_numeric_kg2"),
    ("ما هي المطارات التي يزيد ارتفاعها عن 1000 قدم؟",            "template", "filter_numeric_kg2"),

    # filter_string_kg2
    ("Which airports are located in Italy?",                       "template", "filter_string_kg2"),
    ("Show all large airports.",                                    "template", "filter_string_kg2"),
    ("Quels aéroports sont situés en France?",                     "template", "filter_string_kg2"),
    ("ما هي المطارات الموجودة في تركيا؟",                          "template", "filter_string_kg2"),

    # ranking_kg2
    ("What are the top 5 airports with the highest elevation?",    "template", "ranking_kg2"),
    ("Which airport has the shortest runway?",                     "template", "ranking_kg2"),
    ("Quel aéroport a la piste la plus longue?",                   "template", "ranking_kg2"),
    ("أي مطار لديه أعلى ارتفاع؟",                                  "template", "ranking_kg2"),

    # compare_two_airports
    ("Compare VIE and FRA by elevation.",                          "template", "compare_two_airports"),
    ("Compare MUC and SOF by runway width.",                       "template", "compare_two_airports"),
    ("Comparez CDG et LHR par longueur de piste.",                 "template", "compare_two_airports"),
    ("قارن بين VIE وSOF من حيث الارتفاع.",                        "template", "compare_two_airports"),

    # count_kg1
    ("How many flights have destination city Berlin?",             "template", "count_kg1"),
    ("How many flights depart from Paris?",                        "template", "count_kg1"),
    ("Combien de vols partent de Vienne?",                         "template", "count_kg1"),
    ("كم رحلة تتجه إلى برلين؟",                                   "template", "count_kg1"),

    # filter_numeric_kg1
    ("Which flights have a ground speed above 400 knots?",         "template", "filter_numeric_kg1"),
    ("Which flights have a vertical speed below -1000 feet per minute?", "template", "filter_numeric_kg1"),
    ("Quels vols ont une vitesse au sol supérieure à 450 nœuds?",  "template", "filter_numeric_kg1"),
    ("ما الرحلات ذات السرعة الأرضية فوق 400 عقدة؟",               "template", "filter_numeric_kg1"),

    # cross_kg_filter
    ("Which flights land at airports with elevation above 800 feet?",  "template", "cross_kg_filter"),
    ("Which flights arrive at airports located in Germany?",           "template", "cross_kg_filter"),
    ("Which flights land at large airports?",                          "template", "cross_kg_filter"),
    ("Quels vols atterrissent dans des aéroports en Allemagne?",       "template", "cross_kg_filter"),
    ("ما الرحلات التي تهبط في مطارات فوق 800 قدم؟",                   "template", "cross_kg_filter"),
]

ALL_TESTS = [
    ("FULL SYSTEM TEST — all branches × all languages", FULL_SYSTEM_TESTS),
]

STRATEGY = "zero-shot"

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def ok(s):   return f"{GREEN}{s}{RESET}"
def err(s):  return f"{RED}{s}{RESET}"
def warn(s): return f"{YELLOW}{s}{RESET}"


# ── BRANCH HANDLERS ───────────────────────────────────────────────────────────
# Each handler runs its own pipeline and returns a partial result dict.
# run_single_test() merges these into the final result.

def _run_single_kg1(question: str, routing: dict, lang: str) -> dict:
    """
    Branch B: flight number is known (routing["entity"]).
    Runs: extractor → mapper → generator → executor.
    """
    out = {"sparql": None, "sparql_valid": False,
           "raw_answer": None, "final_answer": None,
           "failure_type": "not_run"}

    entities = extract_entities(question, lang)
    if not validate_extraction(entities) or not is_flight_question(entities):
        out["failure_type"] = "extraction_failure"
        return out

    lexicon      = load_lexicon(get_lexicon("flights"))
    property_uri, mapping_layer, property2_uri = map_property_cascade(
        entities["property"], lexicon, get_lexicon("flights")
    )
    flight_uri = map_flight(entities["entity"])

    if not flight_uri or not property_uri:
        out["failure_type"] = "mapping_failure"
        return out

    BASE           = get_base_uri("flights")
    full_prop_uri  = BASE + property_uri
    full_prop2_uri = (BASE + property2_uri) if property2_uri else None

    sparql = inject_and_generate(
        flight_uri, full_prop_uri, question,
        strategy=STRATEGY, property2_uri=full_prop2_uri
    )
    out["sparql"] = sparql

    is_valid = (
        validate_sparql(sparql)
        and sparql.strip().startswith("SELECT")
        and "PREFIX" not in sparql
        and full_prop_uri in sparql
    )
    if full_prop2_uri:
        is_valid = is_valid and (full_prop2_uri in sparql)
    out["sparql_valid"] = is_valid

    if is_valid:
        raw = execute_sparql(sparql, endpoint=get_endpoint("flights"))
        out["raw_answer"] = raw
        if raw:
            out["final_answer"]  = format_answer(question, raw, lang)
            out["failure_type"]  = "success"
        else:
            out["failure_type"] = "execution_failure"
    else:
        out["failure_type"] = "generation_failure"

    return out


def _run_single_kg2(question: str, routing: dict, lang: str) -> dict:
    """
    Branch C: airport entity is known (routing["entity"]).
    Runs: extractor → mapper → generator → executor (KG2 endpoint).
    """
    out = {"sparql": None, "sparql_valid": False,
           "raw_answer": None, "final_answer": None,
           "failure_type": "not_run"}

    entities = extract_airport_entities(question, lang, routing["entity"])
    if not validate_airport_extraction(entities):
        out["failure_type"] = "extraction_failure"
        return out

    lexicon_path = get_lexicon("airports")
    lexicon      = load_lexicon(lexicon_path)
    property_uri, mapping_layer, property2_uri = map_property_cascade(
        entities["property"], lexicon, lexicon_path
    )
    airport_uri = map_airport(entities["entity"]) if entities["entity"] else None

    if not airport_uri or not property_uri:
        out["failure_type"] = "mapping_failure"
        return out

    BASE           = get_base_uri("airports")
    full_prop_uri  = BASE + property_uri
    full_prop2_uri = (BASE + property2_uri) if property2_uri else None

    sparql = inject_and_generate(
        airport_uri, full_prop_uri, question,
        strategy=STRATEGY, property2_uri=full_prop2_uri
    )
    out["sparql"] = sparql

    is_valid = (
        validate_sparql(sparql)
        and sparql.strip().startswith("SELECT")
        and "PREFIX" not in sparql
        and full_prop_uri in sparql
    )
    out["sparql_valid"] = is_valid

    if is_valid:
        raw = execute_sparql(sparql, endpoint=get_endpoint("airports"))
        out["raw_answer"] = raw
        if raw:
            out["final_answer"]  = format_answer(question, raw, lang)
            out["failure_type"]  = "success"
        else:
            out["failure_type"] = "execution_failure"
    else:
        out["failure_type"] = "generation_failure"

    return out


def _run_cross_kg(question: str, routing: dict, lang: str) -> dict:
    """
    Branch D: flight number + airport property.
    Runs: mapper (flight URI) → extractor (airport property) →
          mapper (airport property) → cross_kg_resolver.
    """
    out = {"sparql": None, "sparql_valid": False,
           "raw_answer": None, "final_answer": None,
           "failure_type": "not_run"}

    flight_uri = map_flight(routing["entity"])
    if not flight_uri:
        out["failure_type"] = "mapping_failure"
        return out

    entities = extract_airport_entities(question, lang, iata_from_router=None)

    lexicon_path = get_lexicon("airports")
    lexicon      = load_lexicon(lexicon_path)
    property_uri, mapping_layer, _ = map_property_cascade(
        entities["property"], lexicon, lexicon_path
    )

    if not property_uri:
        out["failure_type"] = "mapping_failure"
        return out

    full_prop_uri = get_base_uri("airports") + property_uri

    result = resolve_cross_kg(
        flight_uri     = flight_uri,
        direction      = routing["direction"],
        property_uri   = full_prop_uri,
        property_short = property_uri,
    )

    out["raw_answer"]   = result.get("raw_value")
    out["failure_type"] = result.get("failure_type")
    out["sparql_valid"] = result.get("success", False)

    if result["success"]:
        out["final_answer"] = format_answer(question, result["raw_value"], lang)
        out["failure_type"] = "success"

    return out


def _run_template(question: str, routing: dict, lang: str) -> dict:
    """
    Branch E: template query.
    Delegates entirely to resolve_template().
    """
    template_name = routing["template"]
    tr = resolve_template(question, template_name, lang)
    return {
        "template_name":   template_name,
        "template_params": tr.get("params"),
        "sparql":          tr.get("sparql"),
        "sparql_valid":    tr.get("success", False),
        "raw_answer":      tr.get("raw_data"),
        "final_answer":    tr.get("final_answer"),
        "failure_type":    tr.get("failure_type"),
    }


# ── CORE RUNNER ───────────────────────────────────────────────────────────────

def run_single_test(question: str, expected_type: str) -> dict:
    result = {
        "question":        question,
        "expected_type":   expected_type,
        "detected_lang":   None,
        "query_type":      None,
        "routing_ok":      False,
        "template_name":   None,
        "template_params": None,
        "sparql":          None,
        "sparql_valid":    False,
        "raw_answer":      None,
        "final_answer":    None,
        "failure_type":    "not_run",
        "duration_s":      0,
    }

    t0 = time.time()

    try:
        # ── Step 0: language detection ────────────────────────────────────────
        lang = detect_language(question)
        result["detected_lang"] = lang

        # ── Step 1: routing ───────────────────────────────────────────────────
        routing    = route(question)
        query_type = routing["query_type"]
        result["query_type"] = query_type
        result["routing_ok"] = (query_type == expected_type)

        # ── Step 2: dispatch to the correct branch ────────────────────────────
        # Even if routing is wrong we still run the pipeline so we can see
        # what the actual branch produced — helps debugging misroutes.
        if query_type == "single_kg1":
            branch_out = _run_single_kg1(question, routing, lang)

        elif query_type == "single_kg2":
            branch_out = _run_single_kg2(question, routing, lang)

        elif query_type == "cross_kg":
            branch_out = _run_cross_kg(question, routing, lang)

        elif query_type == "template":
            branch_out = _run_template(question, routing, lang)

        else:
            # out_of_scope — nothing to run
            branch_out = {"failure_type": f"out_of_scope"}

        result.update(branch_out)

    except Exception as e:
        result["failure_type"] = f"exception: {str(e)}"

    result["duration_s"] = round(time.time() - t0, 2)
    return result


# ── DISPLAY ───────────────────────────────────────────────────────────────────

def print_result(i: int, question: str, r: dict):
    routing_mark = ok("✓") if r["routing_ok"]               else err("✗")
    success_mark = ok("✓") if r["failure_type"] == "success" else err("✗")

    q_short = question[:55] + "…" if len(question) > 55 else question
    print(f"  [{i+1}] {q_short}")
    print(f"       lang={r['detected_lang']}  "
          f"route={routing_mark} {r['query_type']}  "
          f"template={r.get('template_name') or '-'}  "
          f"valid={r['sparql_valid']}  "
          f"end2end={success_mark}  ({r['duration_s']}s)")
    if r["final_answer"]:
        ans = str(r["final_answer"])
        print(f"       → {ans[:120]}{'…' if len(ans) > 120 else ''}")
    elif r["failure_type"] not in ("success", "not_run"):
        print(f"       {warn('⚠')} failure : {r['failure_type']}")
        if r.get("template_params"):
            print(f"       params  : {r['template_params']}")
    print()


# ── MAIN ─────────────────────────────────────────────────────────────────────

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
            print(f"\n  Running test {i+1}/{len(cases)}  [{note}]...")
            r = run_single_test(question, expected_type)
            r["note"]   = note
            r["branch"] = branch_name
            all_results.append(r)
            log_file.write(json.dumps(r, ensure_ascii=False) + "\n")
            print_result(i, question, r)

            if r["routing_ok"]:                branch_routed += 1
            if r["failure_type"] == "success": branch_passed += 1

        total        += len(cases)
        total_routed += branch_routed
        total_passed += branch_passed

        print(f"  Branch result: routing {branch_routed}/{len(cases)}  "
              f"end-to-end {branch_passed}/{len(cases)}")

    log_file.close()

    # ── Summary ───────────────────────────────────────────────────────────────
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

    # ── Failure breakdown ──────────────────────────────────────────────────────
    failures = [r for r in all_results if r["failure_type"] != "success"]
    if failures:
        print("  Failure breakdown:")
        from collections import Counter
        counts = Counter(r["failure_type"] for r in failures)
        for ft, n in counts.most_common():
            print(f"    {ft:40s} × {n}")
        print()

    # ── Per-branch breakdown ───────────────────────────────────────────────────
    print("  Per expected_type breakdown:")
    from collections import defaultdict
    by_type = defaultdict(lambda: {"total": 0, "routed": 0, "passed": 0})
    for r in all_results:
        t = r["expected_type"]
        by_type[t]["total"]  += 1
        if r["routing_ok"]:                by_type[t]["routed"] += 1
        if r["failure_type"] == "success": by_type[t]["passed"] += 1

    for t, counts in sorted(by_type.items()):
        print(f"    {t:20s}  routing {counts['routed']}/{counts['total']}  "
              f"end2end {counts['passed']}/{counts['total']}")
    print()


if __name__ == "__main__":
    main()