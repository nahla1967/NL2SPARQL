"""
test_pipeline.py
----------------
Tests Branch E (template) of the NL2SPARQL multi-KG pipeline.

HOW TO RUN:
    1. Make sure Fuseki is running with BOTH datasets:
          http://localhost:3030/flights/sparql   (KG1)
          http://localhost:3030/airports/sparql  (KG2)
    2. Make sure Ollama is running:
          ollama serve
    3. From your project root (NL2SPARQL/):
          python test_pipeline.py

WHAT IT TESTS:
    Branch E — template : 21 questions across 7 template types × 3 languages
        filter_numeric_kg2   : airports where numeric property meets condition
        filter_string_kg2    : airports where text property equals value
        ranking_kg2          : top/bottom N airports by property
        compare_two_airports : airport A vs airport B
        count_kg1            : count or list flights matching a condition
        filter_numeric_kg1   : flights where speed meets a threshold
        cross_kg_filter      : flights whose airport property meets condition

OUTPUT:
    Prints a result table per group.
    Saves all results to test_results.jsonl.
    Prints a final summary with pass/fail counts.
"""

import json
import time

from pipeline.language import detect_language
from router            import route
from template_resolver import resolve_template

# ── TEST CASES ────────────────────────────────────────────────────────────────

TEMPLATE_TESTS = [
    # ── filter_numeric_kg2 : testing new signals (exceeds, greater than) ──────
    ("Show airports whose runway length exceeds 10000 feet.", "template", "filter_numeric_kg2"),
    ("List airports with elevation greater than 2000 feet.", "template", "filter_numeric_kg2"),
    ("Which airports have a runway width below 150 feet?",   "template", "filter_numeric_kg2"),
    ("Show airports whose elevation is above 500 feet.",     "template", "filter_numeric_kg2"),

    # ── filter_string_kg2 : testing new signals (located in, municipality) ────
    ("Which airports are located in Germany?",               "template", "filter_string_kg2"),
    ("Show all large airports.",                             "template", "filter_string_kg2"),
    ("Show airports whose municipality is Vienna.",          "template", "filter_string_kg2"),
    ("List airports in France.",                             "template", "filter_string_kg2"),

    # ── ranking_kg2 : variations in phrasing ──────────────────────────────────
    ("Which airport has the shortest runway?",               "template", "ranking_kg2"),
    ("Show the top 3 airports with the widest runways.",     "template", "ranking_kg2"),
    ("List the 5 airports at the lowest elevation.",         "template", "ranking_kg2"),

    # ── compare_two_airports : testing the .strip() fix on IATA codes ─────────
    ("Compare LHR and MAD by elevation.",                    "template", "compare_two_airports"),
    ("Compare VIE and FRA by elevation.",                    "template", "compare_two_airports"),
    ("Compare JFK and CDG by runway length.",                "template", "compare_two_airports"),

    # ── filter_numeric_kg1 : testing new alt property ─────────────────────────
    ("List flights with altitude above 30000 feet.",         "template", "filter_numeric_kg1"),
    ("Which flights are flying above 35000 feet?",           "template", "filter_numeric_kg1"),

    # ── cross_kg_filter : testing new signals (land at large airports) ────────
    ("Which flights land at large airports?",                "template", "cross_kg_filter"),
    ("Show flights whose destination airport has a runway longer than 10000 feet.", "template", "cross_kg_filter"),

    # ── count_kg1 : regression check ─────────────────────────────────────────
    ("How many flights are operated by Lufthansa?",          "template", "count_kg1"),
    ("How many flights have destination city Berlin?",       "template", "count_kg1"),
]
ALL_TESTS = [
    ("BRANCH E — template", TEMPLATE_TESTS),
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
        "question":       question,
        "expected_type":  expected_type,
        "detected_lang":  None,
        "query_type":     None,
        "routing_ok":     False,
        "template_name":  None,
        "template_params": None,
        "sparql":         None,
        "sparql_valid":   False,
        "raw_answer":     None,
        "final_answer":   None,
        "failure_type":   "not_run",
        "duration_s":     0,
    }

    t0 = time.time()

    try:
        # Step 0 — language detection
        lang = detect_language(question)
        result["detected_lang"] = lang

        # Step 1 — routing
        routing    = route(question)
        query_type = routing["query_type"]
        result["query_type"] = query_type
        result["routing_ok"] = (query_type == expected_type)

        if query_type != "template":
            # Router sent this question to the wrong branch
            result["failure_type"] = f"wrong_route:{query_type}"
            return result

        # Step 2 — template resolution (param extraction + SPARQL build + execute + format)
        template_name = routing["template"]
        result["template_name"] = template_name

        tr = resolve_template(question, template_name, lang)

        result["template_params"] = tr.get("params")
        result["sparql"]          = tr.get("sparql")
        result["sparql_valid"]    = tr.get("success", False)
        result["raw_answer"]      = tr.get("raw_data")
        result["final_answer"]    = tr.get("final_answer")
        result["failure_type"]    = tr.get("failure_type")

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
          f"template={r['template_name'] or '-'}  "
          f"valid={r['sparql_valid']}  "
          f"end2end={success_mark}  ({r['duration_s']}s)")
    if r["final_answer"]:
        print(f"       → {r['final_answer']}")
    elif r["failure_type"] != "success":
        print(f"       {warn('⚠')} failure : {r['failure_type']}")
        if r["template_params"]:
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
            print(f"    {ft:35s} × {n}")
        print()


if __name__ == "__main__":
    main()