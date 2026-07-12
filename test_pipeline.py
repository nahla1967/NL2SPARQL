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
import template_resolver

def _deterministic_format_answer(question: str, raw_data: str, lang: str) -> str:
    """
    Test-only override for template_resolver._format_answer.
    Counts and lists are computed in Python from raw_data directly —
    never restated or recounted by the LLM.
    """
    lines = [ln for ln in raw_data.strip().split("\n") if ln.strip()]
    count = len(lines)

    if count == 0:
        return "No results found."
    if count == 1 and lines[0].replace(".", "", 1).isdigit():
        # A single numeric raw_data (e.g. a count_kg1 result) — just state it.
        return f"The answer is {lines[0]}."

    listed = "\n".join(f"{i+1}. {ln}" for i, ln in enumerate(lines))
    return f"There are {count} result(s):\n\n{listed}"

# Override — test only, template_resolver.py itself is untouched.
template_resolver._format_answer = _deterministic_format_answer
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
    get_university_entity_type,
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
from pipeline.extractor import (
    extract_entities,
    validate_extraction,
    is_flight_question,
    extract_airport_entities,
    validate_airport_extraction,
    extract_university_entities,
    validate_university_extraction,
)
from pipeline.mapper import (
    load_lexicon,
    map_property_cascade,
    map_flight,
    map_airport,
    map_university_entity,
)
from pipeline.generator import inject_and_generate
from pipeline.executor  import (
    validate_sparql,
    execute_sparql,
    format_answer,
    format_answer_list,
)
from cross_kg_resolver  import resolve_cross_kg
from kg_registry        import get_base_uri, get_endpoint, get_lexicon

# ── TEST CASES ────────────────────────────────────────────────────────────────

FULL_SYSTEM_TESTS =[
    # ── single_kg3 (direct lookups) ──────────────────────────────────────────
    ("Who is the advisor of UndergraduateStudent4?",       "single_kg3", "kg3b_single_001"),
    ("What courses does AssociateProfessor0 teach?",       "single_kg3", "kg3b_single_002"),
    ("What university is Department5 part of?",            "single_kg3", "kg3b_single_003"),
    ("What is AssociateProfessor0's name?",                 "single_kg3", "kg3b_single_004"),
    ("How many courses does AssociateProfessor0 teach?",    "template",   "kg3b_count_001"),
    ("How many courses does UndergraduateStudent4 take?",   "template",   "kg3b_count_002"),
    ("List the courses that UndergraduateStudent4 takes.",  "template",   "kg3b_count_003"),
    ("How many professors work for Department5?",           "template",   "kg3b_count_004"),
    ("Which professors work for Department5?",               "template",   "kg3b_filter_001"),
    ("Which professors work for Department3?",               "template",   "kg3b_filter_002"),
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
    print(f"[debug] sparql=\n{sparql}")  # ← add this

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
def _run_single_kg3(question: str, routing: dict, lang: str) -> dict:
    """
    Branch B (KG3): university entity is known (routing["entity"]).
    Runs: extractor → mapper → generator → executor (university endpoint).
    """
    out = {"sparql": None, "sparql_valid": False,
           "raw_answer": None, "final_answer": None,
           "failure_type": "not_run"}

    entities = extract_university_entities(question, lang, routing["entity"])
    if not validate_university_extraction(entities):
        out["failure_type"] = "extraction_failure"
        return out

    lexicon_path = get_lexicon("university")
    lexicon      = load_lexicon(lexicon_path)
    property_uri, mapping_layer, property2_uri = map_property_cascade(
        entities["property"], lexicon, lexicon_path
    )
    entity_uri = map_university_entity(entities["entity"]) if entities["entity"] else None

    # Disambiguate "part of" style phrases: memberOf (person -> dept)
    # vs subOrganizationOf (dept -> university) — same fix as main.py.
    if entity_uri and property_uri in ("memberOf", "subOrganizationOf"):
        entity_type = get_university_entity_type(entity_uri)
        if entity_type == "Department" and property_uri == "memberOf":
            property_uri = "subOrganizationOf"
        elif entity_type != "Department" and property_uri == "subOrganizationOf":
            property_uri = "memberOf"
    
    if not entity_uri or not property_uri:
        out["failure_type"] = "mapping_failure"
        return out

    BASE           = get_base_uri("university")
    full_prop_uri  = BASE + property_uri
    full_prop2_uri = (BASE + property2_uri) if property2_uri else None

    sparql = inject_and_generate(
        entity_uri, full_prop_uri, question,
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
        raw = execute_sparql(sparql, endpoint=get_endpoint("university"), multiple=True)
        out["raw_answer"] = raw
        if raw:
            out["final_answer"]  = format_answer_list(question, raw, lang)
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

def _run_open_kg(question: str, routing: dict, lang: str) -> dict:
    """
    Branch F: open_kg — schema-grounded free SPARQL generation.
    No mapping layer. The LLM generates SPARQL directly from the schema
    and also determines the correct target endpoint by inspecting which
    ontology namespace appears in the generated query.

    execute_sparql is called with multiple=True because open_kg questions
    are aggregate or exploratory — they return lists of results, not a
    single value for a known entity. format_answer_list is used accordingly.
    """
    from pipeline.generator import generate_open_kg_sparql
    from pipeline.executor  import format_answer_list
    from kg_registry        import get_open_kg_schema

    out = {"sparql": None, "sparql_valid": False,
           "raw_answer": None, "final_answer": None,
           "failure_type": "not_run"}

    schema = get_open_kg_schema()
    sparql, endpoint = generate_open_kg_sparql(question, lang, schema)
    out["sparql"] = sparql

    if not sparql or not sparql.strip().startswith("SELECT"):
        out["failure_type"] = "generation_failure"
        return out

    is_valid = validate_sparql(sparql)
    out["sparql_valid"] = is_valid

    if not is_valid:
        out["failure_type"] = "generation_failure"
        return out

    raw = execute_sparql(sparql, endpoint=endpoint, multiple=True)
    print(f"[debug] raw={raw}")
    out["raw_answer"] = raw

    if raw:
        out["final_answer"] = format_answer_list(question, raw, lang)
        out["failure_type"] = "success"
    else:
        out["failure_type"] = "execution_failure"

    return out
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
        elif query_type == "single_kg3":
            branch_out = _run_single_kg3(question, routing, lang)
        elif query_type == "open_kg":
            branch_out = _run_open_kg(question, routing, lang)
        
        else:
            # out_of_scope — nothing to run.
            # Success means the router correctly recognized this as out-of-scope.
            if query_type == expected_type:
                branch_out = {"failure_type": "success"}
            else:
                branch_out = {"failure_type": "out_of_scope_misroute"}

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