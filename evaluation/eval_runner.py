"""
eval_runner.py
---------------
Reads NL2SPARQL_Evaluation_Dataset.xlsx, expands each row into the correct
number of runs (x3 languages always, x3 strategies only where
strategy_applicable=TRUE), executes each one through the REAL pipeline
(same branch handlers as test_pipeline.py / main.py), and writes one JSON
line per run to eval_results.jsonl.

PLACE THIS FILE IN THE ROOT OF THE NL2SPARQL REPO (same level as router.py,
main.py, kg_registry.py) — it imports those modules directly, exactly like
test_pipeline.py does. It assumes Fuseki (all 3 endpoints) and Ollama are
running locally, same prerequisites as test_pipeline.py.

WHY THIS DOESN'T DUPLICATE test_pipeline.py:
    test_pipeline.py is your fixed-suite pipeline smoke test — good for
    "does every branch still work". This script is the thesis EVALUATION
    run: it reads the versioned dataset (so results are reproducible and
    the question set is auditable in the same file your supervisors see),
    threads the strategy condition through correctly per-branch (only
    single_kg1/2/3 actually vary by strategy — see kg_registry / generator.py),
    and scores each run against a ground-truth expected_answer for
    Exact Match / F1, which test_pipeline.py does not do at all.

OUTPUT SCHEMA (one JSON object per line in eval_results.jsonl):
    id, tier, category, kg, language, strategy, expected_type, query_type,
    routing_ok, sparql, sparql_valid, raw_answer, final_answer,
    failure_type, error_detail, exact_match, f1, duration_s
"""

import ast
import json
import re
import time
import pandas as pd
import os
import sys
import socket
from rdflib import Graph
import sys
import os
import sys
import os

# ── PATH FIX : eval_runner.py is in evaluation/, but needs root-level modules ──
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ── TTL PATH FIX : ontology files live in data/ ──
_AIRPORT_TTL_PATH = os.path.join(_REPO_ROOT, "data", "airport_ontology_kg1_aligned.ttl")
# Add parent directory to path so root-level modules (router, kg_registry, etc.) are importable
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
FULL_RUN = "--full" in sys.argv

DATASET_PATH = os.path.join(os.path.dirname(__file__), "results", "NL2SPARQL_Evaluation_Dataset.xlsx")

# ── BASELINE / ABLATION MODE ────────────────────────────────────────────────
# None            : normal run (writes to eval_results.jsonl, as always)
# "A"             : Baseline A — LLM without knowledge injection. Forces
#                   every question in BASELINE_A_CATEGORIES through the
#                   schema-guided open_kg generator instead of normal
#                   routing/URI injection.
# "B"             : Baseline B — LLM without templates. Forces every
#                   question in BASELINE_B_CATEGORIES through open_kg
#                   instead of resolve_template().
# "ablation"      : normal routing, but map_property_cascade() only runs
#                   the tiers listed in ABLATION_STAGES (see stages= on
#                   _run_single_kg1/2/3 and _run_cross_kg). Scoped to the
#                   same question population as Baseline A, since that's
#                   the only population where map_property_cascade() is
#                   actually exercised.
BASELINE_MODE = None # None | "A" | "B" | "ablation"

# Which cascade tiers to keep when BASELINE_MODE == "ablation". Edit this
# and rerun for each ablation condition (e.g. {"pre-norm","exact"} for
# "exact-only", {"pre-norm","exact","fuzzy"} for "exact+fuzzy").
ABLATION_STAGES = frozenset({"pre-norm", "exact"})

BASELINE_A_CATEGORIES = {"single_kg1", "single_kg2", "single_kg3", "cross_kg"}
BASELINE_B_CATEGORIES = {"count_kg1", "count_kg3", "filter_numeric_kg1",
                          "filter_numeric_kg2", "filter_string_kg2",
                          "filter_string_kg3", "ranking_kg2",
                          "compare_two_airports"}

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "results",
    f"baseline_{BASELINE_MODE}_results.jsonl" if BASELINE_MODE else "eval_results.jsonl"
)

LANGUAGES = ["en", "fr", "ar"]
STRATEGIES = ["zero-shot", "few-shot", "cot"]
BROKEN_IDS = {
    # Bug 2 fix — group_aggregate_kg2 → ranking_kg2 reroute
    "count_kg2_001", "compare_two_flights_001",
    # Bug 3 + Bug 4 fixes — malformed COUNT SPARQL + hallucination misroute

    # Remaining ids from the last 20 questions — bug category not specified, please confirm
   
    "ranking_kg1_001",
    "single_kg3_009",
    "ranking_kg3_001",
    "filter_numeric_kg3_002",
    "filter_numeric_kg3_003",
    "single_kg3_010",
    "single_kg3_011",
    "single_kg3_012",
    "single_kg3_013",
    "group_aggregate_kg1_001",
    "group_aggregate_kg3_001",
    "ranking_kg1_002",
    "ranking_kg3_002",
    "count_kg2_002",
    "count_kg2_003",
    "compare_two_flights_002",
    "compare_two_departments_002",
}
from template_resolver import resolve_template, resolve_ask_query
# ── PIPELINE IMPORTS (same as test_pipeline.py) ────────────────────────────
from router.router import route, _is_kg_answerable
from router.classifier import _is_kg_answerable

from pipeline.language import detect_language
from pipeline.extractor import (
    extract_entities, validate_extraction, is_flight_question,
    extract_airport_entities, validate_airport_extraction,
    extract_university_entities, validate_university_extraction,
)
from pipeline.mapper import (
    get_university_entity_type, load_lexicon, map_property_cascade,
    map_flight, map_airport, map_university_entity,
)
from pipeline.generator import inject_and_generate, generate_open_kg_sparql
from pipeline.executor import (
    validate_sparql, execute_sparql, format_answer, format_answer_list,
    SURFACE_CODES,
)
from cross_kg_resolver import resolve_cross_kg
from kg_registry import get_base_uri, get_endpoint, get_lexicon, get_open_kg_schema


# ── SCORING HELPERS ─────────────────────────────────────────────────────────
def _internet_reachable(host="8.8.8.8", port=53, timeout=1.0) -> bool:
    """Fast external-connectivity check, logged at the moment of a
    failure so causation is evidenced directly instead of inferred
    from a before/after rerun."""
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except OSError:
        return False
def _strip_trailing_note(text) -> str:
    """Drops a trailing clarifying note in parentheses, e.g.
    '(complete — Greece only has 2)' -- it's metadata, not data.
    Applied once, up front, so both the list-comparison path and the
    plain-string comparison path benefit -- previously this only ran
    inside _split_list_answer(), so a gold string with a trailing note
    but fewer than 2 commas (e.g. a 2-item list joined by a single
    comma) skipped list-detection and compared with the note still
    attached, silently failing exact_match/F1 for a correct answer."""
    if text is None:
        return text
    return re.sub(r"\s*\([^)]*\)\s*$", "", str(text).strip())


def _normalise_for_scoring(text) -> str:
    if text is None:
        return ""
    text = _strip_trailing_note(text)
    text = str(text).strip().lower()
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _load_airport_code_lookup(ttl_path=None):
    """
    Builds a {airport name (lower-cased only) → IATA code} lookup from
    OUR OWN KG2 ontology (58 airports) — not the global airports.csv,
    which has 9006 rows and 49 duplicate names mapping to different IATA
    codes. Using the KG's own 58 airports guarantees no ambiguity.

    Keys are lower-cased only, NOT run through _normalise_for_scoring().
    Full normalisation replaces punctuation (hyphens, en-dashes) with
    spaces, so a name like "Catania-Fontanarossa Airport" would become
    the key "catania fontanarossa airport" — a space where the real
    pipeline text still has a hyphen. That space-vs-hyphen mismatch
    means the regex substitution in _canonicalize_airport_names() never
    matches, silently failing for every hyphenated/en-dashed name (7 of
    the 58 airports: Catania-Fontanarossa, Luxembourg-Findel,
    Falcone–Borsellino, Stockholm-Arlanda, Josep Tarradellas
    Barcelona-El Prat, Paris-Orly, Rome–Fiumicino). Lower-casing alone
    is enough since matching is already case-insensitive
    (re.IGNORECASE) in _canonicalize_airport_names().
    """
    if ttl_path is None:
        ttl_path = _AIRPORT_TTL_PATH
    g = Graph()
    g.parse(ttl_path, format="turtle")
    lookup = {}
    query = """
    PREFIX ao: <http://www.semanticweb.org/ontologies/airport_ontology#>
    SELECT ?name ?code WHERE {
        ?airport ao:airportName ?name ;
                 ao:iataCode ?code .
    }
    """
    for row in g.query(query):
        lookup[str(row.name).strip().lower()] = str(row.code).strip().upper()
    return lookup


AIRPORT_CODE_LOOKUP = _load_airport_code_lookup()


def _canonicalize_airport_names(text: str) -> str:
    """
    Replaces full airport names with their IATA code inside a text blob,
    so scoring can compare 'Esenboğa International Airport' against the
    ground truth's 'ESB' as the same entity.

    Substitutes directly on the ORIGINAL text (case-insensitive), rather
    than through _normalise_for_scoring() first — that would collapse
    every newline/comma into a single space, which destroys the list
    structure that _split_list_answer() needs to detect multi-item
    answers (e.g. ranking_kg2's "Name1, val1\nName2, val2"). Longer
    names are substituted first so a short name that's a substring of a
    longer one (rare, but possible) never partially matches first.
    """
    if not text:
        return text
    for name, code in sorted(AIRPORT_CODE_LOOKUP.items(), key=lambda kv: -len(kv[0])):
        text = re.sub(re.escape(name), code, text, flags=re.IGNORECASE)
    return text


def _canonicalize_university_names(text: str) -> str:
    """
    Shortens LUBM URI-derived names like 'www.University0.edu' or
    'www.Department0.University0.edu' down to the bare entity name
    ('University0', 'Department0'), so scoring can match ground truth
    that uses the short form.

    Matches on the naming pattern itself, not a category list — it can
    only fire on text shaped like www.<Name><Digits>....edu, so it
    can't misfire on unrelated answers.

    Guards against None/empty the same way _canonicalize_airport_names
    does — scored_value_for_match is None whenever a run failed and
    produced no answer at all, and re.sub crashes on None otherwise.
    """
    if not text:
        return text
    return re.sub(r'www\.([A-Za-z]+\d+)(?:\.[A-Za-z]+\d+)*\.edu',
                  r'\1', text, flags=re.IGNORECASE)


def _canonicalize_surface_code(expected):
    """
    Some single_kg2 runway-surface questions have expected_answer set to
    the RAW ontology surface code ("ASP") rather than the decoded value
    the pipeline actually returns ("Asphalt") via SURFACE_CODES in
    pipeline/executor.py (imported above, so this stays in sync with
    the pipeline's own mapping instead of duplicating it).

    Unlike _canonicalize_airport_names, this is applied to the GOLD
    side, not the predicted side. Several codes decode to the same
    value (ASP, ASPH, PEM -> "Asphalt"; CON, CONC -> "Concrete"), so the
    code -> value direction is unambiguous, but value -> code is not —
    canonicalizing the predicted "Asphalt" back into "a" code would mean
    guessing which of three codes was the "real" one. Decoding the gold
    code instead avoids that guess entirely.

    Only fires when expected_answer is literally a known surface code
    (case-insensitive) — any other expected_answer passes through
    unchanged, so this can't misfire on unrelated categories/questions.
    """
    if expected is None:
        return expected
    expected_str = str(expected).strip()
    return SURFACE_CODES.get(expected_str.upper(), expected)


def exact_match(predicted, expected) -> bool | None:
    if expected is None or (isinstance(expected, float) and pd.isna(expected)):
        return None
    expected_list = _split_list_answer(expected)
    if expected_list is not None:
        predicted_list = _split_list_answer(predicted) or [_normalise_for_scoring(predicted)]
        return list_exact_match(predicted_list, expected_list)
    return _normalise_for_scoring(predicted) == _normalise_for_scoring(expected)


def _split_list_answer(text) -> list[str] | None:
    """
    Turns a list-style answer into a normalised list of primary items.
    Returns None if the text doesn't look like a list at all.
    """
    if text is None:
        return None
    text = str(text).strip()

    # Drop a trailing clarifying note in parentheses, e.g.
    # "(top 10 by gspeed, highest first)" — it's metadata, not data.
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text)

    # Gold answers stored as a Python-list literal ("['x']", "['x', 'y']")
    # — parse as a real list regardless of item count. Doing this before
    # the comma-count check below matters: a single-item literal has 0
    # commas and would otherwise be compared as a raw string, brackets
    # and quotes included, against the plain predicted value.
    # Some spreadsheet cells wrap the whole literal in one extra layer
    # of quotes (e.g. the cell content is literally `"['x']"`, quotes
    # included) — strip one such layer before checking, so this works
    # regardless of whether that extra wrapping is present.
    list_candidate = text
    if (len(list_candidate) >= 2
            and list_candidate[0] == list_candidate[-1]
            and list_candidate[0] in ("'", '"')):
        list_candidate = list_candidate[1:-1].strip()

    if list_candidate.startswith("[") and list_candidate.endswith("]"):
        try:
            parsed = ast.literal_eval(list_candidate)
            if isinstance(parsed, list):
                return [_normalise_for_scoring(str(p)) for p in parsed if str(p).strip()]
        except (ValueError, SyntaxError):
            pass  # not a valid literal — fall through to normal handling

    # A "list" is either multiple lines, or a single line with 2+ commas.
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) < 2 and text.count(",") < 2:
        return None  # single value, not a list

    items = lines if len(lines) >= 2 else text.split(",")

    # Each item may carry a secondary value after its own comma
    # ("ESB, 3125.00") — keep only the first field, the identifier.
    primary = [item.split(",")[0].strip() for item in items]
    return [_normalise_for_scoring(p) for p in primary if p.strip()]


def _strip_value_suffix_if_gold_is_bare(predicted, expected) -> str:
    """
    Handles single-item ranking answers (e.g. ranking_kg2 with limit=1),
    where the pipeline correctly returns "name, value" (matching the
    multi-row ranking format), but the dataset's expected_answer is a
    bare identifier only (e.g. "STR"), since a single superlative answer
    doesn't need its value spelled out to be verified as correct.

    Without this, "STR, 98.00" is compared as one whole string against
    "STR" and fails exact_match outright, even though the identifier
    portion is correct — the same class of format mismatch already
    handled separately for compare_two_airports and ask_query.
    """
    if predicted is None or expected is None:
        return predicted
    expected_str = str(expected).strip()
    if "," not in expected_str and isinstance(predicted, str) and "," in predicted:
        return predicted.split(",")[0].strip()
    return predicted


def list_exact_match(predicted: list[str], expected: list[str]) -> bool:
    return set(predicted) == set(expected)


def list_f1(predicted: list[str], expected: list[str]) -> float:
    if not predicted or not expected:
        return 0.0
    overlap = len(set(predicted) & set(expected))
    if overlap == 0:
        return 0.0
    precision = overlap / len(set(predicted))
    recall = overlap / len(set(expected))
    return round(2 * precision * recall / (precision + recall), 4)


def token_f1(predicted, expected) -> float | None:
    if expected is None or (isinstance(expected, float) and pd.isna(expected)):
        return None
    expected_list = _split_list_answer(expected)
    if expected_list is not None:
        predicted_list = _split_list_answer(predicted) or [_normalise_for_scoring(predicted)]
        return list_f1(predicted_list, expected_list)
    pred_tokens = _normalise_for_scoring(predicted).split()
    gold_tokens = _normalise_for_scoring(expected).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    gold_counts = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    for t, c in gold_counts.items():
        overlap += min(c, common.get(t, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return round(2 * precision * recall / (precision + recall), 4)


# ── BRANCH HANDLERS (strategy-parameterised versions of main.py's branches) ─

_DEFAULT_STAGES = frozenset({"pre-norm", "exact", "fuzzy", "semantic"})


def _run_single_kg1(question, entity_hint, strategy, lang, stages=_DEFAULT_STAGES):
    out = {"sparql": None, "sparql_valid": False, "raw_answer": None,
           "final_answer": None, "failure_type": "not_run"}
    entities = extract_entities(question, lang)
    if not validate_extraction(entities) or not is_flight_question(entities):
        out["failure_type"] = "extraction_failure"
        return out
    lexicon = load_lexicon(get_lexicon("flights"))
    property_uri, _, property2_uri = map_property_cascade(
        entities["property"], lexicon, get_lexicon("flights"), stages=stages)
    flight_uri = map_flight(entities["entity"])
    if not flight_uri or not property_uri:
        out["failure_type"] = "mapping_failure"
        return out
    base = get_base_uri("flights")
    full_prop = base + property_uri
    full_prop2 = (base + property2_uri) if property2_uri else None
    sparql = inject_and_generate(flight_uri, full_prop, question,
                                  strategy=strategy, property2_uri=full_prop2)
    out["sparql"] = sparql
    is_valid = (validate_sparql(sparql) and sparql.strip().startswith("SELECT")
                and "PREFIX" not in sparql and full_prop in sparql)
    if full_prop2:
        is_valid = is_valid and (full_prop2 in sparql)
    out["sparql_valid"] = is_valid
    if is_valid:
        result = execute_sparql(sparql, endpoint=get_endpoint("flights"))
        out["raw_answer"] = result["value"]
        if result["error"] is not None:
            out["failure_type"] = "execution_failure"
            out["error_detail"] = result["error"]
        elif result["value"]:
            out["final_answer"] = format_answer(question, result["value"], lang)
            out["failure_type"] = "success"
        else:
            out["failure_type"] = "no_results"
    else:
        out["failure_type"] = "generation_failure"
    return out


def _run_single_kg2(question, entity_hint, strategy, lang, stages=_DEFAULT_STAGES):
    out = {"sparql": None, "sparql_valid": False, "raw_answer": None,
           "final_answer": None, "failure_type": "not_run"}
    entities = extract_airport_entities(question, lang, entity_hint)
    if not validate_airport_extraction(entities):
        out["failure_type"] = "extraction_failure"
        return out
    lexicon_path = get_lexicon("airports")
    lexicon = load_lexicon(lexicon_path)
    property_uri, _, property2_uri = map_property_cascade(
        entities["property"], lexicon, lexicon_path, stages=stages)
    airport_uri = map_airport(entities["entity"]) if entities["entity"] else None
    if not airport_uri or not property_uri:
        out["failure_type"] = "mapping_failure"
        out["error_detail"] = (
            f"entity={entities.get('entity')!r} -> airport_uri={airport_uri!r} | "
            f"property_text={entities.get('property')!r} -> property_uri={property_uri!r}"
        )
        return out
    base = get_base_uri("airports")
    full_prop = base + property_uri
    full_prop2 = (base + property2_uri) if property2_uri else None
    sparql = inject_and_generate(airport_uri, full_prop, question,
                                  strategy=strategy, property2_uri=full_prop2)
    out["sparql"] = sparql
    is_valid = (validate_sparql(sparql) and sparql.strip().startswith("SELECT")
                and "PREFIX" not in sparql and full_prop in sparql)
    out["sparql_valid"] = is_valid
    if is_valid:
        result = execute_sparql(sparql, endpoint=get_endpoint("airports"))
        out["raw_answer"] = result["value"]
        if result["error"] is not None:
            out["failure_type"] = "execution_failure"
            out["error_detail"] = result["error"]
        elif result["value"]:
            out["final_answer"] = format_answer(question, result["value"], lang)
            out["failure_type"] = "success"
        else:
            out["failure_type"] = "no_results"
    else:
        out["failure_type"] = "generation_failure"
    return out


def _run_single_kg3(question, routing, strategy, lang, stages=_DEFAULT_STAGES):
    """
    Branch: university entity is known (routing["entity"]).
    Mirrors test_pipeline.py's _run_single_kg3 — added here because the
    original eval_runner.py dispatch table was missing this case entirely,
    which meant any property_ambiguity or single-entity KG3 question would
    have silently fallen to 'unhandled_branch' instead of actually running.
    """
    out = {"sparql": None, "sparql_valid": False, "raw_answer": None,
           "final_answer": None, "failure_type": "not_run"}
    entities = extract_university_entities(question, lang, routing["entity"])
    if not validate_university_extraction(entities):
        out["failure_type"] = "extraction_failure"
        return out
    lexicon_path = get_lexicon("university")
    lexicon = load_lexicon(lexicon_path)
    property_uri, _, property2_uri = map_property_cascade(
        entities["property"], lexicon, lexicon_path, stages=stages)
    entity_uri = map_university_entity(entities["entity"]) if entities["entity"] else None

    FACULTY_TYPES = {"FullProfessor", "AssociateProfessor", "AssistantProfessor", "Lecturer"}

    if entity_uri and property_uri in ("memberOf", "subOrganizationOf"):
        entity_type = get_university_entity_type(entity_uri)
        if entity_type == "Department" and property_uri == "memberOf":
            property_uri = "subOrganizationOf"
            print(f"[disambiguation] Department entity — corrected memberOf → subOrganizationOf")
        elif entity_type != "Department" and property_uri == "subOrganizationOf":
            property_uri = "memberOf"
            print(f"[disambiguation] Non-department entity — corrected subOrganizationOf → memberOf")
        elif entity_type in FACULTY_TYPES and property_uri == "memberOf":
            property_uri = "worksFor"
            print(f"[disambiguation] Faculty entity ({entity_type}) — corrected memberOf → worksFor")

    if not entity_uri or not property_uri:
        out["failure_type"] = "mapping_failure"
        return out
    base = get_base_uri("university")
    full_prop = base + property_uri
    full_prop2 = (base + property2_uri) if property2_uri else None
    sparql = inject_and_generate(entity_uri, full_prop, question,
                                  strategy=strategy, property2_uri=full_prop2)
    out["sparql"] = sparql
    is_valid = (validate_sparql(sparql) and sparql.strip().startswith("SELECT")
                and "PREFIX" not in sparql and full_prop in sparql)
    out["sparql_valid"] = is_valid
    if is_valid:
        result = execute_sparql(sparql, endpoint=get_endpoint("university"), multiple=True)
        out["raw_answer"] = result["value"]
        if result["error"] is not None:
            out["failure_type"] = "execution_failure"
            out["error_detail"] = result["error"]
        elif result["value"]:
            out["final_answer"] = format_answer_list(question, result["value"], lang)
            out["failure_type"] = "success"
        else:
            out["failure_type"] = "no_results"
    else:
        out["failure_type"] = "generation_failure"
    return out


def _run_cross_kg(question, routing, lang, stages=_DEFAULT_STAGES):
    out = {"sparql": None, "sparql_valid": False, "raw_answer": None,
           "final_answer": None, "failure_type": "not_run"}
    flight_uri = map_flight(routing["entity"])
    if not flight_uri:
        out["failure_type"] = "mapping_failure"
        return out
    entities = extract_airport_entities(question, lang, iata_from_router=None)
    lexicon_path = get_lexicon("airports")
    lexicon = load_lexicon(lexicon_path)
    property_uri, _, property2_uri = map_property_cascade(entities["property"], lexicon, lexicon_path, stages=stages)
    if not property_uri:
        out["failure_type"] = "mapping_failure"
        return out
    full_prop = get_base_uri("airports") + property_uri
    full_prop2 = (get_base_uri("airports") + property2_uri) if property2_uri else None
    result = resolve_cross_kg(flight_uri=flight_uri, direction=routing["direction"],
                               property_uri=full_prop, property_short=property_uri,
                               property2_uri=full_prop2)
    out["raw_answer"] = result.get("raw_value")
    out["failure_type"] = result.get("failure_type")
    out["sparql_valid"] = result.get("success", False)
    if result["success"]:
        out["final_answer"] = format_answer(question, result["raw_value"], lang)
        out["failure_type"] = "success"
    return out


def _run_open_kg(question, lang):
    out = {"sparql": None, "sparql_valid": False, "raw_answer": None,
           "final_answer": None, "failure_type": "not_run"}
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
    result = execute_sparql(sparql, endpoint=endpoint, multiple=True)
    out["raw_answer"] = result["value"]
    if result["error"] is not None:
        out["failure_type"] = "execution_failure"
        out["error_detail"] = result["error"]
    elif result["value"]:
        out["final_answer"] = format_answer_list(question, result["value"], lang)
        out["failure_type"] = "success"
    else:
        out["failure_type"] = "no_results"
    return out

def _run_template(question, routing, lang):
    tr = resolve_template(question, routing["template"], lang, router_params=routing.get("params"))
    out = {
        "sparql": tr.get("sparql"), "sparql_valid": tr.get("success", False),
        "raw_answer": tr.get("raw_data"), "final_answer": tr.get("final_answer"),
        "failure_type": tr.get("failure_type"),
    }
    if not tr.get("success"):
        out["error_detail"] = f"params={tr.get('params')!r}"
    return out


def _run_ask_query(question, routing, lang):
    ar = resolve_ask_query(question, routing, lang)
    return {
        "sparql": ar.get("sparql"), "sparql_valid": ar.get("success", False),
        "raw_answer": ar.get("raw_answer"), "final_answer": ar.get("final_answer"),
        "failure_type": ar.get("failure_type"),
    }


def _dispatch(question, lang, strategy):
    """Routes the question, then calls the matching branch handler."""

    # Baseline A/B: bypass normal routing entirely and force the
    # schema-guided open_kg generator. main() already scopes the
    # dataframe to the right question population (BASELINE_A_CATEGORIES /
    # BASELINE_B_CATEGORIES) before _dispatch is ever called in this mode,
    # so there's no need to route — every question here should go through
    # the no-injection generator regardless of what normal routing would say.
    if BASELINE_MODE in ("A", "B"):
        out = _run_open_kg(question, lang)
        out["query_type"] = "open_kg"
        return out

    routing = route(question)
    query_type = routing["query_type"]

    # Ablation: keep normal routing, but restrict which cascade tiers
    # map_property_cascade() is allowed to use, in whichever branch
    # handler ends up being called below. Every other mode (None) passes
    # _DEFAULT_STAGES, which is a no-op — identical to not passing
    # stages= at all.
    ablation_stages = ABLATION_STAGES if BASELINE_MODE == "ablation" else _DEFAULT_STAGES

    if query_type == "single_kg1":
        out = _run_single_kg1(question, routing["entity"], strategy, lang, stages=ablation_stages)
    elif query_type == "single_kg2":
        out = _run_single_kg2(question, routing["entity"], strategy, lang, stages=ablation_stages)
    elif query_type == "single_kg3":
        out = _run_single_kg3(question, routing, strategy, lang, stages=ablation_stages)
    elif query_type == "cross_kg":
        out = _run_cross_kg(question, routing, lang, stages=ablation_stages)
    elif query_type == "open_kg":
        out = _run_open_kg(question, lang)
    elif query_type == "template":
        out = _run_template(question, routing, lang)
    elif query_type == "ask_query":
        out = _run_ask_query(question, routing, lang)
        # Fallback: a mapping_failure on an ask_query often means the
        # question was never really about a KG property at all. The
        # Arabic 'هل' fast-path (router.py, Priority 1.5) routes any
        # yes/no-structured question with a known entity straight to
        # ask_query, with no scope check at that point — unlike the
        # airport branch (Priority 2.5), which does gate on
        # _is_kg_answerable() before committing. Re-check scope here,
        # only in the failure case, so this doesn't add an extra LLM
        # call to every ask_query — only to the ones that already failed.
        if out["failure_type"] == "mapping_failure" and not _is_kg_answerable(question):
            print(f"[dispatch] ask_query mapping failed and question isn't KG-answerable — reclassifying as out_of_scope")
            query_type = "out_of_scope"
            out = {"sparql": None, "sparql_valid": False, "raw_answer": None,
                   "final_answer": None, "failure_type": "success"}
    else:  # out_of_scope or anything unrecognised
        out = {"sparql": None, "sparql_valid": False, "raw_answer": None,
               "final_answer": None,
               "failure_type": "success" if query_type == "out_of_scope" else "unhandled_branch"}

    out["query_type"] = query_type
    return out


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_excel(DATASET_PATH, sheet_name="Questions")

    if BASELINE_MODE == "A":
        df = df[df["category"].isin(BASELINE_A_CATEGORIES)]
    elif BASELINE_MODE == "B":
        df = df[df["category"].isin(BASELINE_B_CATEGORIES)]
    elif BASELINE_MODE == "ablation":
        df = df[df["category"].isin(BASELINE_A_CATEGORIES)]
    elif not FULL_RUN:
        df = df[df["id"].isin(BROKEN_IDS)]           # only re-run what broke

    old_good_rows = []
    # Baseline/ablation runs always start fresh — no incremental
    # preservation, since baseline_results.jsonl could otherwise end up
    # mixing rows from different ABLATION_STAGES configs in one file.
    if BASELINE_MODE is None and not FULL_RUN and os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec["id"] not in BROKEN_IDS:    # keep everything that already passed
                    old_good_rows.append(line)

    out_f = open(OUTPUT_PATH, "w", encoding="utf-8")
    for line in old_good_rows:
        out_f.write(line)

    total_runs = 0
    skipped = 0

    for _, row in df.iterrows():
        # Baseline/ablation tracks are about the generation approach or
        # the mapping cascade, not prompting strategy — running them
        # across all 3 strategies would conflate two different
        # comparisons, so force zero-shot only.
        if BASELINE_MODE is not None:
            strategies = ["zero-shot"]
        else:
            strategies = STRATEGIES if bool(row["strategy_applicable"]) else ["zero-shot"]
        for lang in LANGUAGES:
            question = row.get(f"question_{lang}")
            if pd.isna(question) or not str(question).strip():
                skipped += len(strategies)
                continue

            for strategy in strategies:
                t0 = time.time()
                try:
                    result = _dispatch(str(question).strip(), lang, strategy)
                except Exception as e:
                    result = {"query_type": None, "sparql": None, "sparql_valid": False,
                              "raw_answer": None, "final_answer": None,
                              "failure_type": "exception", "error_detail": str(e)}

                expected_type = row["expected_type"]
                routing_ok = (expected_type == "VARIES") or (result["query_type"] == expected_type)

                # Fix #1: score the raw extracted value, not the natural-
                # language sentence — comparing "The departure city of
                # flight OS295 is Vienna." against gold "Vienna" destroys
                # exact-match and dilutes F1 with sentence filler tokens.
                # EXCEPTION: compare_two_airports is scored on final_answer,
                # because that's the branch where we specifically built
                # final_answer to BE the gold-matching comparison sentence
                # ("NAP is higher (BLQ: 123, NAP: 294)") — raw_answer there
                # is the unformatted SPARQL row dump, not a scoring target.
                # ask_query is scored separately below on its own boolean
                # raw_answer, bypassing this branch entirely.
                if row["category"] == "compare_two_airports":
                    scored_value = result.get("final_answer")
                else:
                    scored_value = result.get("raw_answer")
                if isinstance(scored_value, list):
                    scored_value = "\n".join(str(v) for v in scored_value)
                elif isinstance(scored_value, bool):
                    scored_value = "yes" if scored_value else "no"
                # Fix #3: ranking_kg2 / filter_numeric_kg2 ground truth uses
                # IATA codes ("ESB"), the pipeline outputs full names
                # ("Esenboğa International Airport") — same entity, different
                # identifier. Canonicalize before comparing, but ONLY for
                # these two categories — applying it everywhere (previous
                # bug) ran every answer through it regardless of whether it
                # needed it, and the canonicalization itself used to funnel
                # through full normalisation first, destroying the comma/
                # newline structure list-scoring depends on. Both are fixed
                # here: scoped to just these 2 categories, and substituting
                # directly on the original text instead.
                #
                # Fix #5: single-item ranking answers (limit=1) come back
                # from the pipeline as "name, value" but the gold answer is
                # a bare identifier only ("STR") — strip the value suffix
                # before comparing so the identifier still matches exactly.
                if row["category"] in ("ranking_kg2", "filter_numeric_kg2"):
                    scored_value_for_match = _canonicalize_airport_names(scored_value)
                    scored_value_for_match = _strip_value_suffix_if_gold_is_bare(
                        scored_value_for_match, row.get("expected_answer")
                    )
                else:
                    scored_value_for_match = scored_value
                # Fix #4: LUBM entity answers ("www.University0.edu") vs.
                # ground truth's short form ("University0") — same entity,
                # different identifier, same shape of problem as Fix #3.
                # Pattern-matched, not category-scoped (see function
                # docstring), and guarded against None since a failed run's
                # scored_value_for_match is None here.
                scored_value_for_match = _canonicalize_university_names(scored_value_for_match)

                # Fix #6: some single_kg2 questions store the raw ontology
                # surface code as expected_answer ("ASP") while the pipeline
                # returns the decoded value ("Asphalt"). Canonicalize the
                # GOLD side (see _canonicalize_surface_code docstring for
                # why not the predicted side). Not category-scoped: it only
                # fires when expected_answer is literally a known code, so
                # it's safe to apply on every row.
                expected_for_match = _canonicalize_surface_code(row.get("expected_answer"))

                # Fix #2: ask_query ground truth is English-only ("Yes."/"No."),
                # but final_answer is localized ("Non.", "لا."). A yes/no
                # answer is a boolean and has no language — compare booleans,
                # not translated text.
                if row["category"] == "ask_query":
                    expected_bool = _normalise_for_scoring(row.get("expected_answer")).startswith("yes")
                    predicted_bool = result.get("raw_answer")
                    if predicted_bool is None:
                        ask_exact, ask_f1 = None, None
                    else:
                        ask_exact = (bool(predicted_bool) == expected_bool)
                        ask_f1 = 1.0 if ask_exact else 0.0

                record = {
                    "id": row["id"],
                    "tier": int(row["tier"]),
                    "category": row["category"],
                    "kg": row["kg"],
                    "language": lang,
                    "strategy": strategy,
                    "expected_type": expected_type,
                    "query_type": result["query_type"],
                    "routing_ok": routing_ok,
                    "sparql": result["sparql"],
                    "sparql_valid": result["sparql_valid"],
                    "raw_answer": result["raw_answer"],
                    "final_answer": result["final_answer"],
                    "failure_type": result["failure_type"],
                    "error_detail": result.get("error_detail"),
                    "exact_match": ask_exact if row["category"] == "ask_query"
                                   else exact_match(scored_value_for_match, expected_for_match),
                    "f1": ask_f1 if row["category"] == "ask_query"
                          else token_f1(scored_value_for_match, expected_for_match),
                    "duration_s": round(time.time() - t0, 2),
                }
                # Fix #7: _dispatch() marks any out_of_scope routing as
                # "success" by design — a question the pipeline decides
                # isn't KG-answerable is correctly resolved. That's only
                # true when out_of_scope really was the expected outcome.
                # When an in-scope question (expected_type != out_of_scope)
                # gets misrouted to out_of_scope, no answer is produced —
                # that's a routing miss, not a success. Pattern-matched on
                # "no answer + wrongly-expected out_of_scope", not on any
                # specific question, so it applies to every future row.
                if (record["failure_type"] == "success"
                        and record["raw_answer"] is None
                        and expected_type != "out_of_scope"):
                    record["failure_type"] = "routing_miss"

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_runs += 1
                print(f"[{total_runs}] {row['id']} | {lang} | {strategy} | "
                      f"{result['failure_type']}")

    out_f.close()
    mode_label = "FULL RUN (all 72 questions, old results discarded)" if FULL_RUN \
        else "INCREMENTAL (only BROKEN_IDS re-run, old results preserved)"
    print(f"\nDone. Mode: {mode_label}")
    print(f"{total_runs} runs written to {OUTPUT_PATH}. "
          f"{skipped} runs skipped (empty question cells — dataset not fully filled in yet).")


if __name__ == "__main__":
    main()