"""
test_broken_rows.py
--------------------
Targeted diagnostic test — NOT a replacement for eval_runner.py.

Two things this script does that your original snippet didn't:

1. RUNS EACH ROUTING CASE N_RUNS TIMES, not once.
   _is_kg_answerable() and _has_ask_signal() are live LLM calls with no
   temperature control, so a single call can pass or fail by chance.
   A "PASS" on one run tells you nothing about reliability — running it
   5x and reporting a percentage does.

2. FOR ask_query ROWS, GOES ONE LAYER DEEPER THAN route().
   route() only tells you if classification landed on "ask_query" — it
   says nothing about whether the property phrase then maps to a real
   KG2 property. Since your last eval showed rows that ROUTE correctly
   but still fail downstream (mapping_failure), this script also calls
   extract_ask_entities() + map_property_cascade() directly and prints
   exactly what was extracted and what the cascade resolved to (or
   didn't). This is the diagnostic we need before deciding whether the
   Arabic ask_query failures are an extraction problem or a matching
   problem — no guessing, just print what actually happened.

Run this locally (it needs your live Fuseki/ollama stack, same as
eval_runner.py normally does).
"""

from collections import Counter

from router import route
from pipeline.extractor import extract_ask_entities, validate_ask_extraction
from pipeline.mapper import load_lexicon, map_property_cascade
from kg_registry import get_lexicon

N_RUNS = 5  # repeat each routing case this many times to catch LLM non-determinism


# ── PART 1: ROUTING STABILITY ────────────────────────────────────────────────
# Includes the 3 out_of_scope rows (previously confirmed unstable) and the
# ask_query rows that showed misrouting (French) or downstream failure (Arabic).

ROUTING_CASES = [
    # out_of_scope — previously flaky even with the answerability-gate fix
    ("What's the weather like at VIE airport today?",              "out_of_scope"),
    ("Quel temps fait-il à l'aéroport VIE aujourd'hui?",            "out_of_scope"),
    ("كيف حال الطقس في مطار VIE اليوم؟",                            "out_of_scope"),
    ("Can I bring a pet on flight FR947?",                          "out_of_scope"),
    ("Puis-je emmener un animal de compagnie sur le vol FR947?",    "out_of_scope"),
    ("هل يمكنني اصطحاب حيوان أليف في الرحلة FR947؟",                "out_of_scope"),
    ("What is the history of the airline industry?",                "out_of_scope"),

    # ask_query — the rows that failed last run (routing OR downstream mapping)
    ("Is KRK located in Poland?",                    "ask_query"),
    ("KRK est-il situé en Pologne?",                 "ask_query"),
    ("هل يقع مطار KRK في بولندا؟",                    "ask_query"),
    ("Is BLQ located in France?",                    "ask_query"),
    ("BLQ est-il situé en France?",                  "ask_query"),
    ("هل يقع مطار BLQ في فرنسا؟",                     "ask_query"),
    ("Is LUX's runway surface concrete?",            "ask_query"),
    ("La surface de piste de LUX est-elle en béton?", "ask_query"),
    ("هل سطح مدرج مطار LUX من الخرسانة؟",             "ask_query"),

    # controls — confirm we haven't broken anything real
    ("What is the elevation of VIE?",                "single_kg2"),
    ("What country does flight LO225 land in?",      "cross_kg"),
    ("What is the gate of flight OS529?",            "single_kg1"),
]


def run_routing_stability():
    print("=" * 78)
    print(f"PART 1 — ROUTING STABILITY  ({N_RUNS} runs per question)")
    print("=" * 78)

    rows = []
    for question, expected in ROUTING_CASES:
        outcomes = Counter()
        for _ in range(N_RUNS):
            result = route(question)
            outcomes[result["query_type"]] += 1

        pass_rate = outcomes.get(expected, 0) / N_RUNS
        flag = "OK  " if pass_rate == 1.0 else ("FLAKY" if pass_rate > 0 else "FAIL")
        rows.append((flag, pass_rate, expected, outcomes, question))

    for flag, pass_rate, expected, outcomes, question in rows:
        outcomes_str = ", ".join(f"{k}×{v}" for k, v in outcomes.items())
        print(f"[{flag}] {pass_rate*100:5.0f}%  expected={expected:12s} "
              f"got={{{outcomes_str}}}")
        print(f"        {question}")

    n_flaky = sum(1 for f, *_ in rows if f == "FLAKY")
    n_fail = sum(1 for f, *_ in rows if f == "FAIL")
    print()
    print(f"Summary: {len(rows) - n_flaky - n_fail} stable OK, "
          f"{n_flaky} flaky (non-deterministic), {n_fail} consistently wrong.")
    print()


# ── PART 2: ASK_QUERY DOWNSTREAM MAPPING ─────────────────────────────────────
# For every ask_query row, regardless of what Part 1 found, extract + map
# directly so we see exactly where the Arabic ones break.

ASK_QUERY_MAPPING_CASES = [
    # (question, lang, entity_from_router)
    ("Is KRK located in Poland?",                     "en", "KRK"),
    ("KRK est-il situé en Pologne?",                  "fr", "KRK"),
    ("هل يقع مطار KRK في بولندا؟",                     "ar", "KRK"),
    ("Is BLQ located in France?",                     "en", "BLQ"),
    ("BLQ est-il situé en France?",                   "fr", "BLQ"),
    ("هل يقع مطار BLQ في فرنسا؟",                      "ar", "BLQ"),
    ("Is LUX's runway surface concrete?",             "en", "LUX"),
    ("La surface de piste de LUX est-elle en béton?", "fr", "LUX"),
    ("هل سطح مدرج مطار LUX من الخرسانة؟",              "ar", "LUX"),
]


def run_ask_query_mapping():
    print("=" * 78)
    print("PART 2 — ask_query DOWNSTREAM MAPPING (extraction + lexicon cascade)")
    print("=" * 78)

    lexicon_path = get_lexicon("airports")
    lexicon = load_lexicon(lexicon_path)

    for question, lang, entity in ASK_QUERY_MAPPING_CASES:
        entities = extract_ask_entities(question, lang, entity)
        extraction_ok = validate_ask_extraction(entities)

        property_uri, tier, property2_uri = (None, None, None)
        if extraction_ok:
            property_uri, tier, property2_uri = map_property_cascade(
                entities["property"], lexicon, lexicon_path
            )

        status = "OK  " if property_uri else ("EXTRACT_FAIL" if not extraction_ok else "MAP_FAIL")
        print(f"[{status}] lang={lang}")
        print(f"    question       : {question}")
        print(f"    extracted prop : {entities.get('property')!r}")
        print(f"    extracted value: {entities.get('value')!r}")
        print(f"    extraction_ok  : {extraction_ok}")
        print(f"    cascade result : property_uri={property_uri!r} tier={tier!r} "
              f"property2_uri={property2_uri!r}")
        print()


if __name__ == "__main__":
    run_routing_stability()
    run_ask_query_mapping()