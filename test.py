"""
test_llm_variance.py
---------------------
Measures RAW single-call sampling variance for the two yes/no prompts that
_llm_yes_no_majority() was built to compensate for: _has_ask_signal() and
_is_kg_answerable() in router.py.

WHY THIS TESTS THE RAW PROMPT, NOT _has_ask_signal()/_is_kg_answerable()
DIRECTLY:
    Those functions already call _llm_yes_no_majority(), which does k=3
    calls internally and returns the MAJORITY vote. Calling the already-
    voted function N times mostly hides the noise the vote was designed
    to smooth over — you'd be testing "does the vote itself flip across
    separate votings", which is a valid but much less informative
    question than "how noisy is a single raw call". This script replicates
    the exact prompt text used inside router.py and calls ollama.chat()
    directly, once per sample, so the raw per-call disagreement rate is
    visible before any voting is applied.

WHY MULTIPLE QUESTIONS, NOT JUST THE ONE FLAKY CASE:
    "Is BLQ located in France?" is where 4-1 was observed once. Testing
    only that question risks a conclusion (stable or noisy) that doesn't
    generalize. This script also runs a question expected to be a clean
    YES and a clean NO, as controls, so a real noise finding on the
    borderline question can be contrasted against stable behavior
    elsewhere rather than assumed to be universal.

WHY RAW TEXT IS LOGGED, NOT JUST THE PARSED BOOLEAN:
    router.py parses responses via:
        response["message"]["content"].strip().upper().startswith("YES")
    If the model is actually consistent but sometimes replies "Yes." and
    sometimes "Yes, it does, since..." — that's a parsing fragility, not
    genuine semantic sampling noise. Logging the raw text lets you tell
    the two apart before concluding "the model is unstable."

PREREQUISITES: Ollama running locally with the 'llama3' model, same as
router.py's own runtime dependency. _is_kg_answerable's prompt also pulls
get_open_kg_schema() from kg_registry.py, which may require Fuseki to be
reachable — if that import fails, this script skips that half and still
runs the _has_ask_signal half standalone.

USAGE: python test_llm_variance.py [N]
    N = number of raw calls per question (default 10).
"""

import sys
import ollama

N_CALLS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
MODEL = "llama3"


# ── Exact prompt text, replicated from router.py (not re-derived) ──────────

def _ask_signal_prompt(question: str) -> str:
    return f"""Does this question ask to CONFIRM whether a specific
property already has a specific value (a yes/no question)? Or does it
ASK for information (what/which/how much)?

Examples:
Q: "Is BR62's callsign EVA062?" → YES
Q: "La porte du vol OS830 est-elle A17?" → YES
Q: "هل مطار فيينا يقع في النمسا؟" → YES
Q: "What is the callsign of BR62?" → NO
Q: "Quelle est la porte du vol OS830?" → NO
Q: "ما هو مطار الوصول؟" → NO
Q: "هل يقع مطار زيورخ في سويسرا؟" → YES
Q: "في أي دولة يقع مطار أثينا؟" → NO
Q: "ما هي إشارة النداء للرحلة TK500؟" → NO

Answer only YES or NO.

Question: "{question}"
"""


def _kg_answerable_prompt(question: str, schema_note: str) -> str:
    return f"""You are a scope classifier for an aviation knowledge graph system.
The question may be in English, French, or Arabic.

The knowledge graph contains:
- Flights: flight number, airline, origin city, destination city,
  aircraft type, gate, terminal, callsign, ground speed, vertical speed
- Airports: name, type, elevation, country, region, city,
  IATA code, ICAO code, coordinates
- Runways: length, width, surface, lighting, identifier

The knowledge graph does NOT contain: weather, prices/tickets, passenger
policies (pets, baggage, check-in), history, news, opinions, safety
records, or anything not explicitly listed above.

EXAMPLES:
Q: "What is the weather forecast for JFK tomorrow?"     → NO (weather not in KG)
Q: "Am I allowed to bring a guitar on flight BR62?"     → NO (policy, not in KG)
Q: "Who invented the first commercial airplane?"        → NO (general knowledge, not in KG)
Q: "Can I bring a pet on flight FR947?"           → NO (policy, not in KG)
Q: "What is the history of the airline industry?" → NO (general knowledge, not in KG)
Q: "Quel temps fait-il à l'aéroport VIE?"          → NO
Q: "هل يمكنني اصطحاب حيوان أليف؟"                  → NO
Q: "What is the elevation of ZRH?"                → YES (elevation is in KG)
Q: "Is ZRH located in Switzerland?"               → YES (country is in KG)
Q: "هل يقع مطار زيورخ في سويسرا؟"                  → YES
Q: "في أي دولة يقع مطار أثينا؟"                    → NO
Answer only YES or NO:
Can this question be answered using only the data described above?

Question: "{question}"
"""


# ── Raw single-call helper (no voting) ──────────────────────────────────────

def raw_call(prompt: str) -> tuple[str, bool]:
    """One un-voted call. Returns (raw_text, parsed_bool) — mirrors exactly
    how router.py parses a response, so any parsing fragility shows up
    here the same way it would in production."""
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    raw = response["message"]["content"]
    parsed = raw.strip().upper().startswith("YES")
    return raw, parsed


def run_variance_check(label: str, prompt: str, n: int = N_CALLS):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    votes = []
    raw_texts = []
    for i in range(n):
        try:
            raw, parsed = raw_call(prompt)
        except Exception as e:
            print(f"  call {i+1}: FAILED ({e})")
            continue
        votes.append(parsed)
        raw_texts.append(raw.strip())
        print(f"  call {i+1}: {'YES' if parsed else 'NO ':3} | raw: {raw.strip()[:60]!r}")

    if not votes:
        print("  No successful calls — skipping summary.")
        return

    yes_count = sum(votes)
    no_count = len(votes) - yes_count
    distinct_raw = len(set(raw_texts))

    print(f"\n  Summary: {yes_count} YES / {no_count} NO out of {len(votes)} calls")
    print(f"  Distinct raw response strings: {distinct_raw} "
          f"({'all identical text' if distinct_raw == 1 else 'wording varies'})")

    if yes_count > 0 and no_count > 0:
        print("  >>> SEMANTIC DISAGREEMENT CONFIRMED: parsed YES/NO actually flips "
              "across calls at temperature=0 — this is genuine sampling noise, "
              "not a parsing artifact.")
    elif distinct_raw > 1:
        print("  >>> Parsed result is stable, but raw wording varies — "
              "no evidence of semantic noise here; majority-vote isn't "
              "doing anything for this question.")
    else:
        print("  >>> Fully stable: identical output every call.")

    k3_majority = sum(votes[:3]) > 1 if len(votes) >= 3 else None
    if k3_majority is not None:
        print(f"  If only k=3 calls had been made (calls 1-3): "
              f"majority = {'YES' if k3_majority else 'NO'}")


if __name__ == "__main__":
    print(f"Running {N_CALLS} raw calls per question at temperature=0.\n")

    # ── _has_ask_signal prompt: 3 test questions ────────────────────────────
    run_variance_check(
        "_has_ask_signal — KNOWN FLAKY CASE (previously observed 4-1 split)",
        _ask_signal_prompt("Is BLQ located in France?"),
    )
    run_variance_check(
        "_has_ask_signal — CONTROL: clear ASK-style question (expect stable YES)",
        _ask_signal_prompt("Is LUX's runway surface concrete?"),
    )
    run_variance_check(
        "_has_ask_signal — CONTROL: clear non-ASK question (expect stable NO)",
        _ask_signal_prompt("What is the elevation of ZRH?"),
    )
    run_variance_check(
        "_has_ask_signal — English, TRUE fact (missing cell: language=EN, truth=TRUE)",
        _ask_signal_prompt("Is ZRH located in Switzerland?"),
    )
    run_variance_check(
        "_has_ask_signal — Arabic, FALSE fact (missing cell: language=AR, truth=FALSE)",
        _ask_signal_prompt("هل يقع مطار بولونيا في فرنسا؟"),
    )

    # ── _is_kg_answerable prompt: needs the real KG schema ──────────────────
    try:
        from kg_registry import get_open_kg_schema
        schema = get_open_kg_schema()  # currently unused in the prompt text
                                        # itself (matches router.py's own
                                        # prompt, which doesn't interpolate
                                        # the schema string) — kept here only
                                        # to mirror the real import path and
                                        # fail the same way router.py would
                                        # if Fuseki isn't reachable.
        run_variance_check(
            "_is_kg_answerable — CONTROL: in-scope question (expect stable YES)",
            _kg_answerable_prompt("What is the elevation of ZRH?", schema),
        )
        run_variance_check(
            "_is_kg_answerable — CONTROL: out-of-scope question (expect stable NO)",
            _kg_answerable_prompt("Can I bring a pet on flight FR947?", schema),
        )
        run_variance_check(
            "_is_kg_answerable — FLAGGED CASE: this exact question is one of the "
            "prompt's own few-shot examples, labelled NO despite country being "
            "listed as in-scope elsewhere in the same prompt (see 'Is ZRH located "
            "in Switzerland?' -> YES). Worth checking if the model is stable on "
            "its own internally-inconsistent example.",
            _kg_answerable_prompt("في أي دولة يقع مطار أثينا؟", schema),
        )
    except Exception as e:
        print(f"\n[skipped _is_kg_answerable tests — kg_registry/Fuseki "
              f"unavailable: {e}]")