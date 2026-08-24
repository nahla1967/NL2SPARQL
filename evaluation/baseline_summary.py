"""
baseline_summary.py
--------------------
Reads baseline_results.jsonl (produced by eval_runner.py when BASELINE_MODE
is set to "A", "B", or "ablation") and prints a plain-text summary straight
to the console -- no Excel file, no recalc step.

Every row in baseline_results.jsonl already has exact_match / f1 /
sparql_valid / duration_s computed by eval_runner.py's normal scoring
pipeline (the same functions used for the main eval_results.jsonl) -- this
script only aggregates what's already there, it doesn't recompute scoring.

Usage:
    python baseline_summary.py
    python baseline_summary.py --path baseline_results.jsonl   (default anyway)

Run this after each baseline/ablation track (Baseline A, Baseline B, or
each ablation config) -- since eval_runner.py overwrites baseline_results.jsonl
fresh every run, this always summarizes whatever's currently in that file.
"""

import json
import sys
import statistics

PATH = "baseline_results.jsonl"
if "--path" in sys.argv:
    PATH = sys.argv[sys.argv.index("--path") + 1]


def _load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _pct(values):
    """values: list of bool/None. Returns % True among non-None, or None if empty."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return round(100 * sum(1 for v in vals if v) / len(vals), 1)


def _avg(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return round(statistics.mean(vals), 3)


def _fmt(x, suffix=""):
    return f"{x}{suffix}" if x is not None else "n/a"


def summarize(rows, label):
    if not rows:
        print(f"  (no rows for {label})")
        return
    exact = _pct([r.get("exact_match") for r in rows])
    f1 = _avg([r.get("f1") for r in rows])
    valid = _pct([r.get("sparql_valid") for r in rows])
    lat = _avg([r.get("duration_s") for r in rows])
    print(f"  {label:28s} n={len(rows):4d}  "
          f"exact_match={_fmt(exact,'%'):>7s}  "
          f"avg_f1={_fmt(f1):>6s}  "
          f"sparql_valid={_fmt(valid,'%'):>7s}  "
          f"avg_latency={_fmt(lat,'s'):>7s}")


def main():
    rows = _load(PATH)
    if not rows:
        print(f"No rows found in {PATH} -- run eval_runner.py with BASELINE_MODE set first.")
        return

    print(f"=== {PATH} -- {len(rows)} total rows ===\n")

    print("OVERALL")
    summarize(rows, "all rows")

    print("\nBY CATEGORY")
    categories = sorted(set(r["category"] for r in rows))
    for cat in categories:
        summarize([r for r in rows if r["category"] == cat], cat)

    print("\nBY LANGUAGE")
    for lang in ("en", "fr", "ar"):
        summarize([r for r in rows if r["language"] == lang], lang)

    print("\nBY CATEGORY x LANGUAGE")
    for cat in categories:
        for lang in ("en", "fr", "ar"):
            subset = [r for r in rows if r["category"] == cat and r["language"] == lang]
            if subset:
                summarize(subset, f"{cat} / {lang}")

    print("\nFAILURE TYPE BREAKDOWN")
    failure_types = sorted(set(r.get("failure_type") for r in rows))
    for ft in failure_types:
        n = sum(1 for r in rows if r.get("failure_type") == ft)
        print(f"  {ft:28s} n={n}")


if __name__ == "__main__":
    main()