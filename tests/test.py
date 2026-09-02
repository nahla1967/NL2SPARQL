import json, pandas as pd

df = pd.read_excel(r"C:\Users\fersi\Desktop\NL2SPARQL\evaluation\results\NL2SPARQL_Evaluation_Dataset.xlsx", sheet_name="Questions")
expected = dict(zip(df["id"], df["expected_answer"]))

target_categories = {"group_aggregate_kg2", "group_aggregate_kg3", "filter_numeric_kg3"}

with open("evaluation/results/eval_results.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["category"] in target_categories:
            print(f"{r['id']} | {r['category']} | lang={r['language']} | failure={r['failure_type']}")
            print(f"  raw_answer   = {r['raw_answer']!r}")
            print(f"  final_answer = {r['final_answer']!r}")
            print(f"  expected     = {expected.get(r['id'])!r}")
            print(f"  error_detail = {r.get('error_detail')!r}")
            print(f"  exact_match  = {r.get('exact_match')!r}")
            print(f"  f1           = {r.get('f1')!r}")
            print()