from eval_runner import _run_open_kg

question = "What department does FullProfessor5 in Department0 work for?"
result = _run_open_kg(question, "en")
print(result)