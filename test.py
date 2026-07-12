from pipeline.language import detect_language
from template_resolver import resolve_template

question = "What is the average number of courses taught per department?"
lang = detect_language(question)
result = resolve_template(question, "group_aggregate_kg3", lang)
print(result)