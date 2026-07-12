from router import route
from template_resolver import resolve_ask_query
from pipeline.language import detect_language

question = "Is CDG located in France?"
lang = detect_language(question)
routing = route(question)
print(resolve_ask_query(question, routing, lang))