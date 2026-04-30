import json
import urllib.parse
import urllib.request
from sentence_transformers import SentenceTransformer, util

embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def load_lexicon():
    with open("lexicon.json", "r", encoding="utf-8") as f:
        return json.load(f)

def map_property(property_text, lexicon):
    property_text = property_text.lower().strip()
    if property_text in lexicon["properties"]:
        return lexicon["properties"][property_text]
    return None

def map_property_with_embeddings(property_text, lexicon):
    known_phrases = list(lexicon["properties"].keys())
    user_embedding = embedding_model.encode(property_text)
    known_embeddings = embedding_model.encode(known_phrases)
    scores = util.cos_sim(user_embedding, known_embeddings)[0]
    best_index = scores.argmax().item()
    best_score = scores[best_index].item()
    if best_score >= 0.5:
        best_phrase = known_phrases[best_index]
        return lexicon["properties"][best_phrase]
    return None

def map_flight(flight_number):
    base = "http://www.semanticweb.org/ontologies/flight_ontology#"
    query = f"""
SELECT ?flight WHERE {{
  ?flight <{base}flightNumber> "{flight_number}" .
}}
LIMIT 1
"""
    url = "http://localhost:3030/flights/sparql"
    data = urllib.parse.urlencode({
        "query": query,
        "format": "application/sparql-results+json"
    }).encode()
    req = urllib.request.Request(url, data=data)
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read())
        bindings = result["results"]["bindings"]
        if bindings:
            return bindings[0]["flight"]["value"]
    return None