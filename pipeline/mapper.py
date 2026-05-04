import json
import urllib.parse
import urllib.request
from sentence_transformers import SentenceTransformer, util

# Problem 3 — Lazy initialization of the embedding model.
#
# Previously, this line ran at import time:
#   embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
#
# That caused the ~400MB model to be loaded into memory every time mapper.py
# was imported — even when lexicon lookup succeeded and embeddings were never needed.
#
# The fix: initialize the variable as None and load the model only on first use,
# inside map_property_with_embeddings(). This is the standard Python pattern
# called "lazy initialization" or "lazy loading."
#
# Practical benefit for thesis evaluation: if all your test questions match
# the lexicon (exact match), the startup time drops from ~10s to near zero,
# and your evaluation loop runs significantly faster.
_embedding_model = None

def _get_embedding_model():
    """
    Returns the embedding model, loading it on first call only.
    All subsequent calls reuse the already-loaded instance.
    """
    global _embedding_model
    if _embedding_model is None:
        print("[mapper] Loading embedding model (first use)...")
        _embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _embedding_model

FUSEKI_URL = "http://localhost:3030/flights/sparql"

def load_lexicon():
    with open("lexicon.json", "r", encoding="utf-8") as f:
        return json.load(f)

def map_property(property_text, lexicon):
    property_text = property_text.lower().strip()
    if property_text in lexicon["properties"]:
        return lexicon["properties"][property_text]
    return None

def map_property_with_embeddings(property_text, lexicon):
    # The model is loaded here, not at import time.
    model = _get_embedding_model()

    known_phrases = list(lexicon["properties"].keys())
    user_embedding = model.encode(property_text)
    known_embeddings = model.encode(known_phrases)
    scores = util.cos_sim(user_embedding, known_embeddings)[0]
    best_index = scores.argmax().item()
    best_score = scores[best_index].item()

    # 0.5 is a manually chosen threshold — documented explicitly
    # for thesis: this value should be justified empirically
    # by plotting score distributions for correct vs incorrect matches
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
    data = urllib.parse.urlencode({
        "query": query,
        "format": "application/sparql-results+json"
    }).encode()

    req = urllib.request.Request(FUSEKI_URL, data=data)

    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read())
            bindings = result["results"]["bindings"]
            if bindings:
                return bindings[0]["flight"]["value"]
    except urllib.error.URLError as e:
        print(f"[map_flight] Fuseki is unreachable: {e}")
    except Exception as e:
        print(f"[map_flight] Unexpected error: {e}")

    return None