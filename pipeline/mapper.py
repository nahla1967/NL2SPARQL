import json
import urllib.parse
import urllib.request
from sentence_transformers import SentenceTransformer, util

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
    # This function is unchanged in logic, but it is now architecturally active
    # for all three languages.
    #
    # Previously, the extractor forced all property output to English, so this
    # function only ever received strings like "departure city" and the French/
    # Arabic keys in the lexicon were never matched.
    #
    # Now that the extractor returns raw surface text in the user's language,
    # this function will correctly match:
    #   "departure city"     → hasOriginCity   (English)
    #   "ville de départ"    → hasOriginCity   (French)
    #   "مدينة المغادرة"     → hasOriginCity   (Arabic)
    #
    # The .lower().strip() normalization is safe for all three languages:
    #   - English and French have case, so lowercasing is necessary.
    #   - Arabic has no case distinction, so lowercasing is a no-op.
    property_text = property_text.lower().strip()
    if property_text in lexicon["properties"]:
        return lexicon["properties"][property_text]
    return None

def map_property_with_embeddings(property_text, lexicon):
    # This function is also unchanged in logic.
    #
    # It uses paraphrase-multilingual-MiniLM-L12-v2, which was specifically
    # chosen because it produces comparable embedding spaces across languages.
    # This means that the Arabic phrase "شركة الطيران" and the English phrase
    # "airline" will have high cosine similarity, even if the exact Arabic
    # string is not in the lexicon.
    #
    # This fallback is now genuinely useful: it handles paraphrases, typos,
    # and surface forms not covered by the lexicon — across all three languages.
    model = _get_embedding_model()

    known_phrases = list(lexicon["properties"].keys())
    user_embedding = model.encode(property_text)
    known_embeddings = model.encode(known_phrases)
    scores = util.cos_sim(user_embedding, known_embeddings)[0]
    best_index = scores.argmax().item()
    best_score = scores[best_index].item()

    # Threshold of 0.5 — to be justified empirically in thesis evaluation.
    # See Issue #7 for the full discussion of this value.
    if best_score >= 0.5:
        best_phrase = known_phrases[best_index]
        return lexicon["properties"][best_phrase]
    return None

def map_flight(flight_number):
    base = "http://www.semanticweb.org/ontologies/flight_ontology#"
    flight_number = flight_number.strip().upper()
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