import json
import urllib.parse
import re
import urllib.request
from sentence_transformers import SentenceTransformer, util

# ── EMBEDDING MODEL ───────────────────────────────────────────────────────────
# Lazy initialization — the model is only loaded on first use.
# This avoids loading ~400MB into memory when the lexicon handles all matches,
# keeping startup time fast for the majority of test conditions.

_embedding_model = None

def _get_embedding_model():
    """
    Returns the multilingual embedding model, loading it on first call only.
    All subsequent calls reuse the already-loaded instance.

    Model:paraphrase-multilingual-mpnet-base-v2 
    Chosen because it produces comparable vector spaces across languages,
    meaning Arabic, French, and English paraphrases of the same concept
    will have high cosine similarity to each other.
    """
    global _embedding_model
    if _embedding_model is None:
        print("[mapper] Loading embedding model (first use)...")
        _embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    return _embedding_model


# ── CONFIGURATION ─────────────────────────────────────────────────────────────

FUSEKI_URL = "http://localhost:3030/flights/sparql"


# ── LEXICON ───────────────────────────────────────────────────────────────────

def load_lexicon():
    """
    Loads the multilingual lexicon from lexicon.json.
    The lexicon maps surface expressions in English, French, and Arabic
    to short Knowledge Graph property names (e.g. "hasOriginCity").
    """
    with open("lexicon.json", "r", encoding="utf-8") as f:
        return json.load(f)


# ── PROPERTY MAPPING ──────────────────────────────────────────────────────────

def map_property(property_text, lexicon):
    """
    Attempts an exact match between the extracted property text and
    the lexicon keys.

    Returns the corresponding KG property name if found, None otherwise.

    Why filter keys starting with '_':
    The lexicon uses _section_* keys as human-readable section separators
    (e.g. "_section_origin"). These are metadata — not valid property expressions.
    Including them in the lookup would risk matching a user query against a
    separator string and returning a garbage value like "── DEPARTURE CITY ──".
    Filtering on the '_' prefix excludes all metadata keys safely.

    Why .lower().strip():
    English and French have case distinctions, so lowercasing is necessary
    for reliable matching. Arabic has no case, so lowercasing is a no-op.
    Stripping removes leading/trailing whitespace that the LLM extractor
    may occasionally introduce.
    """
    property_text = property_text.lower().strip()

    # Exclude metadata keys before lookup
    properties = {k: v for k, v in lexicon["properties"].items()
                  if not k.startswith("_")}

    if property_text in properties:
        return properties[property_text]
    return None

def normalize_arabic(text):
    text = re.sub(r'[\u064B-\u0652]', '', text)  # diacritics
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'[يى]', 'ي', text)
    text = re.sub(r'ـ', '', text)

    # remove punctuation (important)
    text = re.sub(r'[^\w\s]', '', text)

    return text.strip().lower()

def map_property_with_embeddings(property_text, lexicon):
    """
    Semantic fallback using multilingual embeddings.
    """

    model = _get_embedding_model()

    # Normalize input (important for Arabic)
    property_text_norm = normalize_arabic(property_text)

    # Extract valid phrases
    known_phrases = [
        k for k in lexicon["properties"].keys()
        if not k.startswith("_")
    ]

    # Normalize lexicon phrases
    normalized_phrases = [normalize_arabic(p) for p in known_phrases]

    # Encode
    user_embedding = model.encode(property_text_norm)
    known_embeddings = model.encode(normalized_phrases)

    # Similarity
    scores = util.cos_sim(user_embedding, known_embeddings)[0]

    best_index = scores.argmax().item()
    best_score = scores[best_index].item()

    # Debug (keep this for experiments)
    print(f"[embedding] input='{property_text}' → match='{known_phrases[best_index]}' score={best_score:.3f}")

    # Threshold (raise it)
    if best_score >= 0.75:
        return lexicon["properties"][known_phrases[best_index]]

    return None


# ── FLIGHT MAPPING ────────────────────────────────────────────────────────────

def map_flight(flight_number):
    """
    Resolves a flight number string (e.g. "TK1887") to its full KG URI
    by querying the Fuseki endpoint.

    Why normalise with .strip().upper():
    The LLM extractor may return lowercase or mixed-case flight numbers,
    or include leading/trailing whitespace. SPARQL string matching is
    case-sensitive, so both issues would cause a silent lookup failure.
    Normalising to uppercase before querying prevents this.
    """
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