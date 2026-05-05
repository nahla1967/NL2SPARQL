import json
import urllib.parse
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

    Model: paraphrase-multilingual-MiniLM-L12-v2
    Chosen because it produces comparable vector spaces across languages,
    meaning Arabic, French, and English paraphrases of the same concept
    will have high cosine similarity to each other.
    """
    global _embedding_model
    if _embedding_model is None:
        print("[mapper] Loading embedding model (first use)...")
        _embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
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


def map_property_with_embeddings(property_text, lexicon):
    """
    Semantic fallback used when exact lexicon lookup fails.

    Encodes the extracted property text and all lexicon keys into a shared
    multilingual vector space, then returns the KG property name whose
    lexicon key has the highest cosine similarity to the input.

    Why filter keys starting with '_':
    Same reason as in map_property — section separator keys must not be
    included as candidate phrases for embedding matching.

    Threshold: 0.65
    Below this score, the match is considered too uncertain to use.
    The run is then logged as a mapping_failure, which is an informative
    evaluation outcome rather than a silent wrong answer.
    The value 0.65 was chosen empirically — scores below this were observed
    to produce incorrect property matches (e.g. "route" → "runway").
    """
    model = _get_embedding_model()

    # Exclude metadata keys from candidate phrases
    known_phrases = [k for k in lexicon["properties"].keys()
                     if not k.startswith("_")]

    user_embedding = model.encode(property_text)
    known_embeddings = model.encode(known_phrases)
    scores = util.cos_sim(user_embedding, known_embeddings)[0]

    best_index = scores.argmax().item()
    best_score = scores[best_index].item()

    if best_score >= 0.65:
        best_phrase = known_phrases[best_index]
        return lexicon["properties"][best_phrase]

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