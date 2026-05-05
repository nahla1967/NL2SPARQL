# pipeline/mapper.py

import json
import re
import urllib.parse
import urllib.request
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import process, fuzz

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

FUSEKI_URL      = "http://localhost:3030/flights/sparql"
CACHE_EMBEDDINGS = "lexicon_embeddings.npy"   # pre-computed phrase vectors
CACHE_PHRASES    = "lexicon_phrases.json"     # phrase list aligned to the matrix

FUZZY_THRESHOLD    = 87   # rapidfuzz WRatio score out of 100
SEMANTIC_THRESHOLD = 0.75 # cosine similarity


# ── TEXT NORMALISATION ────────────────────────────────────────────────────────
# Moved here as a standalone utility so both tiers can reuse it.
# Arabic normalisation removes diacritics, unifies alef variants, teh marbuta,
# and ya variants — all of which are stylistic, not semantic distinctions.

def _normalise(text: str) -> str:
    text = re.sub(r'[\u064B-\u0652]', '', text)   # diacritics
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'[يى]', 'ي', text)
    text = re.sub(r'ـ', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip().lower()


# ── LEXICON ───────────────────────────────────────────────────────────────────

def load_lexicon():
    with open("lexicon.json", "r", encoding="utf-8") as f:
        return json.load(f)

def _get_phrases(lexicon: dict) -> list[str]:
    """Returns only valid property expressions — excludes _section_* metadata keys."""
    return [k for k in lexicon["properties"] if not k.startswith("_")]


# ── TIER 1 — EXACT MATCH ──────────────────────────────────────────────────────

def map_property(property_text: str, lexicon: dict) -> str | None:
    key = _normalise(property_text)
    properties = {
        _normalise(k): v
        for k, v in lexicon["properties"].items()
        if not k.startswith("_")
    }
    return properties.get(key)


# ── TIER 2 — FUZZY MATCH (rapidfuzz) ─────────────────────────────────────────
# rapidfuzz is a lightweight C-extension — no model, no GPU, loads instantly.
# WRatio is a composite scorer that handles partial matches, transpositions,
# and reorderings, making it robust to minor wording variations without
# needing vector representations at all.
# This tier catches failures that exact match misses:
#   "departure town"  → "departure city"
#   "aéroport d'arrivée" → "ville d'arrivée"  (conceptual overlap)
#   "مدينة الانطلاق"  → "مدينة المغادرة"      (synonym)

def map_property_fuzzy(property_text: str, lexicon: dict) -> str | None:
    phrases  = _get_phrases(lexicon)
    norm_in  = _normalise(property_text)
    norm_phrases = [_normalise(p) for p in phrases]

    result = process.extractOne(norm_in, norm_phrases, scorer=fuzz.WRatio)
    if result is None:
        return None

    matched_phrase, score, index = result
    print(f"[fuzzy] input='{property_text}' → match='{phrases[index]}' score={score}")

    if score >= FUZZY_THRESHOLD:
        return lexicon["properties"][phrases[index]]
    return None


# ── TIER 3 — SEMANTIC EMBEDDINGS (pre-computed, cached) ──────────────────────
# The single biggest change from your original code.
#
# The problem with the original: model.encode(all_phrases) ran on EVERY call.
# The lexicon never changes, so those vectors are identical every time.
# This is equivalent to re-computing a multiplication table from scratch
# before every arithmetic problem.
#
# The solution: compute the matrix once, persist it to disk with numpy.
# Subsequent calls load ~400KB instead of running ~100 forward passes.
# First run: slow (one-time setup). Every run after: instant.

_embedding_model  = None   # lazy-loaded on first use
_cached_embeddings = None  # numpy matrix (n_phrases × 768)
_cached_phrases    = None  # list aligned to the matrix rows


def _get_model():
    global _embedding_model
    if _embedding_model is None:
        print("[mapper] Loading embedding model (first time)...")
        _embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    return _embedding_model


def _load_or_build_cache(lexicon: dict):
    """
    Returns (phrases, embedding_matrix).
    Loads from disk if the cache exists, builds and saves it otherwise.
    
    Why numpy .npy?  It is the standard serialisation format for dense
    numerical arrays in Python — fast to write, fast to load, no
    dependencies beyond numpy itself.
    """
    global _cached_embeddings, _cached_phrases
    if _cached_embeddings is not None:
        return _cached_phrases, _cached_embeddings

    import os
    if os.path.exists(CACHE_EMBEDDINGS) and os.path.exists(CACHE_PHRASES):
        print("[mapper] Loading pre-computed lexicon embeddings from cache...")
        with open(CACHE_PHRASES, "r", encoding="utf-8") as f:
            _cached_phrases = json.load(f)
        _cached_embeddings = np.load(CACHE_EMBEDDINGS)
        return _cached_phrases, _cached_embeddings

    # First run: build the cache
    print("[mapper] Building embedding cache (one-time setup)...")
    model   = _get_model()
    phrases = _get_phrases(lexicon)
    norm_phrases = [_normalise(p) for p in phrases]

    embeddings = model.encode(norm_phrases, show_progress_bar=True)

    np.save(CACHE_EMBEDDINGS, embeddings)
    with open(CACHE_PHRASES, "w", encoding="utf-8") as f:
        json.dump(phrases, f, ensure_ascii=False)

    _cached_phrases    = phrases
    _cached_embeddings = embeddings
    return phrases, embeddings


def map_property_with_embeddings(property_text: str, lexicon: dict) -> str | None:
    """
    Semantic fallback. Only called when tiers 1 and 2 both fail.
    Encodes only the user's query (one forward pass), then computes cosine
    similarity against the pre-loaded phrase matrix.
    """
    model             = _get_model()
    phrases, matrix   = _load_or_build_cache(lexicon)
    norm_in           = _normalise(property_text)

    query_vec  = model.encode(norm_in)
    scores     = util.cos_sim(query_vec, matrix)[0]
    best_index = scores.argmax().item()
    best_score = scores[best_index].item()

    print(f"[semantic] input='{property_text}' → match='{phrases[best_index]}' score={best_score:.3f}")

    if best_score >= SEMANTIC_THRESHOLD:
        return lexicon["properties"][phrases[best_index]]
    return None


# ── FLIGHT MAPPING ────────────────────────────────────────────────────────────

def map_flight(flight_number: str) -> str | None:
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
            result   = json.loads(response.read())
            bindings = result["results"]["bindings"]
            if bindings:
                return bindings[0]["flight"]["value"]
    except urllib.error.URLError as e:
        print(f"[map_flight] Fuseki unreachable: {e}")
    except Exception as e:
        print(f"[map_flight] Unexpected error: {e}")
    return None