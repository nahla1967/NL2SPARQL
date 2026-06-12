"""
mapper.py  (modified — v2)
--------------------------
WHAT CHANGED vs v1:
    load_lexicon() now accepts an optional path argument.
    map_property_cascade() now accepts an optional lexicon argument.
    map_airport() added — resolves an airport IATA code to its KG2 URI.

    All existing functions (map_property, map_property_fuzzy,
    map_property_with_embeddings, map_flight, map_property_cascade)
    are completely unchanged in logic — only signatures updated to
    accept config so the same cascade works for both KG1 and KG2.

WHY THIS APPROACH:
    The cascade (pre-norm → exact → fuzzy → semantic) is language-agnostic
    and property-agnostic. It does not care whether the property belongs to
    flights or airports. Only the lexicon it searches changes.
    Passing the lexicon as a parameter — rather than hardcoding the path —
    is the minimal change needed to make the mapper work for both KGs.
"""

import json
import re
import urllib.parse
import urllib.request
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import process, fuzz
from kg_registry import get_endpoint, get_base_uri

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
FUSEKI_URL           = "http://localhost:3030/flights/sparql"
CACHE_EMBEDDINGS_KG1 = "lexicon_embeddings.npy"
CACHE_PHRASES_KG1    = "lexicon_phrases.json"
CACHE_EMBEDDINGS_KG2 = "lexicon_airports_embeddings.npy"
CACHE_PHRASES_KG2    = "lexicon_airports_phrases.json"

FUZZY_THRESHOLD    = 80
SEMANTIC_THRESHOLD = 0.72


# ── TEXT NORMALISATION ────────────────────────────────────────────────────────
def _normalise(text: str) -> str:
    text = re.sub(r'[\u064B-\u0652]', '', text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'[يى]', 'ي', text)
    text = re.sub(r'ـ', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip().lower()


# ── LEXICON ───────────────────────────────────────────────────────────────────

def load_lexicon(path: str = "lexicon.json") -> dict:
    """
    Loads a lexicon from the given path.
    Defaults to lexicon.json (KG1) for backward compatibility.
    Pass "lexicon_airports.json" for KG2.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _get_phrases(lexicon: dict) -> list[str]:
    return [
        k for k in lexicon["properties"].keys()
        if not k.startswith("_")
    ]


# ── PRE-NORM STEP ─────────────────────────────────────────────────────────────
def _pre_normalise(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r'\?+$', '', text)
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def _pre_map(text: str, lexicon: dict) -> tuple:
    norm = _pre_normalise(text)
    normalised_properties = {
        _normalise(k): v
        for k, v in lexicon["properties"].items()
        if not k.startswith("_")
    }
    result = normalised_properties.get(norm)
    if result:
        print(f"[pre-norm] '{text}' → exact hit: {result}")
    return result, None


# ── TIER 1: EXACT ─────────────────────────────────────────────────────────────
def map_property(property_text: str, lexicon: dict) -> str | None:
    key = _normalise(property_text)
    normalised_properties = {
        _normalise(k): v
        for k, v in lexicon["properties"].items()
        if not k.startswith("_")
    }
    return normalised_properties.get(key)


# ── TIER 2: FUZZY ─────────────────────────────────────────────────────────────
def map_property_fuzzy(property_text: str, lexicon: dict) -> str | None:
    phrases      = _get_phrases(lexicon)
    norm_in      = _normalise(property_text)
    norm_phrases = [_normalise(p) for p in phrases]
    result = process.extractOne(norm_in, norm_phrases, scorer=fuzz.WRatio)
    if result is None:
        return None
    matched_phrase, score, index = result
    print(f"[fuzzy] input='{property_text}' → match='{phrases[index]}' score={score}")
    if score >= FUZZY_THRESHOLD:
        return lexicon["properties"][phrases[index]]
    return None


# ── TIER 3: SEMANTIC ──────────────────────────────────────────────────────────
_embedding_model   = None
_cached_embeddings = {}   # dict: cache_path → numpy matrix
_cached_phrases    = {}   # dict: cache_path → list of phrases

def _get_model():
    global _embedding_model
    if _embedding_model is None:
        print("[mapper] Loading embedding model...")
        _embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    return _embedding_model

def _load_or_build_cache(lexicon: dict, lexicon_path: str):
    """
    Returns (phrases, embedding_matrix).
    Uses separate cache files for KG1 and KG2 lexicons.
    """
    import os
    # Choose cache file based on lexicon path
    if "airport" in lexicon_path:
        emb_cache   = CACHE_EMBEDDINGS_KG2
        phrase_cache = CACHE_PHRASES_KG2
    else:
        emb_cache   = CACHE_EMBEDDINGS_KG1
        phrase_cache = CACHE_PHRASES_KG1

    if emb_cache in _cached_embeddings:
        return _cached_phrases[emb_cache], _cached_embeddings[emb_cache]

    if os.path.exists(emb_cache) and os.path.exists(phrase_cache):
        print(f"[mapper] Loading cached embeddings from {emb_cache}...")
        with open(phrase_cache, "r", encoding="utf-8") as f:
            phrases = json.load(f)
        embeddings = np.load(emb_cache)
        _cached_phrases[emb_cache]    = phrases
        _cached_embeddings[emb_cache] = embeddings
        return phrases, embeddings

    print(f"[mapper] Building embedding cache for {lexicon_path}...")
    model        = _get_model()
    phrases      = _get_phrases(lexicon)
    norm_phrases = [_normalise(p) for p in phrases]
    embeddings   = model.encode(norm_phrases, show_progress_bar=True)
    np.save(emb_cache, embeddings)
    with open(phrase_cache, "w", encoding="utf-8") as f:
        json.dump(phrases, f, ensure_ascii=False)
    _cached_phrases[emb_cache]    = phrases
    _cached_embeddings[emb_cache] = embeddings
    return phrases, embeddings

_ARABIC_RE = re.compile(r'^[\u0600-\u06FF\s\?]+$')

def _is_arabic_phrase(phrase: str) -> bool:
    return bool(_ARABIC_RE.match(phrase))

def _detect_script(text: str) -> str:
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    total        = len([c for c in text if c.isalpha()])
    if total == 0:
        return "latin"
    return "arabic" if arabic_chars / total > 0.3 else "latin"

def map_property_with_embeddings(
    property_text: str,
    lexicon: dict,
    lexicon_path: str = "lexicon.json"
) -> str | None:
    model           = _get_model()
    phrases, matrix = _load_or_build_cache(lexicon, lexicon_path)
    norm_in         = _normalise(property_text)
    script          = _detect_script(norm_in)
    query_vec       = model.encode(norm_in)

    same_script_indices = [
        i for i, ph in enumerate(phrases)
        if (_is_arabic_phrase(ph) if script == "arabic" else not _is_arabic_phrase(ph))
    ]

    if same_script_indices:
        sub_matrix  = matrix[same_script_indices]
        sub_scores  = util.cos_sim(query_vec, sub_matrix)[0]
        best_sub_idx   = sub_scores.argmax().item()
        best_sub_score = sub_scores[best_sub_idx].item()
        best_index     = same_script_indices[best_sub_idx]
        best_phrase    = phrases[best_index]

        print(f"[semantic] '{property_text}' [script={script}] "
              f"→ '{best_phrase}' score={best_sub_score:.3f}")

        if best_sub_score >= SEMANTIC_THRESHOLD:
            return lexicon["properties"][best_phrase]

    all_scores      = util.cos_sim(query_vec, matrix)[0]
    best_full_idx   = all_scores.argmax().item()
    best_full_score = all_scores[best_full_idx].item()
    best_full_phrase = phrases[best_full_idx]

    print(f"[semantic] '{property_text}' full-matrix → "
          f"'{best_full_phrase}' score={best_full_score:.3f}")

    if best_full_score >= SEMANTIC_THRESHOLD + 0.05:
        return lexicon["properties"][best_full_phrase]
    return None


# ── FULL CASCADE ──────────────────────────────────────────────────────────────

def map_property_cascade(
    property_text: str,
    lexicon: dict,
    lexicon_path: str = "lexicon.json"
) -> tuple:
    """
    Returns (prop1, tier, prop2).
    lexicon_path is used to select the correct embedding cache.
    Defaults to KG1 for backward compatibility.
    """
    if not property_text:
        return None, None, None

    def _unpack(uri):
        if isinstance(uri, list):
            return uri[0], uri[1]
        if isinstance(uri, dict):
            # special_query entries like coordinates are not
            # mappable to a single property URI — skip them
            return None, None
        return uri, None

    uri, _ = _pre_map(property_text, lexicon)
    if uri:
        prop1, prop2 = _unpack(uri)
        return prop1, "pre-norm", prop2

    uri = map_property(property_text, lexicon)
    if uri:
        prop1, prop2 = _unpack(uri)
        return prop1, "exact", prop2

    uri = map_property_fuzzy(property_text, lexicon)
    if uri:
        prop1, prop2 = _unpack(uri)
        return prop1, "fuzzy", prop2

    uri = map_property_with_embeddings(property_text, lexicon, lexicon_path)
    if uri:
        prop1, prop2 = _unpack(uri)
        return prop1, "semantic", prop2

    return None, None, None


# ── FLIGHT MAPPING (KG1) — UNCHANGED ─────────────────────────────────────────
_flight_uri_cache: dict[str, str] = {}

def map_flight(flight_number: str) -> str | None:
    base          = "http://www.semanticweb.org/ontologies/flight_ontology#"
    flight_number = flight_number.strip().upper()

    if flight_number in _flight_uri_cache:
        return _flight_uri_cache[flight_number]

    query = f"""
SELECT ?flight WHERE {{
  ?flight <{base}flightNumber> "{flight_number}" .
}}
LIMIT 1
"""
    data = urllib.parse.urlencode({
        "query":  query,
        "format": "application/sparql-results+json"
    }).encode()
    req = urllib.request.Request(FUSEKI_URL, data=data)
    try:
        with urllib.request.urlopen(req) as response:
            result   = json.loads(response.read())
            bindings = result["results"]["bindings"]
            if bindings:
                uri = bindings[0]["flight"]["value"]
                _flight_uri_cache[flight_number] = uri
                return uri
    except urllib.error.URLError as e:
        print(f"[map_flight] Fuseki unreachable: {e}")
    except Exception as e:
        print(f"[map_flight] Unexpected error: {e}")
    return None


# ── AIRPORT MAPPING (KG2) — NEW ───────────────────────────────────────────────
_airport_uri_cache: dict[str, str] = {}

def map_airport(iata: str) -> str | None:
    """
    Resolves an IATA code to its KG2 Airport URI.
    Queries the /airports/sparql endpoint.

    Example: "VIE" → "http://...airport_ontology#Airport/VIE"

    Uses an in-memory cache — the KG is static.
    """
    iata     = iata.strip().upper()
    endpoint = get_endpoint("airports")
    base     = get_base_uri("airports")

    if iata in _airport_uri_cache:
        return _airport_uri_cache[iata]

    query = f"""
SELECT ?airport WHERE {{
  ?airport <{base}iataCode> "{iata}" .
}}
LIMIT 1
"""
    data = urllib.parse.urlencode({
        "query":  query,
        "format": "application/sparql-results+json"
    }).encode()
    req = urllib.request.Request(endpoint, data=data)
    try:
        with urllib.request.urlopen(req) as response:
            result   = json.loads(response.read())
            bindings = result["results"]["bindings"]
            if bindings:
                uri = bindings[0]["airport"]["value"]
                _airport_uri_cache[iata] = uri
                return uri
    except urllib.error.URLError as e:
        print(f"[map_airport] Fuseki unreachable: {e}")
    except Exception as e:
        print(f"[map_airport] Unexpected error: {e}")
    return None