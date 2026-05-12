# pipeline/mapper.py

import json
import re
import urllib.parse
import urllib.request
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import process, fuzz

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

FUSEKI_URL       = "http://localhost:3030/flights/sparql"
CACHE_EMBEDDINGS = "lexicon_embeddings.npy"   # pre-computed phrase vectors
CACHE_PHRASES    = "lexicon_phrases.json"     # phrase list aligned to the matrix

FUZZY_THRESHOLD    = 80   # rapidfuzz WRatio score out of 100
# 80 instead of 87: catches spelling mistakes ("componie" → "compagnie" scores 82)
# while still rejecting random noise. You can lower to 75 for more coverage.

SEMANTIC_THRESHOLD = 0.75 # cosine similarity


# ── TEXT NORMALISATION ────────────────────────────────────────────────────────
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
    """
    Returns all property expression keys from the lexicon.
    No filter needed — the cleaned lexicon stores only real expressions
    inside 'properties'. Metadata lives separately under '_sections'.
    """
    return list(lexicon["properties"].keys())


# ── PRE-NORMALISE STEP ───────────────────────────────────────────────────────
#
# Before calling any cascade tier, try a plain normalisation + exact lookup.
# This is NOT the same as map_property below:
#   - map_property does  _normalise(input) → lexicon lookup (Tier 1)
#   - _pre_normalise     does the same but ALSO handles:
#       * stripping small stopwords ("the", "du", "la", "de", "ال")
#       * collapsing multiple spaces
#       * stripping trailing question marks
#
# The idea: the LLM extractor now returns a raw phrase like "gate" or
# "ville de départ" stripped of question framing. The pre-normalise step
# catches the majority of clean inputs in a single cheap operation,
# keeping the cascade for genuinely difficult cases only.


def _pre_normalise(text: str) -> str:
    """
    Lightweight cleanup before lexicon lookup.
    Strips punctuation, common article words, and extra whitespace.
    """
    text = text.strip().lower()

    # Strip trailing question mark (language-agnostic)
    text = re.sub(r'\?+$', '', text)

    # Collapse internal punctuation to spaces — "ville,d'arrivée" → "ville d'arrivée"
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def _pre_map(text: str, lexicon: dict) -> tuple[str | None, str | None]:
    """
    Tries direct normalised lookup before invoking any cascade tier.
    Returns (property_short_name, None) on success, or (None, None) on failure.
    This step is fast and cheap — no fuzzy, no embeddings.
    """
    norm = _pre_normalise(text)
    normalised_properties = {
        _normalise(k): v
        for k, v in lexicon["properties"].items()
    }
    result = normalised_properties.get(norm)
    if result:
        print(f"[pre-norm] '{text}' → '{norm}' → exact hit: {result}")
    return result, None


# ── TIER 1 — EXACT MATCH ──────────────────────────────────────────────────────

def map_property(property_text: str, lexicon: dict) -> str | None:
    key = _normalise(property_text)
    normalised_properties = {
        _normalise(k): v
        for k, v in lexicon["properties"].items()
    }
    return normalised_properties.get(key)


# ── TIER 2 — FUZZY MATCH (rapidfuzz) ─────────────────────────────────────────
# WRatio is a composite scorer that handles partial matches, transpositions,
# and reorderings — robust to minor wording variations without needing
# vector representations at all.
# Example failures it catches that exact match misses:
#   "departure town"     → "departure city"
#   "aéroport d'arrivée" → "ville d'arrivée"
#   "مدينة الانطلاق"    → "مدينة المغادرة"

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


# ── TIER 3 — SEMANTIC EMBEDDINGS (language-aware) ─────────────────────────────
# The lexicon never changes between runs, so recomputing embeddings every time
# is wasteful. We compute the phrase matrix once, persist it to disk, and load
# it on subsequent runs. Only the user's query needs a fresh forward pass.
#
# IMPORTANT: semantic matching is language-aware.
# Without language filtering, a French query like "componie" could incorrectly
# match an Arabic phrase that happens to be vectorially close. To prevent this:
#   1. Detect the script of the input (Arabic / Latin)
#   2. Build a sub-matrix of only same-script lexicon entries
#   3. Search within that sub-matrix first
#   4. Fall back to full matrix only if no good match found
#
# Known limitation: this heuristic splits the lexicon by script (Arabic vs Latin),
# which means EN/FR share the same semantic space. This is acceptable since they
# are morphologically close and the multilingual model handles cross-lingual cases.

_embedding_model   = None   # lazy-loaded on first use
_cached_embeddings = None   # numpy matrix (n_phrases × 768)
_cached_phrases    = None   # list aligned to the matrix rows


def _get_model():
    global _embedding_model
    if _embedding_model is None:
        print("[mapper] Loading embedding model (first time)...")
        _embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    return _embedding_model


def _load_or_build_cache(lexicon: dict):
    """
    Returns (phrases, embedding_matrix).
    Loads from disk if the cache exists; builds and saves it on first run.
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

    print("[mapper] Building embedding cache (one-time setup)...")
    model        = _get_model()
    phrases      = _get_phrases(lexicon)
    norm_phrases = [_normalise(p) for p in phrases]
    embeddings   = model.encode(norm_phrases, show_progress_bar=True)

    np.save(CACHE_EMBEDDINGS, embeddings)
    with open(CACHE_PHRASES, "w", encoding="utf-8") as f:
        json.dump(phrases, f, ensure_ascii=False)

    _cached_phrases    = phrases
    _cached_embeddings = embeddings
    return phrases, embeddings


def _detect_script(text: str) -> str:
    """
    Heuristic language/script detection for semantic filtering.
    Returns 'arabic' | 'latin' based on dominant Unicode block.
    """
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    total        = len([c for c in text if c.isalpha()])
    if total == 0:
        return "latin"
    return "arabic" if arabic_chars / total > 0.3 else "latin"


_ARABIC_RE  = re.compile(r'^[\u0600-\u06FF\s\?]+$')


def _is_arabic_phrase(phrase: str) -> bool:
    """True if phrase is primarily Arabic script."""
    return bool(_ARABIC_RE.match(phrase))


def map_property_with_embeddings(property_text: str, lexicon: dict) -> str | None:
    """
    Semantic fallback with language-aware search.
    See cascade docstring above for the language-filtering strategy.
    """
    model           = _get_model()#the AI that converts phrases to vectors
    phrases, matrix = _load_or_build_cache(lexicon)#a pre-computed table of vectors for every phrase in your lexicon
    norm_in         = _normalise(property_text)
    script          = _detect_script(norm_in)

    query_vec = model.encode(norm_in)# Convert user input to vector

    # Step 1: same-script candidates only
    same_script_indices = [
        i for i, ph in enumerate(phrases)
        if (_is_arabic_phrase(ph) if script == "arabic" else not _is_arabic_phrase(ph))
    ]

    if same_script_indices:
        sub_matrix      = matrix[same_script_indices]
        sub_scores      = util.cos_sim(query_vec, sub_matrix)[0]
        best_sub_idx    = sub_scores.argmax().item()
        best_sub_score  = sub_scores[best_sub_idx].item()
        best_index      = same_script_indices[best_sub_idx]
        best_phrase     = phrases[best_index]
        best_score      = best_sub_score

        print(f"[semantic] input='{property_text}' [script={script}] "
              f"→ match='{best_phrase}' score={best_score:.3f} "
              f"(searched {len(same_script_indices)} same-script candidates)")

        if best_score >= SEMANTIC_THRESHOLD:
            return lexicon["properties"][best_phrase]

    # Step 2: fallback — search full matrix (cross-script permitted)
    all_scores     = util.cos_sim(query_vec, matrix)[0]
    best_full_idx  = all_scores.argmax().item()
    best_full_score = all_scores[best_full_idx].item()
    best_full_phrase = phrases[best_full_idx]

    print(f"[semantic] input='{property_text}' [script={script}] "
          f"→ full-matrix best: match='{best_full_phrase}' score={best_full_score:.3f} "
          f"(same-script search returned no confident match)")

    # Only return if full-matrix score is notably better AND above threshold
    # This prevents silently accepting a mediocre cross-script match
    if best_full_score >= SEMANTIC_THRESHOLD + 0.05:
        return lexicon["properties"][best_full_phrase]

    return None


# ── FULL CASCADE (returns URI + tier for evaluation) ──────────────────────────

def map_property_cascade(property_text: str, lexicon: dict) -> tuple[str | None, str | None]:
    """
    Full mapping cascade with tier reporting for evaluation.

    Returns (property_short_name, tier_label):
        tier_label = "pre-norm" | "exact" | "fuzzy" | "semantic" | None

    Evaluation use: logs.jsonl records which tier resolved each query, so you
    can compute tier-coverage statistics across your full test set.
    """
    if not property_text:
        return None, None

    # Tier 0 — pre-normalise (fast, no LLM/fuzzy needed)
    uri, _ = _pre_map(property_text, lexicon)
    if uri:
        return uri, "pre-norm"

    # Tier 1 — exact (after full normalisation)
    uri = map_property(property_text, lexicon)
    if uri:
        return uri, "exact"

    # Tier 2 — fuzzy
    uri = map_property_fuzzy(property_text, lexicon)
    if uri:
        return uri, "fuzzy"

    # Tier 3 — semantic
    uri = map_property_with_embeddings(property_text, lexicon)
    if uri:
        return uri, "semantic"

    return None, None


# ── FLIGHT MAPPING ────────────────────────────────────────────────────────────

# In-memory cache: flight_number (str) → KG URI (str)
# The KG is static — a flight number always resolves to the same URI.
# Caching avoids a redundant Fuseki HTTP call on every repeated lookup,
# which matters during evaluation where the same flights appear across
# multiple conditions and languages.
_flight_uri_cache: dict[str, str] = {}


def map_flight(flight_number: str) -> str | None:
    base          = "http://www.semanticweb.org/ontologies/flight_ontology#"
    flight_number = flight_number.strip().upper()

    # Return immediately if already resolved in this session
    if flight_number in _flight_uri_cache:
        print(f"[map_flight] cache hit → {flight_number}")
        return _flight_uri_cache[flight_number]

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
                uri = bindings[0]["flight"]["value"]
                _flight_uri_cache[flight_number] = uri   # store for future calls
                return uri
    except urllib.error.URLError as e:
        print(f"[map_flight] Fuseki unreachable: {e}")
    except Exception as e:
        print(f"[map_flight] Unexpected error: {e}")
    return None

#why didnt we put the flights in the lexicon ? because the flights are dynamic so we need to query fuseki because "too many , random uris" + faster , propertes in lexicon because they are fixed + knwown in advance + instant