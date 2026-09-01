"""
Deterministic entity and signal detectors for the NL2SPARQL router.
"""

import json
import re

from rapidfuzz import process, fuzz

from kg_registry import KG_REGISTRY, get_lexicon
from .rules import (
    _FLIGHT_RE,
    _IATA_RE,
    _UNIVERSITY_ENTITY_RE,
    _KG1_ONLY_SIGNALS,
    _OPEN_KG_SIGNALS,
    _COUNT_SIGNALS,
    _FILTER_SIGNALS,
    _COMPARE_SIGNALS,
    _COMPARE_PROPERTY_KEYWORDS,
    _COMPARE_PROPERTY_KEYWORDS_KG1,
    _GROUP_RANKING_SIGNALS,
    _normalise,
    _normalise_for_signal_match,
    _strip_arabic_al,
)

# ── LEXICON LOAD (airports) ──────────────────────────────────────
def _load_airport_lexicon():
    with open(get_lexicon("airports"), encoding="utf-8") as f:
        return json.load(f)

_airport_lex      = _load_airport_lexicon()
_AIRPORT_ENTITIES = _airport_lex.get("airport_entities", {})
_AIRPORT_TRIGGERS = KG_REGISTRY["airports"].get("triggers", [])


# ── FLIGHT NUMBER DETECTORS ───────────────────────────────────────
def _detect_flight_number(q: str):
    """Returns the LONGEST flight-number match."""
    m = _FLIGHT_RE.findall(q.upper())
    return max(m, key=len) if m else None
def _detect_flight_numbers_all(q: str) -> list:
    """Returns all distinct flight-number matches — used to catch multi-flight
    (comparison) questions that _detect_flight_number would silently collapse
    to a single entity."""
    m = _FLIGHT_RE.findall(q.upper())
    return sorted(set(m))

def _detect_flight_number_first(q: str):
    """ASK-specific variant: returns the FIRST match."""
    m = _FLIGHT_RE.findall(q.upper())
    return m[0] if m else None


# ── AIRPORT ENTITY DETECTOR ───────────────────────────────────────
def _detect_airport_keyword(q: str) -> bool:
    q_lower = q.lower()
    return any(k.lower() in q_lower for k in _AIRPORT_TRIGGERS)


def _detect_airport_entity(q: str):
    """
    Dictionary lookup and fuzzy match for airport names and IATA codes.
    Tiers: exact phrase → IATA code → fuzzy token match.
    """
    q_norm = _normalise(q)
    tokens = q_norm.split()

    # Tier 1: exact phrase match, longest phrase first
    for size in range(6, 0, -1):
        for i in range(len(tokens) - size + 1):
            phrase = " ".join(tokens[i: i + size])
            if phrase in _AIRPORT_ENTITIES:
                return _AIRPORT_ENTITIES[phrase]

    # Tier 2: IATA code
    for code in _IATA_RE.findall(q.upper()):
        if code in _AIRPORT_ENTITIES:
            return _AIRPORT_ENTITIES[code]

    # Tier 3: fuzzy match on individual tokens
    STOP_WORDS = {
        "what", "is", "the", "of", "in", "at", "which", "where",
        "airport", "how", "does", "do", "an", "a", "nation", "town",
        "quel", "est", "le", "la", "de", "du", "quelle", "aéroport",
        "dans", "se", "trouve",
        "ما", "هو", "في", "أي", "يقع", "مطار", "هي", "على",
        "ارتفاع", "نوع", "بلد", "دولة",
    }
    GEOGRAPHIC_NOISE = {
        "france", "italy", "history", "aviation", "naples", "pizza",
        "president", "book", "flight", "best", "germany", "large",
        "vienna", "airports", "runway", "elevation", "municipality",
        "located", "show", "list", "all", "highest", "lowest", "top",
        "longest", "shorter", "exceeds", "above", "below", "whose",
        "flying", "altitude", "speed", "knots", "vertical", "width",
        "length", "country", "continent", "surface", "type", "city",
    }
    candidates = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    for candidate in candidates:
        if candidate.lower() in GEOGRAPHIC_NOISE:
            continue
        result = process.extractOne(
            candidate,
            list(_AIRPORT_ENTITIES.keys()),
            scorer=fuzz.WRatio,
        )
        if result is not None:
            match, score, _ = result
            if score >= 92:
                return _AIRPORT_ENTITIES[match]
    return None


# ── UNIVERSITY ENTITY DETECTOR ──────────────────────────────────
def _detect_university_entity(q: str):
    """Regex-based LUBM entity detection."""
    m = _UNIVERSITY_ENTITY_RE.search(q)
    return m.group(1) if m else None


# ── SIGNAL DETECTORS ──────────────────────────────────────────────
def _has_open_kg_signal(q_lower: str) -> bool:
    return any(sig in q_lower for sig in _OPEN_KG_SIGNALS)


# detectors.py — _has_kg1_signal, replace the two _strip_arabic_al calls

def _has_kg1_signal(q_lower: str) -> bool:
    q_lower = _normalise_for_signal_match(q_lower)          # was _strip_arabic_al
    for sig in _KG1_ONLY_SIGNALS:
        sig_stripped = _normalise_for_signal_match(sig)     # was _strip_arabic_al
        if " " in sig_stripped:
            if sig_stripped in q_lower:
                return True
        else:
            if re.search(rf"\b{re.escape(sig_stripped)}\b", q_lower):
                return True
    return False


def _has_filter_signal(q: str) -> bool:
    q_lower = q.lower()
    return any(sig in q_lower for sig in _FILTER_SIGNALS)


# detectors.py — _has_count_signal, add one regex check before the loop

def _has_count_signal(q: str) -> bool:
    q_lower = q.lower()
    if re.search(r"كم\s+(يبلغ|تبلغ)", q_lower):
        return False
    # "combien de" elides to "combien d'" before a vowel — combien d'étudiants,
    # combien d'aéroports — so a plain substring check for "combien de" misses
    # every elided case. Handle it with a regex instead.
    if re.search(r"\bcombien\s+d[e']", q_lower):
        return True
    for sig in _COUNT_SIGNALS:
        if " " in sig:
            if sig in q_lower:
                return True
        else:
            if re.search(rf"\b{re.escape(sig)}\b", q_lower):
                return True
    return False


def _has_compare_signal(q: str) -> bool:
    q_lower = q.lower()
    for sig in _COMPARE_SIGNALS:
        if " " in sig:
            if sig in q_lower:
                return True
        else:
            if re.search(rf"\b{re.escape(sig)}\b", q_lower):
                return True
    return False


def _detect_compare_property(q: str) -> str | None:
    q_lower = q.lower()
    for keywords, prop in _COMPARE_PROPERTY_KEYWORDS:
        if any(k.lower() in q_lower for k in keywords):
            return prop
    return None


def _detect_two_airport_codes(q: str) -> list[str] | None:
    """
    Finds every IATA code in the question. Handles Arabic 'و' glued to Latin tokens.
    Returns exactly two distinct known codes, or None.
    """
    q = re.sub(r'و(?=[A-Z]{2,})', 'و ', q)
    codes = []
    for code in _IATA_RE.findall(q.upper()):
        if code in _AIRPORT_ENTITIES and code not in codes:
            codes.append(_AIRPORT_ENTITIES[code])
    return codes if len(codes) == 2 else None

def _detect_two_flight_numbers(q: str) -> list[str] | None:
    """Finds every flight-number match. Returns exactly two distinct
    flight numbers, or None."""
    numbers = []
    for num in _FLIGHT_RE.findall(q.upper()):
        if num not in numbers:
            numbers.append(num)
    return numbers if len(numbers) == 2 else None
def _has_group_ranking_signal(q: str) -> bool:
    q_lower = q.lower()
    return any(sig in q_lower for sig in _GROUP_RANKING_SIGNALS)

def _detect_compare_property_kg1(q: str) -> str | None:
    q_lower = q.lower()
    for keywords, prop in _COMPARE_PROPERTY_KEYWORDS_KG1:
        if any(k.lower() in q_lower for k in keywords):
            return prop
    return None