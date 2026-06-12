"""
router.py  (v4 — priority reorder + template disambiguation)
-------------------------------------------------------------
CHANGES vs v3:

    Change 1 — All templates now check at Priority 3, before airport entity detection.
               Previously, only KG1 templates were at P3; all other templates came
               after the airport entity detector at P4. This allowed fuzzy matching
               on words like "Germany", "Vienna", "large" to steal questions that
               should have gone to the template branch.

    Change 2 — _detect_template now enforces an explicit priority order.
               ranking_kg2 is checked before filter_string_kg2, so questions
               containing both "airports with" and "top"/"highest" are
               correctly resolved to ranking_kg2, not filter_string_kg2.

    Change 3 — Numeric template disambiguation by domain.
               A small set of flight-property keywords (speed, knots, altitude,
               vertical) is checked before falling through to filter_numeric_kg2.
               If a flight-property word is present, the question resolves to
               filter_numeric_kg1 instead of the airport numeric template.
"""

from rapidfuzz import process, fuzz
import re
import json
from kg_registry import (
    KG_REGISTRY,
    CROSS_KG_CONFIG,
    TEMPLATE_REGISTRY,
    get_lexicon,
)

# ─────────────────────────────────────────────
# REGEX
# ─────────────────────────────────────────────

_FLIGHT_RE = re.compile(r"\b([A-Z]{2,3}\d+)\b")
_IATA_RE   = re.compile(r"\b([A-Z]{3})\b")

# ─────────────────────────────────────────────
# CHANGE 3 — flight-property keyword set
# These words indicate a KG1 (flight) numeric question,
# not a KG2 (airport) numeric question.
# When any of these appear, filter_numeric_kg1 wins over
# filter_numeric_kg2 for numeric operator questions.
# ─────────────────────────────────────────────

_FLIGHT_NUMERIC_KEYWORDS = {
    # English
    "speed", "knots", "altitude", "vertical", "ground speed",
    "gspeed", "vspeed", "feet per minute",
    "flying",        # ← add this
    "flight level",
    # French
    "vitesse", "nœuds", "altitude",
    # Arabic
    "سرعة", "عقدة", "ارتفاع الطائرة",
}

# ─────────────────────────────────────────────
# CHANGE 2 — explicit template priority order
# This list defines the order in which _detect_template
# checks template signals. First match wins, so more
# specific templates must appear before generic ones.
# ranking_kg2 must be before filter_string_kg2 because
# ranking questions also contain "airports with".
# cross_kg_filter must be early because its signals
# (flights + airport property) overlap with KG1 templates.
# ─────────────────────────────────────────────

_TEMPLATE_PRIORITY_ORDER = [
    "cross_kg_filter",        # most specific: flights + airport property
    "compare_two_airports",   # two IATA codes present
    "ranking_kg2",            # top/bottom/highest/lowest
    "filter_numeric_kg2",     # numeric threshold on airport property
    "filter_string_kg2",      # categorical filter on airport property
    "count_kg1",              # how many flights
    "filter_numeric_kg1",     # numeric threshold on flight property
]

# ─────────────────────────────────────────────
# AIRPORT PROPERTY SIGNALS (cross-KG only)
# ─────────────────────────────────────────────

_AIRPORT_PROPERTY_WORDS = {
    "country", "nation", "elevation", "altitude", "runway", "surface",
    "length", "region", "continent", "type", "city", "name", "coordinates",
    "latitude", "longitude", "lighted", "width", "town", "municipality",
    "pays", "piste", "longueur", "région", "ville", "nom", "coordonnées",
    "élévation", "altitude",
    "بلد", "دولة", "ارتفاع", "مدرج", "سطح", "طول", "منطقة",
    "قارة", "نوع", "مدينة", "اسم", "إحداثيات",
}

# ─────────────────────────────────────────────
# CROSS-KG FLEXIBLE DETECTION
# ─────────────────────────────────────────────

_DIRECTION_WORDS = {
    "departure", "departing", "departs", "origin", "originating",
    "take off", "takeoff", "taking off", "leaves from", "departs from",
    "destination", "arrival", "arriving", "arrives", "landing",
    "lands at", "land at", "lands in", "arriving at",
    "départ", "décolle", "provenance",
    "arrivée", "atterrit", "destination",
    "مغادرة", "إقلاع", "انطلاق",
    "وصول", "هبوط", "وجهة",
}

_AIRPORT_CONTEXT_WORDS = {
    "airport", "aéroport", "مطار",
}

_DIRECTION_TO_VALUE = {
    "departure": "origin",    "departing": "origin",  "departs": "origin",
    "origin": "origin",       "originating": "origin", "take off": "origin",
    "takeoff": "origin",      "taking off": "origin",  "leaves from": "origin",
    "departs from": "origin", "départ": "origin",      "décolle": "origin",
    "provenance": "origin",   "مغادرة": "origin",      "إقلاع": "origin",
    "انطلاق": "origin",

    "destination": "destination", "arrival": "destination",
    "arriving": "destination",    "arrives": "destination",
    "landing": "destination",     "lands at": "destination",
    "land at": "destination",     "lands in": "destination",
    "arriving at": "destination", "arrivée": "destination",
    "atterrit": "destination",    "وصول": "destination",
    "هبوط": "destination",        "وجهة": "destination",
}

# ─────────────────────────────────────────────
# LEXICON LOAD
# ─────────────────────────────────────────────

def _load_airport_lexicon():
    with open(get_lexicon("airports"), encoding="utf-8") as f:
        return json.load(f)

_airport_lex = _load_airport_lexicon()

_AIRPORT_ENTITIES  = _airport_lex.get("airport_entities", {})
_TEMPLATE_SIGNALS  = {
    name: cfg["signals"]
    for name, cfg in _airport_lex.get("template_triggers", {}).items()
}
_AIRPORT_TRIGGERS = KG_REGISTRY["airports"].get("triggers", [])

# ─────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────

def _normalise(text: str) -> str:
    text = text.lower()
    text = text.replace("'", " ").replace("\u2019", " ").replace("\u2018", " ")
    text = text.replace("\u061F", " ")
    text = re.sub(r"[^\w\s\u0600-\u06FE]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ─────────────────────────────────────────────
# MINIMUM STRUCTURE GUARD
# ─────────────────────────────────────────────

def _has_minimum_structure(question: str) -> bool:
    words = question.strip().split()
    if len(words) < 2:
        return False
    if len(words) == 2:
        flight_re = re.compile(r'[A-Za-z]{2,3}\d+')
        iata_re   = re.compile(r'\b[A-Z]{3}\b')
        if flight_re.search(question):
            return True
        if iata_re.search(question.upper()):
            return True
        return False
    return True

# ─────────────────────────────────────────────
# DETECTORS
# ─────────────────────────────────────────────

def _detect_flight_number(q: str):
    m = _FLIGHT_RE.findall(q.upper())
    return max(m, key=len) if m else None


def _detect_cross_kg_signal(q: str):
    q_lower = q.lower()
    has_airport_context = any(ctx in q_lower for ctx in _AIRPORT_CONTEXT_WORDS)
    if not has_airport_context:
        return False, None
    has_property = any(
        (pw in q_lower) if any(ord(c) > 127 for c in pw)
        else bool(re.search(rf"\b{re.escape(pw)}\b", q_lower))
        for pw in _AIRPORT_PROPERTY_WORDS
    )
    if not has_property:
        return False, None
    direction = None
    for dword, dval in sorted(_DIRECTION_TO_VALUE.items(), key=lambda x: -len(x[0])):
        dword_lower = dword.lower()
        if any(ord(c) > 127 for c in dword_lower):
            if dword_lower in q_lower:
                direction = dval
                break
        else:
            if re.search(rf"\b{re.escape(dword_lower)}\b", q_lower):
                direction = dval
                break
    if not direction:
        return False, None
    return True, direction


def _detect_airport_entity(q: str):
    q_norm = _normalise(q)
    tokens = q_norm.split()
    for size in range(6, 0, -1):
        for i in range(len(tokens) - size + 1):
            phrase = " ".join(tokens[i : i + size])
            if phrase in _AIRPORT_ENTITIES:
                return _AIRPORT_ENTITIES[phrase]
    for code in _IATA_RE.findall(q.upper()):
        if code in _AIRPORT_ENTITIES:
            return _AIRPORT_ENTITIES[code]
    STOP_WORDS = {
        "what", "is", "the", "of", "in", "at", "which", "where",
        "airport", "how", "does", "do", "an", "a", "nation", "town",
        "quel", "est", "le", "la", "de", "du", "quelle", "aéroport",
        "dans", "se", "trouve",
        "ما", "هو", "في", "أي", "يقع", "مطار", "هي", "على",
        "ارتفاع", "نوع", "بلد", "دولة",
    }
    candidates = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    GEOGRAPHIC_NOISE = {
        "france", "italy", "history", "aviation", "naples",
        "pizza", "president", "book", "flight", "best",
        # CHANGE 1 — added words that caused wrong_route:single_kg2
        # These are common words in filter/ranking questions that
        # should never match as airport entity names.
        "germany", "large", "vienna", "airports", "runway",
        "elevation", "municipality", "located", "show", "list",
        "all", "highest", "lowest", "top", "longest", "shorter",
        "exceeds", "above", "below", "whose",
    }
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


def _detect_airport_keyword(q: str):
    q_lower = q.lower()
    return any(k.lower() in q_lower for k in _AIRPORT_TRIGGERS)


# CHANGE 2 + CHANGE 3 — rewritten _detect_template
# Now iterates in _TEMPLATE_PRIORITY_ORDER, not dict order.
# For numeric templates, domain disambiguation runs first:
# if flight-property keywords are present, skip filter_numeric_kg2
# and return filter_numeric_kg1 directly.

def _detect_template(q: str):
    q_lower = q.lower()

    # CHANGE 3 — check for flight numeric keywords before looping
    # If the question is about flight speed / altitude, we resolve
    # filter_numeric_kg1 immediately without risking filter_numeric_kg2
    # stealing the match via generic signals like "above" or "below".
    has_flight_numeric = any(
        (kw in q_lower) if " " in kw
        else bool(re.search(rf"\b{re.escape(kw)}\b", q_lower))
        for kw in _FLIGHT_NUMERIC_KEYWORDS
    )
    if has_flight_numeric:
        # Only apply if there are also numeric operator signals present.
        # This prevents bare mentions of "speed" routing here without a filter.
        numeric_operator_signals = ["above", "below", "more than", "less than",
                                    "greater than", "exceeds", "over", "under",
                                    "faster than", "slower than"]
        has_operator = any(sig in q_lower for sig in numeric_operator_signals)
        if has_operator:
            return "filter_numeric_kg1"

    # CHANGE 2 — iterate in explicit priority order, not dict insertion order
    for template_name in _TEMPLATE_PRIORITY_ORDER:
        signals = _TEMPLATE_SIGNALS.get(template_name, [])
        for sig in sorted(signals, key=len, reverse=True):
            s = sig.lower()
            if " " in s:
                if s in q_lower:
                    return template_name
            elif re.search(rf"\b{re.escape(s)}\b", q_lower):
                return template_name

    return None


# ─────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────

def route(question: str) -> dict:

    if not _has_minimum_structure(question):
        return {
            "query_type": "out_of_scope",
            "kg":         None,
            "entity":     None,
            "direction":  None,
            "template":   None,
            "config":     None,
        }

    flight            = _detect_flight_number(question)
    cross, direction  = _detect_cross_kg_signal(question)
    template          = _detect_template(question)    # CHANGE 1 — moved up
    airport           = None                          # CHANGE 1 — deferred
    keyword           = None                          # CHANGE 1 — deferred

    # Priority 1: cross-KG (flight number + airport property + direction)
    if flight and cross:
        return {
            "query_type": "cross_kg",
            "kg":         "cross",
            "entity":     flight,
            "direction":  direction,
            "template":   None,
            "config":     CROSS_KG_CONFIG,
        }

    # Priority 2: KG1 single flight lookup
    if flight:
        return {
            "query_type": "single_kg1",
            "kg":         "flights",
            "entity":     flight,
            "direction":  None,
            "template":   None,
            "config":     KG_REGISTRY["flights"],
        }

    # CHANGE 1 — Priority 3: ALL templates, before any entity detection.
    # Previously this was split: KG1 templates at P3, KG2 templates at P5.
    # Merging them here means fuzzy entity matching never steals template
    # questions, regardless of which domain the template targets.
    if template:
        cfg = TEMPLATE_REGISTRY[template]
        return {
            "query_type": "template",
            "kg":         cfg["kg"],
            "entity":     None,
            "direction":  None,
            "template":   template,
            "config":     cfg,
        }

    # CHANGE 1 — entity detection now runs only after templates failed.
    # This is the key structural fix.
    airport = _detect_airport_entity(question)
    keyword = _detect_airport_keyword(question)

    # Priority 4: KG2 airport entity
    if airport:
        return {
            "query_type": "single_kg2",
            "kg":         "airports",
            "entity":     airport,
            "direction":  None,
            "template":   None,
            "config":     KG_REGISTRY["airports"],
        }

    # Priority 5: airport keyword fallback
    if keyword:
        return {
            "query_type": "single_kg2",
            "kg":         "airports",
            "entity":     None,
            "direction":  None,
            "template":   None,
            "config":     KG_REGISTRY["airports"],
        }

    # Fallback
    return {
        "query_type": "out_of_scope",
        "kg":         None,
        "entity":     None,
        "direction":  None,
        "template":   None,
        "config":     None,
    }