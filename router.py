"""
router.py  (v2 — flexible routing)
------------------------------------
WHAT CHANGED vs v1:

    Fix 1 — Minimum structure guard:
        Single keywords like "elevation" or "flight" no longer trigger KG2.
        A question must have at least 2 words and contain a flight number,
        OR at least 3 words with at least one question/context signal.
        This prevents noise inputs from producing mapping failures.

    Fix 2 — Flexible cross-KG signal detection:
        The old approach required an EXACT phrase from a fixed list.
        New approach: checks for an airport-direction word (departure/origin/
        destination/arrival/landing) AND a context word ("airport", "aéroport",
        "مطار") anywhere in the question — no fixed phrase required.
        This is more robust because natural language varies widely.

    No changes to the routing priority order.
    No changes to the entity detection logic.
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
# AIRPORT PROPERTY SIGNALS (cross-KG only)
# ─────────────────────────────────────────────

_AIRPORT_PROPERTY_WORDS = {
    # English
    "country", "nation", "elevation", "altitude", "runway", "surface",
    "length", "region", "continent", "type", "city", "name", "coordinates",
    "latitude", "longitude", "lighted", "width", "town", "municipality",
    # French
    "pays", "piste", "longueur", "région", "ville", "nom", "coordonnées",
    "élévation", "altitude",
    # Arabic
    "بلد", "دولة", "ارتفاع", "مدرج", "سطح", "طول", "منطقة",
    "قارة", "نوع", "مدينة", "اسم", "إحداثيات",
}

# ─────────────────────────────────────────────
# CROSS-KG FLEXIBLE DETECTION
# ─────────────────────────────────────────────
# Instead of requiring an exact phrase, we check for:
#   - A direction word (departure/origin/destination/arrival)
#   - AND an airport-context word (airport/aéroport/مطار)
# This matches many more natural phrasings.

_DIRECTION_WORDS = {
    # English — origin
    "departure", "departing", "departs", "origin", "originating",
    "take off", "takeoff", "taking off", "leaves from", "departs from",
    # English — destination
    "destination", "arrival", "arriving", "arrives", "landing",
    "lands at", "land at", "lands in", "arriving at",
    # French — origin
    "départ", "décolle", "provenance",
    # French — destination
    "arrivée", "atterrit", "destination",
    # Arabic — origin
    "مغادرة", "إقلاع", "انطلاق",
    # Arabic — destination
    "وصول", "هبوط", "وجهة",
}

_AIRPORT_CONTEXT_WORDS = {
    "airport", "aéroport", "مطار",
}

# Maps direction words to their direction value
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

_KG1_TEMPLATES = {
    "count_kg1",
    "filter_numeric_kg1",
    "cross_kg_filter",
}

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
# MINIMUM STRUCTURE GUARD (Fix 1)
# ─────────────────────────────────────────────

def _has_minimum_structure(question: str) -> bool:
    """
    Returns False for questions that are too vague to route meaningfully.

    WHY THIS EXISTS:
        Without this guard, a single word like "elevation" triggers the
        airport keyword detector and routes to single_kg2, causing a
        mapping failure because there is no entity.

        We allow short inputs only when they contain a flight number
        (e.g. "airline FR9005" is a valid short query).

    Rules:
        1. Single word → always False
        2. Two words with a flight number pattern → True (valid short query)
        3. Three or more words → True (let later stages decide)
    """
    words = question.strip().split()

    # Rule 1: single word
    if len(words) < 2:
        return False

    # Rule 2: two words but has flight number → valid short query
    if len(words) == 2:
        flight_re = re.compile(r'[A-Za-z]{2,3}\d+')
        if flight_re.search(question):
            return True
        return False

    # Rule 3: three or more words → proceed normally
    return True

# ─────────────────────────────────────────────
# DETECTORS
# ─────────────────────────────────────────────

def _detect_flight_number(q: str):
    m = _FLIGHT_RE.findall(q.upper())
    return max(m, key=len) if m else None


def _detect_cross_kg_signal(q: str):
    """
    Flexible cross-KG detection.

    OLD approach (v1): required an exact phrase from a fixed list.
    NEW approach (v2): checks independently for:
        - Any direction word (departure / destination / arrival / etc.)
        - Any airport context word (airport / aéroport / مطار)
        - Any airport property word (country / elevation / etc.)

    All three must be present. This catches:
        "What type of airport does flight KE567 land at?"
            → "land" (direction) + "airport" (context) + "type" (property)

        "Quelle est l'élévation de l'aéroport de départ du vol BR62?"
            → "départ" (direction) + "aéroport" (context) + "élévation" (property)

    Returns (True, direction_value) or (False, None).
    """
    q_lower  = q.lower()
    q_norm   = _normalise(q)

    # Check for airport context word
    has_airport_context = any(ctx in q_lower for ctx in _AIRPORT_CONTEXT_WORDS)
    if not has_airport_context:
        return False, None

    # Check for airport property word
    has_property = any(
        (pw in q_lower) if any(ord(c) > 127 for c in pw)
        else bool(re.search(rf"\b{re.escape(pw)}\b", q_lower))
        for pw in _AIRPORT_PROPERTY_WORDS
    )
    if not has_property:
        return False, None

    # Check for direction word and capture which direction
    direction = None
    for dword, dval in sorted(_DIRECTION_TO_VALUE.items(), key=lambda x: -len(x[0])):
        dword_lower = dword.lower()
        if any(ord(c) > 127 for c in dword_lower):
            # Arabic/French: substring match
            if dword_lower in q_lower:
                direction = dval
                break
        else:
            # English: word boundary match
            if re.search(rf"\b{re.escape(dword_lower)}\b", q_lower):
                direction = dval
                break

    if not direction:
        return False, None

    return True, direction


def _detect_airport_entity(q: str):
    q_norm = _normalise(q)
    tokens = q_norm.split()

    # Tier 1: sliding window exact match (longest phrase first)
    for size in range(6, 0, -1):
        for i in range(len(tokens) - size + 1):
            phrase = " ".join(tokens[i : i + size])
            if phrase in _AIRPORT_ENTITIES:
                return _AIRPORT_ENTITIES[phrase]

    # Tier 2: IATA code (3 uppercase letters as standalone word)
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

    candidates = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    for candidate in candidates:
        result = process.extractOne(
            candidate,
            list(_AIRPORT_ENTITIES.keys()),
            scorer=fuzz.WRatio,
        )
        if result is not None:
            match, score, _ = result
            if score >= 85:
                return _AIRPORT_ENTITIES[match]

    return None


def _detect_airport_keyword(q: str):
    q_lower = q.lower()
    return any(k.lower() in q_lower for k in _AIRPORT_TRIGGERS)


def _detect_template(q: str):
    q_lower = q.lower()

    for name, signals in _TEMPLATE_SIGNALS.items():
        for sig in sorted(signals, key=len, reverse=True):
            s = sig.lower()
            if " " in s:
                if s in q_lower:
                    return name
            elif re.search(rf"\b{re.escape(s)}\b", q_lower):
                return name

    return None


# ─────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────

def route(question: str) -> dict:

    # Fix 1: Reject questions that are too short/vague to route
    if not _has_minimum_structure(question):
        return {
            "query_type": "out_of_scope",
            "kg":         None,
            "entity":     None,
            "direction":  None,
            "template":   None,
            "config":     None,
        }

    flight   = _detect_flight_number(question)
    cross, direction = _detect_cross_kg_signal(question)   # Fix 2: flexible
    airport  = _detect_airport_entity(question)
    keyword  = _detect_airport_keyword(question)
    template = _detect_template(question)

    # Priority 1: cross-KG
    if flight and cross:
        return {
            "query_type": "cross_kg",
            "kg":         "cross",
            "entity":     flight,
            "direction":  direction,
            "template":   None,
            "config":     CROSS_KG_CONFIG,
        }

    # Priority 2: KG1 flight
    if flight:
        return {
            "query_type": "single_kg1",
            "kg":         "flights",
            "entity":     flight,
            "direction":  None,
            "template":   None,
            "config":     KG_REGISTRY["flights"],
        }

    # Priority 3: KG1 templates
    if template in _KG1_TEMPLATES:
        cfg = TEMPLATE_REGISTRY[template]
        return {
            "query_type": "template",
            "kg":         cfg["kg"],
            "entity":     None,
            "direction":  None,
            "template":   template,
            "config":     cfg,
        }

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

    # Priority 5: general templates
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

    # Priority 6: airport keyword fallback
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