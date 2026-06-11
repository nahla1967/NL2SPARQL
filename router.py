"""
router.py
---------
Deterministic rule-based router for KG selection.

No LLM. Pure classification using regex + lexicon signals.
"""

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
_IATA_RE = re.compile(r"\b([A-Z]{3})\b")

# ─────────────────────────────────────────────
# AIRPORT PROPERTY SIGNALS (cross-KG only)
# ─────────────────────────────────────────────

_AIRPORT_PROPERTY_WORDS = {
    "country", "elevation", "altitude", "runway", "surface", "length",
    "region", "continent", "type", "city", "name", "coordinates",
    "latitude", "longitude", "lighted", "width",
    "pays", "piste", "longueur", "région", "ville", "nom", "coordonnées",
    "بلد", "دولة", "ارتفاع", "مدرج", "سطح", "طول", "منطقة",
    "قارة", "نوع", "مدينة", "اسم", "إحداثيات",
}

# ─────────────────────────────────────────────
# LEXICON LOAD
# ─────────────────────────────────────────────

def _load_airport_lexicon():
    with open(get_lexicon("airports"), encoding="utf-8") as f:
        return json.load(f)

_airport_lex = _load_airport_lexicon()

_AIRPORT_ENTITIES = _airport_lex.get("airport_entities", {})
_CROSS_KG_SIGNALS = _airport_lex.get("cross_kg_signals", {})

_TEMPLATE_SIGNALS = {
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
    text = text.replace("'", " ").replace("’", " ").replace("‘", " ")
    text = text.replace("؟", " ")
    text = re.sub(r"[^\w\s\u0600-\u06FE]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ─────────────────────────────────────────────
# DETECTORS
# ─────────────────────────────────────────────

def _detect_flight_number(q: str):
    m = _FLIGHT_RE.findall(q.upper())
    return max(m, key=len) if m else None


def _detect_cross_kg_signal(q: str):
    q_norm = _normalise(q)
    q_lower = q.lower()

    direction = None

    for phrase, meta in _CROSS_KG_SIGNALS.items():
        if _normalise(phrase) in q_norm:
            direction = meta.get("direction")
            break

    if not direction:
        return False, None

    has_property = any(
        (pw in q_lower) if any(ord(c) > 127 for c in pw)
        else bool(re.search(rf"\b{re.escape(pw)}\b", q_lower))
        for pw in _AIRPORT_PROPERTY_WORDS
    )

    if not has_property:
        return False, None

    return True, direction


def _detect_airport_entity(q: str):
    q_norm = _normalise(q)
    tokens = q_norm.split()

    # Sliding window — longest phrase first (handles "frankfurt airport", "مطار فيينا")
    for size in range(4, 0, -1):
        for i in range(len(tokens) - size + 1):
            phrase = " ".join(tokens[i:i+size])
            if phrase in _AIRPORT_ENTITIES:
                return _AIRPORT_ENTITIES[phrase]

    # IATA code fallback — 3 uppercase letters as standalone word
    for code in _IATA_RE.findall(q.upper()):
        if code in _AIRPORT_ENTITIES:
            return _AIRPORT_ENTITIES[code]

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

    flight = _detect_flight_number(question)
    cross, direction = _detect_cross_kg_signal(question)
    airport = _detect_airport_entity(question)
    keyword = _detect_airport_keyword(question)
    template = _detect_template(question)

    # 1. cross-KG (strongest constraint)
    if flight and cross:
        return {
            "query_type": "cross_kg",
            "kg": "cross",
            "entity": flight,
            "direction": direction,
            "template": None,
            "config": CROSS_KG_CONFIG,
        }

    # 2. KG1 flight
    if flight:
        return {
            "query_type": "single_kg1",
            "kg": "flights",
            "entity": flight,
            "direction": None,
            "template": None,
            "config": KG_REGISTRY["flights"],
        }

    # 3. KG1 templates
    if template in _KG1_TEMPLATES:
        cfg = TEMPLATE_REGISTRY[template]
        return {
            "query_type": "template",
            "kg": cfg["kg"],
            "entity": None,
            "direction": None,
            "template": template,
            "config": cfg,
        }

    # 4. KG2 airport entity
    if airport:
        return {
            "query_type": "single_kg2",
            "kg": "airports",
            "entity": airport,
            "direction": None,
            "template": None,
            "config": KG_REGISTRY["airports"],
        }

    # 5. general templates
    if template:
        cfg = TEMPLATE_REGISTRY[template]
        return {
            "query_type": "template",
            "kg": cfg["kg"],
            "entity": None,
            "direction": None,
            "template": template,
            "config": cfg,
        }

    # 6. airport keyword fallback
    if keyword:
        return {
            "query_type": "single_kg2",
            "kg": "airports",
            "entity": None,
            "direction": None,
            "template": None,
            "config": KG_REGISTRY["airports"],
        }

    # fallback
    return {
        "query_type": "out_of_scope",
        "kg": None,
        "entity": None,
        "direction": None,
        "template": None,
        "config": None,
    }