"""
router.py
---------
Reads a question and decides:
  1. Which KG to query (flights, airports, cross, template)
  2. What query type it is (single_kg1, single_kg2, cross_kg, template)
  3. What the main entity is (flight number or IATA code)
  4. What direction for cross-KG (origin or destination)

DESIGN:
    No LLM used. Routing is a classification task with well-defined,
    enumerable categories. Deterministic rule-based detection guarantees
    correctness and eliminates LLM variance at the most critical gate.

PRIORITY ORDER:
    1. cross_kg     — cross-KG signal + flight number + airport property word
    2. single_kg1   — flight number detected (regex only)
    3. single_kg2   — airport IATA or name detected (lexicon lookup)
    4. template     — filter/ranking/comparison/count signal detected
    5. single_kg2   — airport keyword fallback
    6. out_of_scope — nothing matched

CROSS-KG DETECTION RULE:
    A question is cross-KG ONLY when it has:
      (a) a flight number  AND
      (b) a cross-KG direction signal (departure/arrival airport)  AND
      (c) an airport property word (country, elevation, runway...)
    Without (c), 'What is the arrival airport of OS295?' is KG1 only —
    it asks for the IATA code itself, not a property of the airport.
"""

import re
import json
from kg_registry import (
    KG_REGISTRY,
    CROSS_KG_CONFIG,
    TEMPLATE_REGISTRY,
    get_lexicon,
)

# ── REGEX ─────────────────────────────────────────────────────────────────────
_FLIGHT_RE = re.compile(r'\b([A-Z]{2,3}\d+)\b')
_IATA_RE   = re.compile(r'\b([A-Z]{3})\b')

# Airport property words required for cross-KG detection
_AIRPORT_PROPERTY_WORDS = [
    "country", "elevation", "altitude", "runway", "surface", "length",
    "region", "continent", "type", "city", "name", "coordinates",
    "latitude", "longitude", "lighted", "width",
    "pays", "piste", "longueur", "région", "ville", "nom", "coordonnées",
    "بلد", "دولة", "ارتفاع", "مدرج", "سطح", "طول", "منطقة",
    "قارة", "نوع", "مدينة", "اسم", "إحداثيات",
]

# ── LOAD LEXICON ONCE ─────────────────────────────────────────────────────────
def _load_airport_lexicon() -> dict:
    with open(get_lexicon("airports"), encoding="utf-8") as f:
        return json.load(f)

_airport_lex      = _load_airport_lexicon()
_AIRPORT_ENTITIES = _airport_lex.get("airport_entities", {})
_CROSS_KG_SIGNALS = _airport_lex.get("cross_kg_signals", {})
_TEMPLATE_SIGNALS = {
    name: config["signals"]
    for name, config in _airport_lex.get("template_triggers", {}).items()
}
_AIRPORT_TRIGGERS = KG_REGISTRY["airports"].get("triggers", [])


# ── NORMALISE ─────────────────────────────────────────────────────────────────
def _normalise(text: str) -> str:
    """
    Lowercase + normalize apostrophes + strip punctuation + collapse whitespace.
    Handles both Latin and Arabic text.
    Arabic question mark '؟' (U+061F) is explicitly removed since it falls
    inside the Arabic Unicode block but is punctuation, not a letter.
    """
    text = text.lower()
    # Normalize apostrophes to space
    text = text.replace("'", " ").replace("\u2019", " ").replace("\u2018", " ")
    # Remove Arabic question mark explicitly
    text = text.replace("\u061F", " ")
    # Remove all remaining punctuation (keep Arabic letters, Latin, digits, spaces)
    text = re.sub(r"[^\w\s\u0600-\u06FE]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ── DETECTION HELPERS ─────────────────────────────────────────────────────────

def _detect_flight_number(question: str) -> str | None:
    matches = _FLIGHT_RE.findall(question.upper())
    if not matches:
        return None
    return max(matches, key=len)


def _detect_cross_kg_signal(question: str) -> tuple[bool, str | None]:
    q_norm  = _normalise(question)
    q_lower = question.lower()

    signal_found = False
    direction    = None

    for phrase, meta in _CROSS_KG_SIGNALS.items():
        if _normalise(phrase) in q_norm:
            signal_found = True
            direction    = meta.get("direction")
            break

    if not signal_found:
        return False, None

    # Require at least one airport property word
    if not any(pw in q_lower for pw in _AIRPORT_PROPERTY_WORDS):
        return False, None

    return True, direction


def _detect_airport_entity(question: str) -> str | None:
    q_norm   = _normalise(question)
    q_tokens = q_norm.split()

    # Sliding window — longest phrase first
    for size in range(4, 0, -1):
        for i in range(len(q_tokens) - size + 1):
            phrase = " ".join(q_tokens[i:i + size])
            if phrase in _AIRPORT_ENTITIES:
                return _AIRPORT_ENTITIES[phrase]

    # Bare IATA code (3 uppercase letters)
    for code in _IATA_RE.findall(question):
        if code in _AIRPORT_ENTITIES:
            return _AIRPORT_ENTITIES[code]

    return None


def _detect_airport_keyword(question: str) -> bool:
    q_lower = question.lower()
    return any(kw.lower() in q_lower for kw in _AIRPORT_TRIGGERS)


def _detect_template(question: str) -> str | None:
    q_lower = question.lower()
    for template_name, signals in _TEMPLATE_SIGNALS.items():
        for signal in sorted(signals, key=len, reverse=True):
            sig = signal.lower()
            if " " in sig:
                if sig in q_lower:
                    return template_name
            else:
                if re.search(rf"\b{re.escape(sig)}\b", q_lower):
                    return template_name
    return None


# ── MAIN ROUTER ───────────────────────────────────────────────────────────────

def route(question: str) -> dict:
    """
    Returns a routing decision dict:
    {
        query_type : single_kg1 | single_kg2 | cross_kg | template | out_of_scope
        kg         : flights | airports | cross | None
        entity     : flight number or IATA code or None
        direction  : origin | destination | None
        template   : template name or None
        config     : relevant config dict or None
    }
    """
    flight_number           = _detect_flight_number(question)
    cross_signal, direction = _detect_cross_kg_signal(question)
    airport_entity          = _detect_airport_entity(question)
    airport_keyword         = _detect_airport_keyword(question)
    template_name           = _detect_template(question)

    # Priority 1 — cross_kg
    if cross_signal and flight_number:
        return {
            "query_type": "cross_kg",
            "kg":         "cross",
            "entity":     flight_number,
            "direction":  direction,
            "template":   None,
            "config":     CROSS_KG_CONFIG,
        }

    # Priority 2 — single_kg1
    if flight_number:
        return {
            "query_type": "single_kg1",
            "kg":         "flights",
            "entity":     flight_number,
            "direction":  None,
            "template":   None,
            "config":     KG_REGISTRY["flights"],
        }

    # Priority 2b — KG1 template pre-check
    # Must run BEFORE entity detection because questions like
    # 'How many flights go to Munich?' contain an airport name (Munich→MUC)
    # but are fundamentally about flights, not airports.
    # KG1 template signals unambiguously identify these questions.
    _KG1_TEMPLATES = ['count_kg1', 'filter_numeric_kg1', 'cross_kg_filter']
    if template_name and template_name in _KG1_TEMPLATES:
        tmpl_config = TEMPLATE_REGISTRY[template_name]
        return {
            "query_type": "template",
            "kg":         tmpl_config["kg"],
            "entity":     None,
            "direction":  None,
            "template":   template_name,
            "config":     tmpl_config,
        }

    # Priority 3 — single_kg2 (entity match)
    if airport_entity:
        return {
            "query_type": "single_kg2",
            "kg":         "airports",
            "entity":     airport_entity,
            "direction":  None,
            "template":   None,
            "config":     KG_REGISTRY["airports"],
        }

    # Priority 4 — template
    if template_name:
        tmpl_config = TEMPLATE_REGISTRY[template_name]
        return {
            "query_type": "template",
            "kg":         tmpl_config["kg"],
            "entity":     None,
            "direction":  None,
            "template":   template_name,
            "config":     tmpl_config,
        }

    # Priority 4b — single_kg2 keyword fallback
    if airport_keyword:
        return {
            "query_type": "single_kg2",
            "kg":         "airports",
            "entity":     None,
            "direction":  None,
            "template":   None,
            "config":     KG_REGISTRY["airports"],
        }

    # Priority 5 — out of scope
    return {
        "query_type": "out_of_scope",
        "kg":         None,
        "entity":     None,
        "direction":  None,
        "template":   None,
        "config":     None,
    }