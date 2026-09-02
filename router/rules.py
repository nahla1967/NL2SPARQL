"""
Constants, regexes, signal word sets, and normalization utilities
for the NL2SPARQL router.
"""

import re

# ── REGEX ──────────────────────────────────────────────────────────
_FLIGHT_RE = re.compile(r"\b([A-Z]{2,3}\d+)\b")
_IATA_RE   = re.compile(r"\b([A-Z]{3})\b")

# ── KG1-ONLY SIGNAL WORDS ──────────────────────────────────────────
_KG1_ONLY_SIGNALS = {
    # English
    "gate", "terminal", "callsign", "squawk",
    "ground speed", "vertical speed", "airline", "operated by", "operates flight",
    # French
    "porte", "indicatif", "vitesse sol", "vitesse verticale",
    "compagnie aérienne", "opère le vol", "exploité par",
    # Arabic
    "بوابة", "مبنى", "صالة", "إشارة النداء", "الإشارة",
    "سرعة أرضية", "سرعة عمودية", "شركة الطيران",
    "تشغل الرحلة", "تشغل رحلة",
    
    "departure city", "origin city", "destination city",
    "ville de départ", "ville de destination", "ville d'origine",
    "مدينة مغادرة", "مدينة انطلاق", "مدينة وصول",
}

# ── OPEN-KG SIGNAL WORDS ─────────────────────────────────────────
_OPEN_KG_SIGNALS = {
    "registration number", "registration no",
    "numéro d'immatriculation", "numero d'immatriculation", "immatriculation",
    "رقم التسجيل", "رقم تسجيل",
}

# ── COUNT / FILTER / COMPARE SIGNALS ─────────────────────────────
_COUNT_SIGNALS = [
    "how many", "combien", "كم", "list all", "count"
]
_FILTER_SIGNALS = [
    "which professors", "which students", "who works for",
    "who is a member of", "list the professors", "list the students",
    "quels professeurs", "quels étudiants", "أي أستاذ", "أي طالب",
    "من هم", "اذكر", "listez",
]
_COMPARE_SIGNALS = ["compare", "comparer", "comparez", "vs", "versus",
                     "قارن", "مقارنة", "أيهما", "أيها", "أيّ منهما"]
# ── COMPARE PROPERTY KEYWORDS ─────────────────────────────────────
_COMPARE_PROPERTY_KEYWORDS = [
    (["elevation", "altitude", "height", "élévation", "ارتفاع", "أعلى", "أدنى"], "elevationFt"),
    (["runway length", "length", "longer", "longest", "longueur", "أطول", "أقصر", "طول"], "lengthFt"),
    (["runway width", "width", "wider", "widest", "largeur", "أعرض", "أضيق", "عرض"], "widthFt"),
    (["airport type", "type", "kind", "نوع"], "airportType"),
]
# AFTER
# NOTE (fix): the gold Arabic questions phrase these properties WITHOUT
# the definite article ("سرعة عمودية" / "سرعة أرضية"), not with it
# ("السرعة العمودية" / "السرعة الأرضية"). A plain substring match against
# only the definite form fails, so compare_property_kg1 came back None
# and Priority 1.9's `two_flights and compare_property_kg1` guard failed
# silently, falling through to open_kg. See compare_two_flights_001/002
# (ar) in the eval log. Adding the indefinite forms alongside the
# existing definite ones — additive only, can't affect any current match.
_COMPARE_PROPERTY_KEYWORDS_KG1 = [
    (["vertical speed", "vitesse verticale", "vspeed",
      "السرعة العمودية", "سرعة عمودية"], "vspeed"),
    (["ground speed", "vitesse sol", "vitesse au sol", "gspeed",
      "السرعة الأرضية", "سرعة أرضية"], "gspeed"),
]
# ── UNIVERSITY ENTITY REGEX ─────────────────────────────────────
_UNIVERSITY_ENTITY_RE = re.compile(
    r'\b((?:[A-Z][a-zA-Z]*)?(?:Professor|Student|Course|Department|Group|'
    r'University|Lecturer|Publication)\d+)\b'
)

# ── WH-WORDS (fast-path negative checks) ─────────────────────────
_WH_WORDS = (
    "what", "which", "where", "when", "how", "who",
    "quel", "quelle", "quels", "quelles", "où", "quand", "comment", "qui", "combien",
    "ما", "ماذا", "أي", "أين", "متى", "كيف", "من", "كم",
)

# ── RANKING / ASC SIGNALS (used by smart reroutes) ───────────────
# ENHANCED _RANKING_SIGNALS with context-aware expansion
_RANKING_SIGNALS = [
  
    # English
    "highest", "lowest", "most", "least", "top", "bottom",
    "largest", "biggest", "greatest", "smallest",
    "fastest", "slowest", "fastest descent", "highest descent",
    "steepest", "fastest climb", "most rapid",
    # French
    "plus haut", "plus bas", "plus", "moins",
    "plus rapide", "plus lent", "plus élevé", "plus bas",
    "plus raide", "descente la plus rapide",
    # Arabic
    "أعلى", "أدنى", "أكثر", "أقل",
    "أسرع", "أبطأ", "الأسرع", "الأبطأ",
    "أكثر انحدارا", "أكثر ارتفاعا"
]
_ASC_SIGNALS = [
    "shortest", "lowest", "smallest", "narrowest",
    "la plus courte", "la plus basse", "la plus petite", "la plus étroite",
    "أقصر", "أدنى", "أضيق"
]
_GROUP_RANKING_SIGNALS = [
    "the most", "the fewest", "le plus de", "le moins de",
    "أكبر عدد", "أقل عدد",
]
_SUPERLATIVE_COUNT_SIGNALS = [
    "the most", "the least", "le plus de", "le moins de",
    "أكبر عدد", "أقل عدد",
]


def _strip_arabic_al(text: str) -> str:
    """Strip leading Arabic definite article 'ال' from each word."""
    return re.sub(r'(?<=^)ال|(?<=\s)ال', '', text)
# rules.py — add near _strip_arabic_al

_ARABIC_POSSESSIVE_SUFFIXES = ("ها", "هم", "هن", "كم", "كن", "نا", "ه", "ك", "ي")

def _strip_arabic_possessive(word: str) -> str:
    """Strip a trailing possessive pronoun suffix for SIGNAL MATCHING only
    (سرعتها -> سرعة). Not used for entity detection, where surface form
    still matters."""
    for suf in _ARABIC_POSSESSIVE_SUFFIXES:
        if word.endswith(suf) and len(word) > len(suf) + 2:
            stem = word[: -len(suf)]
            if stem.endswith("ت"):          # سرعت + ها -> سرعة (ت->ة)
                stem = stem[:-1] + "ة"
            return stem
    return word

def _normalise_for_signal_match(text: str) -> str:
    text = _strip_arabic_al(text)
    return " ".join(_strip_arabic_possessive(w) for w in text.split())

def _normalise(text: str) -> str:
    """Normalize text for matching: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.replace("'", " ").replace("\u2019", " ").replace("\u2018", " ")
    text = text.replace("\u061F", " ")
    text = re.sub(r"[^\w\s\u0600-\u06FE]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _has_minimum_structure(question: str) -> bool:
    """Reject single-word or meaningless inputs."""
    words = question.strip().split()
    if len(words) < 2:
        return False
    if len(words) == 2:
        if re.search(r'[A-Za-z]{2,3}\d+', question):
            return True
        if re.search(r'\b[A-Z]{3}\b', question.upper()):
            return True
        return False
    return True