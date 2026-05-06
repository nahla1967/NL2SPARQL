import json
import re
import ollama

# KNOWN_FLIGHT_PREFIXES is used by is_flight_question to verify that the
# extracted entity looks like a real flight number before the pipeline
# continues. This avoids wasting a SPARQL query on garbage extraction.
KNOWN_FLIGHT_PREFIXES = [
    "OS", "FR", "TK", "BR", "BA", "AF",
    "KE", "LO", "BT", "PC", "7L", "XQ",
    "DE", "EN", "EW", "AI", "AY",
    "LG", "LX", "MAE", "OU", "PE",
    "SM", "SN", "TO", "VF", "W"
]


def safe_json_parse(text):
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end])
            except Exception:
                return None
    return None


# ── FLIGHT NUMBER REGEX ───────────────────────────────────────────────────────
# Matches any airline code (2-3 letters) followed by digits, any case.
# Covers all prefixes in KNOWN_FLIGHT_PREFIXES and any others present in the KG.
_FLIGHT_RE = re.compile(r'\b([A-Za-z]{2,3})\d+', re.ASCII)


def _extract_flight_number(text: str) -> str | None:
    """
    Extracts a flight number directly from the question text using regex.
    This avoids relying on the LLM to correctly isolate the entity string.
    Priority: longest prefix wins (e.g. MAE123 beats AE123).
    """
    matches = _FLIGHT_RE.findall(text.upper())
    if not matches:
        return None
    return max(matches, key=len)


# ── PROPERTY PHRASE EXTRACTION ───────────────────────────────────────────────
#
# KEY DESIGN CHANGE: the extractor no longer maps to canonical property names.
# It extracts the RAW property phrase as the user wrote it — no translation.
# The cascade (exact → fuzzy → semantic) does the normalisation.
#
# Why this is better:
#   OLD: LLM translates "quand arrive" → "heure d'arrivée"   ← translation step = noise
#   NEW: LLM echoes "quand arrive"      → cascade resolves it ← pure normalisation
#
# Eliminating the translation step means:
#   - Fewer errors: no canonical mapping to get wrong
#   - Faster: exact/fuzzy hit rate rises significantly
#   - Cleaner failure cases: when mapping fails, we know the phrase is genuinely OOV

def extract_entities(question, lang):
    flight = _extract_flight_number(question)

    prompt = f"""You are a property phrase extractor for a flight knowledge graph.

TASK: Read the question and extract ONLY the words that describe
what property of the flight is being asked about.

RULES:
- Extract the phrase AS IT APPEARS in the question (do not translate)
- Return ONLY the extracted phrase — no labels, no canonical forms
- Strip common question framing (e.g. "What is the", "of flight TK1887")

EXAMPLES:
"What is the gate of flight OS235?"              → "gate"
"What is the departure city of flight TK1887?"    → "departure city"
"When does flight AF1739 arrive?"                → "when does it arrive"
"What is the terminal?"                          → "terminal"
"Quel est la porte du vol OS235?"                → "porte"
"Quelle est la ville de départ du vol TK1887?"   → "ville de départ"
"متى تصل الرحلة OS235؟"                          → "متى تصل"
"ما هو مطار المغادرة؟"                           → "مطار المغادرة"

Return ONLY a JSON object with key "property". No explanation. No extra text.

Question: {question}
"""
    try:
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )
        raw  = response["message"]["content"]
        prop = ""
        parsed = safe_json_parse(raw)
        if parsed:
            prop = parsed.get("property", "")
        prop = prop.strip().lower() if prop else ""

        return {
            "entity":   flight,
            "property": prop,
        }

    except Exception as e:
        return {
            "entity":   flight,
            "property": "",
            "reason":   f"ollama_error: {str(e)}",
        }


def validate_extraction(entities):
    if not entities.get("entity"):
        return False
    if str(entities.get("entity")).strip().lower() == "none":
        return False
    if not entities.get("property"):
        return False
    return True


def is_flight_question(entities):
    entity = entities.get("entity", "")
    if not entity:
        return False
    entity = entity.strip().upper()
    for prefix in KNOWN_FLIGHT_PREFIXES:
        if entity.startswith(prefix):
            return True
    return False