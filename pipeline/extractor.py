"""
extractor.py  (modified — v2)
-----------------------------
WHAT CHANGED vs v1:
    Added extract_airport_entity() — extracts an airport identifier
    (IATA code or city/airport name) from the question when the router
    has determined the question targets KG2.

    The original extract_entities() and is_flight_question() are
    completely unchanged — the KG1 pipeline is untouched.

WHY NOT A SEPARATE extractor_airports.py:
    The router already detected the entity type before calling the
    extractor. We pass entity_type from the config so one file handles
    both cases. This preserves the thesis claim that the core pipeline
    required no structural modification.
"""

import json
import re
import ollama

# ── UNCHANGED FROM v1 ─────────────────────────────────────────────────────────

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

_FLIGHT_RE = re.compile(r'\b([A-Za-z]{2,3}\d+)', re.ASCII)

def _extract_flight_number(text: str) -> str | None:
    matches = _FLIGHT_RE.findall(text.upper())
    if not matches:
        return None
    return max(matches, key=len)

def extract_entities(question, lang):
    """
    Original KG1 extractor — unchanged.
    Extracts flight number (via regex) and property phrase (via LLM).
    """
    flight = _extract_flight_number(question)

    prompt = f"""You are a property phrase extractor for a flight knowledge graph.

TASK: Read the question and extract ONLY the words that describe
what property of the flight is being asked about.

RULES:
- Extract the phrase AS IT APPEARS in the question (do not translate)
- Return ONLY the extracted phrase : no labels, no canonical forms
- Strip common question framing (e.g. "What is the", "of flight TK1887")

EXAMPLES:
"What is the gate of flight OS529?"              → "gate"
"What is the departure city of flight OS295?"     → "departure city"
"When does flight BR62 arrive?"                  → "when does it arrive"
"Quel est la porte du vol OS529?"                → "porte"
"Quelle est la ville de départ du vol OS295?"    → "ville de départ"
"متى تصل الرحلة BR62؟"                           → "متى تصل"
"ما هو مطار المغادرة؟"                           → "مطار المغادرة"

Return ONLY a JSON object with key "property". No explanation. No extra text.

Question: {question}
"""
    try:
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0}
        )
        raw    = response["message"]["content"]
        prop   = ""
        parsed = safe_json_parse(raw)
        if parsed:
            prop = parsed.get("property", "")
        prop = prop.strip().lower() if prop else ""
        return {"entity": flight, "property": prop}
    except Exception as e:
        return {"entity": flight, "property": "", "reason": f"ollama_error: {str(e)}"}


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


# ── NEW: KG2 AIRPORT EXTRACTOR ────────────────────────────────────────────────

def extract_airport_entities(question: str, lang: str, iata_from_router: str | None) -> dict:
    """
    Extracts the airport entity and the requested property from an airport question.

    DESIGN:
        - The airport IATA code is already resolved by the router via the
          entity map. We receive it directly (iata_from_router) — no LLM needed.
        - The property phrase is extracted by the LLM, same strategy as KG1.
          Raw phrase extraction, no translation — the cascade handles mapping.

    Args:
        question         : the user's original question
        lang             : detected language code (en / fr / ar)
        iata_from_router : IATA code already resolved by router (e.g. 'VIE')
                           None if router could not identify a specific airport

    Returns:
        {
            "entity":   IATA code string or None,
            "property": raw property phrase from the question,
        }
    """
    prompt = f"""You are a property phrase extractor for an airport knowledge graph.

TASK: Read the question and extract ONLY the words that describe
what property of the airport is being asked about.

RULES:
- Extract the phrase AS IT APPEARS in the question (do not translate)
- Return ONLY the extracted phrase — no labels, no extra explanation
- Strip question framing ("What is the", "of Vienna airport", "at FRA", etc.)

EXAMPLES:
"What is the elevation of Vienna airport?"       → "elevation"
"How long is the runway at FRA?"                 → "runway length"
"What country is Frankfurt airport in?"          → "country"
"What type of airport is LHR?"                   → "airport type"
"Quelle est l'élévation de l'aéroport de VIE?"  → "élévation"
"Quel pays est l'aéroport de Munich?"            → "pays"
"ما هو ارتفاع مطار فيينا؟"                        → "ارتفاع"
"في أي دولة يقع مطار فرانكفورت؟"                 → "البلد"
"Which municipality is Athens airport in?"       → "municipality"
"في أي بلدية يقع مطار أثينا؟"                     → "بلدية"

Return ONLY a JSON object with key "property". No explanation. No extra text.

Question: {question}
"""
    try:
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0}
        )
        raw    = response["message"]["content"]
        prop   = ""
        parsed = safe_json_parse(raw)
        if parsed:
            prop = parsed.get("property", "")
        prop = prop.strip().lower() if prop else ""
        return {"entity": iata_from_router, "property": prop}
    except Exception as e:
        return {
            "entity":   iata_from_router,
            "property": "",
            "reason":   f"ollama_error: {str(e)}"
        }


def validate_airport_extraction(entities: dict) -> bool:
    """
    Validates airport extraction result.
    Entity (IATA) can be None for template queries — only property matters.
    """
    if not entities.get("property"):
        return False
    return True

def extract_university_entities(question: str, lang: str, entity_from_router: str | None) -> dict:
    """
    Extracts the requested property from a university (LUBM) question.

    DESIGN: identical to extract_airport_entities — the entity itself is
    already resolved deterministically by the router (_detect_university_entity).
    Only the property phrase needs LLM extraction here.

    Args:
        question           : the user's original question
        lang                : detected language code (en / fr / ar)
        entity_from_router  : entity name already resolved by router
                              (e.g. 'FullProfessor0'), None if not found

    Returns:
        {
            "entity":   entity name string or None,
            "property": raw property phrase from the question,
        }
    """
    prompt = f"""You are a property phrase extractor for a university knowledge graph.

TASK: Read the question and extract ONLY the words that describe
what property or relationship is being asked about.

RULES:
- Extract the phrase AS IT APPEARS in the question (do not translate)
- Return ONLY the extracted phrase — no labels, no extra explanation
- Strip question framing ("What is the", "of FullProfessor0", "does he", etc.)

EXAMPLES:
"What courses does FullProfessor0 teach?"        → "courses taught"
"Where did GraduateStudent5 get their masters?"  → "masters degree from"
"Is Lecturer3 tenured?"                          → "tenured"
"What is Department2's name?"                    → "name"
"Quel est le titre de AssociateProfessor1?"      → "titre"
"Qui sont les étudiants de Department3?"         → "étudiants de"
"من كتب Publication12؟"                          → "كتب"
"إلى أي جامعة ينتمي Department7؟"                 → "أي جامعة"
# extractor.py — inside extract_university_entities()'s EXAMPLES block, add:
"يتابع GraduateStudent5 درجة علمية متقدمة — من أي جامعة أتم دراسته الجامعية الأولى قبل بدء الماجستير هنا؟" → "درجة البكالوريوس من"

Return ONLY a JSON object with key "property". No explanation. No extra text.

Question: {question}
"""
    try:
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0}
        )
        raw    = response["message"]["content"]
        prop   = ""
        parsed = safe_json_parse(raw)
        if parsed:
            prop = parsed.get("property", "")
        prop = prop.strip().lower() if prop else ""
        return {"entity": entity_from_router, "property": prop}
    except Exception as e:
        return {
            "entity":   entity_from_router,
            "property": "",
            "reason":   f"ollama_error: {str(e)}"
        }


def validate_university_extraction(entities: dict) -> bool:
    """
    Validates university extraction result.
    Mirrors validate_airport_extraction exactly.
    """
    if not entities.get("property"):
        return False
    return True

def extract_ask_entities(question: str, lang: str, entity_from_router: str | None) -> dict:
    """
    Extracts the property phrase AND the comparison value from an
    ASK-style question (e.g. "Is BR62's callsign EVA062?").

    DESIGN: same convention as extract_airport_entities() and
    extract_university_entities() — the entity itself is already
    resolved by the router (Priority 1.5). Only property + value
    need LLM extraction here. Unlike the other extractors, ASK
    questions require TWO pieces of information instead of one,
    since they assert a specific value rather than just requesting one.

    Args:
        question           : the user's original question
        lang                : detected language code (en / fr / ar)
        entity_from_router  : entity already resolved by router
                              (flight number, IATA code, or university entity)

    Returns:
        {
            "entity":   entity string (passed through from router),
            "property": raw property phrase from the question,
            "value":    the comparison value being asserted,
        }
    """
    prompt = f"""You are a property and value extractor for a yes/no (ASK-style) question
about a knowledge graph.

TASK: The question asserts that a specific entity has a specific property
value. Extract two things:
1. "property" — the property being asked about, AS IT APPEARS in the question
2. "value" — the value being asserted, exactly as written (do not translate,
   do not reformat)

RULES:
- Extract phrases AS THEY APPEAR in the question (do not translate)
- Do not include the entity name/ID itself in either field
- Return ONLY the JSON object — no labels, no explanation

EXAMPLES:
"Is BR62's callsign EVA062?"                → {{"property": "callsign", "value": "EVA062"}}
"Is CDG located in France?"                 → {{"property": "located in", "value": "France"}}
"Does flight OS295 depart from Vienna?"     → {{"property": "departure city", "value": "Vienna"}}
"Est-ce que le vol TK1887 atterrit à CDG?"  → {{"property": "atterrit à", "value": "CDG"}}
"هل مطار فيينا يقع في النمسا؟"              → {{"property": "يقع في", "value": "النمسا"}}
"هل مدرج مطار فرانكفورت من الأسفلت؟" → {{"property": "سطح المدرج", "value": "الأسفلت"}}
"La porte du vol TK123 est-elle B5?"        → {{"property": "porte", "value": "B5"}}
"هل بوابة الرحلة TK123 هي B5؟"              → {{"property": "بوابة", "value": "B5"}}
Return ONLY a JSON object with keys "property" and "value". No explanation.

Question: {question}
"""
    try:
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0}
        )
        raw    = response["message"]["content"]
        parsed = safe_json_parse(raw)
        prop  = (parsed.get("property", "") if parsed else "").strip().lower()
        value = (parsed.get("value", "")    if parsed else "").strip()
        return {"entity": entity_from_router, "property": prop, "value": value}
    except Exception as e:
        return {
            "entity":   entity_from_router,
            "property": "",
            "value":    "",
            "reason":   f"ollama_error: {str(e)}"
        }


def validate_ask_extraction(entities: dict) -> bool:
    """
    Validates ASK extraction. Unlike other validators, BOTH property
    AND value are required — an ASK question with a value but no known
    property (or vice versa) cannot be resolved to a SPARQL ASK query.
    """
    if not entities.get("property"):
        return False
    if not entities.get("value"):
        return False
    return True