import json
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
        end = text.rfind("}") + 1
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end])
            except Exception:
                return None
    return None


def extract_entities(question, lang):
    # The allowed-values list is intentionally strict and exhaustive.
    # The LLM must pick exactly one value — no paraphrasing, no invention.
    #
    # The disambiguation block is the critical addition.
    # Without it, the LLM collapses similar concepts:
    #   - "terminal" → "gate"       (both relate to boarding)
    #   - "callsign" → "flight number" (both are identifiers)
    #   - "country"  → "city"       (both are geographic)
    #
    # The examples below train the model on those exact boundaries
    # so it learns to distinguish them from surface-text cues alone.

    prompt = f"""
You are an entity extractor for a flight knowledge graph.
Extract the flight number and the property being asked about.

You MUST return the property using EXACTLY one of these allowed values:

English: departure city, arrival city, airline, pilot, gate, runway,
aircraft, weather, flight number, arrival time, route,
terminal, callsign, departure country, arrival country, flight attendant

French: ville de départ, ville d'arrivée, compagnie aérienne, pilote,
porte, piste, avion, météo, numéro de vol, heure d'arrivée, itinéraire,
terminal, indicatif, pays de départ, pays d'arrivée, personnel de cabine

Arabic: مدينة المغادرة, مدينة الوصول, شركة الطيران, الطيار,
البوابة, المدرج, الطائرة, الطقس, رقم الرحلة, وقت الوصول, المسار,
الصالة, الرمز, بلد المغادرة, بلد الوصول, طاقم الضيافة

── DISAMBIGUATION RULES ──────────────────────────────────────────────
These pairs are easily confused. Read carefully:

gate     = the door/number where passengers board (e.g. "gate", "porte", "البوابة")
terminal = the building at the airport           (e.g. "terminal", "Terminal", "الصالة")
→ "Which gate?" and "Which terminal?" are DIFFERENT questions.

callsign     = the radio identifier used by air traffic control (e.g. "callsign", "indicatif", "الرمز")
flight number = the commercial number printed on a ticket      (e.g. "flight number", "numéro de vol", "رقم الرحلة")
→ "What is the callsign?" and "What is the flight number?" are DIFFERENT questions.

departure city    = the city the flight departs from   (e.g. "departure city", "ville de départ", "مدينة المغادرة")
departure country = the country the flight departs from (e.g. "departure country", "pays de départ", "بلد المغادرة")
arrival city      = the city the flight arrives at     (e.g. "arrival city", "ville d'arrivée", "مدينة الوصول")
arrival country   = the country the flight arrives at  (e.g. "arrival country", "pays d'arrivée", "بلد الوصول")
→ City and country are DIFFERENT levels of geography. Never substitute one for the other.

── EXAMPLES ──────────────────────────────────────────────────────────

Standard cases:
- "Where does OS235 depart from?"              → {{"entity": "OS235",  "property": "departure city"}}
- "Quelle compagnie opère le vol AF1739?"      → {{"entity": "AF1739", "property": "compagnie aérienne"}}
- "ما هو مدرج هبوط الرحلة BR62؟"              → {{"entity": "BR62",   "property": "المدرج"}}

Disambiguation cases (study these carefully):
- "What is the gate of flight OS235?"          → {{"entity": "OS235",  "property": "gate"}}
- "What is the terminal of flight OS235?"      → {{"entity": "OS235",  "property": "terminal"}}
- "What is the callsign of flight BR62?"       → {{"entity": "BR62",   "property": "callsign"}}
- "What is the flight number of flight BR62?"  → {{"entity": "BR62",   "property": "flight number"}}
- "What is the departure city of flight TK1887?"    → {{"entity": "TK1887", "property": "departure city"}}
- "What is the departure country of flight TK1887?" → {{"entity": "TK1887", "property": "departure country"}}
- "What is the arrival city of flight AF1739?"      → {{"entity": "AF1739", "property": "arrival city"}}
- "What is the arrival country of flight AF1739?"   → {{"entity": "AF1739", "property": "arrival country"}}
- "Quel est le terminal du vol OS235?"         → {{"entity": "OS235",  "property": "terminal"}}
- "Quelle est la porte du vol OS235?"          → {{"entity": "OS235",  "property": "porte"}}
- "Quel est le pays de départ du vol AF1739?"  → {{"entity": "AF1739", "property": "pays de départ"}}
- "Quelle est la ville de départ du vol AF1739?" → {{"entity": "AF1739", "property": "ville de départ"}}
- "ما هي الصالة الخاصة بالرحلة AF1739؟"      → {{"entity": "AF1739", "property": "الصالة"}}
- "ما هي البوابة الخاصة بالرحلة AF1739؟"     → {{"entity": "AF1739", "property": "البوابة"}}
- "ما هو بلد المغادرة للرحلة BR62؟"           → {{"entity": "BR62",   "property": "بلد المغادرة"}}
- "ما هي مدينة المغادرة للرحلة BR62؟"         → {{"entity": "BR62",   "property": "مدينة المغادرة"}}

Return ONLY a JSON object. No explanation. No extra text.

Question: {question}
"""
    try:
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response["message"]["content"]
        result = safe_json_parse(raw)
        if result:
            return result
        return {"entity": None, "property": None, "reason": f"parse_failed: {raw[:200]}"}

    except Exception as e:
        return {"entity": None, "property": None, "reason": f"ollama_error: {str(e)}"}


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