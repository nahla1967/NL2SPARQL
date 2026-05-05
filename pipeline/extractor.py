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
        end   = text.rfind("}") + 1
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end])
            except Exception:
                return None
    return None


def extract_entities(question, lang):
    # Why a strict allowed-values list?
    # The LLM must return an exact known string so the lexicon lookup
    # succeeds. Free-form output ("departing airport", "origin") would
    # require fuzzy matching at this stage, which defeats the purpose of
    # a controlled mapping layer.
    #
    # Why keep only disambiguation examples and drop standard ones?
    # Standard cases (departure city, airline, aircraft) are already
    # unambiguous from the allowed-values list. The examples budget is
    # better spent on the pairs the model genuinely confuses:
    # gate vs terminal, callsign vs flight number, city vs country.

    prompt = f"""You are an entity extractor for a flight knowledge graph.
Extract the flight number and the property being asked about.

Return ONLY the property using EXACTLY one of these values:

English : departure city, arrival city, airline, pilot, gate, runway,
          aircraft, weather, flight number, arrival time, route,
          terminal, callsign, departure country, arrival country, flight attendant

French  : ville de départ, ville d'arrivée, compagnie aérienne, pilote,
          porte, piste, avion, météo, numéro de vol, heure d'arrivée, itinéraire,
          terminal, indicatif, pays de départ, pays d'arrivée, personnel de cabine

Arabic  : مدينة المغادرة, مدينة الوصول, شركة الطيران, الطيار,
          البوابة, المدرج, الطائرة, الطقس, رقم الرحلة, وقت الوصول, المسار,
          الصالة, الرمز, بلد المغادرة, بلد الوصول, طاقم الضيافة

── DISAMBIGUATION (easily confused pairs) ────────────────────────────
gate     = boarding door/number  |  terminal = airport building
callsign = ATC radio identifier  |  flight number = ticket number
departure city    ≠ departure country
arrival city      ≠ arrival country

── EXAMPLES (hard cases only) ────────────────────────────────────────
"What is the gate of flight OS235?"          → {{"entity": "OS235",  "property": "gate"}}
"What is the terminal of flight OS235?"      → {{"entity": "OS235",  "property": "terminal"}}
"What is the callsign of flight BR62?"       → {{"entity": "BR62",   "property": "callsign"}}
"What is the flight number of flight BR62?"  → {{"entity": "BR62",   "property": "flight number"}}
"What is the departure city of TK1887?"      → {{"entity": "TK1887", "property": "departure city"}}
"What is the departure country of TK1887?"   → {{"entity": "TK1887", "property": "departure country"}}
"What is the arrival city of AF1739?"        → {{"entity": "AF1739", "property": "arrival city"}}
"What is the arrival country of AF1739?"     → {{"entity": "AF1739", "property": "arrival country"}}
"Quel est le terminal du vol OS235?"         → {{"entity": "OS235",  "property": "terminal"}}
"Quelle est la porte du vol OS235?"          → {{"entity": "OS235",  "property": "porte"}}
"Quel est le pays de départ du vol AF1739?"  → {{"entity": "AF1739", "property": "pays de départ"}}
"Quelle est la ville de départ du vol AF1739?" → {{"entity": "AF1739", "property": "ville de départ"}}
"ما هي الصالة الخاصة بالرحلة AF1739؟"      → {{"entity": "AF1739", "property": "الصالة"}}
"ما هي البوابة الخاصة بالرحلة AF1739؟"     → {{"entity": "AF1739", "property": "البوابة"}}
"ما هو بلد المغادرة للرحلة BR62؟"           → {{"entity": "BR62",   "property": "بلد المغادرة"}}
"ما هي مدينة المغادرة للرحلة BR62؟"         → {{"entity": "BR62",   "property": "مدينة المغادرة"}}

Return ONLY a JSON object. No explanation. No extra text.

Question: {question}
"""
    try:
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )
        raw    = response["message"]["content"]
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