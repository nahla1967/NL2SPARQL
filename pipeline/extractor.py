import json
import ollama

VALID_PROPERTIES = {
    "departure city", "arrival city", "airline", "pilot",
    "gate", "runway", "aircraft", "weather", "flight number",
    "arrival time", "route"
}

KNOWN_FLIGHT_PREFIXES = [
    "OS", "FR", "TK", "BR", "BA", "AF",
    "KE", "LO", "BT", "PC", "7L", "XQ",
    "DE", "EN", "EW", "AI", "AY", "BE"
]

def safe_json_parse(text):
    try:
        return json.loads(text)
    except:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end])
            except:
                return None
    return None

def extract_entities(question, lang):
    prompt = f"""
You are an entity extractor for a flight knowledge graph.
Extract the main entity (flight number) and the property being asked about.

Allowed properties (use exactly these words):
departure city, arrival city, airline, pilot, gate, runway,
aircraft, weather, flight number, arrival time, route

Examples:
- "Where does OS235 depart from?" → {{"entity": "OS235", "property": "departure city"}}
- "What airline operates TK1887?" → {{"entity": "TK1887", "property": "airline"}}
- "Quelle compagnie opère le vol AF1739?" → {{"entity": "AF1739", "property": "airline"}}
- "Quel est l'aéroport de départ du vol 7L280?" → {{"entity": "7L280", "property": "departure city"}}
- "ما هي شركة الطيران التي تشغّل الرحلة BR62؟" → {{"entity": "BR62", "property": "airline"}}
- "من هو طيار الرحلة AI180؟" → {{"entity": "AI180", "property": "pilot"}}

Return ONLY a JSON object. No explanation. No extra text.

Question: {question}
"""
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response["message"]["content"]
    result = safe_json_parse(raw)
    if result:
        return result
    return {"entity": None, "property": None , "reason": f"parse_failed: {raw[:200]}"}

def validate_extraction(entities):
    if not entities.get("entity"):
        return False
    if entities.get("property") not in VALID_PROPERTIES:
        return False
    return True

def is_flight_question(entities):
    entity = entities.get("entity", "")
    if not entity:
        return False
    for prefix in KNOWN_FLIGHT_PREFIXES:
        if entity.startswith(prefix):
            return True
    return False