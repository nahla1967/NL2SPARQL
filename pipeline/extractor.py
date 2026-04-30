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
- "Who is the pilot of BR62?" → {{"entity": "BR62", "property": "pilot"}}
- "What gate is FR12 at?" → {{"entity": "FR12", "property": "gate"}}

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
    return {"entity": None, "property": None}

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