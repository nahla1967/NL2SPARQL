import json
import ollama

# CHANGED — VALID_PROPERTIES removed.
#
# Previously, this set was used to validate the extracted property by checking
# whether it was one of a fixed list of English strings.
#
# This was the root cause of the dead-lexicon problem: the extractor prompt
# forced the LLM to normalize all property names to English before the mapper
# ever saw them. As a result, the French and Arabic keys in lexicon.json
# were never reachable.
#
# Validation is now delegated to the mapping step: a property is considered
# valid if and only if map_property or map_property_with_embeddings can resolve
# it to a known Knowledge Graph URI. This is semantically more correct —
# the ground truth of validity is the Knowledge Graph, not a hardcoded list.

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
    # CHANGED — The prompt no longer contains an "Allowed properties" list.
    #
    # Before:
    #   The prompt said "use exactly these words: departure city, arrival city ..."
    #   This forced the LLM to normalize all output to English, making the
    #   multilingual lexicon permanently unreachable.
    #
    # After:
    #   The prompt instructs the LLM to return the property as it naturally
    #   appears in the question. If the user wrote in Arabic, the property
    #   field will contain Arabic text. If French, French text. If English,
    #   English text.
    #
    #   The examples are updated to reflect this: the expected output for a
    #   French question now contains a French property name, and the expected
    #   output for an Arabic question contains an Arabic property name.
    #
    #   This makes the extractor a faithful surface-text extractor, and
    #   delegates all normalization and resolution to the mapper — which is
    #   where the multilingual lexicon lives.

    prompt = f"""
You are an entity extractor for a flight knowledge graph.
Extract the flight number and the property being asked about.

For the property, extract the key concept describing what is asked — in the same language as the question.
Do not translate. Do not normalize to English. Return the property as it appears in the question.

Examples:
- "Where does OS235 depart from?" → {{"entity": "OS235", "property": "departure city"}}
- "What airline operates TK1887?" → {{"entity": "TK1887", "property": "airline"}}
- "Quelle compagnie opère le vol AF1739?" → {{"entity": "AF1739", "property": "compagnie aérienne"}}
- "Quel est l'aéroport de départ du vol 7L280?" → {{"entity": "7L280", "property": "ville de départ"}}
- "ما هي شركة الطيران التي تشغّل الرحلة BR62؟" → {{"entity": "BR62", "property": "شركة الطيران"}}
- "من هو طيار الرحلة AI180؟" → {{"entity": "AI180", "property": "الطيار"}}
- "Quel est l'itinéraire du vol LX19?" → {{"entity": "LX19", "property": "itinéraire"}}
- "What is the arrival time of flight BA456?" → {{"entity": "BA456", "property": "arrival time"}}

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

    # CHANGED — Added error handling for Ollama connection failures.
    # Previously, if Ollama was not running, this function raised an uncaught
    # exception that crashed the program before any log entry was written.
    # This fix is part of Issue #8 but applied here since we are editing this file.
    except Exception as e:
        return {"entity": None, "property": None, "reason": f"ollama_error: {str(e)}"}

def validate_extraction(entities):
    # CHANGED — Removed the VALID_PROPERTIES check.
    #
    # Before:
    #   if entities.get("property") not in VALID_PROPERTIES: return False
    #   This rejected any non-English property string, preventing Arabic and
    #   French surface forms from ever reaching the mapper.
    #
    # After:
    #   We only verify that both fields are non-empty strings.
    #   Whether the property is meaningful is determined later by map_property
    #   and map_property_with_embeddings. If both return None, the system
    #   correctly logs a mapping failure for that test case.
    if not entities.get("entity"):
        return False
    if not entities.get("property"):
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