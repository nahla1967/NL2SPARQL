from pipeline.language  import detect_language
from pipeline.extractor import extract_entities, validate_extraction, is_flight_question
from pipeline.mapper    import load_lexicon, map_property_cascade, map_flight

question = "What is the departure city of flight OS235?"

print("=" * 60)
print("Q:", question)
print("=" * 60)

# Step 0
lang = detect_language(question)
print(f"\n[0] language     : {lang}")

# Step 1
entities = extract_entities(question, lang)
print(f"[1] entities     : {entities}")

# Step 2a — validate
v1 = validate_extraction(entities)
v2 = is_flight_question(entities)
print(f"[2] validate_extraction  : {v1}")
print(f"[2] is_flight_question   : {v2}")

if not v1 or not v2:
    print("\n❌ STOPPED at extraction validation")
    exit()

# Step 2b — property mapping
lexicon = load_lexicon()
property_uri, mapping_layer = map_property_cascade(entities["property"], lexicon)
print(f"[3] property_uri   : {property_uri}")
print(f"[3] mapping_layer  : {mapping_layer}")

# Step 2c — flight mapping
flight_uri = map_flight(entities["entity"])
print(f"[4] flight_uri     : {flight_uri}")

if not flight_uri:
    print("\n❌ STOPPED — map_flight returned None")
    print("    entity passed to map_flight:", repr(entities['entity']))
elif not property_uri:
    print("\n❌ STOPPED — property_uri is None")
else:
    print("\n✅ Both URIs resolved — pipeline can continue")