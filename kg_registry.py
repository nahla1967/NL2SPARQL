"""
kg_registry.py
--------------
Central registry for all knowledge graphs in the system.

DESIGN PRINCIPLE:
    Adding a new KG = adding one block here.
    The router, mapper, executor, and extractor read from this registry
    and adapt automatically. No other file needs to change.

QUERY TYPES:
    single_kg1   → flight number + property  (existing pipeline)
    single_kg2   → airport name/IATA + property  (new airport branch)
    cross_kg     → flight number + airport property (KG1 → IATA → KG2)
    template     → filter / ranking / comparison / count queries

CHANGES vs v1:
    - Removed redundant flight prefix triggers from KG_REGISTRY["flights"]
      Flight detection is handled purely by regex in router.py.
      Reason: prefix strings like "OS" or "TK" added no value and risked
      false positives on non-flight questions containing those substrings.

    - Removed signals from TEMPLATE_REGISTRY entries.
      Signals now live exclusively in lexicon_airports.json under
      "template_triggers". This enforces Single Source of Truth —
      adding a new signal requires editing only the lexicon.

    - Added specific helper functions get_endpoint(), get_lexicon(),
      get_base_uri() so calling code stays clean and readable.
"""

# ── KNOWLEDGE GRAPH REGISTRY ──────────────────────────────────────────────────

KG_REGISTRY = {

    "flights": {
        # SPARQL endpoint
        "endpoint":    "http://localhost:3030/flights/sparql",

        # Multilingual lexicon file
        "lexicon":     "lexicon.json",

        # What the extractor looks for as the main entity.
        # "flight_number" → router uses regex [A-Za-z]{2,3}\d+ exclusively.
        # No prefix trigger list needed — regex covers all airline codes.
        "entity_type": "flight_number",

        # Base URI prefix for this KG
        "base_uri":    "http://www.semanticweb.org/ontologies/flight_ontology#",
    },

    "airports": {
        # SPARQL endpoint
        "endpoint":    "http://localhost:3030/airports/sparql",

        # Multilingual airport lexicon
        "lexicon":     "lexicon_airports.json",

        # What the extractor looks for as the main entity.
        # "airport_name" → IATA code (regex [A-Z]{3}) or city/airport name
        # looked up in lexicon_airports.json["airport_entities"].
        "entity_type": "airport_name",

        # Base URI prefix for this KG
        "base_uri":    "http://www.semanticweb.org/ontologies/airport_ontology#",

        # Airport keyword fallback — used ONLY when no IATA/name is detected.
        # Signals live here (not duplicated in TEMPLATE_REGISTRY).
        "triggers":    [
            # English
            "airport", "runway", "elevation", "altitude",
            "landing strip", "airfield",
            # French
            "aéroport", "piste", "élévation", "hauteur",
            # Arabic
            "مطار", "مدرج", "ارتفاع",
        ],
    },
}


# ── CROSS-KG CONFIGURATION ────────────────────────────────────────────────────
# Used by cross_kg_resolver.py to bridge KG1 and KG2 via the IATA code.
#
# How it works:
#   Step 1 → query KG1: flight + direction → IATA string (e.g. "MUC")
#   Step 2 → query KG2: IATA string → Airport URI (ao:Airport/MUC)
#   Step 3 → query KG2: Airport URI + property → final answer
#
# If the ontology property names change, only this block needs updating.
# cross_kg_resolver.py reads from here and never hardcodes property names.

CROSS_KG_CONFIG = {
    # KG1: property chain to reach the origin IATA code
    "origin_property":       "hasAirportDetails",
    "origin_iata_prop":      "orig_iata",

    # KG1: property chain to reach the destination IATA code
    "destination_property":  "hasAirportDetails",
    "destination_iata_prop": "dest_iata",

    # KG2: property used to find an airport by its IATA code
    "kg2_iata_property":     "iataCode",

    # Endpoints — referenced from registry to avoid duplication
    "kg1_endpoint":          KG_REGISTRY["flights"]["endpoint"],
    "kg2_endpoint":          KG_REGISTRY["airports"]["endpoint"],

    # Base URIs
    "kg1_base":              KG_REGISTRY["flights"]["base_uri"],
    "kg2_base":              KG_REGISTRY["airports"]["base_uri"],
}


# ── ROUTER PRIORITY ORDER ─────────────────────────────────────────────────────
# The router checks conditions in this exact order. First match wins.
#
# Priority 1: cross-KG signal + flight number detected → cross_kg
# Priority 2: flight number detected (regex)           → single_kg1
# Priority 3: airport IATA or name detected            → single_kg2
# Priority 4: airport keyword detected (fallback)      → single_kg2
# Priority 5: template signal detected                 → template
# Priority 6: nothing matched                          → out_of_scope

ROUTER_PRIORITY = [
    "cross_kg",
   
    "single_kg1",
    
    "single_kg2",
     "template",
    "out_of_scope",
]


# ── TEMPLATE QUERY TYPES ──────────────────────────────────────────────────────
# Maps template names to their target KG and description.
#
# SIGNALS ARE NOT STORED HERE.
# They live in lexicon_airports.json["template_triggers"] — single source of
# truth. The router loads signals from the lexicon at runtime.
# This means adding a new signal = edit the lexicon only, never this file.

TEMPLATE_REGISTRY = {

    "filter_numeric_kg2": {
        "kg":          "airports",
        "endpoint":    KG_REGISTRY["airports"]["endpoint"],
        "base_uri":    KG_REGISTRY["airports"]["base_uri"],
        "description": "Filter airports by numeric property with threshold",
    },

    "filter_string_kg2": {
        "kg":          "airports",
        "endpoint":    KG_REGISTRY["airports"]["endpoint"],
        "base_uri":    KG_REGISTRY["airports"]["base_uri"],
        "description": "Filter airports by categorical property value",
    },

    "ranking_kg2": {
        "kg":          "airports",
        "endpoint":    KG_REGISTRY["airports"]["endpoint"],
        "base_uri":    KG_REGISTRY["airports"]["base_uri"],
        "description": "Rank airports by a numeric property",
    },

    "compare_two_airports": {
        "kg":          "airports",
        "endpoint":    KG_REGISTRY["airports"]["endpoint"],
        "base_uri":    KG_REGISTRY["airports"]["base_uri"],
        "description": "Compare two airports on a property",
    },

    "count_kg1": {
        "kg":          "flights",
        "endpoint":    KG_REGISTRY["flights"]["endpoint"],
        "base_uri":    KG_REGISTRY["flights"]["base_uri"],
        "description": "Count or list flights matching a condition",
    },

    "filter_numeric_kg1": {
        "kg":          "flights",
        "endpoint":    KG_REGISTRY["flights"]["endpoint"],
        "base_uri":    KG_REGISTRY["flights"]["base_uri"],
        "description": "Filter flights by numeric property with threshold",
    },

    "cross_kg_filter": {
        "kg":          "cross",
        "endpoint":    None,
        "base_uri":    None,
        "description": "Filter flights based on a property of their airport",
    },
}


# ── CONVENIENCE ACCESSORS ─────────────────────────────────────────────────────
# Use these in router.py, extractor.py, mapper.py, executor.py
# instead of writing KG_REGISTRY["airports"]["endpoint"] every time.

def get_kg_config(kg_name: str) -> dict:
    """Full config dict for a KG. Raises KeyError if not found."""
    return KG_REGISTRY[kg_name]

def get_endpoint(kg_name: str) -> str:
    """SPARQL endpoint URL for a KG."""
    return KG_REGISTRY[kg_name]["endpoint"]

def get_lexicon(kg_name: str) -> str:
    """Lexicon file path for a KG."""
    return KG_REGISTRY[kg_name]["lexicon"]

def get_base_uri(kg_name: str) -> str:
    """Base URI prefix for a KG."""
    return KG_REGISTRY[kg_name]["base_uri"]

def get_entity_type(kg_name: str) -> str:
    """Entity type the extractor should look for."""
    return KG_REGISTRY[kg_name]["entity_type"]

def get_all_kg_names() -> list:
    """All registered KG names."""
    return list(KG_REGISTRY.keys())

def get_template_config(template_name: str) -> dict:
    """Full config dict for a template type."""
    return TEMPLATE_REGISTRY[template_name]

def get_all_template_names() -> list:
    """All registered template names."""
    return list(TEMPLATE_REGISTRY.keys())