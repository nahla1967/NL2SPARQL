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

    "university": {
        # SPARQL endpoint
        "endpoint":    "http://localhost:3030/university/sparql",

        # Multilingual university lexicon
        "lexicon":     "lexicon_university.json",

        # What the extractor looks for as the main entity.
        "entity_type": "person_name",

        # Base URI prefix for this KG
        "base_uri":    "http://www.lehigh.edu/~zhp2/2004/0401/univ-bench.owl#",
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

    "count_kg3": {
        "kg":          "university",
        "endpoint":    KG_REGISTRY["university"]["endpoint"],
        "base_uri":    KG_REGISTRY["university"]["base_uri"],
        "description": "Count or list university entities linked to a known entity",
    },

    "filter_string_kg3": {
        "kg":          "university",
        "endpoint":    KG_REGISTRY["university"]["endpoint"],
        "base_uri":    KG_REGISTRY["university"]["base_uri"],
        "description": "Filter university people by department membership",
    },

    "group_aggregate_kg1": {
        "kg":          "flights",
        "endpoint":    KG_REGISTRY["flights"]["endpoint"],
        "base_uri":    KG_REGISTRY["flights"]["base_uri"],
        "description": "Aggregate a numeric flight property grouped by airline",
    },

    "group_aggregate_kg2": {
        "kg":          "airports",
        "endpoint":    KG_REGISTRY["airports"]["endpoint"],
        "base_uri":    KG_REGISTRY["airports"]["base_uri"],
        "description": "Aggregate a numeric airport property grouped by country or continent",
    },

    "group_aggregate_kg3": {
        "kg":          "university",
        "endpoint":    KG_REGISTRY["university"]["endpoint"],
        "base_uri":    KG_REGISTRY["university"]["base_uri"],
        "description": "Aggregate a count (courses/students) grouped by department",
    },
}

# ── KG2 PROPERTY HOP TABLE ────────────────────────────────────────────────────
# Describes which KG2 properties require an intermediate node.
# Structure: short_property_name → (hop_property, target_property)
#
# WHY THIS EXISTS HERE AND NOT IN THE LEXICON:
#   The lexicon maps language expressions to concept names.
#   This table maps concept names to ontology structure.
#   These are two different concerns and must stay separate.
#   Adding a new language → edit lexicon only.
#   Adding a new ontology property → edit this table only.

KG2_PROPERTY_HOPS = {
    # Runway properties — must go through hasRunway first
    "lengthFt":    ("hasRunway", "lengthFt"),
    "widthFt":     ("hasRunway", "widthFt"),
    "surface":     ("hasRunway", "surface"),
    "lighted":     ("hasRunway", "lighted"),
    "closed":      ("hasRunway", "closed"),
    "runwayIdent": ("hasRunway", "runwayIdent"),
    # Country properties — must go through locatedInCountry first
    "countryName": ("locatedInCountry", "countryName"),
    "continent":   ("locatedInCountry", "continent"),
    # Region properties — must go through locatedInRegion first
    "regionName":  ("locatedInRegion", "regionName"),
}

# ── GROUP-BY / AGGREGATE PROPERTY MAPS ────────────────────────────────────────
# Describes valid (group_property, numeric_property, aggregate_function)
# combinations for group_aggregate_kg1/kg2/kg3 templates.
#
# WHY SCOPED THIS NARROWLY:
#   KG1 groups by airline only (for now — city/country grouping reuses the
#   same builder shape and can be added later with no design change).
#   KG2 groups by country or continent (both reached via locatedInCountry,
#   same hop already used elsewhere).
#   KG3 has no numeric properties on people — "aggregate" there always means
#   COUNT per professor/student, then AVG/MAX/MIN of those counts per
#   department. This requires a nested subquery, unlike KG1/KG2's direct
#   aggregate. See _build_group_aggregate_kg3() for the SPARQL shape.

GROUP_AGGREGATE_KG1 = {
    "group_by": {
        "airline": {"hop_property": "hasAirline", "name_property": "operating_as"},
    },
    "numeric_properties": {
        "gspeed": {"hop": "hasFlightEvent", "unit": "knots"},
        "vspeed": {"hop": "hasFlightEvent", "unit": "ft/min"},
    },
}

GROUP_AGGREGATE_KG2 = {
    "group_by": {
        "country":   {"hop_property": "locatedInCountry", "name_property": "countryName"},
        "continent": {"hop_property": "locatedInCountry", "name_property": "continent"},
    },
    "numeric_properties": {
        "elevationFt": {"hop": "direct"},
        "lengthFt":    {"hop": "hasRunway"},
        "widthFt":     {"hop": "hasRunway"},
    },
}

GROUP_AGGREGATE_KG3 = {
    "group_by": {
        "department": {"hop_property": "worksFor", "name_property": "name"},
    },
    # Countable relations per person — no true numeric property exists.
    "countable_properties": {
        "teacherOf":    {"label": "courses taught"},
        "takesCourse":  {"label": "courses taken"},
    },
}

AGGREGATE_FUNCTIONS = {"AVG", "SUM", "MAX", "MIN"}
# ── OPEN KG SCHEMA DESCRIPTION ───────────────────────────────────────────────
# Human-readable schema injected into LLM prompts for the open_kg branch.
# Describes exactly what data exists in both KGs — nothing more, nothing less.
# If the ontology changes, update this block to keep prompts accurate.

OPEN_KG_SCHEMA = """
KNOWLEDGE GRAPH 1 — Flights (endpoint: http://localhost:3030/flights/sparql)
Base URI: http://www.semanticweb.org/ontologies/flight_ontology#

Classes and properties:
  Flight:
    flightNumber (string)        — e.g. "OS235"
    hasAirline → Airline
    hasAircraft → Aircraft
    hasOriginCity → City         — city has orig_city (string)
    hasDestinationCity → City    — city has dest_city (string)
    hasOriginCountry → Country   — country has orig_country (string)
    hasDestinationCountry → Country — country has dest_country (string)
    hasGate (string)
    hasTerminal (string)
    hasCallsign (string)
    hasRunway (string)
    hasRoute → Route             — route has orig_iata, dest_iata (strings)
    hasAirportDetails → AirportDetails — has orig_iata, dest_iata (strings)
    hasFlightEvent → FlightEvent — event has gspeed, vspeed, alt (numbers)
    hasWeatherCondition (string)
    hasPilot → Pilot
    hasFlightAttendant → FlightAttendant
    hasTimeInstant → TimeInstant — has eta (datetime)

  Airline: operating_as (string)
  Aircraft: type (string), reg (string)

KNOWLEDGE GRAPH 2 — Airports (endpoint: http://localhost:3030/airports/sparql)
Base URI: http://www.semanticweb.org/ontologies/airport_ontology#

Classes and properties:
  Airport:
    airportName (string)
    airportType (string)         — "large_airport" or "medium_airport"
    iataCode (string)            — 3-letter code e.g. "VIE"
    icaoCode (string)            — 4-letter code e.g. "LOWW"
    elevationFt (integer)
    latitude (decimal)
    longitude (decimal)
    municipality (string)
    hasRunway → Runway
    locatedInCountry → Country
    locatedInRegion → Region

  Runway:
    lengthFt (integer)
    widthFt (integer)
    surface (string)             — "ASP", "CON", "GRS"
    lighted (boolean)
    closed (boolean)
    runwayIdent (string)
    belongsToAirport → Airport

  Country:
    countryName (string)
    isoCode (string)
    continent (string)

  Region:
    regionName (string)
    regionCode (string)

KNOWLEDGE GRAPH 3 — University (endpoint: http://localhost:3030/university/sparql)
Base URI: http://www.lehigh.edu/~zhp2/2004/0401/univ-bench.owl#

Classes and properties:
  FullProfessor / AssociateProfessor / AssistantProfessor / Lecturer:
    name (string)
    teacherOf → Course / GraduateCourse (one-to-many)
    undergraduateDegreeFrom / mastersDegreeFrom / doctoralDegreeFrom → University
    worksFor → Department
    headOf → Department (some professors only)

  GraduateStudent / UndergraduateStudent:
    name (string)
    takesCourse → Course / GraduateCourse (one-to-many)
    advisor → Professor
    memberOf → Department

  Course / GraduateCourse:
    name (string)

  Department:
    name (string)
    subOrganizationOf → University

  Publication:
    name (string)
    publicationAuthor → Professor / Student
"""
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
def get_property_hop(property_short: str, kg_name: str = "airports"):
    """
    Returns (prop1, prop2) if the property requires an intermediate hop,
    or (property_short, None) if it is a direct property on the entity node.
    Only applies to KG2 currently — KG1 handles hops via the lexicon array syntax.
    """
    if kg_name == "airports":
        hop = KG2_PROPERTY_HOPS.get(property_short)
        if hop:
            return hop[0], hop[1]
    return property_short, None



# ── INSERT HERE ────────────────────────────────────────────────────────────
def get_group_aggregate_config(kg_name: str) -> dict:
    """Group-by/aggregate property map for a KG (flights/airports/university)."""
    return {
        "flights":    GROUP_AGGREGATE_KG1,
        "airports":   GROUP_AGGREGATE_KG2,
        "university": GROUP_AGGREGATE_KG3,
    }[kg_name]
# ── END INSERT ─────────────────────────────────────────────────────────────

def get_open_kg_schema() -> str:
    """Returns the schema description for open_kg SPARQL generation."""
    return OPEN_KG_SCHEMA
