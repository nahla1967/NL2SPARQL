"""
NL2SPARQL — pipeline orchestrator, built directly from main.py (v2 — multi-KG)

Everything here mirrors main.py's actual branch logic exactly — same
functions, same imports, same order, same validation checks. The only
difference is structural: main.py runs one hardcoded question and exits;
this wraps each branch in a function returning a dict, so the Streamlit UI
can call it repeatedly with whatever question the user types.

No more guessing — cross_kg uses your real resolve_cross_kg(), which I
didn't know existed until you shared main.py. My earlier version hand-built
a two-hop query, which was wrong; this replaces it entirely.
"""

from pipeline.language import detect_language
from router import route
from pipeline.extractor import (
    extract_entities, validate_extraction, is_flight_question,
    extract_airport_entities, validate_airport_extraction,
    extract_university_entities, validate_university_extraction,
)
from pipeline.mapper import (
    get_university_entity_type, load_lexicon, map_property_cascade,
    map_flight, map_airport, map_university_entity,
)
from pipeline.generator import inject_and_generate, generate_open_kg_sparql
from pipeline.executor import validate_sparql, execute_sparql, format_answer, format_answer_list
from cross_kg_resolver import resolve_cross_kg
from template_resolver import resolve_template, resolve_ask_query
from kg_registry import get_base_uri, get_endpoint, get_lexicon


def _validate(sparql_query: str, full_prop_uri: str, full_prop2_uri: str | None) -> bool:
    """Exact copy of the Step-4 validation block repeated in every main.py branch."""
    is_valid = (
        validate_sparql(sparql_query)
        and sparql_query.strip().startswith("SELECT")
        and "PREFIX" not in sparql_query
        and full_prop_uri in sparql_query
    )
    if full_prop2_uri and full_prop2_uri not in sparql_query:
        is_valid = False
    return is_valid


def _ui_result(**kwargs) -> dict:
    """Fills in every key the UI expects, defaulting anything not passed."""
    base = {
        "language": None, "branch": None, "entity": None, "property_surface": None,
        "resolved_uri": None, "tier": None, "sparql": None,
        "execution": {"value": None, "error": None}, "answer": None, "path": None,
    }
    base.update(kwargs)
    return base


def _run_single_kg1(question: str, lang: str, strategy: str) -> dict:
    entities = extract_entities(question, lang)
    if not validate_extraction(entities) or not is_flight_question(entities):
        return _ui_result(language=lang, branch="single_kg1",
                           execution={"value": None, "error": "extraction_failure"})

    lexicon = load_lexicon(get_lexicon("flights"))
    property_uri, tier, property2_uri = map_property_cascade(
        entities["property"], lexicon, get_lexicon("flights")
    )
    flight_uri = map_flight(entities["entity"])

    if not flight_uri or not property_uri:
        return _ui_result(language=lang, branch="single_kg1", entity=entities["entity"],
                           property_surface=entities["property"], tier=tier,
                           execution={"value": None, "error": "mapping_failure"})

    BASE = get_base_uri("flights")
    full_prop_uri = BASE + property_uri
    full_prop2_uri = (BASE + property2_uri) if property2_uri else None

    sparql_query = inject_and_generate(flight_uri, full_prop_uri, question,
                                        strategy=strategy, property2_uri=full_prop2_uri)

    if not _validate(sparql_query, full_prop_uri, full_prop2_uri):
        return _ui_result(language=lang, branch="single_kg1", entity=entities["entity"],
                           property_surface=entities["property"], resolved_uri=full_prop_uri,
                           tier=tier, sparql=sparql_query,
                           execution={"value": None, "error": "generation_failure"})

    result = execute_sparql(sparql_query, endpoint=get_endpoint("flights"))
    answer = format_answer(question, result["value"], lang) if result["value"] else None

    return _ui_result(language=lang, branch="single_kg1", entity=entities["entity"],
                       property_surface=entities["property"], resolved_uri=full_prop_uri,
                       tier=tier, sparql=sparql_query, execution=result, answer=answer)


def _run_single_kg2(question: str, routing: dict, lang: str, strategy: str) -> dict:
    entities = extract_airport_entities(question, lang, routing["entity"])
    if not validate_airport_extraction(entities):
        return _ui_result(language=lang, branch="single_kg2",
                           execution={"value": None, "error": "extraction_failure"})

    lexicon_path = get_lexicon("airports")
    lexicon = load_lexicon(lexicon_path)
    property_uri, tier, property2_uri = map_property_cascade(
        entities["property"], lexicon, lexicon_path
    )
    airport_uri = map_airport(entities["entity"]) if entities["entity"] else None

    if not airport_uri or not property_uri:
        return _ui_result(language=lang, branch="single_kg2", entity=entities["entity"],
                           property_surface=entities["property"], tier=tier,
                           execution={"value": None, "error": "mapping_failure"})

    BASE = get_base_uri("airports")
    full_prop_uri = BASE + property_uri
    full_prop2_uri = (BASE + property2_uri) if property2_uri else None

    sparql_query = inject_and_generate(airport_uri, full_prop_uri, question,
                                        strategy=strategy, property2_uri=full_prop2_uri)

    if not _validate(sparql_query, full_prop_uri, full_prop2_uri):
        return _ui_result(language=lang, branch="single_kg2", entity=entities["entity"],
                           property_surface=entities["property"], resolved_uri=full_prop_uri,
                           tier=tier, sparql=sparql_query,
                           execution={"value": None, "error": "generation_failure"})

    result = execute_sparql(sparql_query, endpoint=get_endpoint("airports"))
    answer = format_answer(question, result["value"], lang) if result["value"] else None

    return _ui_result(language=lang, branch="single_kg2", entity=entities["entity"],
                       property_surface=entities["property"], resolved_uri=full_prop_uri,
                       tier=tier, sparql=sparql_query, execution=result, answer=answer)


def _run_single_kg3(question: str, routing: dict, lang: str, strategy: str) -> dict:
    entities = extract_university_entities(question, lang, routing["entity"])
    if not validate_university_extraction(entities):
        return _ui_result(language=lang, branch="single_kg3",
                           execution={"value": None, "error": "extraction_failure"})

    lexicon_path = get_lexicon("university")
    lexicon = load_lexicon(lexicon_path)
    property_uri, tier, property2_uri = map_property_cascade(
        entities["property"], lexicon, lexicon_path
    )
    entity_uri = map_university_entity(entities["entity"]) if entities["entity"] else None

    # Same memberOf/subOrganizationOf/worksFor disambiguation as main.py
    FACULTY_TYPES = {"FullProfessor", "AssociateProfessor", "AssistantProfessor", "Lecturer"}
    if entity_uri and property_uri in ("memberOf", "subOrganizationOf"):
        entity_type = get_university_entity_type(entity_uri)
        if entity_type == "Department" and property_uri == "memberOf":
            property_uri = "subOrganizationOf"
        elif entity_type != "Department" and property_uri == "subOrganizationOf":
            property_uri = "memberOf"
        elif entity_type in FACULTY_TYPES and property_uri == "memberOf":
            property_uri = "worksFor"

    if not entity_uri or not property_uri:
        return _ui_result(language=lang, branch="single_kg3", entity=entities["entity"],
                           property_surface=entities["property"], tier=tier,
                           execution={"value": None, "error": "mapping_failure"})

    BASE = get_base_uri("university")
    full_prop_uri = BASE + property_uri
    full_prop2_uri = (BASE + property2_uri) if property2_uri else None

    sparql_query = inject_and_generate(entity_uri, full_prop_uri, question,
                                        strategy=strategy, property2_uri=full_prop2_uri)

    if not _validate(sparql_query, full_prop_uri, full_prop2_uri):
        return _ui_result(language=lang, branch="single_kg3", entity=entities["entity"],
                           property_surface=entities["property"], resolved_uri=full_prop_uri,
                           tier=tier, sparql=sparql_query,
                           execution={"value": None, "error": "generation_failure"})

    # multiple=True — university properties are naturally one-to-many
    result = execute_sparql(sparql_query, endpoint=get_endpoint("university"), multiple=True)
    answer = format_answer_list(question, result["value"], lang) if result["value"] else None

    return _ui_result(language=lang, branch="single_kg3", entity=entities["entity"],
                       property_surface=entities["property"], resolved_uri=full_prop_uri,
                       tier=tier, sparql=sparql_query, execution=result, answer=answer)


def _run_cross_kg(question: str, routing: dict, lang: str) -> dict:
    """
    Real cross_kg logic, straight from main.py — uses your actual
    resolve_cross_kg() rather than the hand-built two-hop query I
    guessed at before seeing this file.
    """
    flight_number = routing["entity"]
    direction = routing["direction"]

    flight_uri = map_flight(flight_number)
    if not flight_uri:
        return _ui_result(language=lang, branch="cross_kg", entity=flight_number,
                           execution={"value": None, "error": "mapping_failure"})

    entities = extract_airport_entities(question, lang, iata_from_router=None)

    lexicon_path = get_lexicon("airports")
    lexicon = load_lexicon(lexicon_path)
    property_uri, tier, property2_uri = map_property_cascade(
        entities["property"], lexicon, lexicon_path
    )

    if not property_uri:
        return _ui_result(language=lang, branch="cross_kg", entity=flight_number,
                           property_surface=entities["property"], tier=tier,
                           execution={"value": None, "error": "mapping_failure"})

    BASE = get_base_uri("airports")
    full_prop_uri = BASE + property_uri

    result = resolve_cross_kg(
        flight_uri=flight_uri, direction=direction,
        property_uri=full_prop_uri, property_short=property_uri,
    )

    # Build the path for the cross-KG graph widget from resolve_cross_kg's
    # own output — iata and airport_uri are the two real hops it exposes.
    path = None
    if result.get("iata"):
        path = [
            {"from": flight_number, "to": result["iata"], "label": f"IATA ({direction})"},
            {"from": result["iata"], "to": property_uri, "label": property_uri},
        ]

    if result["success"]:
        answer = format_answer(question, result["raw_value"], lang)
        return _ui_result(language=lang, branch="cross_kg", entity=flight_number,
                           property_surface=entities["property"], resolved_uri=result.get("airport_uri"),
                           tier=tier, sparql=None,
                           execution={"value": result["raw_value"], "error": None},
                           answer=answer, path=path)

    return _ui_result(language=lang, branch="cross_kg", entity=flight_number,
                       property_surface=entities["property"], resolved_uri=result.get("airport_uri"),
                       tier=tier, execution={"value": None, "error": result.get("failure_type")},
                       path=path)


def process_question(question: str, strategy: str = "zero-shot") -> dict:
    """Single entry point for the Streamlit UI — mirrors main.py's dispatch exactly."""
    lang = detect_language(question)
    routing = route(question)
    branch = routing["query_type"]

    if branch == "single_kg1":
        return _run_single_kg1(question, lang, strategy)
    if branch == "single_kg2":
        return _run_single_kg2(question, routing, lang, strategy)
    if branch == "single_kg3":
        return _run_single_kg3(question, routing, lang, strategy)
    if branch == "cross_kg":
        return _run_cross_kg(question, routing, lang)

    if branch == "template":
        result = resolve_template(question, routing["template"], lang, routing.get("params"))
        return _ui_result(
            language=lang, branch=branch, property_surface=routing["template"],
            sparql=result.get("sparql"),
            execution={"value": result.get("raw_data"),
                       "error": None if result.get("success") else result.get("failure_type")},
            answer=result.get("final_answer"),
        )

    if branch == "ask_query":
        result = resolve_ask_query(question, routing, lang)
        return _ui_result(
            language=lang, branch=branch, entity=routing.get("entity"),
            resolved_uri=result.get("entity_uri"),
            execution={"value": result.get("success"), "error": None},
            answer=result.get("final_answer"),
        )

    if branch == "open_kg":
        sparql = generate_open_kg_sparql(question, lang)
        return _ui_result(language=lang, branch=branch, sparql=sparql,
                           execution={"value": None, "error": "open_kg execution not wired here — see kg-detection fix in generator.py"})

    return _ui_result(language=lang, branch=branch,
                       execution={"value": None, "error": "out_of_scope"})