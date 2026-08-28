"""
Main routing logic for the NL2SPARQL pipeline.

FIXED vs the draft you pasted: _RANKING_SIGNALS and _ASC_SIGNALS were
redefined locally here even though rules.py already defines them —
harmless while the two copies match, but a duplication trap if either
list is ever tuned without the other. Now imported from rules.py so
there's one source of truth.
"""

import re

from template_resolver import KG2_NUMERIC_PROPS, KG2_STRING_PROPS
from kg_registry import KG_REGISTRY, CROSS_KG_CONFIG, TEMPLATE_REGISTRY

from .rules import _has_minimum_structure, _normalise, _RANKING_SIGNALS, _ASC_SIGNALS
from .detectors import (
    _detect_flight_number,
    _detect_flight_number_first,
    _detect_airport_entity,
    _detect_university_entity,
    _detect_two_airport_codes,
    _detect_compare_property,
    _has_open_kg_signal,
    _has_kg1_signal,
    _has_compare_signal,
    _has_count_signal,
    _has_filter_signal,
    _AIRPORT_ENTITIES,
    
)
from .classifier import (
    _has_ask_signal,
    _llm_classify,
    _is_kg_answerable,
)


def route(question: str) -> dict:
    """
    Routes a natural language question to the correct pipeline branch.

    Returns a routing dict consumed by main.py and test_pipeline.py.
    Keys: query_type, kg, entity, direction, template, config, params.

    Final structure:
        Priority 1   → structure guard
        Priority 1.5 → ASK-style question + known entity → ask_query
        Priority 2   → flight number → single_kg1 or cross_kg
        Priority 2.5 → airport entity → single_kg2
        Priority 2.6 → two airports + compare signal → compare_two_airports (deterministic)
        Priority 2.7 → university entity → single_kg3 (unless count/filter signal)
        Priority 3   → LLM classifies → template / single_kg2 / open_kg
                       (with smart reroute for known misclassification patterns)
        Clean gate   → _is_kg_answerable() → open_kg or out_of_scope
    """

    # ── Priority 1: Structure guard ───────────────────────────────────────────
    if not _has_minimum_structure(question):
        return {
            "query_type": "out_of_scope",
            "kg":         None,
            "entity":     None,
            "direction":  None,
            "template":   None,
            "config":     None,
        }

    q_lower = question.lower()

    # ── Priority 1.5: ASK-style question + known entity (any KG) ──────────────
    if _has_ask_signal(question):
        flight_entity      = _detect_flight_number_first(question)
        airport_entity     = _detect_airport_entity(question)
        university_entity  = _detect_university_entity(question)

        if flight_entity:
            return {
                "query_type": "ask_query", "kg": "flights",
                "entity": flight_entity, "direction": None, "template": "ask_query",
            }
        elif airport_entity:
            return {
                "query_type": "ask_query", "kg": "airports",
                "entity": airport_entity, "direction": None, "template": "ask_query",
            }
        elif university_entity:
            return {
                "query_type": "ask_query", "kg": "university",
                "entity": university_entity, "direction": None, "template": "ask_query",
            }

    # ── Priority 2: Flight number detected ────────────────────────────────────
    flight = _detect_flight_number(question)

    if flight:

        # ── Fast path (2a-pre): open_kg signal word present ────────────────────
        if _has_open_kg_signal(q_lower):
            return {
                "query_type": "open_kg",
                "kg":         "cross",
                "entity":     None,
                "direction":  None,
                "template":   None,
                "config":     None,
            }

        # ── Fast path (2a): KG1-only signal word present ──────────────────────
        if _has_kg1_signal(q_lower):
            return {
                "query_type": "single_kg1",
                "kg":         "flights",
                "entity":     flight,
                "direction":  None,
                "template":   None,
                "config":     KG_REGISTRY["flights"],
            }

        # ── Slow path (2b): Ambiguous — ask the LLM ──────────────────────────
        classified = _llm_classify(question)
        query_type = classified.get("query_type", "")
        params     = classified.get("params", {})

        if query_type == "cross_kg_filter":
            direction = params.get("direction", "destination")
            return {"query_type": "cross_kg", "kg": "cross", "entity": flight,
                     "direction": direction, "template": None, "config": CROSS_KG_CONFIG}

        if query_type == "out_of_scope":
            return {"query_type": "out_of_scope", "kg": None, "entity": None,
                     "direction": None, "template": None, "config": None}

        # Fallback: plain KG1 flight question
        return {
            "query_type": "single_kg1",
            "kg":         "flights",
            "entity":     flight,
            "direction":  None,
            "template":   None,
            "config":     KG_REGISTRY["flights"],
        }

    # ── Priority 2.5: Airport entity detected (deterministic) ─────────────────
    airport = _detect_airport_entity(question)
    print(f"[router] Priority 2.5 check: airport={airport!r}")
    if airport and not _has_compare_signal(question) and not _has_count_signal(question):
        if not _is_kg_answerable(question):
            return {
                "query_type": "out_of_scope",
                "kg":         None,
                "entity":     None,
                "direction":  None,
                "template":   None,
                "config":     None,
            }
        return {
            "query_type": "single_kg2",
            "kg":         "airports",
            "entity":     airport,
            "direction":  None,
            "template":   None,
            "config":     KG_REGISTRY["airports"],
        }

    # ── Priority 2.6: Two airports + compare signal (deterministic) ───────────
    if _has_compare_signal(question):
        two_airports = _detect_two_airport_codes(question)
        compare_property = _detect_compare_property(question)
        if two_airports and compare_property:
            print(f"[router] Priority 2.6: compare signal + two airports "
                  f"{two_airports} → compare_two_airports")
            return {
                "query_type": "template",
                "kg":         "airports",
                "entity":     None,
                "direction":  None,
                "template":   "compare_two_airports",
                "config":     TEMPLATE_REGISTRY["compare_two_airports"],
                "params":     {"airport1": two_airports[0], "airport2": two_airports[1],
                               "property": compare_property},
            }

    # ── Priority 2.7: University entity detected (deterministic) ──────────────
    university_entity = _detect_university_entity(question)
    if university_entity and not _has_count_signal(question) and not _has_filter_signal(question):
        return {
            "query_type": "single_kg3",
            "kg":         "university",
            "entity":     university_entity,
            "direction":  None,
            "template":   None,
            "config":     KG_REGISTRY["university"],
        }

    # ── Priority 3: No flight/airport/university match — LLM classifies ───────
    classified = _llm_classify(question)
    query_type = classified.get("query_type", "")
    params     = classified.get("params", {})

    # ── Template branch ───────────────────────────────────────────────────────
    if query_type in TEMPLATE_REGISTRY:
        cfg = TEMPLATE_REGISTRY[query_type]

        KG1_FLIGHT_PROPS = {"gspeed", "vspeed", "alt", "groundSpeed", "speed"}

        # Case 1: KG2 template received a KG1 flight property
        if query_type in ("ranking_kg2", "filter_numeric_kg2"):
            prop = params.get("property", "")
            if prop in KG1_FLIGHT_PROPS:
                print(f"[router] Smart reroute: KG1 property in KG2 template → open_kg")
                return {
                    "query_type": "open_kg",
                    "kg":         "cross",
                    "entity":     None,
                    "direction":  None,
                    "template":   None,
                    "config":     None,
                }

        # Case 1.6b: group_aggregate_kg2 with group_by == the KG's own entity
        # ("which airport has the highest X") — this is a ranking, not a
        # group-aggregate. GROUP_AGGREGATE_KG2 only defines group_by =
        # country/continent, so group_by="airport" always fails template
        # build (_build_group_aggregate_kg2 returns None → sparql_build_failure).
        if query_type == "group_aggregate_kg2":
            from kg_registry import GROUP_AGGREGATE_KG2
            if params.get("group_by") not in GROUP_AGGREGATE_KG2["group_by"]:
                prop = params.get("property", "")
                if prop in KG2_NUMERIC_PROPS:
                    print(f"[router] Smart reroute: group_aggregate_kg2 with invalid "
                          f"group_by='{params.get('group_by')}' → ranking_kg2")
                    order = "ASC" if any(sig in question.lower() for sig in _ASC_SIGNALS) else "DESC"
                    query_type = "ranking_kg2"
                    params = {"property": prop, "order": order, "limit": params.get("limit") or 1}
                    cfg = TEMPLATE_REGISTRY[query_type]
                else:
                    print(f"[router] Smart reroute: group_aggregate_kg2 with invalid "
                          f"group_by='{params.get('group_by')}', unrecognised property → open_kg")
                    return {
                        "query_type": "open_kg", "kg": "cross", "entity": None,
                        "direction": None, "template": None, "config": None,
                    }

        # Case 1.5: cross_kg_filter classified with no flight entity
        if query_type == "cross_kg_filter":
            prop = params.get("airport_property", "")
            if prop in KG2_NUMERIC_PROPS:
                print(f"[router] Smart reroute: cross_kg_filter with no flight entity → filter_numeric_kg2")
                query_type = "filter_numeric_kg2"
                params = {"property": prop, "operator": params.get("operator", ">"),
                          "threshold": params.get("threshold")}
                cfg = TEMPLATE_REGISTRY[query_type]
            elif prop in KG2_STRING_PROPS:
                print(f"[router] Smart reroute: cross_kg_filter with no flight entity → filter_string_kg2")
                query_type = "filter_string_kg2"
                params = {"property": prop, "value": params.get("threshold")}
                cfg = TEMPLATE_REGISTRY[query_type]
            else:
                print(f"[router] Smart reroute: cross_kg_filter with no flight entity, unrecognised property → open_kg")
                return {
                    "query_type": "open_kg",
                    "kg":         "cross",
                    "entity":     None,
                    "direction":  None,
                    "template":   None,
                    "config":     None,
                }

        # Case 1.6: compare_two_airports classified with a missing airport code
        if query_type == "compare_two_airports":
            a1 = (params.get("airport1") or "").strip().upper()
            a2 = (params.get("airport2") or "").strip().upper()

            # was: if not a1 or not a2:
            if not a1 or not a2 or a1 not in _AIRPORT_ENTITIES or a2 not in _AIRPORT_ENTITIES:
                text_codes = _detect_two_airport_codes(question)
                if text_codes:
                    print(f"[router] compare_two_airports missing codes in "
                          f"LLM params, but {text_codes} found in question "
                          f"text -- treating as extraction failure, not "
                          f"rerouting to ranking_kg2")
                    return {
                        "query_type": "extraction_failure",
                        "kg":         "airports",
                        "entity":     None,
                        "direction":  None,
                        "template":   None,
                        "config":     None,
                    }
                prop = (params.get("property") or "").strip()
                if prop in KG2_NUMERIC_PROPS:
                    order = "ASC" if any(sig in question.lower() for sig in _ASC_SIGNALS) else "DESC"
                    limit_match = re.search(r"\b(\d+)\b", question)
                    limit = int(limit_match.group(1)) if limit_match else 1
                    print(f"[router] Smart reroute: compare_two_airports missing airport code(s) → ranking_kg2")
                    query_type = "ranking_kg2"
                    params = {"property": prop, "order": order, "limit": limit}
                    cfg = TEMPLATE_REGISTRY[query_type]
                else:
                    print(f"[router] Smart reroute: compare_two_airports missing airport code(s), unrecognised property → open_kg")
                    return {
                        "query_type": "open_kg",
                        "kg":         "cross",
                        "entity":     None,
                        "direction":  None,
                        "template":   None,
                        "config":     None,
                    }

        # Case 2: filter_numeric_kg1 with ranking intent and no real threshold
        if query_type == "filter_numeric_kg1":
            prop      = params.get("property", "")
            threshold = params.get("threshold")
            if prop in KG1_FLIGHT_PROPS and (
                threshold is None or
                any(sig in question.lower() for sig in _RANKING_SIGNALS)
            ):
                print(f"[router] Smart reroute: ranking signal in filter → open_kg")
                return {
                    "query_type": "open_kg",
                    "kg":         "cross",
                    "entity":     None,
                    "direction":  None,
                    "template":   None,
                    "config":     None,
                }

        # Case 3: filter_string_kg2 with runway surface or closed runway
        if query_type == "filter_string_kg2":
            value = params.get("value", "")
            if any(v in str(value).lower() for v in
                   ["grass", "closed", "grs", "closed_runway", "fermée", "مغلق",
                   "asphalt", "asp", "concrete", "con", "إسفلت", "إسفلتي"]):
                print(f"[router] Smart reroute: runway property → open_kg")
                return {
                    "query_type": "open_kg",
                    "kg":         "cross",
                    "entity":     None,
                    "direction":  None,
                    "template":   None,
                    "config":     None,
                }

        # Case 4: ranking_kg2 received a categorical (non-numeric) property
        if query_type == "ranking_kg2":
            prop = params.get("property", "")
            if prop not in KG2_NUMERIC_PROPS and prop in KG2_STRING_PROPS:
                value = params.get("value") or params.get("threshold")
                if value:
                    print(f"[router] Smart reroute: categorical property in ranking_kg2 → filter_string_kg2")
                    query_type = "filter_string_kg2"
                    params = {"property": prop, "value": value}
                else:
                    print(f"[router] Smart reroute: categorical property in ranking_kg2, no value → open_kg")
                    return {
                        "query_type": "open_kg", "kg": "cross", "entity": None,
                        "direction": None, "template": None, "config": None,
                    }

        # Case 5: filter_string_kg2 received a NUMERIC property
        if query_type == "filter_string_kg2":
            prop = params.get("property", "")
            if prop in KG2_NUMERIC_PROPS:
                raw_value = str(params.get("value", ""))
                operator  = params.get("operator")
                threshold = None
                if operator in (">", "<", ">=", "<="):
                    threshold = raw_value
                else:
                    m = re.match(r"\s*(>=|<=|>|<)?\s*(-?\d+(?:\.\d+)?)", raw_value)
                    if m:
                        operator  = m.group(1) or ">"
                        threshold = m.group(2)
                if threshold is not None:
                    print(f"[router] Smart reroute: filter_string_kg2 with numeric property → filter_numeric_kg2")
                    query_type = "filter_numeric_kg2"
                    params = {"property": prop, "operator": operator,
                              "threshold": threshold}
                    cfg = TEMPLATE_REGISTRY[query_type]
                else:
                    print(f"[router] Smart reroute: filter_string_kg2 with numeric property, "
                          f"no parseable threshold → open_kg")
                    return {
                        "query_type": "open_kg", "kg": "cross", "entity": None,
                        "direction": None, "template": None, "config": None,
                    }

        return {
            "query_type": "template",
            "kg":         cfg["kg"],
            "entity":     None,
            "direction":  params.get("direction"),
            "template":   query_type,
            "config":     cfg,
            "params":     params,
        }

    # ── Single airport branch ─────────────────────────────────────────────────
    if query_type == "single_kg2":
        entity = params.get("entity")
        if entity:
            entity_upper = entity.upper().strip()
            if entity_upper in _AIRPORT_ENTITIES:
                entity = _AIRPORT_ENTITIES[entity_upper]
            else:
                entity_norm = _normalise(entity)
                if entity_norm in _AIRPORT_ENTITIES:
                    entity = _AIRPORT_ENTITIES[entity_norm]
        return {
            "query_type": "single_kg2",
            "kg":         "airports",
            "entity":     entity,
            "direction":  None,
            "template":   None,
            "config":     KG_REGISTRY["airports"],
        }

    # ── open_kg branch ───────────────────────────────────────────────────
    if query_type == "open_kg":
        return {
            "query_type": "open_kg",
            "kg":         "cross",
            "entity":     None,
            "direction":  None,
            "template":   None,
            "config":     None,
        }

    # ── out_of_scope branch ──────────────────────────────────────────────
    if query_type == "out_of_scope":
        return {
            "query_type": "out_of_scope",
            "kg":         None,
            "entity":     None,
            "direction":  None,
            "template":   None,
            "config":     None,
        }

    # ── CLEAN GATE ────────────────────────────────────────────────────────────
    if _is_kg_answerable(question):
        return {
            "query_type": "open_kg",
            "kg":         "cross",
            "entity":     None,
            "direction":  None,
            "template":   None,
            "config":     None,
        }

    return {
        "query_type": "out_of_scope",
        "kg":         None,
        "entity":     None,
        "direction":  None,
        "template":   None,
        "config":     None,
    }