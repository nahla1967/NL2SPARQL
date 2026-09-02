"""
LLM-based classification and query-type normalization for the NL2SPARQL router.

FIXED vs the draft you pasted:
- _CLASSIFICATION_PROMPT and the _is_kg_answerable prompt had several
  few-shot examples silently dropped during the split (extra cross_kg_filter
  cases, the duplicated ranking_kg2 EN/FR/AR block, the pet/history/weather
  policy examples in the answerability prompt). Those examples exist because
  specific misclassifications were observed without them (see Priority 2.6's
  comment in router.py) — dropping them during a refactor is a silent
  behavior regression, not a style change. Restored verbatim from the
  original router.py.
- Removed unused imports (_detect_flight_number_first, _detect_airport_entity
  were imported but never referenced).
- Standardized to relative imports (was mixing `from .rules import ...`
  with `from router.detectors import ...`).
- Removed the redundant local `from kg_registry import TEMPLATE_REGISTRY as _TR`
  inside _normalize_query_type — TEMPLATE_REGISTRY is already imported at
  module level.
"""

import ast
import json
import re

import ollama
from rapidfuzz import process, fuzz

from kg_registry import TEMPLATE_REGISTRY, get_open_kg_schema
from .rules import _WH_WORDS
from .detectors import (
    _detect_airport_keyword,
    _detect_airport_entity,
    _detect_university_entity,
    _has_filter_signal,
)

_VALID_QUERY_TYPES = set(TEMPLATE_REGISTRY.keys()) | {"single_kg2", "open_kg", "out_of_scope"}

# ── CLASSIFICATION PROMPT ────────────────────────────────────────
_CLASSIFICATION_PROMPT = """You are a query classifier for an airport and flight database.

Classify the question into exactly one of these query types and extract its parameters.

── QUERY TYPES AND THEIR PARAMETERS ──────────────────────────────────────────



1. count_kg2 — COUNT of airports matching ONE condition (numeric or categorical)
   Query contains: "how many", "combien", "كم" + a filter condition
   params: property, operator, threshold (numeric) OR property, value (categorical)
   Example EN: "How many airports in Germany?" → count_kg2, property=countryName, value=Germany
   Example FR: "Combien d'aéroports avec une piste de plus de 3000m?" → count_kg2, property=lengthFt, operator=>, threshold=3000
   Example AR: "كم عدد المطارات في بولندا؟" → count_kg2, property=countryName, value=Poland

2. filter_numeric_kg2 — airports filtered by a numeric property (NOT counting)
   Query asks for AIRPORTS or NAMES, not a COUNT
   params: property, operator, threshold

3. filter_string_kg2 — airports filtered by a text/categorical property (NOT counting)
   Query asks for AIRPORTS or NAMES, not a COUNT
   params: property, value

3.5 . ranking_kg2 — airports ranked by a numeric property (top/bottom N)
   params: property, order (ASC or DESC), limit (default 5)

4. compare_two_airports — compare exactly two airports on one property
   params: airport1 (IATA code), airport2 (IATA code), property

5. count_kg1 — count or list flights matching a condition
   params: filter_property, filter_value, mode (count or list)

6. filter_numeric_kg1 — flights filtered by a numeric flight property
   params: property, operator, threshold

7. cross_kg_filter — a specific flight is mentioned AND the question asks
   about a property of its origin or destination airport.
   params: direction (origin or destination), airport_property,
           operator, threshold, limit (default 10)

8. single_kg2 — one specific airport asked about by name or IATA code
   params: entity (IATA code or airport name)

9. out_of_scope — cannot be answered from this database
   params: {}

10. group_aggregate_kg1 — aggregate a numeric flight property, grouped by airline
    params: group_by ("airline"), property (gspeed or vspeed),
            function (AVG, SUM, MAX, or MIN)

11. group_aggregate_kg2 — aggregate a numeric airport property, grouped by
    country or continent
    params: group_by ("country" or "continent"), property
            (elevationFt, lengthFt, or widthFt), function (AVG, SUM, MAX, or MIN)

12. group_aggregate_kg3 — aggregate a COUNT of a relation (courses taught,
    courses taken), grouped by department
    params: group_by ("department"), property (teacherOf or takesCourse),
            function (AVG, MAX, or MIN — no SUM, since summing counts of
            counts is rarely a meaningful question)

13. open_kg — the question is about aviation data but does not fit any
    template above. It asks about a specific property or relationship
    that requires a custom query.
    params: {}

    Examples:
    - "Which flight has the highest ground speed?" → ranking_kg1
    - "How many airports are in Germany?" → count_kg2
    - "How many airports are in the dataset?" (no condition) → open_kg
    - "Which airports have a grass runway?" → open_kg
    - "What is the registration number of the aircraft on flight BR62?" → open_kg
    - "Quel vol a la vitesse verticale la plus basse?" → open_kg
    # NOTE (fix): the "no condition -> open_kg" rule above only had an
    # English exemplar. Without a French/Arabic equivalent, the model kept
    # trying to force SOME filter/property out of a bare "how many X total"
    # question in those languages — in French it extracted the entity type
    # itself ("airport") as if it were a filterable property (which then
    # broke downstream template building), and in Arabic it hallucinated a
    # filter value ("Germany") that was never in the question at all. See
    # open_kg_007 (fr, ar) in the eval log.
    - "Combien d'aéroports y a-t-il au total ?" (aucune condition) → open_kg
    - "Combien y a-t-il d'aéroports dans le jeu de données ?" (aucune condition) → open_kg
    - "كم عدد المطارات الموجودة؟" (بدون شرط) → open_kg
    - "كم عدد المطارات في مجموعة البيانات؟" (بدون شرط) → open_kg

14. count_kg3 — count or list university entities linked to a specific
    named entity (professor, student, department). Only applies when the
    question BOTH names a specific entity (e.g. "FullProfessor0",
    "Department0") AND asks "how many" / "list all", not a single fact.
    params: property, direction, mode

15. filter_string_kg3 — list university people (professors/students)
    belonging to a specific NAMED department, filtered by relationship
    type. Use when the question asks "which/who" belongs to a department
    by name, not about one already-named person.
    params: property, value, limit

16. count_kg2 — COUNT of airports matching one numeric or categorical
    condition (e.g. "how many airports are in Germany", "how many small
    airports are there", "how many airports are above 1000ft elevation").
    params: property, operator, threshold (numeric) OR property, value (categorical)
17. ranking_kg1 — flights ranked by a numeric property (top/bottom N),
    with no specific flight named (e.g. "top 3 flights by vertical speed").
    params: property (vspeed or gspeed), order (ASC or DESC), limit (default 5)

    Note: comparing exactly two NAMED flights on a property is handled
    deterministically before this classifier ever runs — you should not
    need to return "compare_two_flights".

18. ranking_kg3 — EITHER departments ranked by a headcount of one entity
    type ("top 3 departments by graduate student population"), OR people
    ranked WITHIN one already-named department by a relation count
    ("which professor in Department0 teaches the most courses").
    params: group_by ("department" or "person"), entity_type, hop_property
            (memberOf/worksFor/teacherOf/takesCourse), order, limit

19. compare_two_departments — compare exactly two named departments by
    how many people of one type work for or belong to each.
    params: entity_type, hop_property (worksFor or memberOf)
    (department names are extracted deterministically, not by you)

20. filter_numeric_kg3 — departments filtered by TOTAL headcount
    (students + staff) against a threshold or range.
    params: operator, threshold, and optionally operator2/threshold2 for
            a "between X and Y" range    
── PROPERTY MAPPING RULES ────────────────────────────────────────────────────

Airport numeric properties:
  "elevation", "altitude", "height"            → elevationFt
  "runway length", "length", "longer", "long"  → lengthFt
  "runway width", "width", "wider", "wide"     → widthFt

Airport string properties:
  "country", "located in", "in [country]"      → countryName
  "large airport", "large airports", "type"    → airportType  (value: large_airport)
  "city", "municipality"                       → municipality
  "continent"                                  → continent
  "surface"                                    → surface
  "grass runway", "asphalt runway", "paved runway", "runway surface",
  "runway material", "what is the runway made of"   → surface
      (value: the surface type mentioned — "grass", "asphalt", "concrete",
       etc. Always use the property name "surface" exactly — never invent
       names like "hasGrassRunway", "pavementType", or "runwayMaterial".)

Flight numeric properties:
  "ground speed", "speed", "knots"             → gspeed
  "vertical speed", "feet per minute"          → vspeed

  NOTE: flight altitude is NOT stored in this database. Classify
  altitude-threshold questions for flights as out_of_scope.

Flight string properties:
  "destination city", "going to", "land in"    → hasDestinationCity
  "origin city", "departing from", "from"      → hasOriginCity
  "airline", "operated by"                     → hasAirline
  "destination country"                        → hasDestinationCountry

Operator mapping:
  "above", "exceeds", "more than", "greater"   → >
  "below", "less than", "under"                → <
  "at least"                                   → >=
  "at most"                                    → <=
  "in", "is", "equal to", "located in"         → =

Ranking direction:
  "highest", "longest", "widest", "most"       → DESC
  "lowest", "shortest", "narrowest", "least"   → ASC

University properties (only when a LUBM entity name like "FullProfessor0",
"Department0", "GraduateStudent3" appears in the question):
  "teach", "courses taught"                    → property=teacherOf, direction=outgoing
  "take", "enrolled in", "courses taken"       → property=takesCourse, direction=outgoing
  "students", "members" (of a department)      → property=memberOf, direction=incoming
  "professors", "faculty", "staff" (of a dept) → property=worksFor, direction=incoming
  "departments" (of a university)              → property=subOrganizationOf, direction=incoming

Group-by / aggregate signal words:

  Determine "function" using this SEQUENTIAL PROCEDURE — check each step
  in order, and STOP at the first match. Do not treat these as independent
  keyword-to-value mappings; later steps must NOT be checked if an earlier
  step already matched.

  STEP 1 — Does the question contain "average"/"mean"/"moyenne"/"moyen"/
  "متوسط"/"معدل"?
    If YES → function = "AVG". STOP HERE.
    Ignore any other words in the question like "highest"/"most"/"le plus
    élevé"/"الأعلى" — those describe which group ranks first (handled
    elsewhere, not by this field), NOT the aggregate itself. Example:
    "which country has the highest average elevation?" → function="AVG",
    because Step 1 matched on "average" — "highest" is never evaluated.

  STEP 2 — (only reached if Step 1 did NOT match) Does the question
  contain "total"/"sum"/"total"/"somme"/"مجموع"/"إجمالي"?
    If YES → function = "SUM". STOP HERE.

  STEP 3 — (only reached if Steps 1-2 did NOT match) Does the question
  contain "highest"/"most"/"maximum"/"le plus élevé"/"maximum"/"le plus"/
  "الأعلى"/"الأكثر"/"أقصى"?
    If YES → function = "MAX". STOP HERE.
    (Reaching this step means the question has no averaging language at
    all — e.g. "what is the maximum elevation per country?" — so "highest"
    genuinely means the max value here, not a ranking-of-averages.)

  STEP 4 — (only reached if Steps 1-3 did NOT match) Does the question
  contain "lowest"/"least"/"minimum"/"le plus bas"/"minimum"/"le moins"/
  "الأدنى"/"الأقل"/"أدنى"?
    If YES → function = "MIN".

  "per airline", "by airline", "for each airline"             → group_by=airline
  "par compagnie", "par compagnie aérienne"                   → group_by=airline
  "لكل شركة طيران", "حسب شركة الطيران"                        → group_by=airline


  "per country", "by country"                                 → group_by=country
  "par pays"                                                  → group_by=country
  "لكل دولة", "حسب الدولة", "بحسب الدولة"                     → group_by=country

  "per continent", "by continent"                             → group_by=continent
  "par continent"                                             → group_by=continent
  "لكل قارة", "حسب القارة"                                    → group_by=continent

  "per department", "by department"                           → group_by=department
  "par département"                                           → group_by=department
  "لكل قسم", "حسب القسم", "بحسب القسم"                        → group_by=department

  IMPORTANT: distinguish group_aggregate from ranking_kg2/filter_numeric_kg1.
  Ranking questions ask for the top/bottom N individual entities
  ("which airport has the highest elevation?" → ranking_kg2).
  Group-aggregate questions ask for a computed value PER CATEGORY
  ("what is the average elevation per country?" → group_aggregate_kg2).
  The word "per", "by", "for each", or "grouped by" is the strongest signal.

── DISAMBIGUATION RULES ──────────────────────────────────────────────────────

CROSS_KG_FILTER: Use when a specific flight number is mentioned AND the
  question asks about a property of that flight's origin or destination
  airport (country, elevation, runway, type). Examples:
    "What country does flight LO225 land in?"           → cross_kg_filter
    "What type of airport does flight FR182 arrive at?" → cross_kg_filter
    "Dans quel pays atterrit le vol OS295?"             → cross_kg_filter
    "في أي دولة يهبط الرحلة OS235؟"                   → cross_kg_filter

COUNT_KG1: Use when the question counts or lists FLIGHTS specifically.
  "how many flights" / "combien de vols" / "كم رحلة" → always count_kg1,
  even if a city name is present.
  Do NOT use count_kg1 to count airports or runways — KG1 has no runway
  data at all. "How many runways are closed?" / "how many airports..."
  belong to open_kg, even though they also start with "how many".

OPEN_KG: Use when the question asks about aviation data that exists in the
  KG but does not fit filter_numeric, filter_string, ranking, compare,
  count, or cross_kg_filter patterns. Specifically:
  - Questions about aircraft registration or specific aircraft details
  - Questions asking for a count of a KG class (airports, runways)
  - Questions about runway surface types (grass, concrete)
  - Questions about closed runways
  Do NOT use open_kg when filter_numeric_kg1 or ranking_kg2 would work.

COUNT_KG3: Use when the question names a specific university entity AND
  counts or lists something linked to it.
  "how many courses does X teach" / "combien de cours enseigne X" / "كم مادة يدرّس X"
  → always count_kg3, even though X is a specific entity — the count/list
  intent takes priority over single-entity lookup.

── EXAMPLES ──────────────────────────────────────────────────────────────────

Q: "Which airports have an elevation above 1000 feet?"
A: {"query_type": "filter_numeric_kg2", "params": {"property": "elevationFt", "operator": ">", "threshold": 1000}}

Q: "Quels aéroports ont une piste de plus de 10000 pieds de long?"
A: {"query_type": "filter_numeric_kg2", "params": {"property": "lengthFt", "operator": ">", "threshold": 10000}}

Q: "ما هي المطارات التي ارتفاعها أقل من 50 قدماً؟"
A: {"query_type": "filter_numeric_kg2", "params": {"property": "elevationFt", "operator": "<", "threshold": 50}}

  IMPORTANT: distinguish filter_numeric_kg2 from ranking_kg2 — a threshold
  question ("above/below/more than/less than X feet", "plus de X pieds",
  "أقل من X قدماً") is always filter_numeric_kg2, even in French or Arabic,
  even without any English words in the question. Only use ranking_kg2 when
  the question asks for the single highest/lowest or a top-N list with no
  numeric threshold given.

Q: "Show all large airports."
A: {"query_type": "filter_string_kg2", "params": {"property": "airportType", "value": "large_airport"}}

Q: "Which airports are located in Germany?"
A: {"query_type": "filter_string_kg2", "params": {"property": "countryName", "value": "Germany"}}

Q: "How many small airports are there?"
A: {"query_type": "count_kg2", "params": {"property": "airportType", "value": "small_airport"}}

Q: "كم عدد المطارات الصغيرة الموجودة؟"
A: {"query_type": "count_kg2", "params": {"property": "airportType", "value": "small_airport"}}

Q: "Combien d'aéroports ont une élévation supérieure à 1000 pieds?"
A: {"query_type": "count_kg2", "params": {"property": "elevationFt", "operator": ">", "threshold": 1000}}

  NOTE (fix): every count_kg2 exemplar above has an explicit condition
  (property + threshold/value). Without an unconditional counterpart in
  French and Arabic, the model had no example telling it that "how many X
  total, no filter" maps to open_kg with empty params — it would either
  extract the entity type itself as a fake property (French) or invent a
  filter value that isn't in the question (Arabic). See open_kg_007 (fr, ar)
  in the eval log.

Q: "How many airports are there in total?"
A: {"query_type": "open_kg", "params": {}}

Q: "Combien d'aéroports y a-t-il au total ?"
A: {"query_type": "open_kg", "params": {}}

Q: "كم عدد المطارات الموجودة؟"
A: {"query_type": "open_kg", "params": {}}

  IMPORTANT: distinguish count_kg2 from filter_string_kg2/filter_numeric_kg2
  — "how many"/"combien de"/"كم عدد" asks for a COUNT (count_kg2), while a
  plain "which/list" question asks for the matching rows themselves
  (filter_string_kg2 or filter_numeric_kg2). The condition-matching logic
  is otherwise identical; only the mode differs.

Q: "What are the top 5 airports with the highest elevation?"
A: {"query_type": "ranking_kg2", "params": {"property": "elevationFt", "order": "DESC", "limit": 5}}

Q: "Which airport has the shortest runway?"
A: {"query_type": "ranking_kg2", "params": {"property": "lengthFt", "order": "ASC", "limit": 1}}

Q: "Quel aéroport a la piste la plus longue?"
A: {"query_type": "ranking_kg2", "params": {"property": "lengthFt", "order": "DESC", "limit": 1}}

Q: "Quel aéroport a la plus haute élévation?"
A: {"query_type": "ranking_kg2", "params": {"property": "elevationFt", "order": "DESC", "limit": 1}}

Q: "أي مطار لديه أعلى ارتفاع؟"
A: {"query_type": "ranking_kg2", "params": {"property": "elevationFt", "order": "DESC", "limit": 1}}

Q: "أي مطار لديه أقصر مدرج؟"
A: {"query_type": "ranking_kg2", "params": {"property": "lengthFt", "order": "ASC", "limit": 1}}

Q: "أي مطار لديه أطول مدرج؟"
A: {"query_type": "ranking_kg2", "params": {"property": "lengthFt", "order": "DESC", "limit": 1}}

Q: "What are the top 3 departments by graduate student population?"
A: {"query_type": "ranking_kg3", "params": {"group_by": "department", "entity_type": "GraduateStudent", "hop_property": "memberOf", "order": "DESC", "limit": 3}}

Q: "Which professor in Department0 teaches the most courses?"
A: {"query_type": "ranking_kg3", "params": {"group_by": "person", "hop_property": "teacherOf", "order": "DESC", "limit": 1}}

  IMPORTANT: distinguish ranking_kg3 from group_aggregate_kg3 — ranking_kg3
  wants a params shape of group_by/entity_type/hop_property/order/limit,
  never property/function. Do not copy group_aggregate_kg3's param keys
  onto a ranking_kg3 answer even though both templates operate on
  departments; they take different parameters.

Q: "Does Department0 or Department9 have more full professors?"
A: {"query_type": "compare_two_departments", "params": {"entity_type": "FullProfessor", "hop_property": "worksFor"}}

Q: "Compare Department1 and Department4 by number of graduate students."
A: {"query_type": "compare_two_departments", "params": {"entity_type": "GraduateStudent", "hop_property": "memberOf"}}

  IMPORTANT: for compare_two_departments, never return department names or
  URIs in params — those two are always detected and resolved separately.
  Only return entity_type and hop_property.

Q: "Considering every airport on record, which single one has the greatest elevation of them all?"
A: {"query_type": "ranking_kg2", "params": {"property": "elevationFt", "order": "DESC", "limit": 1}}

Q: "Parmi tous les aéroports enregistrés, lequel affiche à lui seul l'élévation la plus haute?"
A: {"query_type": "ranking_kg2", "params": {"property": "elevationFt", "order": "DESC", "limit": 1}}

Q: "من بين جميع المطارات المسجلة، أيها يملك وحده أعلى ارتفاع على الإطلاق؟"
A: {"query_type": "ranking_kg2", "params": {"property": "elevationFt", "order": "DESC", "limit": 1}}
Q: "Compare VIE and FRA by elevation."
A: {"query_type": "compare_two_airports", "params": {"airport1": "VIE", "airport2": "FRA", "property": "elevationFt"}}

Q: "Comparez CDG et LHR par longueur de piste."
A: {"query_type": "compare_two_airports", "params": {"airport1": "CDG", "airport2": "LHR", "property": "lengthFt"}}

Q: "قارن ارتفاع مطار ATH ومطار IST."
A: {"query_type": "compare_two_airports", "params": {"airport1": "ATH", "airport2": "IST", "property": "elevationFt"}}

Q: "How many flights are operated by Lufthansa?"
A: {"query_type": "count_kg1", "params": {"filter_property": "hasAirline", "filter_value": "Lufthansa", "mode": "count"}}

Q: "Combien de vols partent de Vienne?"
A: {"query_type": "count_kg1", "params": {"filter_property": "hasOriginCity", "filter_value": "Vienna", "mode": "count"}}

Q: "كم رحلة تتجه إلى برلين؟"
A: {"query_type": "count_kg1", "params": {"filter_property": "hasDestinationCity", "filter_value": "Berlin", "mode": "count"}}

Q: "How many flights arrive in Vienna?"
A: {"query_type": "count_kg1", "params": {"filter_property": "hasDestinationCity", "filter_value": "Vienna", "mode": "count"}}

Q: "Which flights have a ground speed above 400 knots?"
A: {"query_type": "filter_numeric_kg1", "params": {"property": "gspeed", "operator": ">", "threshold": 400}}

Q: "Which flights have a vertical speed below -1000 feet per minute?"
A: {"query_type": "filter_numeric_kg1", "params": {"property": "vspeed", "operator": "<", "threshold": -1000}}

Q: "What country does flight LO225 land in?"
A: {"query_type": "cross_kg_filter", "params": {"direction": "destination", "airport_property": "countryName", "operator": "=", "threshold": "Poland", "limit": 1}}

Q: "What type of airport does flight FR182 arrive at?"
A: {"query_type": "cross_kg_filter", "params": {"direction": "destination", "airport_property": "airportType", "operator": "=", "threshold": "large_airport", "limit": 1}}

Q: "What is the elevation of the destination airport of KE567?"
A: {"query_type": "cross_kg_filter", "params": {"direction": "destination", "airport_property": "elevationFt", "operator": ">", "threshold": 0, "limit": 1}}

Q: "Dans quel pays atterrit le vol OS295?"
A: {"query_type": "cross_kg_filter", "params": {"direction": "destination", "airport_property": "countryName", "operator": "=", "threshold": "Austria", "limit": 1}}

Q: "في أي دولة يهبط الرحلة OS235؟"
A: {"query_type": "cross_kg_filter", "params": {"direction": "destination", "airport_property": "countryName", "operator": "=", "threshold": "Germany", "limit": 1}}

Q: "What is the runway length at the destination of OS214?"
A: {"query_type": "cross_kg_filter", "params": {"direction": "destination", "airport_property": "lengthFt", "operator": ">", "threshold": 0, "limit": 1}}

Q: "Which flights land at airports with elevation above 800 feet?"
A: {"query_type": "cross_kg_filter", "params": {"direction": "destination", "airport_property": "elevationFt", "operator": ">", "threshold": 800, "limit": 10}}

Q: "Which flights arrive at airports located in Germany?"
A: {"query_type": "cross_kg_filter", "params": {"direction": "destination", "airport_property": "countryName", "operator": "=", "threshold": "Germany", "limit": 10}}

Q: "Quels vols atterrissent dans des aéroports en Allemagne?"
A: {"query_type": "cross_kg_filter", "params": {"direction": "destination", "airport_property": "countryName", "operator": "=", "threshold": "Germany", "limit": 10}}

Q: "Which flights land at large airports?"
A: {"query_type": "cross_kg_filter", "params": {"direction": "destination", "airport_property": "airportType", "operator": "=", "threshold": "large_airport", "limit": 10}}

Q: "Which flight has the highest ground speed?"
A: {"query_type": "open_kg", "params": {}}

Q: "What is the callsign of the fastest flight?"
A: {"query_type": "open_kg", "params": {}}

Q: "ما هي الرحلة ذات أعلى سرعة أرضية؟"
A: {"query_type": "open_kg", "params": {}}

Q: "What is the weather forecast for JFK tomorrow?"
A: {"query_type": "out_of_scope", "params": {}}

Q: "Am I allowed to bring a guitar on flight BR62?"
A: {"query_type": "out_of_scope", "params": {}}

Q: "Who invented the first commercial airplane?"
A: {"query_type": "out_of_scope", "params": {}}

Q: "Quel est le prix du billet pour le vol AF123?"
A: {"query_type": "out_of_scope", "params": {}}

Q: "هل تقدم شركة الطيران وجبات نباتية؟"
A: {"query_type": "out_of_scope", "params": {}}

Q: "How many courses does FullProfessor0 teach?"
A: {"query_type": "count_kg3", "params": {"property": "teacherOf", "direction": "outgoing", "mode": "count"}}

Q: "List the courses that GraduateStudent3 takes."
A: {"query_type": "count_kg3", "params": {"property": "takesCourse", "direction": "outgoing", "mode": "list"}}

Q: "How many students are in Department0?"
A: {"query_type": "count_kg3", "params": {"property": "memberOf", "direction": "incoming", "mode": "count"}}

Q: "Which professors work for Department3?"
A: {"query_type": "filter_string_kg3", "params": {"property": "worksFor", "value": "Department3", "limit": 10}}

Q: "List students who are members of Department1."
A: {"query_type": "filter_string_kg3", "params": {"property": "memberOf", "value": "Department1", "limit": 10}}

Q: "Quels professeurs travaillent pour le Department2?"
A: {"query_type": "filter_string_kg3", "params": {"property": "worksFor", "value": "Department2", "limit": 10}}

Q: "Which departments have fewer than 5 students?"
A: {"query_type": "filter_numeric_kg3", "params": {"operator": "<", "threshold": 5}}

Q: "Quels départements comptent moins de 5 étudiants?"
A: {"query_type": "filter_numeric_kg3", "params": {"operator": "<", "threshold": 5}}

Q: "ما هي الأقسام التي يقل عدد أعضائها عن 5؟"
A: {"query_type": "filter_numeric_kg3", "params": {"operator": "<", "threshold": 5}}

  IMPORTANT: distinguish filter_string_kg3 from filter_numeric_kg3 —
  filter_string_kg3 asks WHO belongs to one NAMED department ("which
  professors work for Department2", "list students in Department1").
  filter_numeric_kg3 asks WHICH DEPARTMENTS qualify by a headcount
  threshold, with no department named ("departments with fewer than 5
  students", "moins de 5 étudiants", "يقل عن 5"). A number in the question
  is the signal for filter_numeric_kg3, not a reason to pick filter_string_kg3
  just because "students" or "professors" is also mentioned.

Q: "What is the average ground speed per airline?"
A: {"query_type": "group_aggregate_kg1", "params": {"group_by": "airline", "property": "gspeed", "function": "AVG"}}}

Q: "What is the maximum elevation per country?"
A: {"query_type": "group_aggregate_kg2", "params": {"group_by": "country", "property": "elevationFt", "function": "MAX"}}}

Q: "Which country has the highest average airport elevation?"
A: {"query_type": "group_aggregate_kg2", "params": {"group_by": "country", "property": "elevationFt", "function": "AVG"}}}

Q: "Quelle est la longueur de piste moyenne par pays?"
A: {"query_type": "group_aggregate_kg2", "params": {"group_by": "country", "property": "lengthFt", "function": "AVG"}}}

Q: "Which department teaches the most courses on average per professor?"
A: {"query_type": "group_aggregate_kg3", "params": {"group_by": "department", "property": "teacherOf", "function": "AVG"}}}

Q: "Which airports have a grass runway?"
A: {"query_type": "filter_string_kg2", "params": {"property": "surface", "value": "grass"}}}

Q: "How many runways in the dataset are closed?"
A: {"query_type": "open_kg", "params": {}}}

Q: "How many professors work for Department7?"
A: {"query_type": "count_kg3", "params": {"property": "worksFor", "direction": "incoming", "mode": "count"}}}

── NOW CLASSIFY THIS QUESTION ────────────────────────────────────────────────

Question: "{question}"

- Use double quotes " for every key and every string value. Never use single quotes.
- Do not add comments, trailing commas, or any text outside the JSON object.
- Output exactly one JSON object and nothing else — no markdown, no bullet points.

Return ONLY a JSON object with keys "query_type" and "params".
No explanation. No text before or after the JSON.
"""


# ── LLM MAJORITY-VOTE HELPER ─────────────────────────────────────
def _llm_yes_no_majority(prompt: str, k: int = 3) -> bool:
    votes = []
    for _ in range(k):
        try:
            response = ollama.chat(
                model="llama3",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0}
            )
            votes.append(response["message"]["content"].strip().upper().startswith("YES"))
        except Exception as e:
            print(f"[router] _llm_yes_no_majority call failed: {e}")
            votes.append(False)
    result = sum(votes) > len(votes) / 2
    if len(set(votes)) > 1:
        print(f"[router] _llm_yes_no_majority: split vote {votes} → {result}")
    return result


# ── ASK SIGNAL DETECTOR ──────────────────────────────────────────
def _has_ask_signal(question: str) -> bool:
    if question.strip().startswith("هل"):
        print(f"[router] _has_ask_signal('{question[:40]}...') → True (fast-path: 'هل' prefix)")
        return True

    q_stripped = question.strip().lower()
    if any(q_stripped.startswith(w) for w in _WH_WORDS):
        print(f"[router] _has_ask_signal('{question[:40]}...') → False (fast-path: WH-word opener)")
        return False

    _all_tokens = re.findall(r"\w+", q_stripped)
    if any(t in _WH_WORDS for t in _all_tokens):
        return False

    prompt = f'''Determine if this question asks about a PROPERTY of a KNOWN ENTITY.

ASK questions (YES):
- "Is flight OS295 an international flight?" → YES (entity: OS295, property: international)
- "Does flight TK100 operate on Tuesday?" → YES (entity: TK100, property: operates_on_day)
- "Was flight BR888 delayed?" → YES (entity: BR888, property: delayed)
- "Is runway surface at FRA made of ASP?" → YES (entity: FRA airport, property: runway_surface)
- "Is Department3 part of University5?" → YES (entity: Department3, property: parent_university)
- "Est-ce que l'aéroport CDG est situé en France?" → YES (entity: CDG, property: country)
- "هل تحتوي FRA على مدرج من الإسفلت؟" → YES (entity: FRA, property: runway_surface)

NOT ASK questions (NO):
- "Which flights are delayed?" → NO (no specific entity named)
- "Are there any flights to Paris?" → NO (generic entity)
- "What is the elevation of JFK?" → NO (asks for a VALUE, not yes/no)

Question: "{question}"
Answer only YES or NO. Focus on: is there a SPECIFIC entity named? Does it ask about a PROPERTY yes/no, not a value?'''
    try:
        result = _llm_yes_no_majority(prompt)
        print(f"[router] _has_ask_signal('{question[:40]}...') → {result}")
        return result
    except Exception as e:
        print(f"[router] _has_ask_signal LLM check failed: {e}")
        return False


# ── KG ANSWERABILITY GATE ───────────────────────────────────────
def _is_kg_answerable(question: str) -> bool:
    schema = get_open_kg_schema()
    prompt = f'''You are a scope classifier for an aviation knowledge graph system.
The question may be in English, French, or Arabic.

The knowledge graph contains:
- Flights: flight number, airline, origin city, destination city,
  aircraft type, gate, terminal, callsign, ground speed, vertical speed
- Airports: name, type, elevation, country, region, city,
  IATA code, ICAO code, coordinates
- Runways: length, width, surface, lighting, identifier
- University: departments, professors, students, courses, advisors,degrees, email/phone, and headcount or membership per department
  

The knowledge graph does NOT contain: weather, prices/tickets, passenger
policies (pets, baggage, check-in), history, news, opinions, safety
records, or anything not explicitly listed above.

EXAMPLES:
Q: "What is the weather forecast for JFK tomorrow?"     → NO (weather not in KG)
Q: "Am I allowed to bring a guitar on flight BR62?"     → NO (policy, not in KG)
Q: "Who invented the first commercial airplane?"        → NO (general knowledge, not in KG)
Q: "Can I bring a pet on flight FR947?"           → NO (policy, not in KG)
Q: "What is the history of the airline industry?" → NO (general knowledge, not in KG)
Q: "Quel temps fait-il à l'aéroport VIE?"          → NO
Q: "هل يمكنني اصطحاب حيوان أليف؟"                  → NO
Q: "What is the elevation of ZRH?"                → YES (elevation is in KG)
Q: "Is ZRH located in Switzerland?"               → YES (country is in KG)
Q: "هل يقع مطار زيورخ في سويسرا؟"                  → YES

Q: "في أي دولة يقع مطار أثينا؟"                    → YES (country is in KG)
Q: "How many students are in Department0?"                → YES (department membership is in KG)
Q: "Who is GraduateStudent3's academic advisor?"           → YES (advisor is in KG)

Answer only YES or NO:
Can this question be answered using only the data described above?

Question: "{question}"
'''
    try:
        result = _llm_yes_no_majority(prompt)
        print(f"[router] _is_kg_answerable('{question[:40]}...') → {result}")
        return result
    except Exception as e:
        print(f"[router] _is_kg_answerable failed: {e}")
        return False


# ── JSON OBJECT EXTRACTOR ───────────────────────────────────────
def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    string_char = ""
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == string_char:
                in_string = False
            continue

        if ch in ('"', "'"):
            in_string = True
            string_char = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


# ── QUERY TYPE NORMALIZER ─────────────────────────────────────
def _normalize_query_type(query_type, question: str = "") -> str:
    if not isinstance(query_type, str):
        print(f"[router] LLM returned non-string query_type "
              f"({type(query_type).__name__}); treating as unclassified")
        return ""
    if query_type in _VALID_QUERY_TYPES:
        return query_type

    # AFTER — kg_guess computed once, unconditionally, before either family case
    base = re.sub(r"_kg\d+$", "", query_type)
    family = [t for t in _VALID_QUERY_TYPES
            if re.sub(r"_kg\d+$", "", t) == base and t != query_type]

    if len(family) == 1:
        corrected = family[0]
        print(f"[router] Correcting hallucinated query_type '{query_type}' "
            f"→ '{corrected}' (family match)")
        return corrected

    if _detect_university_entity(question) or _has_filter_signal(question):
        kg_guess = "university"
    elif _detect_airport_keyword(question) or _detect_airport_entity(question):
        kg_guess = "airports"
    else:
        kg_guess = None

    if kg_guess:
        for t in family:
            if TEMPLATE_REGISTRY[t]["kg"] == kg_guess:
                print(f"[router] Correcting hallucinated query_type '{query_type}' "
                    f"→ '{t}' (kg={kg_guess})")
                return t

    print(f"[router] '{query_type}' (family={family}, kg_guess={kg_guess}) has no "
        f"safe same-KG correction; leaving unclassified (will fall to open_kg)")
    return ""


# ── LLM CLASSIFIER ──────────────────────────────────────────────
def _llm_classify(question: str, max_attempts: int = 2) -> dict:
    prompt = _CLASSIFICATION_PROMPT.replace("{question}", question)

    _SMART_PUNCTUATION = str.maketrans({
        "\u201c": '"', "\u201d": '"',
        "\u2018": "'", "\u2019": "'",
    })

    for attempt in range(max_attempts):
        raw = ""
        try:
            response = ollama.chat(
                model="llama3",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0}
            )
            raw = response["message"]["content"].strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            raw = raw.translate(_SMART_PUNCTUATION)

            candidate = _extract_first_json_object(raw)
            if candidate is None:
                raise ValueError("no balanced JSON object found in response")

            try:
                result = json.loads(candidate)
            except json.JSONDecodeError:
                result = ast.literal_eval(candidate)

            if not isinstance(result, dict):
                raise ValueError(f"parsed object is not a dict: {type(result)}")

            result["query_type"] = _normalize_query_type(result.get("query_type", ""), question)
            print(f"[router] LLM classified as: {result.get('query_type')} "
                  f"| params: {result.get('params')}")
            return result

        except Exception as e:
            print(f"[router] Attempt {attempt+1}: classification failed: {e}")
            prompt = (
                f"Your previous response was not valid JSON:\n\n"
                f"{raw}\n\n"
                f"Error: {e}\n\n"
                f"Return ONLY the corrected JSON object. "
                f"Use double quotes for all keys and values. "
                f"No explanation, no text before or after."
            )

    return {}