# Paste these into FULL_SYSTEM_TESTS in test_pipeline.py, then run:
#     python test_pipeline.py
#
# Two routing risks I could NOT verify without your actual lexicon file —
# watch these specifically in the output:
#   - Q5/Q6/Q7/Q18 depend on "Frankfurt Airport" / "Munich Airport" /
#     "Charles de Gaulle Airport" being exact-match phrases (or fuzzy-
#     matchable) in your airport lexicon. If any of these come back
#     routing_ok=False with query_type="open_kg" or "out_of_scope"
#     instead of single_kg2/ask_query, the lexicon doesn't have that
#     exact phrasing — try the IATA code instead (e.g. "FRA", "MUC", "CDG").
#   - Everything else was traced against your real router.py logic and
#     should route as expected — but "should" isn't "confirmed", which is
#     the whole point of running this first.

NEW_CANDIDATE_TESTS = [

    # ── single_kg1 ────────────────────────────────────────────────────────
    ("For the Austrian Airlines service that operates as flight OS295 "
     "between Vienna and Bucharest, what is the assigned departure gate?",
     "single_kg1", "single_kg1_en_001_gate_os295"),

    ("Flight TK1887 is currently airborne — what is its exact ground "
     "speed reading?",
     "single_kg1", "single_kg1_en_002_gspeed_tk1887"),

    ("I'm trying to find the vertical speed of flight BR62 while it's "
     "climbing out of its departure airport — what is it?",
     "single_kg1", "single_kg1_en_003_vspeed_br62"),

    ("What is the callsign that air traffic control uses to identify "
     "flight LO225?",
     "single_kg1", "single_kg1_en_004_callsign_lo225"),

    # ── single_kg2 ────────────────────────────────────────────────────────
    ("I'm researching runway infrastructure — what is the exact length, "
     "in feet, of the primary runway at Frankfurt Airport?",
     "single_kg2", "single_kg2_en_001_length_fra"),

    ("For urban planning purposes, which municipality is Munich Airport "
     "officially located in?",
     "single_kg2", "single_kg2_en_002_municipality_muc"),

    ("What type of airport classification does Charles de Gaulle Airport "
     "fall under in this dataset?",
     "single_kg2", "single_kg2_en_003_type_cdg"),

    # ── single_kg3 ────────────────────────────────────────────────────────
    ("For FullProfessor0, who is affiliated with the Department of "
     "Computer Science, which specific courses are they currently listed "
     "as teaching?",
     "single_kg3", "single_kg3_en_001_teacherof_fullprof0"),

    ("GraduateStudent5 is pursuing an advanced degree — from which "
     "university did they complete their undergraduate studies before "
     "starting their masters here?",
     "single_kg3", "single_kg3_en_002_undergradfrom_gradstudent5"),

    ("According to the university ontology, is Lecturer3 currently "
     "classified as a tenured faculty member?",
     "single_kg3", "single_kg3_en_003_tenured_lecturer3"),

    ("What is the official, full name of Department2 as recorded in "
     "this knowledge graph?",
     "single_kg3", "single_kg3_en_004_name_dept2"),

    # ── cross_kg ──────────────────────────────────────────────────────────
    ("Flight OS295 is scheduled to land soon — once you determine its "
     "destination airport, what country is that airport officially "
     "located in?",
     "cross_kg", "cross_kg_en_001_country_os295"),

    ("For the airport that flight TK1887 departs from, what is its "
     "elevation above sea level, in feet?",
     "cross_kg", "cross_kg_en_002_elevation_origin_tk1887"),

    ("Considering the airport where flight BR62 will touch down, what "
     "type of airport classification does the dataset assign to it?",
     "cross_kg", "cross_kg_en_003_type_br62"),

    # ── template ──────────────────────────────────────────────────────────
    ("Across all the airports in this dataset, grouped by country, what "
     "is the average runway length recorded?",
     "template", "group_agg_kg2_en_002_avglength_bycountry"),

    # FIXED: uses "list the students" verbatim so _has_filter_signal()
    # actually catches it — original phrasing would have misrouted to
    # single_kg3 via Priority 2.7 (see note above the list).
    ("List the students who are members of Department3.",
     "template", "filter_string_kg3_en_001_members_dept3"),

    ("Out of every airport currently stored in the knowledge graph, "
     "which one holds the record for the single highest elevation above "
     "sea level?",
     "template", "ranking_kg2_en_001_highest_elevation"),

    # ── ask_query ─────────────────────────────────────────────────────────
    ("I want to confirm — is the runway surface at Frankfurt Airport "
     "made of asphalt?",
     "ask_query", "ask_query_en_001_surface_fra"),

    # ── open_kg ───────────────────────────────────────────────────────────
    ("Setting aside any specific flight, how many individual airports in "
     "total are currently stored in this knowledge graph?",
     "open_kg", "open_kg_en_001_count_airports"),
]