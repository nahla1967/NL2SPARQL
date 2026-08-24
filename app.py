"""
NL2SPARQL — Streamlit demo UI
Minimal single-page dashboard: question in, full pipeline trace + answer out.

ASSUMPTION (read this first):
This assumes a single orchestrator function, process_question(question),
that runs your whole pipeline and returns one dict shaped like:

    {
        "language": "ar",                     # detected by langdetect
        "branch": "single_kg1",                # router.py's routing decision
        "entity": "OS295",                     # extracted surface form
        "property_surface": "hasOriginCity",   # property surface form (if any)
        "resolved_uri": "flight_ontology#hasOriginCity",
        "tier": "exact",                       # which cascade tier resolved it
        "sparql": "SELECT ?value WHERE {...}", # generated query
        "execution": {"value": "Vienna", "error": None},  # execute_sparql() output
        "answer": "The departure city of flight OS295 is Vienna.",
    }

For cross_kg questions specifically (flight -> airport bridge -> property),
add one more key so the UI can draw the traversal:

    "path": [
        {"from": "OS295", "to": "VIE", "label": "hasOriginCity (IATA)"},
        {"from": "VIE", "to": "Austria", "label": "locatedInCountry"},
    ]

Omit "path" (or leave it as None) for every other branch — the graph only
renders when it's present, so single_kg1/2/3 etc. are unaffected.

If you don't have one function that returns all of this yet, that's the
first thing to build — a thin wrapper around your existing router/mapper/
generator/executor calls. Keep the UI dumb; keep the logic in the pipeline.

Run with: streamlit run app.py
"""

import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from orchestrator import process_question

st.set_page_config(page_title="NL2SPARQL Assistant", layout="wide")

# --- RTL support for Arabic -------------------------------------------------
# Streamlit renders everything left-to-right by default. When the detected
# language is Arabic, we inject a small CSS override so Arabic text actually
# reads correctly instead of looking broken during a demo.
RTL_CSS = """
<style>
.rtl-text {
    direction: rtl;
    text-align: right;
    font-size: 1.05rem;
}
</style>
"""
st.markdown(RTL_CSS, unsafe_allow_html=True)

# --- Compact styling ----------------------------------------------------
# Smaller fonts and tighter spacing so a full pipeline trace fits in one
# screenshot without scrolling. Also hides Streamlit's default "Deploy"
# button and hamburger menu, which aren't relevant for a local demo.
COMPACT_CSS = """
<style>
#MainMenu {visibility: hidden;}
.stDeployButton {display: none;}
header [data-testid="stToolbar"] {visibility: hidden;}
.block-container {padding-top: 1.5rem; padding-bottom: 1rem;}
h1 {font-size: 1.35rem !important;}
h2 {font-size: 1.05rem !important;}
h3 {font-size: 0.95rem !important;}
p, li, .stMarkdown, .stCaption, .stCode, .stAlert {font-size: 0.85rem !important;}
.answer-box {
    font-size: 1.15rem !important;
    font-weight: 600;
    line-height: 1.5;
    padding: 0.4rem 0;
    margin-bottom: 0.5rem;
}
</style>
"""
st.markdown(COMPACT_CSS, unsafe_allow_html=True)


def render_answer(text: str, lang: str):
    """Same RTL handling as render_text, but styled as the prominent answer box."""
    direction = "rtl" if lang == "ar" else "ltr"
    align = "right" if lang == "ar" else "left"
    st.markdown(
        f'<div class="answer-box" style="direction:{direction}; text-align:{align};">{text}</div>',
        unsafe_allow_html=True,
    )


def render_text(text: str, lang: str):
    """Render text right-to-left if Arabic, left-to-right otherwise."""
    if lang == "ar":
        st.markdown(f'<div class="rtl-text">{text}</div>', unsafe_allow_html=True)
    else:
        st.write(text)


# --- Session state for history ----------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of result dicts, oldest first
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = None  # None = show the newest entry
if "input_key" not in st.session_state:
    st.session_state.input_key = 0  # bumped to force-clear the text_input widget
if "show_result" not in st.session_state:
    st.session_state.show_result = False  # False = nothing shown below the input yet


# --- Header ------------------------------------------------------------------
st.title("NL2SPARQL Assistant")
st.caption("Trilingual (AR / FR / EN) question answering over flight, airport, and university knowledge graphs")

# --- Input ---------------------------------------------------------------
question = st.text_input(
    "Ask a question",
    placeholder="e.g. What is the departure city of flight OS295?",
    key=f"question_input_{st.session_state.input_key}",
)
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    # Arrow icon instead of the "Ask" label — same primary (accent-colored) style.
    submitted = st.button("➜", type="primary", help="Submit question")
with col2:
    # Clears the input AND hides the answer/pipeline panel below — a true
    # blank slate — without deleting anything from st.session_state.history.
    if st.button("New question", type="primary", help="Start fresh (keeps your history)"):
        st.session_state.selected_idx = None
        st.session_state.input_key += 1
        st.session_state.show_result = False
        st.rerun()

if submitted and question.strip():
    result = process_question(question)
    st.session_state.history.append({"question": question, **result})
    st.session_state.selected_idx = None  # always show the freshly-run question
    st.session_state.input_key += 1       # clear the box for the next question
    st.session_state.show_result = True

# --- Show selected result (a history entry if clicked, otherwise the newest) --
if st.session_state.history and st.session_state.show_result:
    idx = st.session_state.selected_idx
    if idx is None or not (0 <= idx < len(st.session_state.history)):
        idx = len(st.session_state.history) - 1
    latest = st.session_state.history[idx]
    lang = latest["language"]

    st.divider()

    # --- Answer first, and made visually prominent, per your request ---
    st.subheader("Answer")
    if latest["execution"].get("error"):
        st.error(f"Execution failed: {latest['execution']['error']}")
    elif latest["answer"]:
        render_answer(latest["answer"], lang)
    else:
        st.caption("No answer available for this entry.")

    # --- Everything else: collapsed by default so the answer stays the focus ---
    with st.expander("Show pipeline trace (routing, mapping, graph, SPARQL)"):
        # Step 1: language
        st.subheader("Detected language")
        st.write(f"`{lang}`")

        # Step 2: routing trace — the most useful transparency element for YOUR
        # pipeline specifically, since your reference UI has no equivalent of this.
        st.subheader("Routing decision")
        st.info(f"Routed to branch: **{latest['branch']}**")

        # Step 3: mapping cascade result.
        # cross_kg resolves TWO things (a bridge airport entity, then a
        # property on it), so it gets its own row instead of overloading
        # the property row's "Resolved URI" with the airport's URI.
        st.subheader("Mapping cascade")
        surface_rows = [latest["entity"], latest.get("property_surface", "—")]
        uri_rows = ["—", latest.get("property_resolved_uri") or latest["resolved_uri"]]
        tier_rows = ["—", latest["tier"]]
        if latest["branch"] == "cross_kg" and latest.get("resolved_uri"):
            surface_rows.insert(1, "→ bridge airport")
            uri_rows.insert(1, latest["resolved_uri"])
            tier_rows.insert(1, "—")
        st.table({"Surface form": surface_rows, "Resolved URI": uri_rows, "Tier": tier_rows})

        # Resolution graph — renders for ANY branch that supplies a "path" (now
        # built for single_kg1/2/3 as a one-hop entity->value edge, and for
        # cross_kg as a two-hop chain). Rename the label below if you'd like
        # something else for screenshots.
        if latest.get("path"):
            st.subheader("Resolution graph")

            # Collect unique node ids from every hop's "from"/"to", in order of
            # first appearance, so the graph shows a clean left-to-right chain.
            seen = []
            for hop in latest["path"]:
                for node_id in (hop["from"], hop["to"]):
                    if node_id not in seen:
                        seen.append(node_id)
            nodes = [Node(id=n, label=n, size=20) for n in seen]

            edges = [
                Edge(source=hop["from"], target=hop["to"], label=hop["label"])
                for hop in latest["path"]
            ]

            config = Config(
                width=1100,
                height=450,
                directed=True,
                physics=False,              # fixed layout is easier to read for a short chain
                hierarchical=True,          # lays nodes out left-to-right / top-to-bottom
                staticGraph=True,           # locks node positions — no drag, no jitter on touch
                staticGraphWithDragAndDrop=False,
            )
            agraph(nodes=nodes, edges=edges, config=config)

        # Step 4: generated SPARQL (debug view)
        st.subheader("Generated SPARQL")
        st.code(latest["sparql"], language="sparql")

# --- History (replaces the reference UI's separate "Mémoire" page) -------
# Sidebar list, newest first. Clicking a question loads its full trace into
# the main panel above; the trash button removes it from history.
with st.sidebar:
    st.subheader("History")
    if st.session_state.history:
        for i in reversed(range(len(st.session_state.history))):
            item = st.session_state.history[i]
            is_active = (st.session_state.selected_idx == i) or (
                st.session_state.selected_idx is None and i == len(st.session_state.history) - 1
            )
            col_q, col_del = st.columns([5, 1])
            with col_q:
                label = ("📍 " if is_active else "") + item["question"]
                if st.button(label, key=f"load_{i}", use_container_width=True):
                    st.session_state.selected_idx = i
                    st.session_state.show_result = True
                    st.rerun()
            with col_del:
                if st.button("🗑", key=f"del_{i}", help="Delete this entry"):
                    st.session_state.history.pop(i)
                    if st.session_state.selected_idx == i:
                        st.session_state.selected_idx = None
                        st.session_state.show_result = bool(st.session_state.history)
                    elif st.session_state.selected_idx is not None and st.session_state.selected_idx > i:
                        st.session_state.selected_idx -= 1
                    st.rerun()
    else:
        st.caption("Past questions will appear here.")