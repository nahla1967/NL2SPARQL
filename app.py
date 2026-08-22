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


def render_text(text: str, lang: str):
    """Render text right-to-left if Arabic, left-to-right otherwise."""
    if lang == "ar":
        st.markdown(f'<div class="rtl-text">{text}</div>', unsafe_allow_html=True)
    else:
        st.write(text)


# --- Session state for history ----------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of result dicts, most recent last


# --- Header ------------------------------------------------------------------
st.title("NL2SPARQL Assistant")
st.caption("Trilingual (AR / FR / EN) question answering over flight, airport, and university knowledge graphs")

# --- Input ---------------------------------------------------------------
question = st.text_input("Ask a question", placeholder="e.g. What is the departure city of flight OS295?")
col1, col2 = st.columns([1, 5])
with col1:
    submitted = st.button("Ask", type="primary")

if submitted and question.strip():
    result = process_question(question)
    st.session_state.history.append({"question": question, **result})

# --- Show most recent result --------------------------------------------
if st.session_state.history:
    latest = st.session_state.history[-1]
    lang = latest["language"]

    st.divider()

    # Step 1: language
    st.subheader("Detected language")
    st.write(f"`{lang}`")

    # Step 2: routing trace — the most useful transparency element for YOUR
    # pipeline specifically, since your reference UI has no equivalent of this.
    st.subheader("Routing decision")
    st.info(f"Routed to branch: **{latest['branch']}**")

    # Step 3: mapping cascade result
    st.subheader("Mapping cascade")
    st.table(
        {
            "Surface form": [latest["entity"], latest.get("property_surface", "—")],
            "Resolved URI": ["—", latest["resolved_uri"]],
            "Tier": ["—", latest["tier"]],
        }
    )

    # Cross-KG path graph — only rendered when the pipeline supplies a "path"
    # list (i.e. only for cross_kg questions). Everything else in this app
    # is untouched by this block.
    if latest.get("path"):
        st.subheader("Cross-KG traversal")

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
            width=700,
            height=250,
            directed=True,
            physics=False,       # fixed layout is easier to read for a short chain
            hierarchical=True,   # lays nodes out left-to-right / top-to-bottom
        )
        agraph(nodes=nodes, edges=edges, config=config)

    # Step 4: generated SPARQL (debug view)
    st.subheader("Generated SPARQL")
    st.code(latest["sparql"], language="sparql")

    # Step 5: execution + final answer
    st.subheader("Answer")
    if latest["execution"].get("error"):
        st.error(f"Execution failed: {latest['execution']['error']}")
    else:
        render_text(latest["answer"], lang)

# --- History (replaces the reference UI's separate "Mémoire" page) -------
if len(st.session_state.history) > 1:
    st.divider()
    st.subheader("History")
    for item in reversed(st.session_state.history[:-1]):
        with st.expander(item["question"]):
            render_text(item["answer"], item["language"])
            st.code(item["sparql"], language="sparql")