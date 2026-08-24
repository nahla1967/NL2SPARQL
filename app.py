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
/* ---------- Hide Streamlit chrome ---------- */
#MainMenu {visibility: hidden;}
.stDeployButton {display: none;}
header [data-testid="stToolbar"] {visibility: hidden;}
footer {visibility: hidden;}

/* ---------- Page ---------- */
.stApp {
    background:
        radial-gradient(circle at 8% 10%, rgba(139, 92, 246, 0.08), transparent 25%),
        #faf9ff;
    color: #182033;
}

.block-container {
    max-width: 1450px;
    padding: 1.8rem 2.2rem 1.2rem;
}

/* ---------- Typography ---------- */
html, body, [class*="css"] {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
                 "Segoe UI", sans-serif;
}

h1 {
    font-size: 2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.035em;
    color: #151a2d;
}

h2, h3 {
    color: #1c2238;
}

p, li, .stMarkdown, .stCaption, .stCode, .stAlert {
    font-size: 0.9rem !important;
    color: #697187;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f7f3ff 0%, #fbfaff 100%);
    border-right: 1px solid #eeeafd;
}

section[data-testid="stSidebar"] .block-container {
    padding: 2rem 1.4rem;
}

section[data-testid="stSidebar"] h2 {
    font-size: 1.05rem !important;
    font-weight: 700 !important;
}

/* History buttons */
section[data-testid="stSidebar"] .stButton > button {
    border: 0 !important;
    background: transparent !important;
    color: #60687d !important;
    text-align: left !important;
    justify-content: flex-start !important;
    border-radius: 12px !important;
    min-height: 2.55rem !important;
    box-shadow: none !important;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: #eee8ff !important;
    color: #7048d8 !important;
}

/* ---------- Main white card ---------- */
.main-card {
    background: rgba(255,255,255,0.96);
    border: 1px solid #f0edf8;
    border-radius: 24px;
    padding: 3.2rem 3.8rem 2.2rem;
    box-shadow: 0 18px 55px rgba(57, 42, 104, 0.08);
    min-height: 720px;
}

/* ---------- Brand / welcome ---------- */
.brand {
    display: flex;
    align-items: center;
    gap: 13px;
    margin-bottom: 2.8rem;
}

.brand-icon {
    width: 43px;
    height: 43px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #f0e9ff;
    color: #8055e8;
    font-size: 22px;
    box-shadow: 0 6px 18px rgba(128, 85, 232, 0.14);
}

.brand-name {
    font-size: 1.22rem;
    line-height: 1.05;
    font-weight: 800;
    color: #17203a;
}

.brand-sub {
    color: #8055e8;
    font-size: 0.92rem;
    font-weight: 600;
    margin-top: 4px;
}

.welcome-title {
    font-size: 1.72rem;
    font-weight: 800;
    color: #182033;
    letter-spacing: -0.025em;
    margin-bottom: 0.55rem;
}

.welcome-text {
    color: #727b90;
    line-height: 1.75;
    margin-bottom: 2.25rem;
}

/* ---------- Search ---------- */
.search-label {
    font-size: 0.88rem;
    font-weight: 700;
    color: #252c42;
    margin-bottom: 0.45rem;
}

div[data-testid="stTextInput"] > div > div {
    border: 1.5px solid #9d70f3 !important;
    border-radius: 15px !important;
    background: white !important;
    box-shadow: 0 0 0 4px rgba(157, 112, 243, 0.07);
}

div[data-testid="stTextInput"] input {
    font-size: 0.98rem !important;
    color: #293148 !important;
    padding: 0.9rem 1rem !important;
}

div[data-testid="stTextInput"] input::placeholder {
    color: #9aa1b1 !important;
}

/* Submit button */
button[kind="primary"] {
    background: #8055e8 !important;
    border: 0 !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 700 !important;
    box-shadow: 0 8px 18px rgba(128, 85, 232, 0.18);
}

button[kind="primary"]:hover {
    background: #7046d5 !important;
    transform: translateY(-1px);
}

/* ---------- Example chips ---------- */
.examples-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: #4e566b;
    margin: 1.65rem 0 0.65rem;
}

.example-chip {
    border: 1px solid #e3dfec;
    border-radius: 11px;
    padding: 0.75rem 0.95rem;
    background: #fff;
    color: #596176;
    font-size: 0.86rem;
    text-align: center;
    min-height: 44px;
    display: flex;
    align-items: center;
    justify-content: center;
}

/* ---------- Pipeline collapsed panel ---------- */
.pipeline-wrap {
    margin-top: 1.35rem;
}

div[data-testid="stExpander"] {
    border: 1px solid #ece8f5 !important;
    border-radius: 14px !important;
    background: #fbfaff !important;
    box-shadow: none !important;
}

div[data-testid="stExpander"] summary {
    color: #454d62 !important;
    font-weight: 700 !important;
}

.answer-box {
    background: #f5f0ff;
    border: 1px solid #e5dafd;
    border-radius: 14px;
    padding: 1rem 1.1rem;
    font-size: 1.02rem !important;
    font-weight: 600;
    line-height: 1.55;
    color: #2b2440;
}

/* ---------- Footer ---------- */
.app-footer {
    text-align: center;
    color: #9299aa;
    font-size: 0.78rem;
    margin-top: 5rem;
}

/* RTL */
.rtl-text {
    direction: rtl;
    text-align: right;
    font-size: 1.05rem;
}
</style>
"""


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


# --- Main dashboard ----------------------------------------------------------
st.markdown("""
<div class="main-card">
    <div class="brand">
        <div class="brand-icon">✦</div>
        <div>
            <div class="brand-name">NL2SPARQL</div>
            <div class="brand-sub">Assistant</div>
        </div>
    </div>

    <div class="welcome-title">✦ Hello! I'm your NL2SPARQL Assistant</div>
    <div class="welcome-text">
        Ask me anything about flights, airports, airlines or universities.<br>
        I'll translate your question and search the knowledge graphs.
    </div>
</div>
""", unsafe_allow_html=True)

# Put the real Streamlit input visually inside the card area.
st.markdown('<div style="margin:-430px 3.8rem 0;">', unsafe_allow_html=True)
st.markdown('<div class="search-label">Ask a question</div>', unsafe_allow_html=True)

question = st.text_input(
    "",
    placeholder="Type your question in natural language...",
    key=f"question_input_{st.session_state.input_key}",
    label_visibility="collapsed",
)

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    submitted = st.button("➜", type="primary", help="Submit question", use_container_width=True)
with col2:
    if st.button("New question", type="primary", help="Start fresh (keeps your history)"):
        st.session_state.selected_idx = None
        st.session_state.input_key += 1
        st.session_state.show_result = False
        st.rerun()

st.markdown('<div class="examples-title">Try an example</div>', unsafe_allow_html=True)

e1, e2, e3 = st.columns(3)
with e1:
    st.markdown('<div class="example-chip">✈️ &nbsp; What is the departure city of flight OS295?</div>', unsafe_allow_html=True)
with e2:
    st.markdown('<div class="example-chip">▦ &nbsp; Find all flights to Paris</div>', unsafe_allow_html=True)
with e3:
    st.markdown('<div class="example-chip">🎓 &nbsp; Which universities are in Berlin?</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

if submitted and question.strip():
    result = process_question(question)
    st.session_state.history.append({"question": question, **result})
    st.session_state.selected_idx = None
    st.session_state.input_key += 1
    st.session_state.show_result = True

# --- Show selected result (a history entry if clicked, otherwise the newest) --
if st.session_state.history and st.session_state.show_result:
    idx = st.session_state.selected_idx
    if idx is None or not (0 <= idx < len(st.session_state.history)):
        idx = len(st.session_state.history) - 1
    latest = st.session_state.history[idx]
    lang = latest["language"]

    # --- Answer first, and made visually prominent, per your request ---
    st.subheader("Answer")
    if latest["execution"].get("error"):
        st.error(f"Execution failed: {latest['execution']['error']}")
    elif latest["answer"]:
        render_answer(latest["answer"], lang)
    else:
        st.caption("No answer available for this entry.")

    # --- Everything else: collapsed by default so the answer stays the focus ---
    with st.expander("✦  Pipeline details · routing, mapping, graph & SPARQL", expanded=False):
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


st.markdown(
    '<div class="app-footer">© 2024 NL2SPARQL Assistant · Powered by Knowledge Graphs</div>',
    unsafe_allow_html=True,
)

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