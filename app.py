import streamlit as st
from orchestrator import process_question

st.set_page_config(page_title="NL2SPARQL Assistant", layout="wide")

UI_CSS = """
<style>
#MainMenu, header, footer, .stDeployButton,
[data-testid="stToolbar"], [data-testid="stHeader"],
[data-testid="stDecoration"] {
    visibility: hidden !important;
    display: none !important;
}
[data-testid="InputInstructions"] { display: none !important; }

html, body, [class*="css"] {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
                 "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
.stApp { background: #f3f2f8 !important; }

.main .block-container,
[data-testid="stAppViewBlockContainer"],
[data-testid="stMainBlockContainer"] {
    padding: 0.6rem 0.7rem 0.6rem 0.6rem !important;
    max-width: 100% !important;
    width: 100% !important;
}

/* ── Sidebar ────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #f8f8fb !important;
    border-right: 1px solid #ecebf3;
    height: 100vh;
}
section[data-testid="stSidebar"] > div:first-child {
    background: #f8f8fb !important;
    height: 100%;
}
/* Force flex column so tip can be pushed to bottom */
section[data-testid="stSidebar"] .block-container {
    display: flex !important;
    flex-direction: column !important;
    height: 100% !important;
    box-sizing: border-box;
    padding: 1.5rem 1.1rem 0rem;
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 2rem;
}
.sidebar-brand-icon { width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; }
.sidebar-brand-title {
    font-size: 1.35rem;
    font-weight: 800;
    color: #1a1a2e;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.sidebar-brand-sub { font-size: 1rem; font-weight: 500; color: #8b5cf6; }

/* Nav buttons with SVG icons via CSS ::before */
.st-key-sidebar_nav {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    margin-bottom: 1.4rem;
    flex-shrink: 0;
    align-items: flex-start;
}
.st-key-sidebar_nav button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #6b6b7b !important;
    font-size: 1.05rem !important;
    font-weight: 500 !important;
    padding: 0.65rem 0.9rem 0.65rem 2.6rem !important;
    border-radius: 10px !important;
    width: auto !important;
    min-height: unset !important;
    height: auto !important;
    line-height: 1.2 !important;
    text-align: left !important;
    position: relative !important;
}
.st-key-nav_ask button {
    background: #f3f0ff !important;
    color: #7c3aed !important;
    font-weight: 600 !important;
}
.st-key-nav_ask button::before {
    content: "";
    position: absolute;
    left: 0.85rem;
    top: 50%;
    transform: translateY(-50%);
    width: 16px;
    height: 16px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%237c3aed' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cpath d='M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3'/%3E%3Cpath d='M12 17h.01'/%3E%3C/svg%3E");
    background-size: contain;
    background-repeat: no-repeat;
}
.st-key-nav_history button::before {
    content: "";
    position: absolute;
    left: 0.85rem;
    top: 50%;
    transform: translateY(-50%);
    width: 16px;
    height: 16px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%239ca3af' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cpolyline points='12 6 12 12 16 14'/%3E%3C/svg%3E");
    background-size: contain;
    background-repeat: no-repeat;
}

/* History: flex-grow pushes tip to bottom */
.st-key-sidebar_history {
    flex: 1 1 auto !important;
    overflow-y: auto;
    min-height: 0;
}
.st-key-sidebar_history button {
    text-align: left !important;
    background: transparent !important;
    border: none !important;
    color: #52525b !important;
    font-size: 1rem !important;
    padding: 0.45rem 0.6rem !important;
    border-radius: 8px !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    box-shadow: none !important;
}
.st-key-sidebar_history button:hover { background: #f4f4f5 !important; color: #7c3aed !important; }

/* Tip pinned to bottom */
.st-key-sidebar_tip {
    flex-shrink: 0 !important;
    margin-top: auto !important;
}
.sidebar-tip {
    background: #ffffff;
    border: 1px solid #ecebf3;
    border-radius: 14px;
    padding: 1rem;
}
.sidebar-tip-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1a1a2e;
    margin-bottom: 0.3rem;
    display: flex; align-items: center; gap: 6px;
}
.sidebar-tip-text { font-size: 0.95rem; color: #8a8a9a; line-height: 1.5; }

/* ── Main card ──────────────────────────────────────────────────────── */
.st-key-main_card {
    background: #ffffff;
    border-radius: 24px;
    padding: 7rem 2.2rem 2.3rem;  /* MORE top padding to drop content down */
    box-shadow: 0 4px 24px rgba(0,0,0,0.04);
    min-height: calc(100vh - 1.2rem);
    box-sizing: border-box;
}
.st-key-main_card .main-inner { max-width: 100%; margin: 0; }

.main-title {
    font-size: 1.65rem; font-weight: 800; color: #1a1a2e;
    letter-spacing: -0.025em; margin-bottom: 1.2rem;
    display: flex; align-items: center; gap: 10px;
}
.main-sub {
    font-size: 0.95rem;
    color: #8a8a9a;
    line-height: 1.6;
    margin-bottom: 6rem;  /* MORE space before search bar */
}

/* ── Search bar ─────────────────────────────────────────────────────── */
.st-key-search_wrap {
    position: relative;
    width: 100%;
}
.st-key-search_wrap [data-testid="stTextInput"],
.st-key-search_wrap [data-testid="stTextInput"] > div,
.st-key-search_wrap [data-testid="stTextInput"] > div > div,
.st-key-search_wrap [data-baseweb="input"] {
    width: 100% !important;
    height: 82px !important;
    min-height: 82px !important;
    max-height: 82px !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}
.st-key-search_wrap [data-testid="stTextInput"] input {
    background: #ffffff !important;
    color: #1f2937 !important;
    font-size: 1rem !important;
    height: 82px !important;
    min-height: 82px !important;
    border: 1.5px solid #c4b5fd !important;
    border-radius: 18px !important;
    padding: 0 5rem 0 2.8rem !important;
    outline: none !important;
    box-shadow: none !important;
    width: 100% !important;
    box-sizing: border-box !important;
}
.st-key-search_wrap [data-testid="stTextInput"] input::placeholder {
    color: #a1a1aa !important;
}
.st-key-search_wrap [data-testid="stTextInput"] input:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 3px rgba(167, 139, 250, 0.12) !important;
}
/* Loupe */
.st-key-search_wrap [data-testid="stTextInput"] > div {
    position: relative;
}
.st-key-search_wrap [data-testid="stTextInput"] > div::before {
    content: "";
    position: absolute; left: 20px; top: 50%; transform: translateY(-50%);
    width: 20px; height: 20px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='%23a1a1aa' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cpath d='m21 21-4.3-4.3'/%3E%3C/svg%3E");
    background-size: contain; background-repeat: no-repeat; z-index: 3; pointer-events: none;
}

/* Arrow button: centered vertically on the right */
.st-key-search_btn {
    position: absolute !important;
    right: 12px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    z-index: 5;
    width: 48px !important;
    height: 48px !important;
    pointer-events: none !important;
}
.st-key-search_btn button {
    width: 48px !important;
    height: 48px !important;
    min-height: 48px !important;
    border-radius: 14px !important;
    background: #ede9fe !important;
    border: 1px solid #ddd6fe !important;
    color: #7c3aed !important;
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    pointer-events: auto !important;
    cursor: pointer !important;
}
.st-key-search_btn button:hover { background: #ddd6fe !important; }
.st-key-search_btn button:active { background: #c4b5fd !important; }

/* ── Examples (violet outline, wider, tight gap, dropped down) ─────── */
.st-key-examples_row {
    margin-top: 4.5rem;  /* DROPPED down */
}
.st-key-examples_row [data-testid="stHorizontalBlock"] {
    gap: 0px !important;
    width: 100%;
}
.st-key-examples_row [data-testid="stColumn"] {
    padding: 0 2px !important;
    gap: 0px !important;
}
.st-key-examples_row button {
    background: #ffffff !important;
    border: 1.5px solid #c4b5fd !important;
    border-radius: 14px !important;
    color: #52525b !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    padding: 0 1.5rem 0 2.6rem !important;
    height: 54px !important;
    min-height: 54px !important;
    max-height: 54px !important;
    width: 100% !important;
    text-align: left !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    position: relative !important;
    box-shadow: none !important;
    display: flex !important;
    align-items: center !important;
    transition: all 0.15s ease !important;
}
.st-key-examples_row button:hover {
    border-color: #a78bfa !important;
    background: #faf9ff !important;
    color: #7c3aed !important;
}
.st-key-ex_flight button::before,
.st-key-ex_building button::before,
.st-key-ex_grad button::before,
.st-key-ex_fourth button::before {
    content: "";
    position: absolute;
    left: 0.9rem;
    top: 50%;
    transform: translateY(-50%);
    width: 16px;
    height: 16px;
    background-size: contain;
    background-repeat: no-repeat;
}
.st-key-ex_flight button::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%2352525b' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M2 12h20'/%3E%3Cpath d='M13 2l9 10-9 10'/%3E%3C/svg%3E");
}
.st-key-ex_building button::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%2352525b' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='4' y='2' width='16' height='20' rx='2'/%3E%3Cpath d='M9 22v-4h6v4'/%3E%3Cpath d='M8 6h.01'/%3E%3Cpath d='M16 6h.01'/%3E%3Cpath d='M12 6h.01'/%3E%3Cpath d='M12 10h.01'/%3E%3Cpath d='M12 14h.01'/%3E%3Cpath d='M16 10h.01'/%3E%3Cpath d='M16 14h.01'/%3E%3Cpath d='M8 10h.01'/%3E%3Cpath d='M8 14h.01'/%3E%3C/svg%3E");
}
.st-key-ex_grad button::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%2352525b' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M22 10v6M2 10l10-5 10 5-10 5z'/%3E%3Cpath d='M6 12v5c0 2 2 3 6 3s6-1 6-3v-5'/%3E%3C/svg%3E");
}
.st-key-ex_fourth button::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%2352525b' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 2v20M2 12h20'/%3E%3C/svg%3E");
}
.st-key-ex_flight button:hover::before,
.st-key-ex_building button:hover::before,
.st-key-ex_grad button:hover::before,
.st-key-ex_fourth button:hover::before {
    filter: brightness(0) saturate(100%) invert(27%) sepia(82%) saturate(2162%) hue-rotate(248deg) brightness(95%) contrast(95%);
}

/* ── Answer ───────────────────────────────────────────────────────── */
.answer-box {
    background: #f5f3ff; border: 1px solid #e9e5ff; border-radius: 14px;
    padding: 1rem 1.2rem; font-size: 0.95rem !important; font-weight: 600;
    line-height: 1.6; color: #2e1065;
}

/* ── Resolution flow ─────────────────────────────────────────────────── */
.flow-wrap { display: flex; align-items: center; flex-wrap: wrap; gap: 0.6rem; padding: 0.8rem 0 1.2rem; }
.flow-node {
    background: #f5f3ff; border: 1px solid #ddd6fe; border-radius: 10px;
    padding: 0.5rem 0.9rem; font-size: 0.85rem; font-weight: 600; color: #4c1d95;
}
.flow-edge { font-size: 0.78rem; color: #7c3aed; font-weight: 600; white-space: nowrap; }

/* ── Expander / Pipeline ────────────────────────────────────────────── */
.pipeline-expander [data-testid="stExpander"] {
    background: #ffffff !important; border: 1px solid #e5e7eb !important; border-radius: 14px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important; overflow: hidden;
}
.pipeline-expander [data-testid="stExpander"] summary {
    color: #4b5563 !important; font-weight: 600 !important; font-size: 0.85rem !important; padding: 0.7rem 1rem !important;
}
.pipeline-expander [data-testid="stExpander"] summary:hover { color: #7c3aed !important; }
.pipeline-expander [data-testid="stExpander"] > div[data-testid="stVerticalBlock"] { padding: 0 1rem 1rem !important; }
.pipeline-expander table { background: #fafafa; border-radius: 10px; overflow: hidden; font-size: 0.82rem; }
.pipeline-expander th { background: #f3f4f6; color: #374151; font-weight: 600; padding: 0.5rem 0.7rem; text-align: left; }
.pipeline-expander td { color: #4b5563; padding: 0.5rem 0.7rem; border-top: 1px solid #e5e7eb; }
.pipeline-expander pre { background: #f8f9fa !important; border: 1px solid #e5e7eb !important; border-radius: 10px !important; font-size: 0.8rem !important; }

.rtl-text { direction: rtl; text-align: right; font-size: 1rem; }
</style>
"""
st.markdown(UI_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SVG icons (for HTML markdown only)
# ─────────────────────────────────────────────────────────────────────────────
ICON_SPARKLE = """<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="#8b5cf6" stroke="#7c3aed" stroke-width="1.5"><path d="M12 2l1.5 4.5L18 8l-4.5 1.5L12 14l-1.5-4.5L6 8l4.5-1.5z"/><path d="M5 16l1 3 3 1-3 1-1 3-1-3-3-1 3-1z"/><path d="M19 16l1 3 3 1-3 1-1 3-1-3-3-1 3-1z"/></svg>"""
ICON_TIP = """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="#8b5cf6" stroke="#7c3aed" stroke-width="1.5"><path d="M12 2l1.5 4.5L18 8l-4.5 1.5L12 14l-1.5-4.5L6 8l4.5-1.5z"/><path d="M5 16l1 3 3 1-3 1-1 3-1-3-3-1 3-1z"/><path d="M19 16l1 3 3 1-3 1-1 3-1-3-3-1 3-1z"/></svg>"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def render_answer(text: str, lang: str):
    direction = "rtl" if lang == "ar" else "ltr"
    align = "right" if lang == "ar" else "left"
    st.markdown(
        f'<div class="answer-box" style="direction:{direction};text-align:{align};">{text}</div>',
        unsafe_allow_html=True,
    )


def render_text(text: str, lang: str):
    if lang == "ar":
        st.markdown(f'<div class="rtl-text">{text}</div>', unsafe_allow_html=True)
    else:
        st.write(text)


def render_resolution_flow(path):
    if not path:
        return
    parts = []
    for i, hop in enumerate(path):
        if i == 0:
            parts.append(f'<div class="flow-node">{hop["from"]}</div>')
        parts.append(f'<div class="flow-edge">→ {hop["label"]} →</div>')
        parts.append(f'<div class="flow-node">{hop["to"]}</div>')
    st.markdown(f'<div class="flow-wrap">{"".join(parts)}</div>', unsafe_allow_html=True)


def run_question(question_text: str):
    try:
        result = process_question(question_text)
        st.session_state.history.append({"question": question_text, **result})
        st.session_state.selected_idx = None
        st.session_state.show_result = True
    except Exception as e:
        st.session_state.pending_error = str(e)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = None
if "show_result" not in st.session_state:
    st.session_state.show_result = False
if "pending_error" not in st.session_state:
    st.session_state.pending_error = None


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">{ICON_SPARKLE}</div>
        <div>
            <div class="sidebar-brand-title">NL2SPARQL</div>
            <div class="sidebar-brand-sub">Assistant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container(key="sidebar_nav"):
        if st.button("Ask a question", key="nav_ask", use_container_width=False):
            st.session_state.selected_idx = None
            st.session_state.show_result = False
            st.rerun()
        if st.button("History", key="nav_history", use_container_width=False):
            pass

    with st.container(key="sidebar_history"):
        if st.session_state.history:
            for i in reversed(range(len(st.session_state.history))):
                item = st.session_state.history[i]
                is_active = (st.session_state.selected_idx == i) or (
                    st.session_state.selected_idx is None and i == len(st.session_state.history) - 1
                )
                label = ("● " if is_active else "") + item["question"]
                if st.button(label, key=f"load_{i}", use_container_width=True):
                    st.session_state.selected_idx = i
                    st.session_state.show_result = True
                    st.rerun()

    with st.container(key="sidebar_tip"):
        st.markdown(f"""
        <div class="sidebar-tip">
            <div class="sidebar-tip-title">{ICON_TIP} &nbsp; Tip</div>
            <div class="sidebar-tip-text">
                You can ask about flights, airports, airlines, universities and more.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
with st.container(key="main_card"):
    st.markdown('<div class="main-inner">', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="main-title">{ICON_SPARKLE} Hello! I'm your NL2SPARQL Assistant</div>
    <div class="main-sub">
        Ask me anything about flights, airports, airlines or universities.<br>
        I'll translate your question and search the knowledge graphs.
    </div>
    """, unsafe_allow_html=True)

    with st.container(key="search_wrap"):
        question = st.text_input(
            "", placeholder="Type your question in natural language...",
            label_visibility="collapsed",
        )
        with st.container(key="search_btn"):
            submitted = st.button("→", key="submit_search")

    with st.container(key="examples_row"):
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            with st.container(key="ex_flight"):
                if st.button("Departure city of OS295", key="ex_flight_btn"):
                    run_question("What is the departure city of flight OS295?")
        with e2:
            with st.container(key="ex_building"):
                if st.button("Flights to Paris", key="ex_building_btn"):
                    run_question("Find all flights to Paris")
        with e3:
            with st.container(key="ex_grad"):
                if st.button("Universities in Berlin", key="ex_grad_btn"):
                    run_question("Which universities are in Berlin?")
        with e4:
            with st.container(key="ex_fourth"):
                if st.button("Arrival city of LH400", key="ex_fourth_btn"):
                    run_question("What is the arrival city of flight LH400?")

    if submitted and question.strip():
        run_question(question)

    if st.session_state.pending_error:
        st.error(f"Something went wrong while processing your question: {st.session_state.pending_error}")
        st.session_state.pending_error = None

    if st.session_state.history and st.session_state.show_result:
        idx = st.session_state.selected_idx
        if idx is None or not (0 <= idx < len(st.session_state.history)):
            idx = len(st.session_state.history) - 1
        latest = st.session_state.history[idx]
        lang = latest["language"]

        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

        st.subheader("Answer")
        if latest["execution"].get("error"):
            st.error(f"Execution failed: {latest['execution']['error']}")
        elif latest["answer"]:
            render_answer(latest["answer"], lang)
        else:
            st.caption("No answer available for this entry.")

        with st.expander("✦  Pipeline details · routing, mapping, graph & SPARQL", expanded=False):
            st.subheader("Detected language")
            st.write(f"`{lang}`")

            st.subheader("Routing decision")
            st.info(f"Routed to branch: **{latest['branch']}**")

            st.subheader("Mapping cascade")
            surface_rows = [latest["entity"], latest.get("property_surface", "—")]
            uri_rows = ["—", latest.get("property_resolved_uri") or latest["resolved_uri"]]
            tier_rows = ["—", latest["tier"]]
            if latest["branch"] == "cross_kg" and latest.get("resolved_uri"):
                surface_rows.insert(1, "→ bridge airport")
                uri_rows.insert(1, latest["resolved_uri"])
                tier_rows.insert(1, "—")
            st.table({"Surface form": surface_rows, "Resolved URI": uri_rows, "Tier": tier_rows})

            if latest.get("path"):
                st.subheader("Resolution graph")
                render_resolution_flow(latest["path"])

            st.subheader("Generated SPARQL")
            st.code(latest["sparql"], language="sparql")

    st.markdown('</div>', unsafe_allow_html=True)