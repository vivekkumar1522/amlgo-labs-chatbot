import streamlit as st
from src.pipeline import RAGPipeline
from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Amlgo Labs · AI Assistant",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500&display=swap');

:root {
    --bg:         #08080a;
    --surface:    #101014;
    --surface2:   #16161c;
    --border:     #1e1e26;
    --border2:    #2a2a36;
    --accent:     #b8ff57;
    --accent-dim: rgba(184,255,87,0.10);
    --text:       #f0f0f5;
    --text-dim:   #55556a;
    --text-mid:   #9898b0;
    --r-lg:       18px;
    --r-sm:       8px;
    --user-color: #08080a;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    height: 100% !important;
}

#MainMenu, footer, header, .stDeployButton,
[data-testid="collapsedControl"] {
    visibility: hidden !important;
    display: none !important;
}

/* Remove ALL Streamlit default padding */
.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
    height: 100vh !important;
}

/* ══════════════════════════════════════════
   FIXED HEADER
══════════════════════════════════════════ */
.chat-header {
    position: fixed;
    top: 0;
    left: 50%; transform: translateX(-50%);
    width: 100%; max-width: 760px;
    z-index: 100;
    background: rgba(8,8,10,0.93);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border-bottom: 1px solid var(--border);
    padding: 13px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.brand-row {
    display: flex; align-items: center; gap: 10px;
}
.brand-hex {
    width: 32px; height: 32px;
    background: var(--accent);
    clip-path: polygon(50% 0%,93% 25%,93% 75%,50% 100%,7% 75%,7% 25%);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Syne', sans-serif;
    font-size: 0.62rem; font-weight: 800;
    color: #08080a; flex-shrink: 0;
}
.brand-name {
    font-family: 'Syne', sans-serif;
    font-size: 1rem; font-weight: 700;
    color: var(--text); letter-spacing: -0.3px;
}
.brand-name span { color: var(--accent); }
.status-pill {
    font-family: 'Space Mono', monospace;
    font-size: 0.5rem; letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--accent);
    background: var(--accent-dim);
    border: 1px solid rgba(184,255,87,0.22);
    padding: 4px 10px; border-radius: 100px;
    display: flex; align-items: center; gap: 5px;
}
.status-dot {
    width: 5px; height: 5px;
    background: var(--accent); border-radius: 50%;
    animation: pulse 2s ease infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.4; transform:scale(0.7); }
}

/* ══════════════════════════════════════════
   CHAT BODY — fixed between header & input
══════════════════════════════════════════ */
.chat-body {
    position: fixed;
    top: 60px;
    bottom: 108px;
    left: 50%; transform: translateX(-50%);
    width: 100%; max-width: 760px;
    overflow-y: auto;
    padding: 20px 20px 8px 20px;
    display: flex;
    flex-direction: column;
    gap: 14px;
    scrollbar-width: thin;
    scrollbar-color: var(--border2) transparent;
}
.chat-body::-webkit-scrollbar { width: 3px; }
.chat-body::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

/* ══════════════════════════════════════════
   MESSAGE ROWS + AVATARS
══════════════════════════════════════════ */
.msg-row {
    display: flex;
    align-items: flex-end;
    gap: 8px;
    animation: popIn 0.22s cubic-bezier(0.34,1.4,0.64,1);
}
@keyframes popIn {
    from { opacity:0; transform:translateY(10px) scale(0.97); }
    to   { opacity:1; transform:translateY(0) scale(1); }
}
.msg-row.user { flex-direction: row-reverse; }
.msg-row.ai   { flex-direction: row; }

/* Avatars */
.av {
    width: 28px; height: 28px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.6rem; font-weight: 700;
    flex-shrink: 0; margin-bottom: 2px;
}
.av-user {
    border-radius: 50%;
    background: #1e1e2a;
    border: 1px solid var(--border2);
    color: var(--text-mid);
    font-family: 'Syne', sans-serif;
}
.av-ai {
    background: var(--accent);
    clip-path: polygon(50% 0%,93% 25%,93% 75%,50% 100%,7% 75%,7% 25%);
    color: #08080a;
    font-family: 'Syne', sans-serif;
    font-size: 0.52rem; font-weight: 800;
}

/* Bubbles */
.bubble {
    max-width: 78%;
    padding: 11px 15px;
    font-size: 0.875rem; line-height: 1.65;
    word-break: break-word;
}
.bubble.user {
    background: var(--accent);
    color: var(--user-color);
    border-radius: var(--r-lg) var(--r-lg) 4px var(--r-lg);
    font-weight: 450;
}
.bubble.ai {
    background: var(--surface2);
    border: 1px solid var(--border2);
    color: var(--text);
    border-radius: var(--r-lg) var(--r-lg) var(--r-lg) 4px;
}

/* Blinking cursor for streaming */
.cursor {
    display: inline-block;
    width: 2px; height: 13px;
    background: var(--accent);
    margin-left: 3px; border-radius: 1px;
    vertical-align: middle;
    animation: blink 0.8s steps(1) infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }

/* Source pills */
.src-row {
    display: flex; flex-wrap: wrap; gap: 5px;
    margin-top: 9px; padding-top: 8px;
    border-top: 1px solid var(--border2);
}
.src-pill {
    font-family: 'Space Mono', monospace;
    font-size: 0.55rem; letter-spacing: 0.5px;
    color: var(--accent);
    background: var(--accent-dim);
    border: 1px solid rgba(184,255,87,0.2);
    border-radius: 100px; padding: 2px 9px;
}
.src-card {
    background: var(--surface);
    border: 1px solid var(--border2);
    border-left: 2px solid var(--accent);
    border-radius: var(--r-sm);
    padding: 9px 12px; margin-top: 5px;
    font-size: 0.75rem; color: var(--text-mid);
    font-family: 'Space Mono', monospace;
    line-height: 1.6;
}

/* ── Welcome screen ── */
.welcome {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    text-align: center;
    height: 100%; gap: 10px; padding: 20px;
}
.welcome-hex {
    width: 64px; height: 64px;
    background: var(--accent-dim);
    border: 1.5px solid rgba(184,255,87,0.2);
    clip-path: polygon(50% 0%,93% 25%,93% 75%,50% 100%,7% 75%,7% 25%);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem; font-weight: 800;
    color: var(--accent); margin-bottom: 6px;
}
.welcome h2 {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem; font-weight: 700;
    color: var(--text); letter-spacing: -0.4px;
}
.welcome p {
    font-size: 0.82rem;
    color: var(--text-dim);
    line-height: 1.6; max-width: 300px;
}
.chip-row {
    display: flex; flex-wrap: wrap; gap: 7px;
    justify-content: center; margin-top: 14px;
}
.chip {
    font-size: 0.75rem; color: var(--text-mid);
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-radius: 100px; padding: 6px 13px;
}

/* ══════════════════════════════════════════
   FIXED INPUT BAR
══════════════════════════════════════════ */
.input-bar {
    position: fixed;
    bottom: 0;
    left: 50%; transform: translateX(-50%);
    width: 100%; max-width: 760px;
    background: rgba(8,8,10,0.96);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border-top: 1px solid var(--border);
    padding: 12px 20px 18px 20px;
    z-index: 100;
}

/* Input field */
.stTextInput label { display: none !important; }
.stTextInput > div { margin-bottom: 0 !important; }
.stTextInput > div > div > input {
    background: var(--surface2) !important;
    border: 1.5px solid var(--border2) !important;
    border-radius: 100px !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    padding: 0 20px !important;
    caret-color: var(--accent) !important;
    height: 46px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(184,255,87,0.07) !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder { color: var(--text-dim) !important; }

/* Send button — circular icon button */
.stButton > button {
    background: var(--accent) !important;
    color: #08080a !important;
    border: none !important;
    border-radius: 50% !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    height: 46px !important;
    width: 46px !important;
    min-width: 46px !important;
    padding: 0 !important;
    line-height: 1 !important;
    transition: all 0.15s !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.stButton > button:hover {
    background: #cbff6e !important;
    transform: scale(1.1) !important;
    box-shadow: 0 0 20px rgba(184,255,87,0.4) !important;
}
.stButton > button:active { transform: scale(0.93) !important; }

/* Align columns vertically */
[data-testid="column"] {
    display: flex !important;
    align-items: center !important;
    padding: 0 !important;
}
[data-testid="column"]:first-child { padding-right: 8px !important; }

/* Source expand buttons styling */
.stButton.src-expand-btn > button {
    background: transparent !important;
    border: 1px solid var(--border2) !important;
    border-radius: 6px !important;
    color: var(--text-mid) !important;
    font-size: 0.7rem !important;
    height: 26px !important;
    width: auto !important;
    min-width: unset !important;
    padding: 0 10px !important;
    border-radius: 100px !important;
}

/* Spinner */
.stSpinner > div { border-top-color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline" not in st.session_state:
    with st.spinner("Connecting to knowledge base…"):
        st.session_state.pipeline = RAGPipeline()
if "show_sources" not in st.session_state:
    st.session_state.show_sources = {}
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
# pending_query is set by on_change (Enter key) and read below
if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""

pipeline: RAGPipeline = st.session_state.pipeline
stats = pipeline.get_stats()


# ─────────────────────────────────────────────
#  FIXED HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="chat-header">
  <div class="brand-row">
    <div class="brand-hex">A</div>
    <span class="brand-name">Amlgo <span>Labs</span></span>
  </div>
  <div class="status-pill">
    <span class="status-dot"></span>
    {stats.get('chunks', '–')} chunks · online
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CHAT BODY
# ─────────────────────────────────────────────
st.markdown('<div class="chat-body">', unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
    <div class="welcome">
        <div class="welcome-hex">A</div>
        <h2>How can I help you?</h2>
        <p>Ask me anything about the indexed document. I'll surface the most relevant passages.</p>
        <div class="chip-row">
            <span class="chip">📄 Summarize</span>
            <span class="chip">🔍 Key findings</span>
            <span class="chip">❓ Deep dive</span>
            <span class="chip">📊 Extract data</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-row user">
                <div class="bubble user">{msg['content']}</div>
                <div class="av av-user">You</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            sources = msg.get("sources", [])
            src_pills = ""
            if sources:
                src_pills = (
                    '<div class="src-row">'
                    + "".join(f'<span class="src-pill">⬡ src {j+1}</span>' for j in range(len(sources)))
                    + "</div>"
                )
            st.markdown(f"""
            <div class="msg-row ai">
                <div class="av av-ai">A</div>
                <div class="bubble ai">{msg['content']}{src_pills}</div>
            </div>
            """, unsafe_allow_html=True)

            # Expandable source cards
            if sources:
                for j, src in enumerate(sources):
                    tkey = f"open_{i}_{j}"
                    label = f"▾ Hide source {j+1}" if st.session_state.show_sources.get(tkey) else f"▸ Source {j+1}"
                    if st.button(label, key=f"srcbtn_{i}_{j}"):
                        st.session_state.show_sources[tkey] = not st.session_state.show_sources.get(tkey, False)
                    if st.session_state.show_sources.get(tkey, False):
                        st.markdown(f'<div class="src-card">{src}</div>', unsafe_allow_html=True)

# Streaming live-text placeholder sits at bottom of chat-body
stream_slot = st.empty()

st.markdown('</div>', unsafe_allow_html=True)  # end .chat-body


# ─────────────────────────────────────────────
#  FIXED INPUT BAR
#  on_change handles Enter key → sets pending_query
# ─────────────────────────────────────────────
def _capture_enter():
    st.session_state.pending_query = st.session_state.user_input_field

st.markdown('<div class="input-bar">', unsafe_allow_html=True)

col_in, col_btn = st.columns([11, 1])
with col_in:
    user_input = st.text_input(
        label="msg",
        placeholder="Ask anything about the document…",
        key="user_input_field",
        label_visibility="collapsed",
        on_change=_capture_enter,
    )
with col_btn:
    send_btn = st.button("↑", use_container_width=False)

st.markdown('</div>', unsafe_allow_html=True)  # end .input-bar


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:Syne,sans-serif;font-size:0.95rem;font-weight:700;
                padding:10px 0 14px;border-bottom:1px solid #1e1e26;margin-bottom:12px;'>
        ⬡ Amlgo Labs
    </div>
    """, unsafe_allow_html=True)
    if st.button("🗑  Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.show_sources = {}
        st.session_state.last_query = ""
        st.session_state.pending_query = ""
        st.rerun()


# ─────────────────────────────────────────────
#  QUERY HANDLER
#  Triggered by Enter (pending_query) OR Send button
# ─────────────────────────────────────────────
# Determine the active query
raw = ""
if send_btn and user_input:
    raw = user_input.strip()
elif st.session_state.pending_query:
    raw = st.session_state.pending_query.strip()

# Always clear pending_query so it doesn't fire again next rerun
st.session_state.pending_query = ""

# Only process if non-empty and not a duplicate of last query
if raw and raw != st.session_state.last_query:
    st.session_state.last_query = raw  # lock immediately

    # Append user message
    st.session_state.messages.append({"role": "user", "content": raw})

    # ── STREAMING ──────────────────────────────
    full_response = ""
    sources = []

    try:
        for chunk, retrieved_sources in pipeline.stream_answer(raw):
            full_response += chunk
            sources = retrieved_sources
            # Render live inside the chat-body placeholder
            stream_slot.markdown(f"""
            <div class="msg-row ai" style="padding:0 20px;">
                <div class="av av-ai">A</div>
                <div class="bubble ai">
                    {full_response}<span class="cursor"></span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    except Exception as exc:
        full_response = f"⚠ Error: {exc}"

    stream_slot.empty()

    # Save finished response
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": sources,
    })

    st.rerun()