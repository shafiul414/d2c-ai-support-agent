import streamlit as st

import agent

# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="D2C AI Support Agent",
    page_icon="🤖",
    layout="centered",
)

try:
    agent.get_data()
    DATA_ERROR = None
except Exception as exc:
    DATA_ERROR = str(exc)


def to_history(messages):
    """Only role + content go to the AI (not the extra info we show in the UI)."""
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def meta_text(result) -> str:
    if result.mode == "ai":
        tools = result.tools_text() or "no tool needed"
        return f"🔧 {tools}  ·  ⏱ {result.seconds:.1f}s  ·  {result.model}"
    return f"Basic mode (AI not used)  ·  ⏱ {result.seconds:.1f}s"


# -----------------------------
# User interface
# -----------------------------
st.title("🤖 D2C AI Support Agent")
st.caption("Customer support prototype: Gemini + tool calling over your CSV data")

if DATA_ERROR:
    st.error(f"Could not load the store data: {DATA_ERROR}")
    st.stop()

if agent.AI_ENABLED:
    st.success(
        f"AI mode is on (model: {agent.MODEL_NAME}, thinking: {agent.THINKING_LEVEL}). "
        "The AI calls tools to read the data. Sample data is fictional."
    )
elif agent.GENAI_IMPORT_ERROR:
    st.warning(
        "Basic mode: the AI package is not installed. "
        "Run: python -m pip install -r requirements.txt"
    )
else:
    st.warning(
        "Basic mode: no GEMINI_API_KEY found. "
        "Add it to the .env file in the same folder as app.py, then restart the app."
    )

with st.expander("Try these example questions"):
    st.markdown(
        """
        - Where is my order ORD-1001?
        - What is the status of ORD-1002?
        - What is your return policy?
        - How long does shipping take?
        - Recommend headphones
        - Show me a smartwatch
        - Do you have a laptop stand?
        - I need something to listen to music while running
        - Do you sell shoes?
        - আমার ORD-1002 অর্ডারটা কবে আসবে?
        """
    )

if "messages" not in st.session_state:
    st.session_state.messages = []
if "ai_error" not in st.session_state:
    st.session_state.ai_error = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("meta"):
            st.caption(message["meta"])

if st.session_state.ai_error:
    st.warning(
        "The AI had a problem, so the basic answer was used instead. "
        f"Details: {st.session_state.ai_error}"
    )

user_input = st.chat_input("Ask the store a question...")

if user_input:
    history = to_history(st.session_state.messages)
    with st.spinner("Thinking..."):
        result = agent.respond(user_input, history)
    st.session_state.ai_error = result.error or None
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "meta": meta_text(result)}
    )
    st.rerun()

if st.button("Clear conversation"):
    st.session_state.messages = []
    st.session_state.ai_error = None
    st.rerun()
