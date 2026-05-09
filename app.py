import os

import chromadb
import streamlit as st

from rag import get_rag_response, COLLECTIONS

VECTORSTORE_DIR = "vectorstore"


def collection_ready(collection_name: str) -> bool:
    if not os.path.exists(VECTORSTORE_DIR):
        return False
    try:
        chroma = chromadb.PersistentClient(path=VECTORSTORE_DIR)
        return chroma.get_collection(collection_name).count() > 0
    except Exception:
        return False


st.set_page_config(
    page_title="Liquid Handler Assistant",
    page_icon="🧪",
    layout="centered",
)

st.title("🧪 Liquid Handler Assistant")
st.caption("Ask me anything about your liquid handler — I'll answer based on the manual.")

# --- Sidebar: provider selector ---
with st.sidebar:
    st.header("Settings")

    openai_ready = collection_ready(COLLECTIONS["openai"])
    gemini_ready = collection_ready(COLLECTIONS["gemini"])

    def label(name: str, ready: bool) -> str:
        return f"{name} {'✅' if ready else '❌ not ingested'}"

    available = []
    if openai_ready:
        available.append("openai")
    if gemini_ready:
        available.append("gemini")

    st.markdown("**Provider status:**")
    st.markdown(f"- {label('OpenAI', openai_ready)}")
    st.markdown(f"- {label('Gemini', gemini_ready)}")

    if not available:
        st.error("No provider ready. Run `python ingest.py` first.")
        st.stop()

    provider = st.radio(
        "Choose AI provider:",
        options=available,
        format_func=lambda x: "OpenAI (gpt-4o)" if x == "openai" else "Gemini (gemini-1.5-flash)",
    )

# Reset chat history when provider changes
if "current_provider" not in st.session_state:
    st.session_state.current_provider = provider

if st.session_state.current_provider != provider:
    st.session_state.messages = []
    st.session_state.current_provider = provider

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your liquid handler..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"Searching manual with {provider}..."):
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            response = get_rag_response(prompt, history, provider=provider)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
