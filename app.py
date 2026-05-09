import os

import chromadb
import streamlit as st

from rag import get_rag_response

VECTORSTORE_DIR = "vectorstore"
COLLECTION_NAME = "liquid_handler_manual"


def vectorstore_ready() -> bool:
    if not os.path.exists(VECTORSTORE_DIR):
        return False
    try:
        chroma = chromadb.PersistentClient(path=VECTORSTORE_DIR)
        return chroma.get_collection(COLLECTION_NAME).count() > 0
    except Exception:
        return False


st.set_page_config(
    page_title="Liquid Handler Assistant",
    page_icon="🧪",
    layout="centered",
)

st.title("🧪 Liquid Handler Assistant")
st.caption("Ask me anything about your liquid handler — I'll answer based on the manual.")

if not vectorstore_ready():
    st.error(
        "Manual not loaded yet. Drop your PDF into the `data/` folder and run:\n\n"
        "```bash\npython ingest.py\n```"
    )
    st.stop()

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
        with st.spinner("Searching manual..."):
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            response = get_rag_response(prompt, history)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
