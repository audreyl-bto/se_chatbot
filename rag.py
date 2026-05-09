import os

import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "liquid_handler_manual"
VECTORSTORE_DIR = "vectorstore"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"
TOP_K = 5

SYSTEM_PROMPT = (
    "You are a helpful assistant specialized in liquid handler operation and maintenance. "
    "Answer questions strictly based on the manual excerpts provided as context. "
    "If the answer is not in the context, say so clearly — do not guess. "
    "Always cite the page number when referencing information from the manual."
)


def _get_collection():
    chroma = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    return chroma.get_collection(COLLECTION_NAME)


def get_rag_response(query: str, chat_history: list[dict]) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    collection = _get_collection()

    query_embedding = (
        client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas"],
    )

    context = "\n\n---\n\n".join(
        f"[{meta['source']} — Page {meta['page']}]\n{doc}"
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(chat_history[-6:])  # keep last 3 exchanges for multi-turn context
    messages.append(
        {
            "role": "user",
            "content": f"Context from manual:\n\n{context}\n\nQuestion: {query}",
        }
    )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )

    return response.choices[0].message.content
