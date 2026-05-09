import os

import chromadb
from dotenv import load_dotenv

load_dotenv()

VECTORSTORE_DIR = "vectorstore"
TOP_K = 5

COLLECTIONS = {
    "openai": "liquid_handler_openai",
    "gemini": "liquid_handler_gemini",
}

SYSTEM_PROMPT = (
    "You are a helpful assistant specialized in liquid handler operation and maintenance. "
    "Answer questions strictly based on the manual excerpts provided as context. "
    "If the answer is not in the context, say so clearly — do not guess. "
    "Always cite the page number when referencing information from the manual."
)


def _get_collection(provider: str):
    chroma = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    return chroma.get_collection(COLLECTIONS[provider])


def _embed_query_openai(query: str) -> list[float]:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return client.embeddings.create(
        model="text-embedding-3-small", input=[query]
    ).data[0].embedding


def _embed_query_gemini(query: str) -> list[float]:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=query,
        task_type="retrieval_query",
    )
    return result["embedding"]


def _chat_openai(context: str, query: str, chat_history: list[dict]) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(chat_history[-6:])
    messages.append({
        "role": "user",
        "content": f"Context from manual:\n\n{context}\n\nQuestion: {query}",
    })
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content


def _chat_gemini(context: str, query: str, chat_history: list[dict]) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=SYSTEM_PROMPT,
    )
    # Convert history: "assistant" → "model" (Gemini's role name)
    history = [
        {
            "role": "model" if msg["role"] == "assistant" else "user",
            "parts": [msg["content"]],
        }
        for msg in chat_history[-6:]
    ]
    chat = model.start_chat(history=history)
    response = chat.send_message(
        f"Context from manual:\n\n{context}\n\nQuestion: {query}"
    )
    return response.text


def get_rag_response(query: str, chat_history: list[dict], provider: str = "openai") -> str:
    collection = _get_collection(provider)

    embed_fn = _embed_query_openai if provider == "openai" else _embed_query_gemini
    chat_fn = _chat_openai if provider == "openai" else _chat_gemini

    query_embedding = embed_fn(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas"],
    )

    context = "\n\n---\n\n".join(
        f"[{meta['source']} — Page {meta['page']}]\n{doc}"
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    )

    return chat_fn(context, query, chat_history)
