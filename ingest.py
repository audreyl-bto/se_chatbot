import os
import glob
import argparse
from pathlib import Path

import pypdf
import chromadb
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"
BATCH_SIZE = 100

COLLECTIONS = {
    "openai": "liquid_handler_openai",
    "gemini": "liquid_handler_gemini",
}


def extract_pages(pdf_path: str) -> list[dict]:
    pages = []
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page": i + 1, "text": text})
    return pages


def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if c.strip()]


def embed_openai(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in response.data]


def embed_gemini(texts: list[str]) -> list[list[float]]:
    import time
    from google import genai
    from google.genai import types
    from google.genai.errors import ClientError
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    embeddings = []
    for text in texts:
        while True:
            try:
                result = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=text,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
                )
                embeddings.append(result.embeddings[0].values)
                time.sleep(0.65)  # stay under 100 req/min free tier limit
                break
            except ClientError as e:
                if e.code == 429:
                    print("    Rate limit hit, waiting 40s...")
                    time.sleep(40)
                else:
                    raise
    return embeddings


def ingest_provider(provider: str):
    embed_fn = embed_openai if provider == "openai" else embed_gemini
    collection_name = COLLECTIONS[provider]

    chroma = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    try:
        chroma.delete_collection(collection_name)
        print(f"  Removed existing '{collection_name}' collection")
    except Exception:
        pass

    collection = chroma.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    pdf_files = glob.glob(f"{DATA_DIR}/*.pdf")
    if not pdf_files:
        print(f"  No PDFs found in {DATA_DIR}/")
        return

    doc_id = 0
    for pdf_path in pdf_files:
        filename = Path(pdf_path).name
        print(f"  Processing {filename}...")
        pages = extract_pages(pdf_path)

        chunks, metadatas, ids = [], [], []
        for page_data in pages:
            for chunk in chunk_text(page_data["text"]):
                chunks.append(chunk)
                metadatas.append({"source": filename, "page": page_data["page"]})
                ids.append(f"doc_{doc_id}")
                doc_id += 1

        for i in range(0, len(chunks), BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, len(chunks))
            print(f"    Embedding chunks {i + 1}–{batch_end} of {len(chunks)}...")
            embeddings = embed_fn(chunks[i:batch_end])
            collection.add(
                embeddings=embeddings,
                documents=chunks[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end],
            )

        print(f"  Done — {len(chunks)} chunks from {filename}")

    print(f"  Total chunks stored: {doc_id}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        choices=["openai", "gemini", "both"],
        default="both",
        help="Which provider to ingest for (default: both)",
    )
    args = parser.parse_args()

    providers = ["openai", "gemini"] if args.provider == "both" else [args.provider]

    for provider in providers:
        print(f"\n=== Ingesting for {provider.upper()} ===")
        ingest_provider(provider)

    print("Ingestion complete.")


if __name__ == "__main__":
    main()
