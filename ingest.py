import os
import glob
from pathlib import Path

import pypdf
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
COLLECTION_NAME = "liquid_handler_manual"
DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100


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


def embed(texts: list[str], client: OpenAI) -> list[list[float]]:
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def ingest():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    chroma = chromadb.PersistentClient(path=VECTORSTORE_DIR)

    try:
        chroma.delete_collection(COLLECTION_NAME)
        print(f"Removed existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = chroma.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    pdf_files = glob.glob(f"{DATA_DIR}/*.pdf")
    if not pdf_files:
        print(f"No PDFs found in {DATA_DIR}/  — drop your manual there and re-run.")
        return

    doc_id = 0
    for pdf_path in pdf_files:
        filename = Path(pdf_path).name
        print(f"\nProcessing {filename}...")
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
            print(f"  Embedding chunks {i + 1}–{batch_end} of {len(chunks)}...")
            embeddings = embed(chunks[i:batch_end], client)
            collection.add(
                embeddings=embeddings,
                documents=chunks[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end],
            )

        print(f"  Done — {len(chunks)} chunks from {filename}")

    print(f"\nIngestion complete. Total chunks stored: {doc_id}")


if __name__ == "__main__":
    ingest()
