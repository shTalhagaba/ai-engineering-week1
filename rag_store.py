import asyncio
import json
import os
import sys
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
INDEX_DIR = Path("index")


def read_pdf(path: str) -> list[tuple[int, str]]:
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}. Replace 'your_file.pdf' with the path to your PDF."
        )
    reader = PdfReader(str(pdf_path))
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    text = " ".join(text.split())  # extra spaces/newlines saaf
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap  # overlap: baat chunk ke beech mein na kat jaye
    return chunks


async def embed(texts: list[str], task: str, batch: int = 50) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), batch):
        result = await client.aio.models.embed_content(
            model=EMBED_MODEL,
            contents=texts[i : i + batch],
            config=types.EmbedContentConfig(task_type=task),
        )
        vecs.extend(e.values for e in result.embeddings)
    arr = np.array(vecs, dtype="float32")
    faiss.normalize_L2(arr)  # normalize karne ke baad inner product = cosine similarity
    return arr


async def build_index(pdf_path: str) -> None:
    chunks = []
    for page, text in read_pdf(pdf_path):
        for piece in chunk_text(text):
            chunks.append({"source": Path(pdf_path).name, "page": page, "text": piece})

    if not chunks:
        raise RuntimeError("PDF mein text nahi mila (scanned PDF ho sakta hai)")

    vecs = await embed([c["text"] for c in chunks], "RETRIEVAL_DOCUMENT")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "docs.faiss"))
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False))
    print(f"{len(chunks)} chunks index ho gaye.")


async def search(query: str, top_k: int = 3):
    index_path = INDEX_DIR / "docs.faiss"
    chunks_path = INDEX_DIR / "chunks.json"
    if not index_path.is_file() or not chunks_path.is_file():
        raise FileNotFoundError(
            "RAG index not found. Build it first with: "
            "uv run python rag_store.py build /path/to/your.pdf"
        )
    index = faiss.read_index(str(index_path))
    chunks = json.loads(chunks_path.read_text())
    q = await embed([query], "RETRIEVAL_QUERY")
    scores, ids = index.search(q, top_k)
    return [(chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]


async def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("build", "ask"):
        raise SystemExit(
            'Usage: uv run python rag_store.py build /path/to/file.pdf  |  '
            'uv run python rag_store.py ask "question"'
        )
    if sys.argv[1] == "build":
        await build_index(sys.argv[2])
    else:
        for chunk, score in await search(sys.argv[2]):
            print(f"\n[{score:.3f}] {chunk['source']} page {chunk['page']}")
            print(chunk["text"][:300], "...")


if __name__ == "__main__":
    asyncio.run(main())
