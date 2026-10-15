import asyncio
import os

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

DOCS = [
    "Client ko invoice har mahine ki 5 tareekh tak bhejna zaroori hai.",
    "Office ka wifi password reception par likha hota hai.",
    "Payment late hone par 2% penalty lagti hai.",
    "Chhutti ke liye HR ko 3 din pehle email karo.",
    "Production deploy se pehle tests pass hona lazmi hai.",
]


async def embed(texts: list[str], task: str) -> np.ndarray:
    result = await client.aio.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task),
    )
    return np.array([e.values for e in result.embeddings])


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (n, d) documents, b: (d,) query
    return (a @ b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b))


async def search(query: str, top_k: int = 2):
    doc_vecs = await embed(DOCS, "RETRIEVAL_DOCUMENT")
    query_vec = (await embed([query], "RETRIEVAL_QUERY"))[0]
    scores = cosine(doc_vecs, query_vec)
    best = np.argsort(scores)[::-1][:top_k]
    return [(DOCS[i], float(scores[i])) for i in best]


async def main():
    for q in ["Payment ki deadline kya hai?", "Chhutti kaise milegi?"]:
        print(f"\nSawal: {q}")
        for text, score in await search(q):
            print(f"  {score:.3f}  {text}")


if __name__ == "__main__":
    asyncio.run(main())