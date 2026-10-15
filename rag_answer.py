import asyncio
import os
import sys

from dotenv import load_dotenv
from google.genai import errors, types
from pydantic import BaseModel, Field, ValidationError

from rag_store import client, search

load_dotenv()

MODELS = tuple(
    m.strip()
    for m in os.getenv(
        "GEMINI_MODELS", "gemini-3.8-flash,gemini-3.5-flash-lite"
    ).split(",")
    if m.strip()
)
MIN_SCORE = 0.45  # isse kam score wale chunks ko irrelevant maan lo (apne data par tune karna)


class Answer(BaseModel):
    answer: str
    found_in_documents: bool
    pages_used: list[int] = Field(default_factory=list)


SYSTEM = """You answer questions using ONLY the provided context chunks.
Rules:
- If the context does not contain the answer, set found_in_documents=false
  and say you could not find it in the documents. Do not use outside knowledge.
- Otherwise answer concisely and list the page numbers you actually used.
- Reply in the same language style as the question."""


async def ask(question: str, top_k: int = 4) -> Answer:
    hits = [(c, s) for c, s in await search(question, top_k) if s >= MIN_SCORE]
    if not hits:
        return Answer(answer="Documents mein is sawal ka jawab nahi mila.", found_in_documents=False)

    context = "\n\n".join(
        f"[Page {c['page']}, {c['source']}]\n{c['text']}" for c, _ in hits
    )
    prompt = f"Context:\n{context}\n\nQuestion: {question}"

    last_error = None
    for model in MODELS:
        for attempt in range(3):
            try:
                resp = await client.aio.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM,
                        response_mime_type="application/json",
                        response_schema=Answer,
                        temperature=0,
                    ),
                )
                result = Answer.model_validate_json(resp.text)
                # Guardrail: sirf wohi pages citation mein rakho jo retrieve hue the
                valid_pages = {c["page"] for c, _ in hits}
                result.pages_used = [p for p in result.pages_used if p in valid_pages]
                return result
            except (ValidationError, ValueError) as exc:
                last_error = exc
            except errors.ServerError as exc:
                last_error = exc
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
            except errors.ClientError as exc:
                last_error = exc
                if exc.code == 429:
                    if attempt < 2:
                        await asyncio.sleep(2**attempt)
                else:
                    break
    raise RuntimeError(f"All models failed: {last_error}")


async def main():
    if len(sys.argv) < 2:
        print('Use: uv run python rag_answer.py "sawal"')
        return
    result = await ask(sys.argv[1])
    print("\nJawab:", result.answer)
    print("Mila documents mein:", result.found_in_documents)
    print("Pages:", result.pages_used)


if __name__ == "__main__":
    asyncio.run(main())
