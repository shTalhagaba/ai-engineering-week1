from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from rag_answer import Answer, ask
from rag_store import INDEX_DIR, build_index

app = FastAPI(title="Document Q&A (RAG)")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_MB = 10


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@app.get("/health")
async def health():
    return {"status": "ok", "index_ready": (INDEX_DIR / "docs.faiss").exists()}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Sirf PDF files allowed hain")

    data = await file.read()
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"File {MAX_MB}MB se choti honi chahiye")

    path = UPLOAD_DIR / Path(file.filename).name  # path traversal se bachne ke liye sirf naam
    path.write_bytes(data)

    try:
        await build_index(str(path))
    except RuntimeError as exc:  # jaise scanned PDF, text nahi mila
        raise HTTPException(422, str(exc))
    return {"status": "indexed", "file": path.name}


@app.post("/ask", response_model=Answer)
async def ask_question(body: Question):
    if not (INDEX_DIR / "docs.faiss").exists():
        raise HTTPException(409, "Pehle /upload se PDF upload karo")
    try:
        return await ask(body.question)
    except RuntimeError:
        raise HTTPException(503, "AI service temporarily unavailable")