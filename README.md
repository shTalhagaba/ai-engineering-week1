# Document Q&A with RAG

## Problem

This project lets a user upload one text-based PDF and ask questions grounded in
that document. Gemini creates embeddings and generates answers; retrieved text
chunks provide the context for each answer.

## Architecture

```text
PDF upload ──> FastAPI (/upload) ──> PyPDF text extraction
                                      │
                                      v
                               chunking (800 chars,
                               150 char overlap)
                                      │
                                      v
                           Gemini embeddings ──> FAISS index
                                                   │
Question ──> FastAPI (/ask) ──> Gemini query embedding ──> top chunks
                                                               │
                                                               v
                                                     Gemini answer (JSON)

FastAPI (/health) ──> API and index readiness status
```

The FAISS index and chunk metadata are stored in `index/`; uploaded PDFs are
stored in `uploads/`.

## Tech Stack

- Python 3.11 and `uv`
- FastAPI, Uvicorn, and Pydantic
- Gemini API through `google-genai`
- PyPDF for PDF text extraction
- FAISS and NumPy for vector search

## Features

- PDF upload with a 10 MB limit
- Text extraction and overlapping chunks
- Gemini-based document and query embeddings
- FAISS cosine-similarity retrieval
- Grounded answers with page citations and an out-of-document response
- Persistent local index and upload directories

## API Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Returns API status and whether the FAISS index file exists. |
| `POST` | `/upload` | Upload a PDF as multipart form field `file` and build the index. |
| `POST` | `/ask` | Ask a question with JSON body `{"question":"..."}`. Upload a PDF first. |

Interactive API documentation is available at `/docs` while the service runs.

## Evaluation

`eval_rag.py` uses the questions in `eval_set.json`. It checks retrieval by
expected page, answer text using required keywords, and refusal for out-of-scope
questions. This is keyword-based evaluation; run it against a prepared index
with a configured Gemini API key to collect results.

| Baseline | chunk size | MIN_SCORE |
| --- | --- | --- |
| _To be measured_ | _To be measured_ | _To be measured_ |

## Run with Docker

1. Copy `.env.example` to `.env` and set a valid `GEMINI_API_KEY`.
2. Start the API:

   ```bash
   docker compose up --build
   ```

3. Check `http://localhost:8000/health` or open `http://localhost:8000/docs`.
4. Upload a PDF before asking questions. The Compose mounts preserve `index/`
   and `uploads/` on the host.

Stop the foreground service with `Ctrl+C`, or run `docker compose down` in
another terminal.

## Run locally with uv

1. Install `uv` and Python 3.11.
2. Copy `.env.example` to `.env` and set a valid `GEMINI_API_KEY`.
3. Install the locked dependencies and start the API:

   ```bash
   uv sync --frozen
   uv run uvicorn rag_api:app --host 0.0.0.0 --port 8000
   ```

4. Open `http://localhost:8000/docs`, upload a PDF, then ask a question.
5. To run the evaluation against the current index:

   ```bash
   uv run python eval_rag.py
   ```

## Limitations

- The service supports one document at a time; uploading another PDF replaces
  the current FAISS index.
- Scanned PDFs and OCR are not supported; the PDF must contain extractable
  text.
- Evaluation is keyword-based and does not provide semantic or human scoring.
