# DualLensAnalytics RAG (AI Initiatives)

Production-ready Python modules converted from the original notebook. This builds a persistent vector index over a local folder of PDF files and answers questions via retrieval-augmented generation (RAG).

## Quickstart

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Ensure you have credentials for OpenAI-compatible access:
   - Option A (recommended): populate `config.json` in this folder with keys:
     - `API_KEY`
     - `OPENAI_API_BASE` (optional but supported)
   - Option B: set environment variables:
     - `OPENAI_API_KEY`
     - `OPENAI_API_BASE` (optional)
3. Index your PDFs:
   - `python3 main.py index --documents-dir "./Companies-AI-Initiatives"`
4. Ask a question:
   - `python3 main.py ask "What are the top AI initiatives for IBM?"`

## CLI

### `index`
Builds/updates a persistent Chroma vector store under `--persist-directory` (default: `./chroma_db`).

### `ask <question>`
Retrieves top-k relevant chunks from the vector store and generates an answer grounded in that context.

## FastAPI Server
Run:
```bash
uvicorn rag_api:app --reload --host 0.0.0.0 --port 8000
```

Endpoints:
- `POST /upload` (multipart form-data: `files` as one or more PDFs) -> ingests PDFs into the persistent vector store.
- `POST /query` (JSON: `question`, optional `top_k`, etc.) -> returns `{ "answer": ..., "sources": [...], "explanation": "...", "confidence": 0.0 }`.

Vector backend selection (environment variables):
- `VECTOR_BACKEND=chroma` (default): uses the existing local persistent Chroma folder (`persist_directory`)
- `VECTOR_BACKEND=pgvector`: uses PostgreSQL + pgvector
  - `PGVECTOR_CONNECTION_STRING` (required), e.g. `postgresql+psycopg://user:pass@host:5432/db`
  - `PGVECTOR_DISTANCE_STRATEGY=cosine` (or `l2` / `inner`)
- `PGVECTOR_EMBEDDING_LENGTH` (optional; e.g. 1536 for `text-embedding-ada-002`)
  - `PGVECTOR_PRE_DELETE_COLLECTION=false` (optional)

## Configuration knobs

- `--chunk-size` / `--chunk-overlap`: controls PDF chunking (defaults match the notebook).
- `--top-k`: number of retrieved chunks to include in the prompt.
- `--collection-name`: must match between `index` and `ask` (default: `AI_Initiatives`).
- `--embedding-model`: must match between `index` and `ask` (default: `text-embedding-ada-002`).

## Notes

- This pipeline intentionally omits the notebook’s exploratory plotting/finance code and evaluation rater logic.
- Vector store persistence is enabled; re-running `index` will write to the configured collection directory.
