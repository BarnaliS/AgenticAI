from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from generation import GenerationConfig, answer_question
from ingestion import IngestionConfig, ingest_documents
from main import load_openai_credentials
from retrieval import build_context, format_sources
from repositories.chroma_repository import ChromaRepository, ChromaRepositoryConfig
from repositories.pgvector_repository import PGVectorRepository, PGVectorRepositoryConfig


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_ROOT = BASE_DIR / "uploaded_pdfs"
DEFAULT_PERSIST_DIRECTORY = BASE_DIR / "chroma_db"


logger = logging.getLogger("rag_api")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class UploadResponse(BaseModel):
    status: str = "ok"
    documents: int
    chunks: int
    collection_name: str
    persist_directory: str
    documents_dir: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)

    # Optional overrides; defaults match the notebooks/modules.
    collection_name: str = "AI_Initiatives"
    persist_directory: str = str(DEFAULT_PERSIST_DIRECTORY)
    embedding_model: str = "text-embedding-ada-002"
    chat_model: str = "gpt-4o-mini"
    max_context_chars: int = 20_000
    temperature: float = 0.0


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    explanation: str
    confidence: float


app = FastAPI(title="DualLensAnalytics RAG API", version="1.0.0")


@app.on_event("startup")
async def _startup() -> None:
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    # Load OPENAI_API_KEY / OPENAI_API_BASE from `config.json` if set.
    load_openai_credentials(BASE_DIR / "config.json")


@app.middleware("http")
async def _log_requests(request: Any, call_next: Any) -> Any:
    request_id = str(uuid.uuid4())
    start = time.time()
    try:
        logger.info("[%s] %s %s", request_id, request.method, request.url.path)
        response = await call_next(request)
        return response
    finally:
        elapsed_ms = int((time.time() - start) * 1000)
        # response might not exist if call_next raised; still log duration.
        logger.info("[%s] completed in %sms", request_id, elapsed_ms)


def _require_openai_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is required. Set it as an env var or in config.json.",
        )


def _get_vector_backend() -> str:
    return os.getenv("VECTOR_BACKEND", "chroma").strip().lower()


def _require_pgvector_connection_string() -> str:
    connection_string = os.getenv("PGVECTOR_CONNECTION_STRING", "").strip()
    if not connection_string:
        raise HTTPException(
            status_code=503,
            detail="PGVECTOR_CONNECTION_STRING is required when VECTOR_BACKEND=pgvector.",
        )
    return connection_string


def _get_pgvector_distance_strategy() -> str:
    val = os.getenv("PGVECTOR_DISTANCE_STRATEGY", "cosine").strip().lower()
    allowed = {"cosine", "l2", "inner"}
    if val not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"PGVECTOR_DISTANCE_STRATEGY must be one of {sorted(allowed)}.",
        )
    return val


def _get_pgvector_pre_delete_collection() -> bool:
    val = os.getenv("PGVECTOR_PRE_DELETE_COLLECTION", "").strip().lower()
    return val in {"1", "true", "yes"}


def _get_pgvector_embedding_length() -> Optional[int]:
    raw = os.getenv("PGVECTOR_EMBEDDING_LENGTH", "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="PGVECTOR_EMBEDDING_LENGTH must be an integer.",
        ) from exc


def _build_repository(
    *,
    collection_name: str,
    persist_directory: str,
    embedding_model: str,
) -> Any:
    backend = _get_vector_backend()

    if backend == "chroma":
        return ChromaRepository(
            ChromaRepositoryConfig(
                persist_directory=persist_directory,
                collection_name=collection_name,
                embedding_model=embedding_model,
            )
        )

    if backend == "pgvector":
        return PGVectorRepository(
            PGVectorRepositoryConfig(
                connection_string=_require_pgvector_connection_string(),
                collection_name=collection_name,
                embedding_model=embedding_model,
                distance_strategy=_get_pgvector_distance_strategy(),
                pre_delete_collection=_get_pgvector_pre_delete_collection(),
                embedding_length=_get_pgvector_embedding_length(),
            )
        )

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported VECTOR_BACKEND='{backend}'. Use 'chroma' or 'pgvector'.",
    )


async def _save_uploads_to_dir(files: List[UploadFile], target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    for upload in files:
        if not upload.filename:
            raise HTTPException(status_code=400, detail="One of the uploaded files has no filename.")

        filename = Path(upload.filename).name  # prevent path traversal
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file '{filename}'. Only PDF uploads are supported.",
            )

        data = await upload.read()
        dest_path = target_dir / filename
        await asyncio.to_thread(dest_path.write_bytes, data)


@app.post("/upload", response_model=UploadResponse)
async def upload(
    files: List[UploadFile] = File(...),
    chunk_size: int = 500,
    chunk_overlap: int = 20,
    collection_name: str = "AI_Initiatives",
    persist_directory: str = str(DEFAULT_PERSIST_DIRECTORY),
    embedding_model: str = "text-embedding-ada-002",
) -> UploadResponse:
    """
    Upload PDFs to ingest them into the configured vector store backend.
    """
    _require_openai_api_key()

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    request_dir = UPLOAD_ROOT / str(uuid.uuid4())
    try:
        await _save_uploads_to_dir(files=files, target_dir=request_dir)

        ingestion_config = IngestionConfig(
            documents_dir=request_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        chunks = await asyncio.to_thread(ingest_documents, ingestion_config)

        repo = _build_repository(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_model=embedding_model,
        )
        await asyncio.to_thread(repo.index_documents, chunks)

        return UploadResponse(
            documents=len(files),
            chunks=len(chunks),
            collection_name=collection_name,
            persist_directory=persist_directory,
            documents_dir=str(request_dir),
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        logger.exception("Upload failed with bad request: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        logger.exception("Upload failed: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error during ingestion.") from exc


@app.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest) -> QueryResponse:
    """
    Query the vector store and return an answer grounded in retrieved context.
    """
    _require_openai_api_key()

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty.")

    generation_config = GenerationConfig(
        model=payload.chat_model,
        temperature=payload.temperature,
        max_context_chars=payload.max_context_chars,
    )

    try:
        repo = _build_repository(
            collection_name=payload.collection_name,
            persist_directory=payload.persist_directory,
            embedding_model=payload.embedding_model,
        )
        documents, confidences = await asyncio.to_thread(
            repo.similarity_search, question, payload.top_k
        )
        context = build_context(documents)
        answer = await asyncio.to_thread(
            answer_question,
            question=question,
            context=context,
            config=generation_config,
        )
        sources = format_sources(documents)

        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            best_conf = max(confidences)
            confidence = float(max(0.0, min(1.0, avg_conf)))
            explanation = (
                f"Used the {len(documents)} top retrieved chunks (top_k={payload.top_k}) as context "
                f"because they had the highest relevance to the question (avg confidence={avg_conf:.2f}, "
                f"best={best_conf:.2f}). Only this context was provided to the generator."
            )
        else:
            # Scores aren't available from this vector store API; use a heuristic confidence.
            confidence = float(min(0.9, 0.25 + 0.02 * len(documents)))
            explanation = (
                f"Used the {len(documents)} top retrieved chunks (top_k={payload.top_k}) as context "
                "because they were returned by the vector store for the given question. "
                "Only this context was provided to the generator."
            )

        return QueryResponse(
            answer=answer,
            sources=sources,
            explanation=explanation,
            confidence=confidence,
        )
    except FileNotFoundError as exc:
        logger.exception("Vector store not found: %s", exc)
        raise HTTPException(
            status_code=400,
            detail=str(exc) + " Run /upload first to ingest documents.",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        logger.exception("Query failed: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error during query.") from exc

