from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from langchain.schema import Document


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 10
    search_type: str = "similarity"


def create_retriever(
    vector_store: Any,
    config: RetrievalConfig,
) -> Any:
    return vector_store.as_retriever(
        search_type=config.search_type,
        search_kwargs={"k": config.top_k},
    )


def retrieve_documents(
    query: str,
    retriever: Any,
) -> list[Document]:
    if not query.strip():
        raise ValueError("Query must not be empty.")
    return list(retriever.invoke(query))


def retrieve_documents_with_scores(
    query: str,
    vector_store: Any,
    config: RetrievalConfig,
) -> tuple[list[Document], list[float]]:
    """
    Retrieve documents plus relevance scores when the underlying vector store
    supports it (e.g. Chroma).
    """
    if not query.strip():
        raise ValueError("Query must not be empty.")

    # LangChain / Chroma typically supports `similarity_search_with_relevance_scores`.
    if hasattr(vector_store, "similarity_search_with_relevance_scores"):
        results: Iterable[tuple[Document, float]] = vector_store.similarity_search_with_relevance_scores(
            query, k=config.top_k
        )
        documents = [doc for doc, _score in results]
        scores = [float(score) for _doc, score in results]
        return documents, scores

    # Fallback for stores that return (Document, distance) where lower is better.
    if hasattr(vector_store, "similarity_search_with_score"):
        results2: Iterable[tuple[Document, float]] = vector_store.similarity_search_with_score(
            query, k=config.top_k
        )
        documents = [doc for doc, _score in results2]
        raw_scores = [float(score) for _doc, score in results2]
        # Convert "distance" to a confidence-like score (heuristic).
        # This keeps confidence in [0, 1] and degrades smoothly as distance grows.
        confidences = [1.0 / (1.0 + max(0.0, d)) for d in raw_scores]
        return documents, confidences

    # Last resort: no scores available.
    retriever = create_retriever(vector_store, config)
    documents = retrieve_documents(query, retriever)
    return documents, []


def build_context(documents: list[Document]) -> str:
    # Match the notebook behavior: concatenate retrieved chunks with separators.
    return ". ".join(doc.page_content.strip() for doc in documents if doc.page_content.strip())


def format_sources(documents: list[Document]) -> list[str]:
    sources: list[str] = []
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        label = f"{source} (page {page + 1})" if isinstance(page, int) else str(source)
        if label not in sources:
            sources.append(label)
    return sources
