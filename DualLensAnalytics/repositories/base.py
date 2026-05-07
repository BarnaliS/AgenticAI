from __future__ import annotations

from typing import Protocol, Tuple

from langchain_core.documents import Document


class VectorRepository(Protocol):
    """
    Repository layer for vector-backed document retrieval.

    similarity_search returns:
    - documents: retrieved chunks
    - confidences: per-chunk confidence scores normalized to ~[0, 1]
    """

    def index_documents(self, documents: list[Document]) -> int:
        ...

    def similarity_search(self, query: str, top_k: int) -> Tuple[list[Document], list[float]]:
        ...

