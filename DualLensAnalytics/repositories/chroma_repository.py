from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from langchain_core.documents import Document

from embeddings import EmbeddingConfig, build_vector_store, load_vector_store
from retrieval import RetrievalConfig, retrieve_documents_with_scores


@dataclass(frozen=True)
class ChromaRepositoryConfig:
    persist_directory: str
    collection_name: str
    embedding_model: str = "text-embedding-ada-002"


class ChromaRepository:
    def __init__(self, config: ChromaRepositoryConfig):
        self.config = config

    def index_documents(self, documents: list[Document]) -> int:
        embedding_config = EmbeddingConfig(
            model=self.config.embedding_model,
            collection_name=self.config.collection_name,
            persist_directory=Path(self.config.persist_directory),
        )
        vector_store = build_vector_store(documents, embedding_config)
        # We don't have an exact "count indexed" from Chroma directly here.
        # The number of input documents is the best approximation.
        return len(documents)

    def similarity_search(self, query: str, top_k: int) -> Tuple[list[Document], list[float]]:
        embedding_config = EmbeddingConfig(
            model=self.config.embedding_model,
            collection_name=self.config.collection_name,
            persist_directory=Path(self.config.persist_directory),
        )
        retrieval_config = RetrievalConfig(top_k=top_k)
        vector_store = load_vector_store(embedding_config)

        documents, raw_scores = retrieve_documents_with_scores(
            query=query,
            vector_store=vector_store,
            config=retrieval_config,
        )

        # For Chroma, raw_scores are typically already "relevance-like".
        # Clamp to ~[0, 1] to serve as confidences.
        confidences = [max(0.0, min(1.0, float(s))) for s in raw_scores]
        return documents, confidences

