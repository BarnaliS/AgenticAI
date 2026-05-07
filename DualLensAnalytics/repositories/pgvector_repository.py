from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_community.vectorstores.pgvector import DistanceStrategy

from embeddings import EmbeddingConfig


@dataclass(frozen=True)
class PGVectorRepositoryConfig:
    connection_string: str
    collection_name: str
    embedding_model: str = "text-embedding-ada-002"
    distance_strategy: str = "cosine"  # "cosine" | "l2" | "inner"
    embedding_length: Optional[int] = None
    pre_delete_collection: bool = False


def _distance_to_confidence(distance: float) -> float:
    """
    PGVector returns a distance where smaller is better.
    Convert to a [0, 1] confidence via a simple monotonically decreasing transform.
    """
    d = max(0.0, float(distance))
    return max(0.0, min(1.0, 1.0 / (1.0 + d)))


class PGVectorRepository:
    def __init__(self, config: PGVectorRepositoryConfig):
        self.config = config

        # LangChain expects the DistanceStrategy enum (cosine/l2/inner).
        self._distance_strategy = DistanceStrategy(self.config.distance_strategy)
        self._embedding_length = self.config.embedding_length or self._default_embedding_length(
            self.config.embedding_model
        )

    def _default_embedding_length(self, embedding_model: str) -> Optional[int]:
        # Common OpenAI embedding dimensions.
        # If you use a custom embedding model, set PGVECTOR_EMBEDDING_LENGTH explicitly.
        model = embedding_model.lower()
        if model in {"text-embedding-ada-002", "text-embedding-3-small"}:
            return 1536
        if model in {"text-embedding-3-large"}:
            return 3072
        return None

    def _embedding_function(self) -> OpenAIEmbeddings:
        # Reuse our EmbeddingConfig defaults so the rest of the pipeline stays consistent.
        embedding_config = EmbeddingConfig(
            model=self.config.embedding_model,
            collection_name=self.config.collection_name,
            persist_directory=Path("."),  # Persist dir is unused by PGVector but required by our shared config.
        )
        return OpenAIEmbeddings(model=embedding_config.model)

    def index_documents(self, documents: list[Document]) -> int:
        embedding = self._embedding_function()

        # PGVector.from_documents embeds and stores (chunks as `page_content`, metadata as JSONB).
        PGVector.from_documents(
            documents=documents,
            embedding=embedding,
            collection_name=self.config.collection_name,
            connection_string=self.config.connection_string,
            pre_delete_collection=self.config.pre_delete_collection,
            distance_strategy=self._distance_strategy,
            embedding_length=self._embedding_length,
            use_jsonb=True,
        )
        return len(documents)

    def similarity_search(self, query: str, top_k: int) -> Tuple[list[Document], list[float]]:
        embedding = self._embedding_function()

        # `from_existing_index` does not forward `embedding_length` in this
        # LangChain implementation, so instantiate directly to keep schema/index
        # behavior consistent.
        vector_store = PGVector(
            connection_string=self.config.connection_string,
            embedding_function=embedding,
            collection_name=self.config.collection_name,
            embedding_length=self._embedding_length,
            distance_strategy=self._distance_strategy,
            pre_delete_collection=self.config.pre_delete_collection,
            use_jsonb=True,
        )

        docs_and_scores = vector_store.similarity_search_with_score(query=query, k=top_k)
        documents = [doc for doc, _score in docs_and_scores]
        distances = [float(score) for _doc, score in docs_and_scores]
        confidences = [_distance_to_confidence(d) for d in distances]
        return documents, confidences

