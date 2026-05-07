from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


@dataclass(frozen=True)
class EmbeddingConfig:
    # Match the notebook default to keep embeddings consistent.
    model: str = "text-embedding-ada-002"
    collection_name: str = "AI_Initiatives"
    persist_directory: Path = Path("./chroma_db")


def get_embedding_model(config: EmbeddingConfig) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=config.model)


def build_vector_store(
    documents: list[Document],
    config: EmbeddingConfig,
) -> Chroma:
    persist_directory = config.persist_directory.expanduser().resolve()
    persist_directory.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents,
        get_embedding_model(config),
        collection_name=config.collection_name,
        persist_directory=str(persist_directory),
    )
    vectorstore.persist()
    return vectorstore


def load_vector_store(config: EmbeddingConfig) -> Chroma:
    persist_directory = config.persist_directory.expanduser().resolve()
    if not persist_directory.exists():
        raise FileNotFoundError(
            f"Vector store directory does not exist: {persist_directory}. "
            "Run ingestion first."
        )

    return Chroma(
        collection_name=config.collection_name,
        embedding_function=get_embedding_model(config),
        persist_directory=str(persist_directory),
    )
