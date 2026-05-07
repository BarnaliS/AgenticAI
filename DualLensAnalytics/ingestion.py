from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader


@dataclass(frozen=True)
class IngestionConfig:
    documents_dir: Path
    chunk_size: int = 500
    chunk_overlap: int = 20
    encoding_name: str = "cl100k_base"


def load_documents(documents_dir: Path) -> list[Document]:
    documents_dir = documents_dir.expanduser().resolve()
    if not documents_dir.exists():
        raise FileNotFoundError(f"Documents directory does not exist: {documents_dir}")
    if not documents_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory of PDFs: {documents_dir}")

    loader = PyPDFDirectoryLoader(str(documents_dir))
    documents = loader.load()
    if not documents:
        raise ValueError(f"No PDF documents found in {documents_dir}")
    return documents


def build_text_splitter(config: IngestionConfig) -> RecursiveCharacterTextSplitter:
    # Use a pure character-based splitter to avoid any external network
    # calls for tokenizer downloads (e.g. tiktoken encodings).
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )


def split_documents(
    documents: Iterable[Document],
    config: IngestionConfig,
) -> list[Document]:
    splitter = build_text_splitter(config)
    chunks = splitter.split_documents(list(documents))
    if not chunks:
        raise ValueError("Document splitting produced no chunks.")
    return chunks


def ingest_documents(config: IngestionConfig) -> list[Document]:
    documents = load_documents(config.documents_dir)
    return split_documents(documents, config)
