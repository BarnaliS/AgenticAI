from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

from embeddings import EmbeddingConfig, build_vector_store, load_vector_store
from generation import GenerationConfig, answer_question
from ingestion import IngestionConfig, ingest_documents
from retrieval import RetrievalConfig, build_context, create_retriever, format_sources, retrieve_documents


@dataclass(frozen=True)
class AppConfig:
    documents_dir: Path
    persist_directory: Path
    collection_name: str
    embedding_model: str
    chat_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int


def load_openai_credentials(config_path: Path) -> None:
    """
    Load OpenAI credentials from a JSON file containing keys like:
    - API_KEY
    - OPENAI_API_BASE

    This is optional; env vars take precedence and are never overwritten.
    """

    if os.getenv("OPENAI_API_KEY"):
        return

    if not config_path.exists():
        return

    data = json.loads(config_path.read_text())
    if "API_KEY" in data and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = str(data["API_KEY"])
    if "OPENAI_API_BASE" in data and not os.getenv("OPENAI_API_BASE"):
        os.environ["OPENAI_API_BASE"] = str(data["OPENAI_API_BASE"])


def load_app_config(args: argparse.Namespace) -> AppConfig:
    base_dir = Path(__file__).resolve().parent
    documents_dir = Path(
        args.documents_dir
        or os.getenv("AI_INITIATIVES_DOCS_DIR")
        or (base_dir / "Companies-AI-Initiatives")
    )
    persist_directory = Path(
        args.persist_directory
        or os.getenv("AI_INITIATIVES_VECTOR_DB")
        or (base_dir / "chroma_db")
    )

    return AppConfig(
        documents_dir=documents_dir,
        persist_directory=persist_directory,
        collection_name=args.collection_name or os.getenv("AI_INITIATIVES_COLLECTION", "AI_Initiatives"),
        embedding_model=args.embedding_model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
        chat_model=args.chat_model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
    )


def build_index(config: AppConfig) -> None:
    ingestion_config = IngestionConfig(
        documents_dir=config.documents_dir,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    embedding_config = EmbeddingConfig(
        model=config.embedding_model,
        collection_name=config.collection_name,
        persist_directory=config.persist_directory,
    )

    chunks = ingest_documents(ingestion_config)
    build_vector_store(chunks, embedding_config)
    print(
        f"Indexed {len(chunks)} chunks from '{config.documents_dir}' "
        f"into '{config.persist_directory}'."
    )


def run_query(config: AppConfig, question: str) -> None:
    embedding_config = EmbeddingConfig(
        model=config.embedding_model,
        collection_name=config.collection_name,
        persist_directory=config.persist_directory,
    )
    retrieval_config = RetrievalConfig(top_k=config.top_k)
    generation_config = GenerationConfig(model=config.chat_model)

    vector_store = load_vector_store(embedding_config)
    retriever = create_retriever(vector_store, retrieval_config)
    documents = retrieve_documents(question, retriever)
    context = build_context(documents)
    answer = answer_question(question=question, context=context, config=generation_config)

    print("\nAnswer:\n")
    print(answer)

    sources = format_sources(documents)
    if sources:
        print("\nSources:")
        for source in sources:
            print(f"- {source}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG pipeline for AI initiative PDFs.")
    parser.add_argument(
        "--config",
        help="JSON file for OpenAI credentials (optional; env vars take precedence).",
        default=str(Path(__file__).resolve().parent / "config.json"),
    )
    parser.add_argument("--documents-dir")
    parser.add_argument("--persist-directory")
    parser.add_argument("--collection-name")
    parser.add_argument("--embedding-model")
    parser.add_argument("--chat-model")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("index", help="Ingest PDFs and build the vector store.")

    ask_parser = subparsers.add_parser("ask", help="Query the indexed documents.")
    ask_parser.add_argument("question")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    load_openai_credentials(Path(args.config))
    config = load_app_config(args)

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is required.")

    if args.command == "index":
        build_index(config)
        return

    if args.command == "ask":
        run_query(config, args.question)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
