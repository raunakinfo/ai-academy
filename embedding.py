import os
import argparse
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# Gemini embeddings (LangChain integration)
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def load_pdf(pdf_path: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    return loader.load()  # List[Document]


def split_docs(docs, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def ingest_to_chroma(
    pdf_path: str,
    persist_dir: str,
    collection_name: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing GOOGLE_API_KEY env var.")

    # 1) Load
    docs = load_pdf(pdf_path)

    # 2) Split
    chunks = split_docs(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # 3) Embeddings
    # Common Gemini embedding model names include "models/embedding-001".
    # If your account uses a different embedding model name, change embedding_model.
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)

    # 4) Store in Chroma (persistent)
    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    # Add documents (Chroma will embed via embedding_function)
    vectordb.add_documents(chunks)

    # Persist to disk
    vectordb.persist()

    return len(docs), len(chunks)


def main():
    parser = argparse.ArgumentParser(description="Ingest PDF into Chroma using Gemini embeddings.")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--persist_dir", default="./chroma_db", help="Chroma persistence directory")
    parser.add_argument("--collection", default="pdf_collection", help="Chroma collection name")
    parser.add_argument("--embedding_model", default="models/embedding-001", help="Gemini embedding model name")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size for splitting")
    parser.add_argument("--chunk_overlap", type=int, default=150, help="Chunk overlap for splitting")
    args = parser.parse_args()

    doc_count, chunk_count = ingest_to_chroma(
        pdf_path=args.pdf,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print("✅ Ingestion complete")
    print(f"PDF: {args.pdf}")
    print(f"Docs loaded: {doc_count}")
    print(f"Chunks stored: {chunk_count}")
    print(f"Chroma dir: {args.persist_dir}")
    print(f"Collection: {args.collection}")


if __name__ == "__main__":
    main()
