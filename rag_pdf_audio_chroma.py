import os
import glob
import argparse
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# Gemini embeddings + chat model
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Local Whisper transcription
from faster_whisper import WhisperModel

load_dotenv()
print(load_dotenv())


# -----------------------------
# A) Load & Process: PDF
# -----------------------------
def load_pdf_documents(pdf_path: str) -> List[Document]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()  # each page becomes a Document
    # Make metadata a little more explicit
    for d in docs:
        d.metadata = {**d.metadata, "source_type": "pdf", "source_file": pdf_path}
    return docs


# -----------------------------
# B) Load & Process: Audio -> Text
# -----------------------------
def transcribe_audio_files(
    audio_paths: List[str],
    whisper_model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
) -> List[Document]:
    """
    Uses faster-whisper to transcribe audio into text.
    Returns a list of Documents (one per audio file) with transcript in page_content.
    """
    if not audio_paths:
        return []

    model = WhisperModel(whisper_model_size, device=device, compute_type=compute_type)

    out_docs: List[Document] = []
    for ap in audio_paths:
        if not os.path.exists(ap):
            raise FileNotFoundError(f"Audio not found: {ap}")

        segments, info = model.transcribe(ap, beam_size=5)
        transcript_parts = []
        for seg in segments:
            # seg.text already has punctuation-ish; keep it simple
            transcript_parts.append(seg.text.strip())

        transcript = " ".join([t for t in transcript_parts if t])
        if not transcript.strip():
            transcript = "[No transcript produced]"

        out_docs.append(
            Document(
                page_content=transcript,
                metadata={
                    "source_type": "audio",
                    "source_file": ap,
                    "language": getattr(info, "language", None),
                },
            )
        )

    return out_docs


# -----------------------------
# C) Chunking
# -----------------------------
def chunk_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


# -----------------------------
# D) Embed + Store (Chroma)
# -----------------------------
def build_or_load_chroma(
    persist_dir: str,
    collection_name: str,
    embeddings,
) -> Chroma:
    return Chroma(
        persist_directory=persist_dir,
        collection_name=collection_name,
        embedding_function=embeddings,
    )


def ingest_documents_to_chroma(vdb: Chroma, docs: List[Document]) -> int:
    if not docs:
        return 0
    vdb.add_documents(docs)
    # vdb.persist()
    return len(docs)


# -----------------------------
# E) Retrieve + Generate (RAG)
# -----------------------------
def answer_question(
    vdb: Chroma,
    question: str,
    llm,
    k: int = 5,
) -> Dict[str, Any]:
    # 1) Retrieve
    retrieved_docs = vdb.similarity_search(question, k=k)

    # 2) Prepare context
    context_blocks = []
    for i, d in enumerate(retrieved_docs, start=1):
        meta = d.metadata or {}
        src = meta.get("source_file", "unknown")
        stype = meta.get("source_type", "unknown")
        page = meta.get("page", None)
        loc = f"{stype}:{os.path.basename(src)}"
        if page is not None:
            loc += f":page={page}"
        context_blocks.append(f"[{i}] ({loc})\n{d.page_content}")

    context = "\n\n".join(context_blocks) if context_blocks else "[No context retrieved]"

    # 3) Ask LLM with context
    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the context provided.
If the answer is not in the context, say you don't know and suggest what to look up.

Context:
{context}

Question: {question}
Answer:
"""

    response = llm.invoke(prompt)

    return {
        "answer": getattr(response, "content", str(response)),
        "retrieved": retrieved_docs,
    }


# -----------------------------
# CLI / Main
# -----------------------------
def expand_inputs(patterns: List[str]) -> List[str]:
    """
    Expand file patterns like data/*.pdf into concrete paths.
    Keeps order stable-ish by sorting each glob expansion.
    """
    paths: List[str] = []
    for p in patterns:
        matches = sorted(glob.glob(p))
        if matches:
            paths.extend(matches)
        else:
            # if no glob match, treat it as literal file path
            paths.append(p)
    # de-dup while preserving order
    seen = set()
    out = []
    for x in paths:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def main():
    parser = argparse.ArgumentParser(description="RAG: PDF + Audio (Whisper) -> Chroma -> Gemini answer")
    parser.add_argument("--pdf", nargs="*", default=[], help="PDF path(s) or glob(s) e.g. docs/*.pdf")
    parser.add_argument("--audio", nargs="*", default=[], help="Audio path(s) or glob(s) e.g. audio/*.mp3")
    parser.add_argument("--persist_dir", default="./chroma_db", help="Where Chroma persists")
    parser.add_argument("--collection", default="rag_collection", help="Chroma collection name")

    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--chunk_overlap", type=int, default=150)

    # Whisper options
    parser.add_argument("--whisper_model", default="small", help="tiny/base/small/medium/large-v3 etc")
    parser.add_argument("--whisper_device", default="cpu", help="cpu or cuda")
    parser.add_argument("--whisper_compute_type", default="int8", help="int8/float16 etc")

    # Retrieval
    parser.add_argument("--k", type=int, default=5, help="Top-k chunks to retrieve")
    parser.add_argument("--question", default=None, help="Ask a question after ingestion (optional)")

    # Gemini models
    parser.add_argument("--embedding_model", default="gemini-embedding-001")
    parser.add_argument("--chat_model", default="gemini-2.5-flash")

    args = parser.parse_args()

    if not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"]="AIzaSyBN0RdBFiZVog1_qpG4lVFo_n75hMiZ17Y"
        # raise EnvironmentError("Missing GOOGLE_API_KEY in environment.")

    pdf_paths = expand_inputs(args.pdf)
    audio_paths = expand_inputs(args.audio)

    # 1) Load PDF docs
    pdf_docs: List[Document] = []
    for pdf in pdf_paths:
        pdf_docs.extend(load_pdf_documents(pdf))

    # 2) Transcribe audio to docs
    audio_docs = transcribe_audio_files(
        audio_paths=audio_paths,
        whisper_model_size=args.whisper_model,
        device=args.whisper_device,
        compute_type=args.whisper_compute_type,
    )

    # 3) Combine + chunk
    all_docs = pdf_docs + audio_docs
    if not all_docs:
        raise ValueError("No input documents found. Provide --pdf and/or --audio.")

    chunked_docs = chunk_documents(
        all_docs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    # 4) Embeddings + Chroma
    embeddings = GoogleGenerativeAIEmbeddings(model=args.embedding_model)
    vdb = build_or_load_chroma(
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        embeddings=embeddings,
    )

    added = ingest_documents_to_chroma(vdb, chunked_docs)

    print("✅ Ingestion complete")
    print(f"PDFs: {len(pdf_paths)} | Audio files: {len(audio_paths)}")
    print(f"Loaded docs: {len(all_docs)} | Chunks stored: {added}")
    print(f"Chroma persist_dir: {args.persist_dir} | collection: {args.collection}")

    # 5) Optional QA
    if args.question:
        llm = ChatGoogleGenerativeAI(model=args.chat_model, temperature=0.2)
        result = answer_question(vdb=vdb, question=args.question, llm=llm, k=args.k)
        print("\n=== ANSWER ===")
        print(result["answer"])


if __name__ == "__main__":
    main()
