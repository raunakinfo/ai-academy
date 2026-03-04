"""
RAG Pipeline (PDF + Audio/Video)
--------------------------------
1) Load & extract text from PDFs (robust: PyPDFLoader -> pdfplumber fallback)
2) Transcribe audio/video to text (faster-whisper; uses ffmpeg under the hood)
3) Chunk text into semantically meaningful pieces
4) Embed + store in a vector DB (Chroma)
5) Retrieve relevant chunks + generate an answer with an LLM (Gemini)

Folder layout (edit if you want):
./docs   -> PDFs
./audio  -> mp3/wav/m4a/mp4/mkv/etc

Env:
  GOOGLE_API_KEY=...

Run:
  python rag_pipeline.py --ingest
  python rag_pipeline.py --ask "What are the key points?"
  python rag_pipeline.py --chat

Notes:
- Install ffmpeg on your machine for best media support.
- Re-ingestion is idempotent-ish via deterministic chunk IDs (won’t duplicate on rerun).
"""

import os
import glob
import hashlib
import argparse
from typing import List, Dict, Any, Optional, Iterable, Tuple

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector DB
from langchain_chroma import Chroma

# Gemini embeddings + chat model
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# PDF loading (primary)
from langchain_community.document_loaders import PyPDFLoader

# PDF fallback
try:
    import pdfplumber  # pip install pdfplumber
except Exception:
    pdfplumber = None

# Whisper transcription
from faster_whisper import WhisperModel


# -----------------------------
# Config (edit these)
# -----------------------------
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "rag_collection"

PDF_DIR = "./docs"
MEDIA_DIR = "./audio"

CHUNK_SIZE = 3000
CHUNK_OVERLAP = 300

# Whisper options
WHISPER_MODEL_SIZE = "small"  # tiny/base/small/medium/large-v3 etc
WHISPER_DEVICE = "cpu"        # "cpu" or "cuda"
WHISPER_COMPUTE_TYPE = "int8" # "int8", "float16", etc

# Retrieval / QA
TOP_K = 5

# Gemini models
EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"


# -----------------------------
# Utilities
# -----------------------------
def discover_files(folder: str, patterns: List[str]) -> List[str]:
    matches: List[str] = []
    for pat in patterns:
        matches.extend(glob.glob(os.path.join(folder, pat)))
    # stable sort + dedupe
    out, seen = [], set()
    for p in sorted(matches):
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def stable_chunk_id(source_file: str, source_type: str, chunk_index: int, content: str) -> str:
    """
    Deterministic ID so reruns don't create duplicates.
    """
    h = hashlib.sha256()
    h.update(source_file.encode("utf-8", errors="ignore"))
    h.update(b"|")
    h.update(source_type.encode("utf-8", errors="ignore"))
    h.update(b"|")
    h.update(str(chunk_index).encode("utf-8"))
    h.update(b"|")
    # include a hash of content (content can be huge)
    h.update(hashlib.sha256(content.encode("utf-8", errors="ignore")).digest())
    return h.hexdigest()


def format_seconds(s: float) -> str:
    s = max(0.0, float(s))
    mm = int(s // 60)
    ss = s - (mm * 60)
    return f"{mm:02d}:{ss:05.2f}"


# -----------------------------
# A) Load & Process: PDF (robust)
# -----------------------------
def load_pdf_documents(pdf_path: str) -> List[Document]:
    """
    Tries PyPDFLoader first. If it yields empty-ish text, falls back to pdfplumber (if installed).
    Returns one Document per page with page metadata.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # 1) Primary path
    docs = PyPDFLoader(pdf_path).load()
    for d in docs:
        d.metadata = {**(d.metadata or {}), "source_type": "pdf", "source_file": pdf_path}

    # Check if extraction looks empty
    nonempty = sum(1 for d in docs if (d.page_content or "").strip())
    if nonempty > 0:
        return docs

    # 2) Fallback: pdfplumber
    if pdfplumber is None:
        # return whatever we got (even if empty) with a hint
        for d in docs:
            d.page_content = (d.page_content or "").strip() or "[Empty PDF text extraction. Install pdfplumber for fallback.]"
        return docs

    out: List[Document] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").strip()
            if not text:
                text = "[No text extracted from this page]"
            out.append(
                Document(
                    page_content=text,
                    metadata={"source_type": "pdf", "source_file": pdf_path, "page": i},
                )
            )
    return out


# -----------------------------
# B) Load & Process: Media -> Text
# -----------------------------
def transcribe_media_files(
    media_paths: List[str],
    whisper_model_size: str = WHISPER_MODEL_SIZE,
    device: str = WHISPER_DEVICE,
    compute_type: str = WHISPER_COMPUTE_TYPE,
) -> List[Document]:
    """
    Uses faster-whisper to transcribe media files into text.
    Produces one Document per file, with timestamps included so retrieval is more useful.
    """
    if not media_paths:
        return []

    model = WhisperModel(whisper_model_size, device=device, compute_type=compute_type)

    out_docs: List[Document] = []
    for path in media_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Media file not found: {path}")

        segments, info = model.transcribe(path, beam_size=5)

        parts: List[str] = []
        seg_count = 0
        for seg in segments:
            seg_count += 1
            start = format_seconds(getattr(seg, "start", 0.0))
            end = format_seconds(getattr(seg, "end", 0.0))
            txt = (getattr(seg, "text", "") or "").strip()
            if txt:
                parts.append(f"[{start} → {end}] {txt}")

        transcript = "\n".join(parts).strip() if parts else "[No transcript produced]"

        out_docs.append(
            Document(
                page_content=transcript,
                metadata={
                    "source_type": "media",
                    "source_file": path,
                    "language": getattr(info, "language", None),
                    "segment_count": seg_count,
                },
            )
        )

    return out_docs


# -----------------------------
# C) Chunking (semantically meaningful)
# -----------------------------
def chunk_documents(
    docs: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Recursive splitter tries to keep paragraphs/sentences intact before splitting by spaces.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


# -----------------------------
# D) Embed + Store (Chroma)
# -----------------------------
def build_or_load_chroma(persist_dir: str, collection_name: str, embeddings) -> Chroma:
    return Chroma(
        persist_directory=persist_dir,
        collection_name=collection_name,
        embedding_function=embeddings,
    )


def upsert_documents_to_chroma(vdb: Chroma, chunked_docs: List[Document]) -> int:
    """
    Adds docs with deterministic IDs; if IDs already exist, Chroma will avoid duplicates
    depending on backend behavior. We try to be safe by providing IDs.
    """
    if not chunked_docs:
        return 0

    ids: List[str] = []
    docs: List[Document] = []

    # Assign chunk_index per (file,type) grouping for stable IDs
    counters: Dict[Tuple[str, str], int] = {}

    for d in chunked_docs:
        meta = d.metadata or {}
        src = meta.get("source_file", "unknown")
        st = meta.get("source_type", "unknown")
        key = (src, st)
        idx = counters.get(key, 0)
        counters[key] = idx + 1

        cid = stable_chunk_id(src, st, idx, d.page_content or "")
        ids.append(cid)

        # keep metadata + add chunk_index + chunk_id
        d.metadata = {**meta, "chunk_index": idx, "chunk_id": cid}
        docs.append(d)

    vdb.add_documents(docs, ids=ids)
    return len(docs)


# -----------------------------
# E) Retrieve + Generate (RAG)
# -----------------------------
def answer_question(vdb: Chroma, question: str, llm, k: int = TOP_K) -> Dict[str, Any]:
    retrieved_docs = vdb.similarity_search(question, k=k)

    context_blocks: List[str] = []
    for i, d in enumerate(retrieved_docs, start=1):
        meta = d.metadata or {}
        src = meta.get("source_file", "unknown")
        stype = meta.get("source_type", "unknown")
        page = meta.get("page", None)
        chunk_idx = meta.get("chunk_index", None)

        loc = f"{stype}:{os.path.basename(src)}"
        if page is not None:
            loc += f":page={page}"
        if chunk_idx is not None:
            loc += f":chunk={chunk_idx}"

        context_blocks.append(f"[{i}] ({loc})\n{(d.page_content or '').strip()}")

    context = "\n\n".join(context_blocks) if context_blocks else "[No context retrieved]"

    prompt = f"""You are a helpful assistant.
Answer the user's question using ONLY the context provided.
If the answer is not in the context, say you don't know and suggest what to look up next.

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
# Main
# -----------------------------
def ingest(pdfs: List[str], media: List[str]) -> None:
    pdf_docs: List[Document] = []
    for pdf in pdfs:
        pdf_docs.extend(load_pdf_documents(pdf))

    media_docs = transcribe_media_files(media)

    all_docs = pdf_docs + media_docs
    if not all_docs:
        raise ValueError(
            f"No input documents found.\n"
            f"- Put PDFs in: {PDF_DIR}\n"
            f"- Put media files in: {MEDIA_DIR}"
        )

    chunked = chunk_documents(all_docs)

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vdb = build_or_load_chroma(PERSIST_DIR, COLLECTION_NAME, embeddings)

    added = upsert_documents_to_chroma(vdb, chunked)

    print("✅ Ingestion complete")
    print(f"PDFs found: {len(pdfs)} | Media files found: {len(media)}")
    print(f"Loaded docs: {len(all_docs)} | Chunks stored (this run): {added}")
    print(f"Chroma persist_dir: {PERSIST_DIR} | collection: {COLLECTION_NAME}")


def run_single_question(question: str) -> None:
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vdb = build_or_load_chroma(PERSIST_DIR, COLLECTION_NAME, embeddings)
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.2)

    result = answer_question(vdb, question, llm, k=TOP_K)
    print("\n=== ANSWER ===")
    print(result["answer"])


def run_chat() -> None:
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vdb = build_or_load_chroma(PERSIST_DIR, COLLECTION_NAME, embeddings)
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.2)

    while True:
        q = input("\nAsk a question (or type 'exit'): ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        result = answer_question(vdb, q, llm, k=TOP_K)
        print("\n=== ANSWER ===")
        print(result["answer"])


def main():
    load_dotenv()

    if not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = "xxx"  

    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", action="store_true", help="Ingest PDFs + media into the vector DB")
    parser.add_argument("--ask", type=str, default=None, help="Ask a single question (requires prior ingestion)")
    parser.add_argument("--chat", action="store_true", help="Interactive Q&A loop (requires prior ingestion)")
    args = parser.parse_args()

    pdfs = discover_files(PDF_DIR, ["*.pdf", "*.PDF"])
    media = discover_files(
        MEDIA_DIR,
        ["*.mp3", "*.wav", "*.m4a", "*.aac", "*.flac", "*.ogg", "*.mp4", "*.mkv", "*.mov", "*.webm", "*.MP4", "*.MKV", "*.MOV", "*.WEBM"],
    )

    if args.ingest:
        ingest(pdfs, media)

    if args.ask:
        run_single_question(args.ask)

    if args.chat:
        run_chat()

    if not (args.ingest or args.ask or args.chat):
        print("Nothing to do. Try one of: --ingest, --ask \"...\", --chat")


if __name__ == "__main__":
    main()