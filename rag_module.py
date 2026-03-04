"""
rag_module.py

Usage examples:
  # transcribe video files (mp4/mkv/etc) in ./audio and save transcripts
  python rag_module.py --transcribe-only

  # ingest PDFs + media (transcribes media if transcripts missing) into Chroma
  python rag_module.py --ingest

  # interactive chat loop
  python rag_module.py --chat
"""

import os
import glob
import hashlib
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv

# langchain-ish imports (match your environment)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

from faster_whisper import WhisperModel

# Gemini connectors
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# optional fallback
try:
    import pdfplumber
except Exception:
    pdfplumber = None

# -----------------------------
# Config (edit as needed)
# -----------------------------
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "rag_collection"

PDF_DIR = "./docs"
MEDIA_DIR = "./audio"
TRANSCRIPTS_DIR = "./transcripts"

CHUNK_SIZE = 3000
CHUNK_OVERLAP = 300

# Whisper options
WHISPER_MODEL_SIZE = "medium"  # tiny/base/small/medium/large-v3 etc
WHISPER_DEVICE = "cpu"        # "cpu" or "cuda"
WHISPER_COMPUTE_TYPE = "int8" # "int8", "float16", etc

# Retrieval / QA
TOP_K = 5

# Gemini models (can be changed)
EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"

# Video -> audio extraction
EXTRACT_AUDIO_FOR_VIDEO = True
AUDIO_EXTRACT_FORMAT = "wav"    # wav recommended

# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def discover_files(folder: str, patterns: List[str]) -> List[str]:
    matches: List[str] = []
    for pat in patterns:
        matches.extend(glob.glob(os.path.join(folder, pat)))
    out, seen = [], set()
    for p in sorted(matches):
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out

import re

_TS_LINE = re.compile(r"^\s*\[\d{2}:\d{2}(?:\.\d{2})?\s*→\s*\d{2}:\d{2}(?:\.\d{2})?\]\s*", re.MULTILINE)

def strip_timestamps(transcript: str) -> str:
    # removes prefixes like: [00:00.00 → 00:09.00]
    return re.sub(_TS_LINE, "", transcript).strip()

def save_two_transcripts(original_media_path: str, transcript_timed: str) -> tuple[str, str]:
    ensure_dir(TRANSCRIPTS_DIR)
    base = Path(original_media_path).stem

    timed_path = os.path.join(TRANSCRIPTS_DIR, f"{base}.timed.txt")
    clean_path = os.path.join(TRANSCRIPTS_DIR, f"{base}.clean.txt")

    with open(timed_path, "w", encoding="utf-8") as f:
        f.write(transcript_timed.strip() + "\n")

    clean_text = strip_timestamps(transcript_timed)
    with open(clean_path, "w", encoding="utf-8") as f:
        f.write(clean_text.strip() + "\n")

    return timed_path, clean_path

def stable_chunk_id(source_file: str, source_type: str, chunk_index: int, content: str) -> str:
    """
    Deterministic ID: includes source path and a hash of the content.
    Avoids relying on ephemeral ordering.
    """
    h = hashlib.sha256()
    h.update(source_file.encode("utf-8", errors="ignore"))
    h.update(b"|")
    h.update(source_type.encode("utf-8", errors="ignore"))
    h.update(b"|")
    h.update(str(chunk_index).encode("utf-8"))
    h.update(b"|")
    h.update(hashlib.sha256((content or "").encode("utf-8", errors="ignore")).digest())
    return h.hexdigest()

def format_seconds(s: float) -> str:
    s = max(0.0, float(s))
    mm = int(s // 60)
    ss = s - (mm * 60)
    return f"{mm:02d}:{ss:05.2f}"

def is_video_file(path: str) -> bool:
    ext = Path(path).suffix.lower()
    return ext in {".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv", ".wmv"}

# -----------------------------
# Video -> audio (ffmpeg) + save transcript
# -----------------------------
def extract_audio_with_ffmpeg(video_path: str, out_dir: str, fmt: str = "wav") -> str:
    ensure_dir(out_dir)
    base = Path(video_path).stem
    audio_path = os.path.join(out_dir, f"{base}.audio.{fmt}")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        audio_path,
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Install ffmpeg and ensure it is on PATH.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed for {video_path}: {e.stderr.decode(errors='ignore')}")
    return audio_path

def save_transcript(source_path: str, transcript: str, out_dir: str) -> str:
    ensure_dir(out_dir)
    base = Path(source_path).stem
    out_path = os.path.join(out_dir, f"{base}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(transcript.strip() + "\n")
    return out_path

# -----------------------------
# A) Load & Process: PDFs
# -----------------------------
def load_pdf_documents(pdf_path: str) -> List[Document]:
    """
    Try PyPDFLoader first; if returned text looks empty, fall back to pdfplumber.
    Returns a Document per page with metadata.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    docs = PyPDFLoader(pdf_path).load()
    for d in docs:
        d.metadata = {**(d.metadata or {}), "source_type": "pdf", "source_file": pdf_path}

    nonempty = sum(1 for d in docs if (d.page_content or "").strip())
    if nonempty > 0:
        return docs

    if pdfplumber is None:
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
# B) Transcribe media -> text (faster-whisper)
# -----------------------------
def transcribe_media_files(
    media_paths: List[str],
    whisper_model_size: str = WHISPER_MODEL_SIZE,
    device: str = WHISPER_DEVICE,
    compute_type: str = WHISPER_COMPUTE_TYPE,
    extract_audio_for_video: bool = EXTRACT_AUDIO_FOR_VIDEO,
) -> List[Document]:
    if not media_paths:
        return []

    model = WhisperModel(whisper_model_size, device=device, compute_type=compute_type)
    ensure_dir(TRANSCRIPTS_DIR)
    tmp_audio_dir = os.path.join(TRANSCRIPTS_DIR, "_tmp_audio")
    ensure_dir(tmp_audio_dir)

    out_docs: List[Document] = []
    for path in media_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Media file not found: {path}")

        transcribe_input = path
        extracted_audio_path: Optional[str] = None

        if extract_audio_for_video and is_video_file(path):
            extracted_audio_path = extract_audio_with_ffmpeg(path, tmp_audio_dir, fmt=AUDIO_EXTRACT_FORMAT)
            transcribe_input = extracted_audio_path

        segments, info = model.transcribe(transcribe_input, beam_size=5)

        parts: List[str] = []
        seg_count = 0
        for seg in segments:
            seg_count += 1
            start = format_seconds(getattr(seg, "start", 0.0))
            end = format_seconds(getattr(seg, "end", 0.0))
            txt = (getattr(seg, "text", "") or "").strip()
            if txt:
                parts.append(f"[{start} → {end}] {txt}")

        transcript_timed = "\n".join(parts).strip() if parts else "[No transcript produced]"

        # ✅ Save both timed and clean versions
        timed_path, clean_path = save_two_transcripts(path, transcript_timed)

        # ✅ Use CLEAN text for embeddings
        clean_text = strip_timestamps(transcript_timed)

        out_docs.append(
            Document(
                page_content=clean_text,  # ← clean version used for embedding
                metadata={
                    "source_type": "media",
                    "source_file": path,
                    "timed_transcript": timed_path,
                    "clean_transcript": clean_path,
                    "language": getattr(info, "language", None),
                    "segment_count": seg_count,
                    "used_audio_extract": bool(extracted_audio_path),
                    "audio_extract_file": extracted_audio_path,
                },
            )
        )
    return out_docs

# -----------------------------
# C) Chunking
# -----------------------------
def chunk_documents(
    docs: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)

# -----------------------------
# D) Embedding & Chroma
# -----------------------------
def build_or_load_chroma(persist_dir: str, collection_name: str, embeddings) -> Chroma:
    ensure_dir(persist_dir)
    return Chroma(
        persist_directory=persist_dir,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

def upsert_documents_to_chroma(vdb: Chroma, chunked_docs: List[Document]) -> int:
    if not chunked_docs:
        return 0

    ids: List[str] = []
    docs: List[Document] = []

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

        d.metadata = {**meta, "chunk_index": idx, "chunk_id": cid}
        docs.append(d)

    # add_documents -> upsert behavior depends on underlying chroma; this supplies IDs
    vdb.add_documents(docs, ids=ids)
    return len(docs)

# -----------------------------
# E) Retrieve + RAG prompt
# -----------------------------
def default_rag_prompt(context: str, question: str) -> str:
    """
    RAG prompt template: strictly use only the provided context. If unknown, say so.
    """
    return f"""You are a helpful assistant.
Answer the user's question using ONLY the context provided below.
If the answer is not contained in the context, say you don't know and suggest what to look up next.

Context:
{context}

Question: {question}

Answer:
"""

def answer_question(vdb: Chroma, question: str, llm, k: int = TOP_K, prompt_template=default_rag_prompt) -> Dict[str, Any]:
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

    prompt = prompt_template(context, question)
    response = llm.invoke(prompt)
    # response may be object or string depending on integration
    ans = getattr(response, "content", None)
    if ans is None:
        ans = str(response)
    return {"answer": ans, "retrieved": retrieved_docs}

# -----------------------------
# High-level ingest orchestration
# -----------------------------
def ingest(pdfs: List[str], media: List[str]) -> None:
    load_dotenv()

    # enforce API key presence
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("Missing GOOGLE_API_KEY. Set it in the environment or in a .env file.")

    # Load PDFs
    pdf_docs: List[Document] = []
    for pdf in pdfs:
        pdf_docs.extend(load_pdf_documents(pdf))

    # For media: prefer existing transcripts. If missing, transcribe.
    media_docs: List[Document] = []
    if media:
        # check if transcript already exists
        for m in media:
            base = Path(m).stem
            transcript_candidate = os.path.join(TRANSCRIPTS_DIR, f"{base}.txt")
            if os.path.exists(transcript_candidate):
                with open(transcript_candidate, "r", encoding="utf-8") as fh:
                    txt = fh.read().strip()
                media_docs.append(
                    Document(
                        page_content=txt,
                        metadata={
                            "source_type": "media",
                            "source_file": m,
                            "transcript_file": transcript_candidate,
                            "restored_from_disk": True,
                        },
                    )
                )
            else:
                # transcribe (this will also save transcript)
                media_docs.extend(transcribe_media_files([m]))

    all_docs = pdf_docs + media_docs
    if not all_docs:
        raise ValueError(
            f"No input documents found.\n- Put PDFs in: {PDF_DIR}\n- Put media files in: {MEDIA_DIR}\n"
        )

    # Chunk
    chunked = chunk_documents(all_docs)

    # Embeddings & Chroma
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vdb = build_or_load_chroma(PERSIST_DIR, COLLECTION_NAME, embeddings)

    added = upsert_documents_to_chroma(vdb, chunked)

    print("✅ Ingestion complete")
    print(f"PDFs found: {len(pdfs)} | Media files found: {len(media)}")
    print(f"Loaded docs: {len(all_docs)} | Chunks stored (this run): {added}")
    print(f"Chroma persist_dir: {PERSIST_DIR} | collection: {COLLECTION_NAME}")

# -----------------------------
# CLI / main wiring
# -----------------------------
def run_single_question(question: str) -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("Missing GOOGLE_API_KEY. Set it in the environment or in a .env file.")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vdb = build_or_load_chroma(PERSIST_DIR, COLLECTION_NAME, embeddings)
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.2)

    result = answer_question(vdb, question, llm, k=TOP_K)
    print("\n=== ANSWER ===")
    print(result["answer"])

def run_chat() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("Missing GOOGLE_API_KEY. Set it in the environment or in a .env file.")
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

def transcribe_only() -> None:
    # transcribe all media files found
    media = discover_files(MEDIA_DIR, ["*.mp3", "*.wav", "*.m4a", "*.aac", "*.flac", "*.ogg", "*.mp4", "*.mkv", "*.mov", "*.webm"])
    print(f"Found {len(media)} media files. Transcribing (and saving transcripts) ...")
    transcribe_media_files(media)
    print(f"Transcripts saved to {TRANSCRIPTS_DIR}")

def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--transcribe-only", action="store_true", help="Transcribe videos/audio and save transcripts")
    parser.add_argument("--ingest", action="store_true", help="Ingest PDFs + media into the vector DB")
    parser.add_argument("--ask", type=str, default=None, help="Ask a single question (requires prior ingestion)")
    parser.add_argument("--chat", action="store_true", help="Interactive Q&A loop (requires prior ingestion)")
    args = parser.parse_args()

    pdfs = discover_files(PDF_DIR, ["*.pdf", "*.PDF"])
    media = discover_files(
        MEDIA_DIR,
        ["*.mp3", "*.wav", "*.m4a", "*.aac", "*.flac", "*.ogg", "*.mp4", "*.mkv", "*.mov", "*.webm", "*.MP4", "*.MKV", "*.MOV", "*.WEBM"],
    )

    if args.transcribe_only:
        transcribe_only()
        return

    if args.ingest:
        ingest(pdfs, media)

    if args.ask:
        run_single_question(args.ask)

    if args.chat:
        run_chat()

    if not (args.transcribe_only or args.ingest or args.ask or args.chat):
        print("Nothing to do. Try one of: --transcribe-only, --ingest, --ask \"...\", --chat")

if __name__ == "__main__":
    main()