import os
import glob
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


# -----------------------------
# Config (edit these)
# -----------------------------
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "rag_collection"

PDF_DIR = "./docs"       # PDFs will be picked from here
VIDEO_DIR = "./audio"    # MP4s will be picked from here

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Whisper options
WHISPER_MODEL_SIZE = "small"     # tiny/base/small/medium/large-v3 etc
WHISPER_DEVICE = "cpu"           # cpu or cuda
WHISPER_COMPUTE_TYPE = "int8"    # int8/float16 etc

# Retrieval / QA
TOP_K = 5
QUESTION: Optional[str] = None  # e.g. "What are the key points?" or keep None

# Gemini models
EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"


# -----------------------------
# A) Load & Process: PDF
# -----------------------------
def load_pdf_documents(pdf_path: str) -> List[Document]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()  # each page becomes a Document
    for d in docs:
        d.metadata = {**(d.metadata or {}), "source_type": "pdf", "source_file": pdf_path}
    return docs


# -----------------------------
# B) Load & Process: Video/Audio -> Text (MP4)
# -----------------------------
def transcribe_media_files(
    media_paths: List[str],
    whisper_model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
) -> List[Document]:
    """
    Uses faster-whisper to transcribe media (e.g., .mp4) into text.
    Returns a list of Documents (one per file) with transcript in page_content.
    """
    if not media_paths:
        return []

    model = WhisperModel(whisper_model_size, device=device, compute_type=compute_type)

    out_docs: List[Document] = []
    for path in media_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Media file not found: {path}")

        segments, info = model.transcribe(path, beam_size=5)

        transcript_parts = []
        for seg in segments:
            transcript_parts.append(seg.text.strip())

        transcript = " ".join([t for t in transcript_parts if t]).strip()
        if not transcript:
            transcript = "[No transcript produced]"

        out_docs.append(
            Document(
                page_content=transcript,
                metadata={
                    "source_type": "mp4",
                    "source_file": path,
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
    retrieved_docs = vdb.similarity_search(question, k=k)

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
# Helpers: Auto-discover files
# -----------------------------
def discover_files(folder: str, patterns: List[str]) -> List[str]:
    """
    Discover files in a folder using one or more glob patterns.
    Returns a de-duped, sorted list.
    """
    all_matches: List[str] = []
    for pat in patterns:
        all_matches.extend(glob.glob(os.path.join(folder, pat)))
    # stable-ish sort + dedupe
    out = []
    seen = set()
    for p in sorted(all_matches):
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def main():
    load_dotenv()

    # If you want to hardcode key fallback (not recommended), keep it here:
    if not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = "AIzaSyBN0RdBFiZVog1_qpG4lVFo_n75hMiZ17Y"  # replace or remove
        # raise EnvironmentError("Missing GOOGLE_API_KEY in environment.")

    pdf_paths = discover_files(PDF_DIR, ["*.pdf", "*.PDF"])
    mp4_paths = discover_files(VIDEO_DIR, ["*.mp4", "*.MP4"])

    # 1) Load PDF docs
    pdf_docs: List[Document] = []
    for pdf in pdf_paths:
        pdf_docs.extend(load_pdf_documents(pdf))

    # 2) Transcribe MP4 to docs
    mp4_docs = transcribe_media_files(
        media_paths=mp4_paths,
        whisper_model_size=WHISPER_MODEL_SIZE,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )

    # 3) Combine + chunk
    all_docs = pdf_docs + mp4_docs
    if not all_docs:
        raise ValueError(
            f"No input documents found.\n"
            f"- Put PDFs in: {PDF_DIR}\n"
            f"- Put MP4s in: {VIDEO_DIR}"
        )

    chunked_docs = chunk_documents(all_docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # 4) Embeddings + Chroma
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vdb = build_or_load_chroma(
        persist_dir=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
    )

    added = ingest_documents_to_chroma(vdb, chunked_docs)

    print("✅ Ingestion complete")
    print(f"PDFs found: {len(pdf_paths)} | MP4 files found: {len(mp4_paths)}")
    print(f"Loaded docs: {len(all_docs)} | Chunks stored: {added}")
    print(f"Chroma persist_dir: {PERSIST_DIR} | collection: {COLLECTION_NAME}")

    # 5) Optional QA
    # if QUESTION:
    #     llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.2)
    #     result = answer_question(vdb=vdb, question=QUESTION, llm=llm, k=TOP_K)
    #     print("\n=== ANSWER ===")
    #     print(result["answer"])
# 5) Interactive QA
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.2)

    while True:
        user_question = input("\nAsk a question (or type 'exit'): ")

        if user_question.lower() == "exit":
            break

        result = answer_question(vdb=vdb, question=user_question, llm=llm, k=TOP_K)
        print("\n=== ANSWER ===")
        print(result["answer"])

if __name__ == "__main__":
    main()
