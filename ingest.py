import os
import pypdf
import chromadb
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────
PDF_FOLDER   = "pdfs"
CHROMA_PATH  = "chroma_store"
COLLECTION   = "bio_chapters"
EMBED_MODEL  = "all-MiniLM-L6-v2"
CHUNK_SIZE   = 400   # words per chunk
CHUNK_OVERLAP = 80   # words of overlap between chunks
# ─────────────────────────────────────────────────────────


def extract_text_from_pdf(pdf_path: str) -> str:
    """Read all pages of a PDF and return as one string."""
    text = ""
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping chunks by word count.
    Each chunk is chunk_size words, stepping forward by (chunk_size - overlap).
    """
    words = text.split()
    step  = chunk_size - overlap
    chunks = []

    for start in range(0, len(words), step):
        end   = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break

    return chunks
