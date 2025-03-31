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


