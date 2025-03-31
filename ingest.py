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

