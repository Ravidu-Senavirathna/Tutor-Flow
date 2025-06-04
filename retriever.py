import os
import chromadb
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_store")
COLLECTION  = "chapters"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K       = 5  # number of chunks to retrieve per query
# ─────────────────────────────────────────────────────────




if __name__ == "__main__":
