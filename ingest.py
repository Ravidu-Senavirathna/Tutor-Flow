import os
import pypdf
import chromadb
from sentence_transformers import SentenceTransformer


# ── Config ────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER   = os.path.join(BASE_DIR, "pdfs")
CHROMA_PATH  = os.path.join(BASE_DIR, "chroma_store")
COLLECTION   = "chapters"
EMBED_MODEL  = "all-MiniLM-L6-v2"
CHUNK_SIZE   = 1000
CHUNK_OVERLAP = 300
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


def ingest():
    # 1. Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)

    # 2. Connect to ChromaDB (creates chroma_store/ if it doesn't exist)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete existing collection so re-running ingest starts clean
    try:
        client.delete_collection(COLLECTION)
        print("Cleared existing collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    # 3. Process each PDF
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"No PDFs found in '{PDF_FOLDER}/' — add your chapter PDFs and re-run.")
        return

    total_chunks = 0

    for pdf_file in sorted(pdf_files):
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        print(f"\nProcessing: {pdf_file}")

        # Extract text
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            print(f"  Warning: no text extracted from {pdf_file} — skipping.")
            continue

        # Chunk it
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"  {len(chunks)} chunks created from {len(text.split())} words")

        # Embed all chunks for this PDF in one batch (faster)
        embeddings = model.encode(chunks, show_progress_bar=True).tolist()

        # Build unique IDs and metadata for each chunk
        ids        = [f"{pdf_file}__chunk_{i}" for i in range(len(chunks))]
        metadatas  = [{"source": pdf_file, "chunk_index": i} for i in range(len(chunks))]

        # Store in ChromaDB
        collection.add(
            ids        = ids,
            documents  = chunks,
            embeddings = embeddings,
            metadatas  = metadatas,
        )

        total_chunks += len(chunks)

    print(f"\nDone! {total_chunks} total chunks stored in ChromaDB.")
    print(f"Vector store saved to '{CHROMA_PATH}/'")


if __name__ == "__main__":
    ingest()
