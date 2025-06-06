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


def load_retriever():
    """Load the embedding model and ChromaDB collection."""

    model      = SentenceTransformer(EMBED_MODEL)
    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION)
    return model, collection


def retrieve(query: str, model, collection, top_k: int = TOP_K) -> list[dict]:
    """
    Given a question, return the top_k most relevant chunks.
    Each result has 'text', 'source', and 'score'.
    """
    # Convert question to vector
    query_embedding = model.encode(query).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # Format results
    chunks = []
    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "text":   text,
            "source": metadata["source"],
            "score":  round(1 - (distance / 2), 4)
        })

    return chunks


if __name__ == "__main__":
    model, collection = load_retriever()

    # Test with a sample question
    query = "reinforcement learning"
    print(f"\nQuery: {query}\n")

    results = retrieve(query, model, collection)

    for i, chunk in enumerate(results):
        print(f"── Result {i+1} (score: {chunk['score']}) from {chunk['source']}")
        print(chunk["text"][:1000])
        print()