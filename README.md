# Tutor-Flow — GCE A/L Biology Study Assistant

A RAG-based quiz chatbot over the A/L Biology syllabus, built with Python, ChromaDB and sentence-transformers.

## Setup

1. Clone the repo and open in VS Code
2. Reopen in Dev Container when prompted
3. Copy `.env.example` to `.env` and add your LLM API key
4. Add your PDF chapters to the `pdfs/` folder
5. Run `python ingest.py` to index the PDFs
6. Run `python quiz.py` to start the chatbot

## Stack
- **LLM:** Any online or local(LLM API)
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Vector store:** ChromaDB (local)
- **PDF parsing:** pypdf