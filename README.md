# RAG Summarizer (Offline)

A fully offline Retrieval-Augmented Generation (RAG) system for summarizing
research papers and textbooks using FAISS and Ollama.

## Features
- Local PDF ingestion
- Semantic chunking
- FAISS vector retrieval
- Offline LLM inference using Ollama (Mistral)
- No OpenAI or cloud API dependency

## Tech Stack
- Python
- FAISS
- HuggingFace Embeddings
- Ollama (Mistral)
- PyMuPDF

## How to Run
```bash
pip install -r requirements.txt
ollama run mistral
python main.py
