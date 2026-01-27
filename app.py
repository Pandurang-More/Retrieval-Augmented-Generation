import streamlit as st
import os

from ingest import load_pdf, chunk_text
from rag_pipeline import run_rag
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(page_title="Offline RAG Document QA", layout="centered")

st.title("📄 Offline RAG Document Q&A")
st.write("Upload a PDF and ask any question about the document using a local LLM (Ollama).")

# ---------- Upload PDF ----------
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully")

    # ---------- Build Index (ONLY ONCE) ----------
    if "faiss_ready" not in st.session_state:
        with st.spinner("⏳ Reading and indexing document (first time only)..."):
            text = load_pdf("temp.pdf")

            if not text:
                st.error("❌ No readable text found in PDF (may be scanned/image-based).")
                st.stop()

            chunks = chunk_text(text)

            if len(chunks) == 0:
                st.error("❌ Failed to split document into text chunks.")
                st.stop()

            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )

            db = FAISS.from_texts(chunks, embeddings)
            db.save_local("faiss_index")

            st.session_state.faiss_ready = True

        st.success("Document indexed successfully")

    # ---------- Question Input ----------
    question = st.text_input(
        "Ask a question about this document",
        placeholder="e.g. What problem does this paper solve?"
    )

    # ---------- Answer ----------
    if question:
        with st.spinner("🤖 Generating answer using local LLM..."):
            answer = run_rag(question)

        st.subheader("📌 Answer")
        st.write(answer)

    # ---------- Cleanup ----------
    if os.path.exists("temp.pdf"):
        os.remove("temp.pdf")
