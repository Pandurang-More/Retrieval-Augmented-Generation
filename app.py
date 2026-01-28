import streamlit as st
import os

from ingest import load_pdf, chunk_text
from rag_pipeline import run_rag
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Offline RAG Document Q&A",
    layout="wide"
)

# ---------------- Session Init ----------------
if "faiss_ready" not in st.session_state:
    st.session_state.faiss_ready = False

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("📄 Offline RAG Q & A")
    st.caption("Ask questions from a PDF")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Upload a text-based PDF"
    )

    if st.session_state.faiss_ready:
        st.success("✅ Document indexed")

    st.markdown("---")
    st.caption("⚙️")

# ---------------- Main Area ----------------
st.title("💬 Document Question Answering")
st.write("Upload a PDF and chat with your document.")

# ---------------- Handle PDF Upload ----------------
if uploaded_file and not st.session_state.faiss_ready:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("⏳ Reading and indexing document (first time only)..."):
        text = load_pdf("temp.pdf")

        if not text:
            st.error("❌ No readable text found (scanned/image-based PDF).")
            st.stop()

        chunks = chunk_text(text)
        if not chunks:
            st.error("❌ Failed to split document into chunks.")
            st.stop()

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        db = FAISS.from_texts(chunks, embeddings)
        db.save_local("faiss_index")

        st.session_state.faiss_ready = True

    st.success("📚 Document indexed successfully")

    if os.path.exists("temp.pdf"):
        os.remove("temp.pdf")

# ---------------- Chat UI ----------------
if st.session_state.faiss_ready:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_question = st.chat_input("Ask something about the document...")

    if user_question:
        st.session_state.messages.append(
            {"role": "user", "content": user_question}
        )

        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Generating answer..."):
                answer = run_rag(user_question)
                st.write(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

else:
    st.info("⬅️ Upload a PDF from the sidebar to get started")
