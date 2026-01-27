from langchain_huggingface import HuggingFaceEmbeddings
from ingest import load_pdf, chunk_text
from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import HuggingFaceEmbeddings

from rag_pipeline import run_rag

text = load_pdf("data/sample.pdf")
chunks = chunk_text(text)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

db = FAISS.from_texts(chunks, embeddings)
db.save_local("faiss_index")

print(run_rag("Summarize this document in simple words"))
