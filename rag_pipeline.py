from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama


def run_rag(query):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Retrieve top chunks
    docs = db.similarity_search(query, k=4)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a research assistant.
Using ONLY the context below, generate a clear summary.

Context:
{context}

Question:
{query}
"""

    llm = Ollama(model="mistral")  # or "llama3"
    return llm.invoke(prompt)
