# 📄 Offline RAG Document Summarizer

A fully **offline Retrieval-Augmented Generation (RAG)** system for summarizing
research papers and documents using **FAISS** for retrieval and **Ollama** for
local LLM inference.  
🚫 No OpenAI key • 🚫 No cloud dependency • ✅ Privacy-friendly

---

## ✨ Features

- 🔍 Semantic retrieval using **FAISS**
- 🧩 **Semantic chunking** for better context understanding
- 🎯 **Cross-encoder reranking** to improve summary quality
- 🤖 **Local LLM inference using Ollama (Mistral / LLaMA)**
- 📄 PDF upload and summarization
- 🖥️ Interactive **Streamlit UI**
- ⚠️ Handles empty / scanned PDF edge cases
- 🚀 Optimized with caching for faster repeated runs

---

## 🧠 System Architecture

PDF → Text Extraction  
→ Semantic Chunking  
→ FAISS Vector Index  
→ Reranking (Cross-Encoder)  
→ Ollama Local LLM  
→ Summary Output  

---

## 🛠️ Tech Stack

- Python
- FAISS
- LangChain
- Sentence-Transformers
- Ollama (Local LLM)
- PyMuPDF
- Streamlit

---

## 📂 Project Structure

```
RAG/
├── app.py            # Streamlit UI
├── ingest.py         # PDF loading & chunking
├── rag_pipeline.py   # Retrieval + reranking + LLM
├── requirements.txt
├── README.md
├── .gitignore
└── venv/             # ignored
```

---

## 🚀 How to Run This Project (Local)

### 1️⃣ Prerequisites

- Python 3.10+
- Git
- Ollama installed  
  👉 https://ollama.com/download

---

### 2️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/rag-summarizer.git
cd rag-summarizer
```

---

### 3️⃣ Create & Activate Virtual Environment

```bash
python -m venv venv
```

**Windows**
```powershell
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

---

### 4️⃣ Install Dependencies

```bash
python -m pip install -r requirements.txt
```

---

### 5️⃣ Start Ollama (Local LLM)

Open a separate terminal and run:

```bash
ollama run mistral
```

⚠️ Keep this terminal running.

---

### 6️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

Open in browser:
```
http://localhost:8501
```

---

## 📌 Usage

1. Upload a PDF (research paper or document)
2. Click **Generate Summary**
3. View AI-generated summary using a **local LLM**

---

## ⚠️ Notes

- This project runs **fully offline**
- Ollama must be running locally
- Scanned/image-only PDFs require OCR (not included)

---

## 📈 Future Enhancements

- OCR support for scanned PDFs
- Page-level citations
- Multi-document summarization
- GPU acceleration
- Optional cloud deployment

---


