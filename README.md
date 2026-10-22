# 🧬 BioMed-RAG: Production Biomedical Question Answering System

> Inspired by: *Efficient and Reproducible Biomedical Question Answering using Retrieval Augmented Generation*

A scalable, production-grade **Retrieval-Augmented Generation (RAG)** system for biomedical question answering with PubMed evidence, PMID citations, and hallucination reduction.

---

## 🏛️ Architecture Overview

```text 
Frontend (Streamlit / HTML)
         | 
         v 
FastAPI Backend 
         | 
         v 
PDF Processing Pipeline 
         | 
         v 
Chunking + Cleaning 
         | 
         +-------------------+ 
         |                   | 
         v                   v 
 BM25 Index            Dense Embeddings (MedCPT) 
         |                   | 
         v                   v 
 Sparse Retrieval      ChromaDB + FAISS 
         |                   | 
         +--------Hybrid Retrieval--------+ 
                          | 
                          v 
                   Reranking Layer (MedCPT Cross-Encoder)
                          | 
                          v 
                Gemini API / LLM Response Generation 
                          | 
                          v 
                Dynamic Medical Answer 
                          | 
                          v 
 Frontend Chat Interface 
```

---

## 📂 Project Structure

```text
medical_rag_system/
├── src/                    # Source code
│   ├── api/                # API endpoints (chat_api.py, inference_api.py)
│   ├── core/               # Configuration & Prompts
│   ├── models/             # Model definitions (LLMs, Embeddings, Rerankers)
│   ├── services/           # Business logic & Pipelines (QA, Ingestion, Training)
│   ├── stores/             # Data store wrappers (Vector DBs, BM25)
│   └── utils/              # Shared utilities
├── data/                   # Data storage (Raw, Processed, Indices)
├── deployment/             # Docker & Deployment scripts
├── evaluation/             # Evaluation scripts & plots
├── frontend/               # Frontend application (Streamlit)
├── notebooks/              # Research notebooks
├── configs/                # External configurations
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

---

## 🛠️ Tech Stack

### Frontend
* Streamlit / HTML5 / CSS3 / JavaScript
* Fetch API
* Responsive UI & Chat Interface
* Markdown Rendering

### Backend
* Python / FastAPI / Uvicorn / Pydantic

### AI / NLP Components
* Google Gemini API / Mistral-7B
* MedCPT Embeddings (Query & Article Encoders)
* Hybrid Retrieval (BM25 + Dense Vector Search)
* FAISS & ChromaDB
* Reranking (MedCPT Cross-Encoder / BGE Reranker)

### Processing
* PyMuPDF / pdfplumber
* LangChain text splitter

---

## 🚀 Quick Start

### 1. Security Note
Never hardcode API keys. Use environment variables:
```env
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Build Indices
```python
from src.utils.helpers import load_config, set_seed
from src.services.data_ingestion import run_ingestion
from src.services.embedding_pipeline import BiomedEmbeddingPipeline
from src.stores.faiss_store import FAISSVectorStore
from src.stores.bm25_retrieval import BM25Retriever

cfg = load_config()
set_seed(cfg["system"]["seed"])

# Ingest data
# chunks, train_qa, val_qa, test_qa = run_ingestion(cfg)

# Build embeddings + FAISS
# emb  = BiomedEmbeddingPipeline(cfg)
# embs = emb.build(chunks)
# FAISSVectorStore(cfg).build(embs, chunks)

# Build BM25
# BM25Retriever(cfg).build(chunks)
```

### 4. Run Inference
```bash
python -m src.api.chat_api
# → http://localhost:8000
```

### 5. Access UI
Open [http://localhost:8000](http://localhost:8000) in your browser.


---

## 🧠 System Prompt

The assistant follows a strict medical system prompt ensuring safety and evidence grounding:

> "You are an advanced AI-powered Medical Retrieval-Augmented Generation (RAG) assistant specialized in biomedical and healthcare question answering..."

(Full prompt available in [prompts.py](file:///c:/Users/nagam/OneDrive/Desktop/medical_rag_system/medical_rag_system/src/core/prompts.py))

---

## 📊 Evaluation & Metrics

**Retrieval**: Recall@K, Precision@K, MAP, MRR, NDCG@K
**QA**: Exact Match, Token F1, ROUGE-1/2/L, BLEU
**Classification**: Accuracy, F1, Confusion Matrix, ROC/AUC

---

## ⚠️ Medical Disclaimer

This system is for **research and informational purposes only**. It does **not** constitute medical advice, diagnosis, or treatment. Always consult a licensed healthcare professional for medical decisions.
