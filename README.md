# AI Assistant (RAG) – arXiv Scientific Documentation

A **RAG (Retrieval-Augmented Generation)** assistant designed to automatically query abstracts [arXiv](https://arxiv.org/) in Computer Science.

The user can ask questions in natural language, and the system generates answers:  

- **Structurées** : concise summary followed by bullet points of details.
- **Sourcées** : each element comes from the retrieved documentary context. 
- **Accessibles** : via a minimalist and intuitive web interface.  

The project is developed in **Python (3.11)** and relies on several modern components :  

- **LangChain** → orchestration of prompts, conversational memory and RAG logic.
- **FAISS** → vector search engine to index and find the abstracts.  
- **OpenAI** → language models for augmented generation. 
- **FastAPI** → powerful backend API serving the answers and web interface.  

---

## Project Pipeline

This project follows a structured pipeline of Retrieval-Augmented Generation (RAG) made for scientific articles in computer science (arXiv).

### 1. Data Collection

- **Source** : Automated scraping of articles via the official [arXiv API](https://info.arxiv.org/help/api/).
- **Preprocessing** : Retention of information deemed essential &rarr; Abstract + Metadatas.
- **Post Preprocessing Storage** : Data is stored locally in the `data\`.

### 2. Embeddings & Indexing

- **Model** : The embedding model used is [allenai-specter](https://huggingface.co/allenai/specter), which is very good for arXiv documentation.
- **Vector Store** : Using a FAISS indexing method to vectorized articles.
- **Post Treatment** : Indexed articles saved in `faiss_index\`.

### 3. RAG Pipeline

- **User Query**  
  The user asks a question in natural language through the web interface.
- **Query Expansion (optional)**  
  The query can be reformulated into several variants to improve the retrieval of relevant documents.
- **Retrieval**  
  The different query formulations are used to search for similar abstracts in the vector database (**FAISS**).  
  The results are merged (e.g., with **Reciprocal Rank Fusion**) to keep only the most relevant documents.
- **Context Building**  
  The selected abstracts and metadata are assembled into a **structured context**.
- **Answer Generation (LLM)**  
  The language model (**OpenAI via LangChain**) generates a response:  
  - Short summary in 1–2 sentences.
  - Detailed bullet points for key ideas.
  - Embedded references (title + arXiv link).  
- **Conversation Memory**  
  The history of questions and answers is stored to maintain context across multiple turns.

### 4. Web Application (FastAPI)

- **Backend**: FastAPI serves the API (`/chat`, `/docs`) and the HTML interface.  
- **Frontend**: a simple `index.html` with a chat-style interface to ask questions and display answers.  
- **Integration**: user queries are sent to the RAG pipeline, and answers are shown directly in the browser.

---

## Project Structure

```bash
├── .dockerignore               # Files and folders excluded from Docker build context  
├── .env                        # Environment variables (API keys, LangSmith config, etc.)  
├── .gitignore                  # Git ignored files and folders  
├── .python-version             # Python version specification (e.g., pyenv/Poetry)  
├── Dockerfile                  # Docker image definition  
├── docker-compose.yml          # Orchestration of the app in Docker  
├── requirements.txt            # Python dependencies  
├── README.md                   # Project documentation 
│
├── app/                        # FastAPI application  
│   └── app.py                  # API endpoints (serves UI and /chat)  
│
├── chains/                     # LangChain pipelines  
│   ├── conversational_qa.py    # Main RAG chain (retrieval + answer generation)  
│   ├── prompts.py              # Prompt templates for rephrasing and answering  
│   ├── utils.py                # Helper functions for RAG pipelines  
│   └── __init__.py  
│
├── config.py                   # Global configuration (API keys, settings)  
│
├── constants/                  # Global constants  
│   ├── constants.py            # Constant values used across the project  
│   └── __init__.py  
│
├── data/                       # Local dataset storage  
│   └── articles.parquet        # Collected and preprocessed arXiv abstracts  
│
├── data_collection/            # Data acquisition pipeline  
│   ├── collector.py            # Fetch articles from arXiv API  
│   ├── preprocess.py           # Clean and normalize raw data  
│   └── __init__.py  
│
├── factory/                    # Factory methods / dependency injection  
│   ├── factory.py              # Build LangChain components (LLM, retriever, etc.)  
│   └── __init__.py  
│
├── faiss_index/                # Prebuilt FAISS vector index  
│   ├── index.faiss             # FAISS binary index  
│   └── index.pkl               # Metadata for the index  
│
├── index.html                  # Minimal web chat interface (frontend)  
│
├── ingests/                    # Embeddings & indexing pipeline  
│   ├── embeddings.py           # Embedding model setup (sentence-transformers, etc.)  
│   ├── indexing.py             # Index creation and persistence (FAISS + metadata)  
│   └── __init__.py  
│
├── models/                     # Vector store retriever logic  
│   ├── vstore_retriever.py     # Custom retriever using FAISS  
│   └── __init__.py  
│
└── utils/                      # General utility functions  
    ├── functions.py            # Misc helper functions  
    └── __init__.py  
```


## Local Installation

Follow these steps to run the project locally without Docker:

### 1. Clone the repository
```bash
git clone https://github.com/khedidata/ia-rag-scientific-articles.git
cd ia_rag_arxiv
```
### 2. Create & Activate Virtual Environnement using [uv](https://github.com/astral-sh/uv)
```bash
# Check versions
python --version
pip install uv
uv --version

# Create virtual environment in .venv
uv venv .venv

# Activate environment
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows PowerShell
```

### 3. Install Dependencies
```bash
uv pip install -r requirements.txt
```

### 4. Run FastAPI Server
```bash
uvicorn app.app:app --reload --port 8000
```
