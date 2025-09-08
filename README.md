# AI Assistant (RAG) – arXiv Scientific Documentation

Un assistant **RAG (Retrieval-Augmented Generation)** conçu pour interroger automatiquement les abstracts [arXiv](https://arxiv.org/) en Computer Science.  

L’utilisateur peut poser des questions en langage naturel, et le système génère des réponses :  

- **Structurées** : résumé concis suivi de détails en puces.
- **Sourcées** : chaque élément provient du contexte documentaire récupéré. 
- **Accessibles** : via une interface web minimaliste et intuitive.  

Le projet est développé en **Python 3.11** et repose sur plusieurs composants modernes :  

- **LangChain** → orchestration des prompts, mémoire conversationnelle et logique RAG.
- **FAISS** → moteur de recherche vectorielle pour indexer et retrouver les abstracts.  
- **OpenAI** → modèles de langage pour la génération augmentée. 
- **FastAPI** → API backend performante servant les réponses et l’interface web.  

---

## Project Pipeline

This project follows a structured pipeline of Retrieval-Augmented Generation (RAG) made for scientific articles in computer science (arXiv).

### 1. Data Collection

- **Source** : Automated scraping of articles via the official [arXiv API](https://info.arxiv.org/help/api/).
- **Preporcessing** : Retention of information deemed essential ---> *Abstract* + *Metadatas* (title, author, publication, URL PDF).

---

## 📦 Installation locale

### Prérequis
- Python ≥ 3.11
- [Poetry](https://python-poetry.org/) ou `pip`
- Une clé OpenAI valide (`OPENAI_API_KEY`)

### Étapes
```bash
git clone https://github.com/monuser/ia_rag_arxiv.git
cd ia_rag_arxiv

# Créer un venv
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\activate   # Windows

# Installer les dépendances
pip install -r requirements.txt

# Lancer l’app
uvicorn app.app:app --reload --port 8000
