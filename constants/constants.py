from typing import List

ARXIV_CATEGORIES: List[str] = [
    "cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR", "cs.CV",
    "cs.CY", "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS", "cs.ET", "cs.FL",
    "cs.GL", "cs.GR", "cs.GT", "cs.HC", "cs.IR", "cs.IT", "cs.LG", "cs.LO",
    "cs.MA", "cs.MM", "cs.MS", "cs.NA", "cs.NE", "cs.NI", "cs.OH", "cs.OS",
    "cs.PF", "cs.PL", "cs.RO", "cs.SC", "cs.SD", "cs.SE", "cs.SI", "cs.SY"
]

ARXIV_API_BASE_URL = "http://export.arxiv.org/api/query?" 

DATA_PATH = "./data/articles.parquet"

EMBEDDINGS_MODEL_NAME = "allenai-specter"

LLM_MODEL_NAME = "gpt-4o-mini"

FAISS_INDEX_PATH = "./faiss_index"