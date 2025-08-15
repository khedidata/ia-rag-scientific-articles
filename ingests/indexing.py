import os
from typing import List

import pandas as pd
from langchain.schema import Document
from langchain.vectorstores import FAISS, VectorStore
from tqdm import tqdm

from constants import DATA_PATH, FAISS_INDEX_PATH
from ingests.embbedings import get_embeddings_model 

 
def transform_to_docs(data: pd.DataFrame) -> List[Document]:
    """
    Transform a pandas DataFrame into a list of LangChain Document objects.

    Each row in the DataFrame is converted into a Document where:
    - `page_content` is taken from the "summary" column.
    - `metadata` is a dictionary containing:
        - title (str): from the "title" column
        - authors (str or list): from the "authors" column
        - published (str): from the "date" column, formatted as YYYY-MM-DD
        - category (str): from the "category" column
        - pdf_url (str): from the "pdf_url" column

    Args:
        data (pd.DataFrame): A DataFrame containing at least the columns:
            ["summary", "title", "authors", "date", "category", "pdf_url"].

    Returns:
        list[Document]: A list of LangChain Document objects ready for use in
                        vector stores, RAG pipelines, or other NLP workflows.

    Raises:
        ValueError: If the resulting list of documents is empty or
                    its length does not match the number of rows in the DataFrame.
    """
    docs = [
        Document(
            page_content=row["summary"],
            metadata={
                "title": row["title"],
                "authors": row["authors"],
                "published": row["date"].strftime("%Y-%m-%d") 
                             if hasattr(row["date"], "strftime") 
                             else str(row["date"]),
                "category": row["category"],
                "pdf_url": row["pdf_url"]
            }
        )
        for _, row in data.iterrows()
    ]
    
    if not docs:
        raise ValueError("No documents were created from the DataFrame.")
    if len(docs) != len(data):
        raise ValueError("Mismatch between number of docs and DataFrame rows.")

    print(f"{len(docs)} documents created successfully.")
    return docs

def get_ingests(docs: List[Document], 
                batch_size: int = 512) -> VectorStore:
    """
    Create a FAISS vector store from a list of LangChain Documents and save it locally.

    This function initializes the FAISS index with the first batch of documents, 
    then iteratively adds the remaining documents in batches to avoid memory issues.
    Finally, the index is saved locally for later use.

    Args:
        docs (List[Document]): A list of LangChain Document objects to index.
        BATCH_SIZE (int, optional): Number of documents to process at a time. Default is 512.

    Returns:
        VectorStore: The FAISS vector store containing all the document embeddings.

    Raises:
        ValueError: If the docs list is empty.
    """
    if not docs:
        raise ValueError("The docs list is empty. Cannot create FAISS index.")
    
    embeddings_model = get_embeddings_model()

    # Initialize FAISS with the first batch
    initial_batch = docs[:batch_size]
    faiss_store = FAISS.from_documents(initial_batch, embedding=embeddings_model)

    # Add remaining documents in batches
    for idx in tqdm(range(batch_size, len(docs), batch_size), desc="Adding documents to FAISS"):
        batch_docs = docs[idx: idx + batch_size]
        faiss_store.add_documents(batch_docs)

    # Save the FAISS index locally
    faiss_store.save_local(FAISS_INDEX_PATH)
    
    return faiss_store


if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        data = pd.read_parquet(DATA_PATH)
        print("File loaded successfully!")
        
        docs = transform_to_docs(data)
        faiss_index = get_ingests(docs=docs)

    else:
        print(f"Unknown file: {DATA_PATH}")