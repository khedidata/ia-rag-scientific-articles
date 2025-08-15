import os
from typing import List, Optional

import pandas as pd
from collector import main_articles_collection

from constants import DATA_PATH 
from utils import regex_for_title


# ---------------- Main Data Preprocessor ----------------
def data_preprocessor(export: bool = False) -> pd.DataFrame:
    """
    Fetch and preprocess articles from the main collection.

    This function collects articles, extracts key information, cleans duplicates,
    filters categories to only Computer Science fields (cs.*), and converts the 
    publication date to a datetime object. It also generates a direct PDF URL for each article.

    Parameters
    ----------
    export : bool, default False
        If True, the resulting DataFrame is saved as a parquet file at DATA_PATH.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the following columns:
        - paper_id : str, unique identifier of the paper
        - title : str, paper title
        - summary : str, abstract of the paper
        - authors : str, comma-separated list of authors
        - category : str, comma-separated Computer Science categories
        - date : datetime, publication date
        - pdf_url : str, direct link to the PDF of the paper
    """
    all_entries = main_articles_collection()
    
    data = pd.DataFrame([{
        "paper_id": e.get("id", ""),
        "title": e.get("title", ""),
        "summary": e.get("summary", ""),
        "authors": ", ".join(a.name for a in e.get("authors", [])),
        "category": ", ".join(t["term"] for t in e.get("tags", [])),
        "published": e.get("published", ""),
        "pdf_url": e.get("id", "").replace("/abs/", "/pdf/") if e.get("id") else ""
    } for e in all_entries])
    
    data = data[~data.duplicated(subset=["summary"])]
    
    data['title'] = data['title'].apply(regex_for_title)
    data["category"] = data["category"].apply(
        lambda cats: ", ".join(cat for cat in cats.split(", ") if cat.startswith("cs.") and isinstance(cats, str))
        )

    data = (data
            .assign(date = lambda x: pd.to_datetime(x["published"].str[:10],errors="coerce"))
            .drop(columns=["published"])
            .reset_index(drop=True)
            )
    
    if export:
        data.to_parquet(DATA_PATH, engine="fastparquet")
        
    return data


if __name__ == "__main__":
    data = data_preprocessor()

