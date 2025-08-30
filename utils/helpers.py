from collections import defaultdict
from typing import List, Tuple

from langchain.schema import Document


def _stable_doc_id(doc: Document) -> str:
    """
    Generate a stable and unique identifier for a given Document.

    This function attempts to create a reliable ID for the document by checking
    several metadata fields in order of priority:
      1. 'paper_id' if available,
      2. 'pdf_url' if 'paper_id' is missing,
      3. a combination of 'title' and 'published' date as a fallback.

    Args:
        doc (Document): The Document object for which to generate the ID.

    Returns:
        str: A string that can serve as a stable identifier for the document.
    """
    return (
        doc.metadata.get("paper_id")
        or doc.metadata.get("pdf_url")
        or (doc.metadata.get("title","") + "|" + doc.metadata.get("published",""))
    )
    
def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[Document, float]]],
    k: int = 60,
    top_n: int = 5
) -> List[Tuple[Document, float]]:
    """
    Fuse multiple ranked lists of documents using Reciprocal Rank Fusion (RRF).

    RRF is a technique to combine results from multiple retrieval queries or
    models. Each document receives a score based on its rank in each list, using
    the formula: score += 1 / (k + rank). The final score is the sum across all
    lists, producing a fused ranking.

    Args:
        ranked_lists (List[List[Tuple[Document, float]]]): A list of ranked lists,
            where each inner list contains tuples of (Document, score) from a
            single query or retrieval method.
        k (int, optional): The RRF parameter that dampens the effect of ranks.
            Defaults to 60.
        top_n (int, optional): Number of top documents to return after fusion.
            Defaults to 5.

    Returns:
        List[Tuple[Document, float]]: A list of tuples of the top `top_n` Documents
            and their fused RRF scores, sorted in descending order.
    """
    scores = defaultdict(float)
    keep_doc = {}

    for results in ranked_lists:
        for rank, doc in enumerate(results, start=1):
            doc_id = _stable_doc_id(doc)
            keep_doc[doc_id] = doc
            scores[doc_id] += 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [(keep_doc[doc_id], score) for doc_id, score in fused]

def format_context(docs: List[Document]) -> str:
    """
    Format a list of Document objects into a structured context string.

    Each document is converted into a numbered block containing its title, 
    publication date, PDF URL, and main content (e.g., abstract). The blocks 
    are concatenated with double newlines, producing a readable context string 
    suitable for input to a language model.

    Args:
        docs (List[Document]): A list of Document objects to format.

    Returns:
        str: A single string representing all documents in a structured, 
             human-readable format, with numbered entries.
    """
    blocks = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata
        title = meta.get("title", "Untitled")
        pub = meta.get("published", "NA")
        url = meta.get("pdf_url", "NA")
        blocks.append(f"[{i}] Title: {title}\nDate: {pub}\nURL: {url}\nAbstract: {d.page_content}")
    return "\n\n".join(blocks)    