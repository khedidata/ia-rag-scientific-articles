import re
from typing import List, Optional

import pandas as pd

from constants import DATA_PATH
from factory import CS_CATEGORIES

def get_categories_labels(category: str) -> Optional[str]:
    """ 
    Map raw category codes to their human-readable labels.

    Splits a comma-separated string of categories, removes duplicates, 
    and converts each category code to its corresponding label using CS_CATEGORIES.

    Parameters
    ----------
    category : str
        Comma-separated string of category codes (e.g., "cs.AI, cs.CL").

    Returns
    -------
    Optional[str]
        Comma-separated string of category labels, or None if input is NaN.
    """
    if pd.isna(category):
        return None
    cats = list(set(category.split(', ')))
    labels: List[str] = [CS_CATEGORIES.get(c, c) for c in cats]
    return ", ".join(labels)

def save_parquet(data: pd.DataFrame,
                 PATH: str = DATA_PATH) -> pd.DataFrame:
    """
    Save a DataFrame as a parquet file using fastparquet.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame to be saved.
    PATH : str, default DATA_PATH
        Path where the parquet file will be written.

    Returns
    -------
    pd.DataFrame
        The input DataFrame (unchanged) for convenience in method chaining.
    """
    return data.to_parquet(PATH, engine="fastparquet")

def regex_for_title(title_: str) -> Optional[str]:
    """
    Clean a title string by removing extra whitespace.

    This function replaces all sequences of whitespace characters
    (spaces, tabs, newlines) with a single space and trims leading
    and trailing spaces. If the input is not a string, it returns None.

    Parameters
    ----------
    title_ : str
        The title string to clean.

    Returns
    -------
    Optional[str]
        The cleaned title string, or None if the input is not a string.
    """
    if not isinstance(title_, str):
        return None
    return re.sub(r"\s+", " ", title_).strip()
