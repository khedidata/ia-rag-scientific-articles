import time
import urllib.parse
import urllib.request
from typing import List

import feedparser

from constants import ARXIV_API_BASE_URL, ARXIV_CATEGORIES 


# ---------------- Constants ----------------
MAX_RESULTS_PER_REQUEST = 300
REQUEST_DELAY_SECONDS = 3

# ---------------- Functions ----------------
def fetch_arxiv_feed(category: str, start: int = 0, max_results: int = MAX_RESULTS_PER_REQUEST) -> feedparser.FeedParserDict:
    """
    Fetch a batch of arXiv entries for a given category.

    Args:
        category: ArXiv category code (e.g., 'cs.AI').
        start: Index to start fetching results from.
        max_results: Maximum number of results per request.

    Returns:
        Parsed feed as a FeedParserDict.
    """
    query = urllib.parse.quote(f"cat:{category}")
    url = f"{ARXIV_API_BASE_URL}search_query={query}&start={start}&max_results={max_results}"
    with urllib.request.urlopen(url) as response:
        return feedparser.parse(response.read())

def fetch_all_category_entries(category: str) -> List[feedparser.FeedParserDict]:
    """
    Fetch all entries for a single arXiv category.

    Handles pagination automatically until no more results are returned.
    """
    entries: List = []
    start = 0
    while True:
        print(f"Fetching {category} starting at {start}...")
        feed = fetch_arxiv_feed(category, start=start)
        if not feed.entries:
            break
        entries.extend(feed.entries)
        start += MAX_RESULTS_PER_REQUEST
        time.sleep(REQUEST_DELAY_SECONDS)
    return entries

def fetch_all_cs_entries(categories: List[str]) -> List[feedparser.FeedParserDict]:
    """
    Fetch all entries for a list of Computer Science categories.

    Loops over each category and aggregates all entries.
    """
    all_entries: List = []
    for cat in categories:
        print(f"Processing category: {cat}")
        all_entries.extend(fetch_all_category_entries(cat))
    return all_entries

def main_articles_collection():
    return fetch_all_cs_entries(ARXIV_CATEGORIES)

