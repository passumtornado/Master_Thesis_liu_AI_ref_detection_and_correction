"""
OpenAlex search integration for academic paper discovery.
Used as a second fallback when DBLP results are weak.
"""

import re
from typing import Any, Optional
from difflib import SequenceMatcher
import dotenv
import requests
import os

dotenv.load_dotenv()




OPENALEX_API_URL = f"https://api.openalex.org/works?api_key={os.getenv('OPENALEX_API_KEY')}"

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def normalize_title(value: Any) -> str:
    """Normalize title for comparison."""
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_openalex_similarity(
    query_title: str,
    candidate_title: str,
    query_author: str = "",
    candidate_authors: str = "",
    query_year: str = "",
    candidate_year: str = "",
) -> float:
    """Compute similarity score for OpenAlex results (0.0 to 1.0)."""
    score = 0.0
    
    # Title similarity (50% weight)
    if query_title and candidate_title:
        normalized_query = normalize_title(query_title)
        normalized_candidate = normalize_title(candidate_title)
        title_match = SequenceMatcher(None, normalized_query, normalized_candidate).ratio()
        score += title_match * 0.5
    
    # Author similarity (30% weight)
    if query_author and candidate_authors:
        query_authors_norm = normalize_title(query_author)
        candidate_norm = normalize_title(candidate_authors)
        author_match = SequenceMatcher(None, query_authors_norm, candidate_norm).ratio()
        score += author_match * 0.3
    
    # Year exact match (20% weight)
    if query_year and candidate_year:
        year_match = 1.0 if str(query_year).strip() == str(candidate_year).strip() else 0.0
        score += year_match * 0.2
    
    return round(score, 3)


def extract_openalex_hits(
    api_response: dict,
    query_title: str,
    query_author: str = "",
    query_year: str = "",
) -> list[dict]:
    """Extract normalized hits from OpenAlex API response."""
    hits = []
    
    if not api_response or "results" not in api_response:
        return hits
    
    for work in api_response.get("results", []):
        if not isinstance(work, dict):
            continue
        
        # Extract basic info
        title = work.get("title", "").strip()
        if not title:
            continue
        
        # Extract authors
        authors = []
        for authorship in work.get("authorships", []):
            if isinstance(authorship, dict) and "author" in authorship:
                author_info = authorship.get("author", {})
                if isinstance(author_info, dict):
                    name = author_info.get("display_name", "")
                    if name:
                        authors.append(name)
        authors_str = "; ".join(authors[:5])  # Limit to first 5 authors
        
        # Extract year
        year = str(work.get("publication_year", "")).strip()
        
        # Extract venue
        venue = ""
        primary_location = work.get("primary_location", {})
        if isinstance(primary_location, dict):
            source = primary_location.get("source", {})
            if isinstance(source, dict):
                venue = source.get("display_name", "")
        
        # Extract DOI
        doi = work.get("doi", "").replace("https://doi.org/", "").strip()
        
        # OpenAlex URL
        url = work.get("id", "").replace("https://openalex.org/", "")
        
        # Compute similarity
        similarity = compute_openalex_similarity(
            query_title=query_title,
            candidate_title=title,
            query_author=query_author,
            candidate_authors=authors_str,
            query_year=query_year,
            candidate_year=year,
        )
        
        hits.append({
            "title": title,
            "authors": authors_str,
            "venue": venue,
            "year": year,
            "url": f"https://openalex.org/{url}" if url else "",
            "doi": doi,
            "similarity_score": similarity,
            "source": "openalex",
            "cited_by_count": work.get("cited_by_count", 0),
        })
    
    # Sort by similarity score descending
    hits.sort(key=lambda item: item.get("similarity_score", 0.0), reverse=True)
    return hits


def search_openalex(
    title: str = "",
    author: str = "",
    year: str = "",
    max_results: int = 3,
) -> list[dict]:
    """
    Search OpenAlex API and return normalized hits.
    
    Args:
        title: Paper title to search for
        author: Optional author name(s)
        year: Optional publication year
        max_results: Maximum number of results to return
    
    Returns:
        List of normalized result dicts with title, authors, year, venue, etc.
    """
    
    if not title or not title.strip():
        return []
    
    # Build query for OpenAlex
    # OpenAlex uses a query string format: title:"..." OR author.display_name:"..." 
    filters = []
    
    # Primary filter: title
    title_clean = title.strip().replace('"', '\\"')
    filters.append(f'title.search:"{title_clean}"')
    
    # Optional author filter
    if author and author.strip():
        author_clean = author.strip().split()[0]  # Take first author name
        filters.append(f'author.display_name.search:"{author_clean}"')
    
    # Optional year filter (range: query_year - 1 to query_year + 1 to account for variations)
    if year and year.strip():
        try:
            year_int = int(year.strip())
            filters.append(f'publication_year:{year_int-1}-{year_int+1}')
        except ValueError:
            pass
    
    query_str = " AND ".join(filters)
    
    try:
        response = requests.get(
            OPENALEX_API_URL,
            params={
                "search": query_str,
                "per_page": max(max_results, 10),  # Request slightly more for filtering
            },
            timeout=10,
            headers={"User-Agent": "BibTeX-Validation-Agent/1.0"},
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print(f"  OpenAlex API error: {e}")
        return []
    except ValueError as e:
        print(f"  OpenAlex JSON parse error: {e}")
        return []
    
    hits = extract_openalex_hits(data, title, author, year)
    return hits[:max_results]
