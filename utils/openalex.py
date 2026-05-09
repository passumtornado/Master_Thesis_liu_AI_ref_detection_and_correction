# """
# OpenAlex search integration for academic paper discovery.
# Used as a second fallback when DBLP results are weak.
# """

# import re
# import time
# from difflib import SequenceMatcher
# from typing import Any, Optional
# import requests

# OPENALEX_API_URL = "https://api.openalex.org/works"

# # Rate limiter: keep track of last request time
# _last_request_time = 0.0

# _YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


# def normalize_title(value: Any) -> str:
#     """Normalize title for comparison."""
#     text = str(value or "").lower()
#     text = re.sub(r"[^a-z0-9\s]", " ", text)
#     return re.sub(r"\s+", " ", text).strip()


# def compute_openalex_similarity(
#     query_title: str,
#     candidate_title: str,
#     query_author: str = "",
#     candidate_authors: str = "",
#     query_year: str = "",
#     candidate_year: str = "",
# ) -> float:
#     """Compute similarity score for OpenAlex results (0.0 to 1.0)."""
#     score = 0.0

#     # Title similarity (50% weight)
#     if query_title and candidate_title:
#         normalized_query = normalize_title(query_title)
#         normalized_candidate = normalize_title(candidate_title)
#         title_match = SequenceMatcher(None, normalized_query, normalized_candidate).ratio()
#         score += title_match * 0.5

#     # Author similarity (30% weight)
#     if query_author and candidate_authors:
#         query_authors_norm = normalize_title(query_author)
#         candidate_norm = normalize_title(candidate_authors)
#         author_match = SequenceMatcher(None, query_authors_norm, candidate_norm).ratio()
#         score += author_match * 0.3

#     # Year exact match (20% weight)
#     if query_year and candidate_year:
#         year_match = 1.0 if str(query_year).strip() == str(candidate_year).strip() else 0.0
#         score += year_match * 0.2

#     return round(score, 3)


# def extract_openalex_hits(
#     api_response: dict,
#     query_title: str,
#     query_author: str = "",
#     query_year: str = "",
# ) -> list[dict]:
#     """Extract normalized hits from OpenAlex API response."""
#     hits = []

#     if not api_response or "results" not in api_response:
#         return hits

#     for work in api_response.get("results", []):
#         if not isinstance(work, dict):
#             continue

#         # Extract basic info
#         title = work.get("title", "").strip()
#         if not title:
#             continue

#         # Extract authors
#         authors = []
#         for authorship in work.get("authorships", []):
#             if isinstance(authorship, dict) and "author" in authorship:
#                 author_info = authorship.get("author", {})
#                 if isinstance(author_info, dict):
#                     name = author_info.get("display_name", "")
#                     if name:
#                         authors.append(name)
#         authors_str = "; ".join(authors[:5])  # Limit to first 5 authors

#         # Extract year
#         year = str(work.get("publication_year", "")).strip()

#         # Extract venue
#         venue = ""
#         primary_location = work.get("primary_location", {})
#         if isinstance(primary_location, dict):
#             source = primary_location.get("source", {})
#             if isinstance(source, dict):
#                 venue = source.get("display_name", "")

#         # Extract DOI
#         doi = work.get("doi", "").replace("https://doi.org/", "").strip()

#         # OpenAlex URL
#         url = work.get("id", "").replace("https://openalex.org/", "")

#         # Compute similarity
#         similarity = compute_openalex_similarity(
#             query_title=query_title,
#             candidate_title=title,
#             query_author=query_author,
#             candidate_authors=authors_str,
#             query_year=query_year,
#             candidate_year=year,
#         )

#         hits.append({
#             "title": title,
#             "authors": authors_str,
#             "venue": venue,
#             "year": year,
#             "url": f"https://openalex.org/{url}" if url else "",
#             "doi": doi,
#             "similarity_score": similarity,
#             "source": "openalex",
#             "cited_by_count": work.get("cited_by_count", 0),
#         })

#     # Sort by similarity score descending
#     hits.sort(key=lambda item: item.get("similarity_score", 0.0), reverse=True)
#     return hits


# def _rate_limit():
#     """Ensure at most 10 requests per second (0.1 second gap)."""
#     global _last_request_time
#     now = time.time()
#     min_interval = 0.1  # 10 requests per second
#     elapsed = now - _last_request_time
#     if elapsed < min_interval:
#         sleep_time = min_interval - elapsed
#         time.sleep(sleep_time)
#     _last_request_time = time.time()


# def search_openalex(
#     title: str = "",
#     author: str = "",
#     year: str = "",
#     max_results: int = 3,
# ) -> list[dict]:
#     """
#     Search OpenAlex API and return normalized hits.

#     Args:
#         title: Paper title to search for
#         author: Optional author name(s)
#         year: Optional publication year
#         max_results: Maximum number of results to return

#     Returns:
#         List of normalized result dicts with title, authors, year, venue, etc.
#     """
#     if not title or not title.strip():
#         return []

#     # Build query for OpenAlex
#     filters = []

#     # Primary filter: title
#     title_clean = title.strip().replace('"', '\\"')
#     filters.append(f'title.search:"{title_clean}"')

#     # Optional author filter
#     if author and author.strip():
#         author_clean = author.strip().split()[0]  # Take first author name
#         filters.append(f'author.display_name.search:"{author_clean}"')

#     # Optional year filter (range: query_year - 1 to query_year + 1 to account for variations)
#     if year and year.strip():
#         try:
#             year_int = int(year.strip())
#             filters.append(f'publication_year:{year_int-1}-{year_int+1}')
#         except ValueError:
#             pass

#     query_str = " AND ".join(filters)

#     # Apply rate limiting before making the request
#     _rate_limit()

#     try:
#         response = requests.get(
#             OPENALEX_API_URL,
#             params={
#                 "search": query_str,
#                 "per_page": max(max_results, 10),  # Request slightly more for filtering
#             },
#             timeout=10,
#             headers={"User-Agent": "BibTeX-Validation-Agent/1.0"},
#         )
#         response.raise_for_status()
#         data = response.json()
#     except requests.RequestException as e:
#         print(f"  OpenAlex API error: {e}")
#         return []
#     except ValueError as e:
#         print(f"  OpenAlex JSON parse error: {e}")
#         return []

#     hits = extract_openalex_hits(data, title, author, year)
#     return hits[:max_results]

# """
# OpenAlex search integration for academic paper discovery.
# Used as a second fallback when DBLP results are weak.
# """

# import re
# import time
# from difflib import SequenceMatcher
# from typing import Any, Optional, List, Dict
# import requests
# import urllib.parse

# OPENALEX_API_URL = "https://api.openalex.org/works"

# # Rate limiter (optional but recommended)
# _last_request_time = 0.0

# def normalize_title(value: Any) -> str:
#     """Normalize title for comparison."""
#     text = str(value or "").lower()
#     text = re.sub(r"[^a-z0-9\s]", " ", text)
#     return re.sub(r"\s+", " ", text).strip()


# def compute_openalex_similarity(
#     query_title: str,
#     candidate_title: str,
#     query_author: str = "",
#     candidate_authors: str = "",
#     query_year: str = "",
#     candidate_year: str = "",
# ) -> float:
#     """Compute similarity score for OpenAlex results (0.0 to 1.0)."""
#     score = 0.0

#     if query_title and candidate_title:
#         norm_q = normalize_title(query_title)
#         norm_c = normalize_title(candidate_title)
#         title_match = SequenceMatcher(None, norm_q, norm_c).ratio()
#         score += title_match * 0.5

#     if query_author and candidate_authors:
#         norm_q = normalize_title(query_author)
#         norm_c = normalize_title(candidate_authors)
#         author_match = SequenceMatcher(None, norm_q, norm_c).ratio()
#         score += author_match * 0.3

#     if query_year and candidate_year:
#         year_match = 1.0 if str(query_year).strip() == str(candidate_year).strip() else 0.0
#         score += year_match * 0.2

#     return round(score, 3)


# def _rate_limit():
#     """Ensure at most 10 requests per second (0.1 second gap)."""
#     global _last_request_time
#     now = time.time()
#     min_interval = 0.1
#     elapsed = now - _last_request_time
#     if elapsed < min_interval:
#         time.sleep(min_interval - elapsed)
#     _last_request_time = time.time()


# def search_openalex(
#     title: str = "",
#     author: str = "",
#     year: str = "",
#     max_results: int = 3,
# ) -> List[Dict[str, Any]]:
#     """
#     Search OpenAlex API and return normalized hits.

#     Args:
#         title: Paper title to search for
#         author: Optional author name(s)
#         year: Optional publication year
#         max_results: Maximum number of results to return

#     Returns:
#         List of normalized result dicts with title, authors, year, venue, etc.
#     """
#     if not title or not title.strip():
#         return []

#     # Build a plain text search query
#     # OpenAlex search works best with just the title keywords
#     search_terms = title.strip()
#     if author and author.strip():
#         first_author = author.split()[0].strip(',;')
#         search_terms += f" {first_author}"
#     if year and year.strip():
#         search_terms += f" {year}"

#     # Apply rate limiting
#     _rate_limit()

#     params = {
#         "search": search_terms,          # plain text search
#         "per_page": max_results,
#         "sort": "relevance_score:desc"
#     }
#     headers = {
#         "User-Agent": "BibTeX-Validation-Agent/1.0 (mailto:your-email@example.com)"
#     }

#     # Build full URL for debugging
#     url_with_params = f"{OPENALEX_API_URL}?{urllib.parse.urlencode(params)}"
#     print(f"OpenAlex request URL: {url_with_params}")

#     try:
#         response = requests.get(
#             OPENALEX_API_URL,
#             params=params,
#             headers=headers,
#             timeout=10
#         )
#         response.raise_for_status()
#         data = response.json()
#     except Exception as e:
#         print(f"  OpenAlex API error: {e}")
#         return []

#     results = data.get("results", [])
#     hits = []

#     for work in results:
#         work_title = work.get("title", "").strip()
#         if not work_title:
#             continue

#         # Extract authors
#         authors = []
#         for authorship in work.get("authorships", []):
#             if isinstance(authorship, dict):
#                 author_info = authorship.get("author", {})
#                 if isinstance(author_info, dict):
#                     name = author_info.get("display_name", "")
#                     if name:
#                         authors.append(name)
#         authors_str = "; ".join(authors[:5])

#         work_year = str(work.get("publication_year", "")).strip()

#         # Venue
#         venue = ""
#         primary_location = work.get("primary_location", {})
#         if isinstance(primary_location, dict):
#             source = primary_location.get("source", {})
#             if isinstance(source, dict):
#                 venue = source.get("display_name", "")
#         if not venue:
#             venue = work.get("host_venue", {}).get("display_name", "")

#         doi = work.get("doi", "").replace("https://doi.org/", "").strip()
#         work_id = work.get("id", "").replace("https://openalex.org/", "")

#         similarity = compute_openalex_similarity(
#             query_title=title,
#             candidate_title=work_title,
#             query_author=author,
#             candidate_authors=authors_str,
#             query_year=year,
#             candidate_year=work_year,
#         )

#         hits.append({
#             "title": work_title,
#             "authors": authors_str,
#             "venue": venue,
#             "year": work_year,
#             "url": f"https://openalex.org/{work_id}" if work_id else "",
#             "doi": doi,
#             "similarity_score": similarity,
#             "source": "openalex",
#             "cited_by_count": work.get("cited_by_count", 0),
#         })

#     hits.sort(key=lambda item: item.get("similarity_score", 0.0), reverse=True)
#     return hits[:max_results]


# """
# OpenAlex search integration for academic paper discovery.
# Used as a second fallback when DBLP results are weak.

# Implements:
#   1) Exact title phrase filter (most accurate)
#   2) Keyword search with re‑ranking (fallback)
#   3) DOI direct lookup (if available)
# """

# import re
# import time
# import urllib.parse
# from difflib import SequenceMatcher
# from typing import Any, Optional, List, Dict
# import requests

# OPENALEX_API_URL = "https://api.openalex.org/works"

# # Optional rate limiter (max 10 req/s)
# _last_request_time = 0.0

# def normalize_text(value: Any) -> str:
#     """Normalize text for comparison."""
#     text = str(value or "").lower()
#     text = re.sub(r"[^a-z0-9\s]", " ", text)
#     return re.sub(r"\s+", " ", text).strip()

# def _rate_limit():
#     """Ensure at most 10 requests per second."""
#     global _last_request_time
#     now = time.time()
#     min_interval = 0.1
#     elapsed = now - _last_request_time
#     if elapsed < min_interval:
#         time.sleep(min_interval - elapsed)
#     _last_request_time = time.time()

# def _headers():
#     return {"User-Agent": "BibTeX-Validation-Agent/1.0 (mailto:your-email@example.com)"}

# def _normalize_work(
#     work: dict,
#     query_title: str,
#     query_author: str,
#     query_year: str,
# ) -> dict:
#     """Convert OpenAlex work to our internal hit format."""
#     title = work.get("title", "").strip()
#     if not title:
#         return None

#     # Extract authors
#     authors = []
#     for auth in work.get("authorships", []):
#         author_info = auth.get("author", {})
#         name = author_info.get("display_name", "")
#         if name:
#             authors.append(name)
#     authors_str = "; ".join(authors[:5])

#     year = str(work.get("publication_year", "")).strip()

#     # Venue
#     venue = ""
#     primary_loc = work.get("primary_location", {})
#     if isinstance(primary_loc, dict):
#         source = primary_loc.get("source", {})
#         if isinstance(source, dict):
#             venue = source.get("display_name", "")
#     if not venue:
#         venue = work.get("host_venue", {}).get("display_name", "")

#     doi = work.get("doi", "").replace("https://doi.org/", "").strip()
#     work_id = work.get("id", "").replace("https://openalex.org/", "")

#     # Compute similarity score
#     title_sim = SequenceMatcher(None, normalize_text(query_title), normalize_text(title)).ratio()
#     author_sim = 0.0
#     if query_author and authors_str:
#         q_norm = normalize_text(query_author)
#         a_norm = normalize_text(authors_str)
#         author_sim = SequenceMatcher(None, q_norm, a_norm).ratio()
#     year_match = 1.0 if query_year and year and str(query_year).strip() == str(year).strip() else 0.0

#     # Weighted combination: 50% title, 30% authors, 20% year
#     similarity = (title_sim * 0.5) + (author_sim * 0.3) + (year_match * 0.2)

#     return {
#         "title": title,
#         "authors": authors_str,
#         "venue": venue,
#         "year": year,
#         "url": f"https://openalex.org/{work_id}" if work_id else "",
#         "doi": doi,
#         "similarity_score": round(similarity, 3),
#         "source": "openalex",
#         "cited_by_count": work.get("cited_by_count", 0),
#     }

# def search_openalex(
#     title: str = "",
#     author: str = "",
#     year: str = "",
#     doi: str = "",
#     max_results: int = 3,
# ) -> List[Dict[str, Any]]:
#     """
#     Search OpenAlex and return normalized hits.
#     Priority:
#       1) DOI lookup (if provided)
#       2) Exact title phrase filter (try with year, then without year if needed)
#       3) Keyword search with re‑ranking (fallback)
#     """
#     if not title and not doi:
#         return []

#     _rate_limit()

#     # ---------- 0) DOI lookup (most precise) ----------
#     if doi and doi.strip():
#         try:
#             url = f"https://api.openalex.org/works/https://doi.org/{doi.strip()}"
#             resp = requests.get(url, headers=_headers(), timeout=10)
#             if resp.status_code == 200:
#                 work = resp.json()
#                 hit = _normalize_work(work, title, author, year)
#                 if hit:
#                     return [hit]
#         except Exception as e:
#             print(f"  OpenAlex DOI lookup error: {e}")

#     # ---------- 1) Exact title phrase filter ----------
#     if title and title.strip():
#         # First attempt: with year if provided
#         filters = [f'title.search:"{title}"']
#         if year and year.strip():
#             try:
#                 y = int(year)
#                 filters.append(f'publication_year:{y}')
#             except:
#                 pass
#         if author and author.strip():
#             first_author = author.split()[0].strip(',;')
#             filters.append(f'authorships.author.display_name.search:"{first_author}"')

#         filter_str = " AND ".join(filters)
#         params = {
#             "filter": filter_str,
#             "per_page": max_results,
#             "sort": "relevance_score:desc"
#         }
#         try:
#             resp = requests.get(OPENALEX_API_URL, params=params, headers=_headers(), timeout=10)
#             if resp.status_code == 200:
#                 data = resp.json()
#                 results = data.get("results", [])
#                 if results:
#                     hits = [_normalize_work(w, title, author, year) for w in results[:max_results]]
#                     hits = [h for h in hits if h is not None]
#                     if hits:
#                         return hits
#         except Exception as e:
#             print(f"  OpenAlex exact‑title search error: {e}")

#         # Second attempt: same but without year (in case year was wrong)
#         if year and year.strip():
#             filters_no_year = [f'title.search:"{title}"']
#             if author and author.strip():
#                 first_author = author.split()[0].strip(',;')
#                 filters_no_year.append(f'authorships.author.display_name.search:"{first_author}"')
#             filter_str_no_year = " AND ".join(filters_no_year)
#             params_no_year = {
#                 "filter": filter_str_no_year,
#                 "per_page": max_results,
#                 "sort": "relevance_score:desc"
#             }
#             try:
#                 resp = requests.get(OPENALEX_API_URL, params=params_no_year, headers=_headers(), timeout=10)
#                 if resp.status_code == 200:
#                     data = resp.json()
#                     results = data.get("results", [])
#                     if results:
#                         hits = [_normalize_work(w, title, author, year) for w in results[:max_results]]
#                         hits = [h for h in hits if h is not None]
#                         if hits:
#                             return hits
#             except Exception as e:
#                 print(f"  OpenAlex exact‑title (no‑year) search error: {e}")

#     # ---------- 2) Keyword search with re‑ranking (fallback) ----------
#     if title and title.strip():
#         query_words = [title]
#         if author and author.strip():
#             first_author = author.split()[0].strip(',;')
#             query_words.append(first_author)
#         if year and year.strip():
#             query_words.append(year)
#         query_str = " ".join(query_words)

#         params = {
#             "search": query_str,
#             "per_page": max_results * 2,
#             "sort": "relevance_score:desc"
#         }
#         try:
#             resp = requests.get(OPENALEX_API_URL, params=params, headers=_headers(), timeout=10)
#             if resp.status_code == 200:
#                 data = resp.json()
#                 candidates = data.get("results", [])
#                 if candidates:
#                     hits = []
#                     for w in candidates[:20]:
#                         hit = _normalize_work(w, title, author, year)
#                         if hit:
#                             hits.append(hit)
#                     hits.sort(key=lambda x: x["similarity_score"], reverse=True)
#                     return hits[:max_results]
#         except Exception as e:
#             print(f"  OpenAlex keyword search error: {e}")

#     return []

"""
OpenAlex search integration – returns the single best match with complete metadata.
Used by validation and correction agents for field‑by‑field comparison.
"""

import re
import time
from difflib import SequenceMatcher
from typing import Any, Optional, List, Dict
import requests

OPENALEX_API_URL = "https://api.openalex.org/works"
_last_request_time = 0.0

def normalize_text(value: Any) -> str:
    """Normalize text: lowercase, remove punctuation, collapse spaces."""
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def _rate_limit():
    global _last_request_time
    now = time.time()
    if now - _last_request_time < 0.1:
        time.sleep(0.1 - (now - _last_request_time))
    _last_request_time = time.time()

def _headers():
    return {"User-Agent": "BibTeX-Validation-Agent/1.0 (mailto:your-email@example.com)"}

def _score_work(work: dict, query_title: str, query_author: str, query_year: str) -> float:
    """
    Compute a relevance score (0-1) for a single OpenAlex work.
    Higher is better. Uses title exact match, author overlap, year closeness,
    and has DOI as a bonus.
    """
    title = work.get("title", "").strip()
    if not title:
        return 0.0

    norm_q_title = normalize_text(query_title)
    norm_w_title = normalize_text(title)

    # Title score: exact match = 1.0, otherwise ratio
    if norm_q_title == norm_w_title:
        title_score = 1.0
    else:
        title_score = SequenceMatcher(None, norm_q_title, norm_w_title).ratio()

    # Author score
    authors = []
    for auth in work.get("authorships", []):
        name = auth.get("author", {}).get("display_name", "")
        if name:
            authors.append(name)
    authors_str = "; ".join(authors[:5])
    author_score = 0.0
    if query_author and authors_str:
        norm_q_author = normalize_text(query_author)
        norm_w_author = normalize_text(authors_str)
        author_score = SequenceMatcher(None, norm_q_author, norm_w_author).ratio()

    # Year score: exact = 1.0, within ±1 = 0.8, else 0.0
    work_year = work.get("publication_year")
    year_score = 0.0
    if query_year and work_year:
        try:
            qy = int(query_year)
            wy = int(work_year)
            if qy == wy:
                year_score = 1.0
            elif abs(qy - wy) == 1:
                year_score = 0.8
            else:
                year_score = 0.0
        except:
            pass

    # DOI bonus: has DOI -> +0.1
    doi_bonus = 0.1 if work.get("doi") else 0.0

    # Venue bonus: if venue exists (not empty) -> +0.05
    venue = ""
    primary_loc = work.get("primary_location", {})
    if isinstance(primary_loc, dict):
        source = primary_loc.get("source", {})
        if isinstance(source, dict):
            venue = source.get("display_name", "")
    if not venue:
        venue = work.get("host_venue", {}).get("display_name", "")
    venue_bonus = 0.05 if venue else 0.0

    # Weighted sum: title (0.5), author (0.2), year (0.2), doi (0.05), venue (0.05)
    score = (title_score * 0.5) + (author_score * 0.2) + (year_score * 0.2) + doi_bonus + venue_bonus
    return min(score, 1.0)

def _extract_best_hit(works: list, query_title: str, query_author: str, query_year: str) -> Optional[Dict]:
    """From a list of works, return the one with highest score and all fields."""
    if not works:
        return None
    best_work = max(works, key=lambda w: _score_work(w, query_title, query_author, query_year))
    score = _score_work(best_work, query_title, query_author, query_year)
    # Only accept if title similarity >= 0.7 (to avoid garbage)
    norm_q = normalize_text(query_title)
    norm_w = normalize_text(best_work.get("title", ""))
    title_sim = 1.0 if norm_q == norm_w else SequenceMatcher(None, norm_q, norm_w).ratio()
    if title_sim < 0.7:
        return None

    # Extract full metadata
    title = best_work.get("title", "").strip()
    authors = []
    for auth in best_work.get("authorships", []):
        name = auth.get("author", {}).get("display_name", "")
        if name:
            authors.append(name)
    authors_str = "; ".join(authors[:5])
    year = str(best_work.get("publication_year", "")).strip()
    venue = ""
    primary_loc = best_work.get("primary_location", {})
    if isinstance(primary_loc, dict):
        source = primary_loc.get("source", {})
        if isinstance(source, dict):
            venue = source.get("display_name", "")
    if not venue:
        venue = best_work.get("host_venue", {}).get("display_name", "")
    doi = best_work.get("doi", "").replace("https://doi.org/", "").strip()
    work_id = best_work.get("id", "").replace("https://openalex.org/", "")

    return {
        "title": title,
        "authors": authors_str,
        "venue": venue,
        "year": year,
        "url": f"https://openalex.org/{work_id}" if work_id else "",
        "doi": doi,
        "similarity_score": round(score, 3),
        "source": "openalex",
        "cited_by_count": best_work.get("cited_by_count", 0),
    }

def search_openalex(
    title: str = "",
    author: str = "",
    year: str = "",
    doi: str = "",
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    """
    Search OpenAlex and return the best matching hit(s).
    Returns a list of up to max_results, each with full metadata.
    """
    if not title and not doi:
        return []

    _rate_limit()

    # 1) DOI lookup (most reliable)
    if doi and doi.strip():
        try:
            url = f"https://api.openalex.org/works/https://doi.org/{doi.strip()}"
            resp = requests.get(url, headers=_headers(), timeout=10)
            if resp.status_code == 200:
                work = resp.json()
                hit = _extract_best_hit([work], title, author, year)
                if hit:
                    return [hit]
        except Exception:
            pass

    # 2) Exact title phrase filter (no year, to get candidates)
    if title and title.strip():
        # Build filter: exact title phrase + optional author
        filters = [f'title.search:"{title}"']
        if author and author.strip():
            first_author = author.split()[0].strip(',;')
            filters.append(f'authorships.author.display_name.search:"{first_author}"')
        filter_str = " AND ".join(filters)
        params = {
            "filter": filter_str,
            "per_page": max(10, max_results * 2),
            "sort": "relevance_score:desc"
        }
        try:
            resp = requests.get(OPENALEX_API_URL, params=params, headers=_headers(), timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("results", [])
                if candidates:
                    # Score all candidates and return the best one(s)
                    scored = []
                    for w in candidates:
                        score = _score_work(w, title, author, year)
                        # Only keep if title similarity >= 0.6 (some flexibility)
                        norm_q = normalize_text(title)
                        norm_w = normalize_text(w.get("title", ""))
                        title_sim = 1.0 if norm_q == norm_w else SequenceMatcher(None, norm_q, norm_w).ratio()
                        if title_sim >= 0.6:
                            scored.append((w, score))
                    if scored:
                        scored.sort(key=lambda x: x[1], reverse=True)
                        # Return the best match as a single hit (or up to max_results)
                        hits = []
                        for w, s in scored[:max_results]:
                            hit = _extract_best_hit([w], title, author, year)
                            if hit:
                                hits.append(hit)
                        return hits
        except Exception as e:
            print(f"  OpenAlex exact‑title error: {e}")

    # 3) Keyword search fallback (if exact title returns nothing)
    if title and title.strip():
        query_words = [title]
        if author and author.strip():
            query_words.append(author.split()[0])
        if year and year.strip():
            query_words.append(year)
        query_str = " ".join(query_words)
        params = {
            "search": query_str,
            "per_page": max_results * 2,
            "sort": "relevance_score:desc"
        }
        try:
            resp = requests.get(OPENALEX_API_URL, params=params, headers=_headers(), timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("results", [])
                if candidates:
                    scored = []
                    for w in candidates[:20]:
                        norm_q = normalize_text(title)
                        norm_w = normalize_text(w.get("title", ""))
                        title_sim = 1.0 if norm_q == norm_w else SequenceMatcher(None, norm_q, norm_w).ratio()
                        if title_sim >= 0.6:
                            score = _score_work(w, title, author, year)
                            scored.append((w, score))
                    if scored:
                        scored.sort(key=lambda x: x[1], reverse=True)
                        hits = []
                        for w, s in scored[:max_results]:
                            hit = _extract_best_hit([w], title, author, year)
                            if hit:
                                hits.append(hit)
                        return hits
        except Exception as e:
            print(f"  OpenAlex keyword error: {e}")

    return []