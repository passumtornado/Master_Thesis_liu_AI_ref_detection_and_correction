import html
import re
from difflib import SequenceMatcher
from typing import Any
from urllib.parse import urljoin

import requests


GOOGLE_SCHOLAR_URL = "https://scholar.google.com/scholar"

_TAG_RE = re.compile(r"<[^>]+>")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _strip_tags(value: str) -> str:
    text = _TAG_RE.sub(" ", value or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def normalize_title(value: Any) -> str:
    text = str(value or "").lower()
    text = html.unescape(text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_scholar_hits(html_text: str, query_title: str, query_author: str = "", query_year: str = "") -> list[dict]:
    hits: list[dict] = []
    if not html_text:
        return hits

    parts = html_text.split('<div class="gs_ri"')
    for part in parts[1:]:
        block = part.split('<div class="gs_fl"', 1)[0]
        title_match = re.search(r'<h3 class="gs_rt"[^>]*>(.*?)</h3>', block, re.S)
        if not title_match:
            continue

        title_fragment = title_match.group(1)
        link_match = re.search(r'<a href="([^"]+)"[^>]*>(.*?)</a>', title_fragment, re.S)
        if link_match:
            url = html.unescape(link_match.group(1))
            title = _strip_tags(link_match.group(2))
        else:
            url = ""
            title = _strip_tags(title_fragment)

        if not title:
            continue

        meta_match = re.search(r'<div class="gs_a"[^>]*>(.*?)</div>', block, re.S)
        snippet_match = re.search(r'<div class="gs_rs"[^>]*>(.*?)</div>', block, re.S)
        meta = _strip_tags(meta_match.group(1)) if meta_match else ""
        snippet = _strip_tags(snippet_match.group(1)) if snippet_match else ""

        authors = ""
        venue = ""
        if meta:
            meta_parts = [segment.strip() for segment in meta.split(" - ") if segment.strip()]
            if meta_parts:
                authors = meta_parts[0]
            if len(meta_parts) > 1:
                venue = meta_parts[1]

        year_match = _YEAR_RE.search(meta)
        year = year_match.group(0) if year_match else ""

        similarity = compute_scholar_similarity(
            query_title=query_title,
            candidate_title=title,
            query_author=query_author,
            candidate_authors=authors,
            query_year=query_year,
            candidate_year=year,
        )

        hits.append(
            {
                "title": title,
                "authors": authors,
                "venue": venue,
                "year": year,
                "url": urljoin(GOOGLE_SCHOLAR_URL, url) if url else "",
                "snippet": snippet,
                "similarity_score": similarity,
                "source": "google_scholar",
                "raw_metadata": meta,
            }
        )

    hits.sort(key=lambda item: item.get("similarity_score", 0.0), reverse=True)
    return hits


def compute_scholar_similarity(
    query_title: str,
    candidate_title: str,
    query_author: str = "",
    candidate_authors: str = "",
    query_year: str = "",
    candidate_year: str = "",
) -> float:
    norm_query = normalize_title(query_title)
    norm_candidate = normalize_title(candidate_title)
    if not norm_query or not norm_candidate:
        return 0.0

    sequence_score = SequenceMatcher(None, norm_query, norm_candidate).ratio()
    query_tokens = set(norm_query.split())
    candidate_tokens = set(norm_candidate.split())
    token_score = len(query_tokens & candidate_tokens) / max(len(query_tokens | candidate_tokens), 1)

    author_score = 0.0
    query_surnames = _extract_surnames(query_author)
    candidate_surnames = _extract_surnames(candidate_authors)
    if query_surnames and candidate_surnames:
        author_score = len(query_surnames & candidate_surnames) / max(len(query_surnames), len(candidate_surnames), 1)

    year_score = 0.0
    if query_year and candidate_year:
        try:
            diff = abs(int(query_year) - int(candidate_year))
            if diff == 0:
                year_score = 1.0
            elif diff == 1:
                year_score = 0.5
        except ValueError:
            year_score = 0.0

    weighted = (0.75 * max(sequence_score, token_score)) + (0.15 * author_score) + (0.10 * year_score)
    return round(min(weighted, 1.0), 3)


def search_google_scholar(
    title: str,
    author: str = "",
    year: str = "",
    max_results: int = 3,
    timeout: int = 15,
    session: requests.Session | None = None,
) -> list[dict]:
    if not title:
        return []

    query_parts = [f'"{title}"']
    first_author = (author.split(" and ")[0] if author else "").strip()
    if first_author:
        query_parts.append(first_author)
    if year:
        query_parts.append(str(year))

    params = {
        "hl": "en",
        "q": " ".join(part for part in query_parts if part),
        "num": max(max_results, 1),
    }
    if year:
        params["as_ylo"] = str(year)
        params["as_yhi"] = str(year)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    http = session or requests.Session()
    try:
        response = http.get(GOOGLE_SCHOLAR_URL, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException:
        return []

    hits = extract_scholar_hits(response.text, query_title=title, query_author=author, query_year=str(year or ""))
    return hits[:max_results]


def _extract_surnames(value: str) -> set[str]:
    if not value:
        return set()

    surnames: set[str] = set()
    for chunk in re.split(r"\band\b|,|;", value, flags=re.I):
        words = [word for word in normalize_title(chunk).split() if len(word) > 1]
        if words:
            surnames.add(words[-1])
    return surnames