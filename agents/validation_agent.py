"""
DBLP Validation Agent
---------------------
Validates prepared BibTeX entries using DBLP MCP tools.

Tools used:
  - fuzzy_title_search       → primary lookup (similarity_threshold=0.20, sort by score)
  - get_author_publications  → fallback when title score is weak

Classification:
  valid          confidence >= 0.75 AND no critical issues
  partially_valid confidence >= 0.35 OR (dblp found AND ≤2 issues)
  invalid        everything else / not found in DBLP

Outputs (all written by pipeline.py):
  evaluation/validation_report.json
  evaluation/validation_report.md
  evaluation/validation_statistics.json
"""

import asyncio
import os
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from utils.google_scholar import search_google_scholar


# ── Thresholds ────────────────────────────────────────────────────────────────
#
# IMPORTANT: real-world testing shows fuzzy_title_search requires a LOW threshold
# (0.20) to return any results. The tool sorts by similarity score, so we filter
# ourselves on the returned score rather than relying on the threshold to filter.
#
SEARCH_THRESHOLD   = 0.20   # passed to the tool (must be low to get candidates)
MIN_ACCEPT_SCORE   = 0.75   # minimum score we accept as a genuine title match
AUTHOR_THRESHOLD   = 0.50   # passed to get_author_publications
YEAR_TOLERANCE     = 1      # ±1 year is still a soft issue, not hard error
SCHOLAR_ACCEPT_SCORE = 0.85

# Confidence weights (must sum to 1.0)
W_TITLE  = 0.50
W_AUTHOR = 0.25
W_YEAR   = 0.15
W_VENUE  = 0.10


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _norm(text: Any) -> str:
    return str(text).lower().strip() if text else ""


def _parse_dblp_text_response(text: str) -> list[dict]:
    """
    Parse DBLP MCP tool's formatted text response into structured data.
    
    Expected format:
    Found N publications with similar titles:

    1. Title. [Similarity: 0.99]
       Authors: Author1, Author2
       Venue: Venue (Year)
       
    2. Title2. [Similarity: 0.85]
       ...
    """
    if not text or "Found" not in text:
        return []
    
    results = []
    lines = text.split("\n")
    
    current_item = None
    for line in lines:
        line = line.strip()
        if not line:
            if current_item:
                results.append(current_item)
                current_item = None
            continue
        
        # Check if this is a title line: "N. Title [Similarity: X.XX]"
        if line and line[0].isdigit() and "." in line and "[Similarity:" in line:
            if current_item:
                results.append(current_item)
            
            # Parse title and similarity
            parts = line.split("[Similarity:")
            title = parts[0].split(". ", 1)
            if len(title) > 1:
                title = title[1].strip()
            else:
                title = line
            
            similarity_str = parts[1].split("]")[0].strip()
            try:
                similarity = float(similarity_str)
            except:
                similarity = 0.0
            
            current_item = {
                "title": title,
                "similarity_score": similarity,
                "similarity": similarity,
                "authors": "",
                "venue": "",
                "year": "",
            }
        
        elif current_item and line.startswith("Authors:"):
            current_item["authors"] = line.replace("Authors:", "").strip()
        
        elif current_item and line.startswith("Venue:"):
            venue_str = line.replace("Venue:", "").strip()
            # Extract year if present: "Venue (Year)"
            if "(" in venue_str and ")" in venue_str:
                venue_part, year_part = venue_str.rsplit("(", 1)
                current_item["venue"] = venue_part.strip()
                current_item["year"] = year_part.rstrip(")").strip()
            else:
                current_item["venue"] = venue_str
    
    if current_item:
        results.append(current_item)
    
    return results


def _parse_similarity(hit: dict) -> float:
    """
    fuzzy_title_search returns similarity as a float in the dict.
    Guard against it being a string like '0.99'.
    """
    raw = hit.get("similarity_score", hit.get("similarity", 0.0))
    try:
        return float(raw)
    except (TypeError, ValueError):
        # Try to extract from string "Similarity: 0.99"
        m = re.search(r"(\d+\.?\d*)", str(raw))
        return float(m.group(1)) if m else 0.0


def _score_entry(
    entry: dict,
    hit: dict,
    source_label: str = "DBLP",
    min_title_score: float = MIN_ACCEPT_SCORE,
) -> tuple[float, list[str], list[str]]:
    """
    Deterministic field-by-field comparison.
    Returns (confidence, issues, suggestions).
    """
    issues:      list[str] = []
    suggestions: list[str] = []
    confidence = 0.0

    title_score = _parse_similarity(hit)

    # ── Title ─────────────────────────────────────────────────────────────
    if title_score >= min_title_score:
        confidence += W_TITLE * title_score
    else:
        issues.append(f"Title similarity too low ({title_score:.2f} < {min_title_score})")
        suggestions.append(f'Update title to: "{hit.get("title", "")}"')

    # ── Year ──────────────────────────────────────────────────────────────
    entry_year = _norm(entry.get("year"))
    dblp_year  = _norm(hit.get("year", ""))
    if entry_year and dblp_year:
        try:
            diff = abs(int(entry_year) - int(dblp_year))
            if diff == 0:
                confidence += W_YEAR
            elif diff <= YEAR_TOLERANCE:
                confidence += W_YEAR * 0.5
                issues.append(f"Year off by {diff} (entry={entry_year}, {source_label}={dblp_year})")
                suggestions.append(f"Change year from {entry_year} to {dblp_year}")
            else:
                issues.append(f"Wrong year (entry={entry_year}, {source_label}={dblp_year})")
                suggestions.append(f"Change year from {entry_year} to {dblp_year}")
        except ValueError:
            issues.append(f"Non-numeric year: '{entry_year}'")
    elif not entry_year:
        issues.append("Missing year")
        suggestions.append(f"Add year: {dblp_year}" if dblp_year else "Add year field")

    # ── Authors ───────────────────────────────────────────────────────────
    dblp_authors_raw = hit.get("authors", "")
    # authors may be a list or a string
    if isinstance(dblp_authors_raw, list):
        dblp_authors_str = ", ".join(dblp_authors_raw)
        dblp_lastnames   = {a.split()[-1].lower() for a in dblp_authors_raw if a.split()}
    else:
        dblp_authors_str = str(dblp_authors_raw)
        dblp_lastnames   = {w.lower() for w in dblp_authors_str.split() if len(w) > 2}

    entry_author = _norm(entry.get("author", ""))
    if entry_author and dblp_lastnames:
        entry_words = set(entry_author.replace(",", " ").split())
        overlap = len(dblp_lastnames & entry_words) / max(len(dblp_lastnames), 1)
        confidence += W_AUTHOR * overlap
        if overlap < 0.5:
            issues.append(f"Author names differ significantly from {source_label}")
            suggestions.append(f"Update authors to: {dblp_authors_str}")
    elif not entry_author:
        issues.append("Missing author field")
        suggestions.append(f"Add authors: {dblp_authors_str}" if dblp_authors_str else "Add author field")

    # ── Venue ─────────────────────────────────────────────────────────────
    entry_venue = _norm(entry.get("journal") or entry.get("booktitle") or "")
    dblp_venue  = _norm(hit.get("venue", ""))
    if entry_venue and dblp_venue:
        if entry_venue in dblp_venue or dblp_venue in entry_venue:
            confidence += W_VENUE
        else:
            issues.append(f"Venue mismatch (entry='{entry_venue}', {source_label}='{dblp_venue}')")
            suggestions.append(f'Update venue to: "{hit.get("venue", "")}"')
    elif not entry_venue:
        issues.append("Missing journal/booktitle")
        suggestions.append(f'Add venue: "{hit.get("venue", "")}"' if dblp_venue else "Add venue field")

    return round(min(confidence, 1.0), 3), issues, suggestions


def _classify(confidence: float, issues: list[str], dblp_found: bool) -> str:
    if not dblp_found:
        return "invalid"
    if confidence >= 0.75 and not issues:
        return "valid"
    if confidence >= 0.35 or len(issues) <= 2:
        return "partially_valid"
    return "invalid"


# ── Validation Agent ──────────────────────────────────────────────────────────

class DBLPValidationAgent:
    """
    Validates prepared BibTeX entries via DBLP MCP tools.
    Scoring/classification: deterministic Python.
    LLM: one-sentence explanation only.
    """

    def __init__(self, mcp_config_path: str):
        self.mcp_config_path = mcp_config_path
        self.llm = None

    async def initialize(self):
        self.llm = ChatOllama(
            model="qwen3-coder:480b-cloud",
            base_url="https://ollama.com",
            temperature=0.1,
            #add the API key for authentication
            client_kwargs={
                "headers": {"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"}
            }
        )
        print("✓ Validation agent initialised")

    # ── MCP tool callers ──────────────────────────────────────────────────

    async def _title_search(self, tools: dict, title: str, year: str) -> list[dict]:
        tool = tools.get("fuzzy_title_search")
        if not tool or not title:
            return []

        kwargs: dict[str, Any] = {
            "title":                title,
            "similarity_threshold": SEARCH_THRESHOLD,
            "max_results":          5,
            "include_bibtex":       False,
        }
        if year:
            try:
                y = int(year)
                kwargs["year_from"] = y - YEAR_TOLERANCE
                kwargs["year_to"]   = y + YEAR_TOLERANCE
            except ValueError:
                pass

        try:
            result = await tool.ainvoke(kwargs)
            
            # Unwrap LangChain message structure: [{"type": "text", "text": "..."}]
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                if "text" in result[0] and isinstance(result[0]["text"], str):
                    text_content = result[0]["text"]
                    # Try to parse as JSON first
                    try:
                        result = json.loads(text_content)
                    except (json.JSONDecodeError, TypeError):
                        # If not JSON, parse as formatted text response
                        result = _parse_dblp_text_response(text_content)
            
            # Parse JSON string if needed
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except (json.JSONDecodeError, TypeError):
                    result = _parse_dblp_text_response(result)
            
            # Extract results from various possible structures
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                return result.get("results", result.get("publications", result.get("entries", [])))
            return []
        except Exception as e:
            print(f"    [fuzzy_title_search error] {e}")
            return []

    async def _author_search(self, tools: dict, author: str, title: str) -> dict | None:
        tool = tools.get("get_author_publications")
        if not tool or not author:
            return None

        first_author = author.split(" and ")[0].split(",")[0].strip()
        try:
            result = await tool.ainvoke({
                "author_name":          first_author,
                "similarity_threshold": AUTHOR_THRESHOLD,
                "max_results":          15,
                "include_bibtex":       False,
            })
            
            # Unwrap LangChain message structure: [{"type": "text", "text": "..."}]
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
               if "text" in result[0] and isinstance(result[0]["text"], str):
                    text_content = result[0]["text"]
                    # Try to parse as JSON first
                    try:
                        result = json.loads(text_content)
                    except (json.JSONDecodeError, TypeError):
                        # If not JSON, parse as formatted text response
                        result = _parse_dblp_text_response(text_content)
            
            # Parse JSON string if needed
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except (json.JSONDecodeError, TypeError):
                    result = _parse_dblp_text_response(result)
            
            pubs = result.get("publications", []) if isinstance(result, dict) else result if isinstance(result, list) else []
        except Exception as e:
            print(f"    [get_author_publications error] {e}")
            return None

        if not pubs:
            return None

        # Find best title match among author's pubs
        norm_title  = _norm(title)
        title_words = set(norm_title.split())
        best_score  = 0.0
        best_pub    = None

        for pub in pubs:
            pub_words = set(_norm(pub.get("title", "")).split())
            if not pub_words:
                continue
            overlap = len(title_words & pub_words) / max(len(title_words), 1)
            if overlap > best_score:
                best_score = overlap
                best_pub   = pub

        if best_pub and best_score >= 0.5:
            best_pub = dict(best_pub)
            best_pub["similarity_score"] = round(best_score, 3)
            return best_pub
        return None

    # ── Single entry ──────────────────────────────────────────────────────

    async def _validate_entry(self, tools: dict, entry: dict) -> dict:
        title  = entry.get("title", "")
        author = entry.get("author", "")
        year   = str(entry.get("year", ""))

        # Step 1 — title search
        hits       = await self._title_search(tools, title, year)
        best_hit   = hits[0] if hits else None
        best_score = _parse_similarity(best_hit) if best_hit else 0.0
        scholar_hit = None
        scholar_hits: list[dict] = []
        match_source = ""

        # Step 2 — author fallback if title score is weak
        if best_score < MIN_ACCEPT_SCORE and author:
            author_hit = await self._author_search(tools, author, title)
            if author_hit:
                a_score = _parse_similarity(author_hit)
                if a_score > best_score:
                    best_hit   = author_hit
                    best_score = a_score

        dblp_found = best_hit is not None and best_score >= MIN_ACCEPT_SCORE
        if dblp_found:
            match_source = "dblp"

        # Step 3 — Google Scholar fallback if DBLP did not find a strong match
        if not dblp_found:
            scholar_hits = search_google_scholar(title=title, author=author, year=year, max_results=3)
            if scholar_hits:
                candidate = scholar_hits[0]
                if candidate.get("similarity_score", 0.0) >= SCHOLAR_ACCEPT_SCORE:
                    scholar_hit = candidate
                    match_source = "google_scholar"

        matched_hit = best_hit if dblp_found else scholar_hit
        matched = matched_hit is not None

        # Step 4 — score
        if dblp_found:
            confidence, issues, suggestions = _score_entry(entry, best_hit, source_label="DBLP")
        elif scholar_hit:
            confidence, issues, suggestions = _score_entry(
                entry,
                scholar_hit,
                source_label="Google Scholar",
                min_title_score=SCHOLAR_ACCEPT_SCORE,
            )
        else:
            issues      = []
            suggestions = []
            for field, label in [("title","title"),("author","author"),
                                  ("year","year"),("journal","journal/booktitle")]:
                if not entry.get(field):
                    issues.append(f"Missing {label}")
                    suggestions.append(f"Add {label} field")
            issues.append("Entry not found in DBLP or Google Scholar")
            confidence = 0.0

        status = _classify(confidence, issues, matched)

        # Step 5 — LLM explanation
        explanation = await self._explain(entry, matched_hit, match_source, status, confidence, issues)

        return {
            "entry_id":    entry.get("id", "unknown"),
            "entry_type":  entry.get("type", ""),
            "title":       title,
            "author":      author,
            "year":        year,
            "journal":     entry.get("journal", ""),
            "status":      status,
            "confidence":  confidence,
            "match_source": match_source,
            "dblp_match":  {
                "title":            best_hit.get("title", ""),
                "authors":          str(best_hit.get("authors", "")),
                "year":             str(best_hit.get("year", "")),
                "venue":            best_hit.get("venue", ""),
                "similarity_score": round(best_score, 3),
            } if dblp_found else {},
            "scholar_match": {
                "title":            scholar_hit.get("title", ""),
                "authors":          str(scholar_hit.get("authors", "")),
                "year":             str(scholar_hit.get("year", "")),
                "venue":            scholar_hit.get("venue", ""),
                "url":              scholar_hit.get("url", ""),
                "snippet":          scholar_hit.get("snippet", ""),
                "similarity_score": round(float(scholar_hit.get("similarity_score", 0.0)), 3),
            } if scholar_hit else {},
            "issues":      issues,
            "suggestions": suggestions,
            "explanation": explanation,
        }

    async def _explain(
        self,
        entry: dict,
        hit: dict | None,
        match_source: str,
        status: str,
        confidence: float,
        issues: list[str],
    ) -> str:
        prompt = (
            f"BibTeX entry \"{entry.get('title','')}\" was classified as '{status}' "
            f"with confidence {confidence:.0%}. "
            f"Issues: {', '.join(issues) if issues else 'none'}. "
            f"Match source: {match_source or 'none'}. "
            f"Reference match: {'found' if hit else 'not found'}. "
            "Write ONE concise sentence explaining this classification."
            f'Write ONE sentence explaining what the author should fix.'
        )
        try:
            resp = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return resp.content.strip()
        except Exception:
            return f"Classified as {status} (confidence {confidence:.0%})."

    # ── Batch ─────────────────────────────────────────────────────────────

    async def validate_entries(self, entries: list[dict]) -> dict:
        print(f"\n{'='*60}")
        print(f"DBLP VALIDATION AGENT — {len(entries)} entries")
        print(f"{'='*60}\n")

        all_results: list[dict] = []

        # Load config file
        with open(self.mcp_config_path, "r") as f:
            config = json.load(f)
        
        # MultiServerMCPClient expects the mcpServers dict
        mcp_servers_config = config.get("mcpServers", config)
        client = MultiServerMCPClient(mcp_servers_config)
        
        # Get tools from mcp-dblp server
        tools_list = await client.get_tools(server_name="mcp-dblp")
        tools      = {t.name: t for t in tools_list}
        print(f"✓ DBLP tools loaded: {list(tools.keys())}\n")

        for i, entry in enumerate(entries, 1):
                eid = entry.get("id", "unknown")
                try:
                    r = await self._validate_entry(tools, entry)
                    all_results.append(r)
                    print(
                        f"  [{i:>3}/{len(entries)}] {eid:<30} "
                        f"{r['status']:<16} conf={r['confidence']:.0%}"
                    )
                except Exception as e:
                    err = self._error_result(entry, str(e))
                    all_results.append(err)
                    print(f"  [{i:>3}/{len(entries)}] {eid:<30} ERROR: {str(e)[:55]}")

        grouped = self._group(all_results)
        stats   = self._statistics(all_results, grouped)
        md      = self._markdown(grouped, stats)

        print(
            f"\n✓ Valid: {stats['valid_count']}  "
            f"Partial: {stats['partially_valid_count']}  "
            f"Invalid: {stats['invalid_count']}\n"
        )

        return {
            "all_results":     all_results,
            "grouped_results": grouped,
            "statistics":      stats,
            "markdown_report": md,
            "total_entries":   len(all_results),
        }

    # ── Helpers ───────────────────────────────────────────────────────────

    def _group(self, results: list[dict]) -> dict:
        g: dict[str, list] = {"valid": [], "partially_valid": [], "invalid": []}
        for r in results:
            g.setdefault(r.get("status", "invalid"), []).append(r)
        return g

    def _statistics(self, all_results: list[dict], grouped: dict) -> dict:
        total   = len(all_results)
        valid   = grouped["valid"]
        partial = grouped["partially_valid"]
        invalid = grouped["invalid"]

        all_issues   = [i for r in all_results for i in r.get("issues", [])]
        issue_counts = Counter(all_issues)

        def avg(lst: list[dict]) -> float:
            return round(sum(r.get("confidence", 0) for r in lst) / len(lst), 3) if lst else 0.0

        return {
            "total_entries":              total,
            "valid_count":                len(valid),
            "partially_valid_count":      len(partial),
            "invalid_count":              len(invalid),
            "valid_percentage":           round(len(valid)   / total * 100, 1) if total else 0,
            "partially_valid_percentage": round(len(partial) / total * 100, 1) if total else 0,
            "invalid_percentage":         round(len(invalid) / total * 100, 1) if total else 0,
            "dblp_match_rate":            round(
                sum(1 for r in all_results if r.get("dblp_match")) / total * 100, 1
            ) if total else 0,
            "scholar_match_rate":         round(
                sum(1 for r in all_results if r.get("scholar_match")) / total * 100, 1
            ) if total else 0,
            "average_confidence_overall": avg(all_results),
            "average_confidence_valid":   avg(valid),
            "average_confidence_partial": avg(partial),
            "total_issues_found":         len(all_issues),
            "top_5_issues":               [i for i, _ in issue_counts.most_common(5)],
            "most_common_issue":          issue_counts.most_common(1)[0][0] if issue_counts else "None",
        }

    def _markdown(self, grouped: dict, stats: dict) -> str:
        md  = "# BibTeX Validation Report\n\n"
        md += "## Summary\n\n"
        md += "| Metric | Value |\n|---|---|\n"
        for label, key in [
            ("Total Entries",       "total_entries"),
            ("✓ Valid",             "valid_count"),
            ("⚠ Partially Valid",   "partially_valid_count"),
            ("✗ Invalid",           "invalid_count"),
            ("DBLP Match Rate",     "dblp_match_rate"),
            ("Scholar Match Rate",  "scholar_match_rate"),
            ("Avg Confidence (valid)", "average_confidence_valid"),
            ("Total Issues Found",  "total_issues_found"),
            ("Most Common Issue",   "most_common_issue"),
        ]:
            v = stats[key]
            pct_key = key.replace("_count", "_percentage")
            pct = f" ({stats[pct_key]}%)" if pct_key in stats else ""
            md += f"| {label} | {v}{pct} |\n"
        md += "\n"

        for section, label, icon in [
            ("valid",          "Valid Citations",                   "✓"),
            ("partially_valid","Partially Valid — Needs Fixes",     "⚠"),
            ("invalid",        "Invalid — Not Found in DBLP or Google Scholar",      "✗"),
        ]:
            entries = grouped[section]
            if not entries:
                continue
            md += f"## {icon} {label}\n\n"
            for e in entries:
                md += f"### `{e['entry_id']}`\n\n"
                md += f"**Status**: `{e['status']}` | **Confidence**: {e['confidence']:.0%}\n\n"
                if e.get("match_source"):
                    md += f"- **Match Source**: {e['match_source']}\n"
                for f in ("title","author","year","journal"):
                    if e.get(f):
                        md += f"- **{f.capitalize()}**: {e[f]}\n"
                if e.get("issues"):
                    md += "\n**Issues**:\n" + "".join(f"- {i}\n" for i in e["issues"])
                if e.get("suggestions"):
                    md += "\n**Suggestions**:\n" + "".join(f"- {s}\n" for s in e["suggestions"])
                if e.get("dblp_match") and e["dblp_match"].get("title"):
                    m = e["dblp_match"]
                    md += (
                        f"\n**DBLP Match** (score={m['similarity_score']:.0%}):\n"
                        f"- Title: {m['title']}\n"
                        f"- Year: {m['year']} | Venue: {m['venue']}\n"
                    )
                if e.get("scholar_match") and e["scholar_match"].get("title"):
                    m = e["scholar_match"]
                    md += (
                        f"\n**Google Scholar Match** (score={m['similarity_score']:.0%}):\n"
                        f"- Title: {m['title']}\n"
                        f"- Year: {m['year']} | Venue: {m['venue']}\n"
                    )
                    if m.get("url"):
                        md += f"- URL: {m['url']}\n"
                if e.get("explanation"):
                    md += f"\n> {e['explanation']}\n"
                md += "\n---\n\n"
        return md

    @staticmethod
    def _error_result(entry: dict, error: str) -> dict:
        return {
            "entry_id":    entry.get("id", "unknown"),
            "entry_type":  entry.get("type", ""),
            "title":       entry.get("title", ""),
            "author":      entry.get("author", ""),
            "year":        str(entry.get("year", "")),
            "journal":     entry.get("journal", ""),
            "status":      "invalid",
            "confidence":  0.0,
            "match_source": "",
            "dblp_match":  {},
            "scholar_match": {},
            "issues":      [f"Validation error: {error}"],
            "suggestions": ["Retry validation or check entry format"],
            "explanation": error,
        }