import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import bibtexparser
from fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.google_scholar import search_google_scholar


current_dir = Path(__file__).parent
EXPORT_PATH = current_dir / "exported_entries"

mcp = FastMCP("reference_mcp")


class BibtexParser:
    def __init__(self):
        self.entries: List[Dict[str, Any]] = []

    def parse_string(self, bibtex_str: str) -> List[Dict[str, Any]]:
        """Parse BibTeX content and convert entries to plain dicts."""
        try:
            library = bibtexparser.parse_string(bibtex_str)
            raw_entries = library.entries
            self.entries = []
            for entry in raw_entries:
                parsed_entry = self.convert_entry(entry)
                if parsed_entry:
                    self.entries.append(parsed_entry)
            return self.entries
        except Exception as e:
            print(f"Error parsing BibTeX string: {e}")
            return []

    def convert_entry(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert one BibTeX entry to JSON-serializable dictionary."""

        def maybe_str(val: Any) -> str:
            if val is None:
                return ""
            if not isinstance(val, str):
                try:
                    return str(val)
                except Exception:
                    return ""
            return val

        try:
            return {
                "type": maybe_str(entry.get("ENTRYTYPE", "")).lower(),
                "id": maybe_str(entry.get("ID", "")),
                "title": maybe_str(entry.get("title", "")),
                "author": maybe_str(entry.get("author", "")),
                "year": maybe_str(entry.get("year", "")),
                "journal": maybe_str(entry.get("journal", "")),
                "booktitle": maybe_str(entry.get("booktitle", "")),
                "doi": maybe_str(entry.get("doi", "")),
                "pages": maybe_str(entry.get("pages", "")),
                "volume": maybe_str(entry.get("volume", "")),
                "number": maybe_str(entry.get("number", "")),
                "publisher": maybe_str(entry.get("publisher", "")),
            }
        except Exception as e:
            print(f"Error converting entry {entry.get('ID', '')}: {e}")
            return None

    def parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse BibTeX from a file path."""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
            return self.parse_string(content)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return []

    def export_json(self, output_path: str):
        """Export currently parsed entries to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            with open(output_path, "w", encoding="utf-8") as json_file:
                json.dump(self.entries, json_file, indent=4)
            print(f"Exported {len(self.entries)} entries to {output_path}")
        except Exception as e:
            print(f"Error exporting to JSON: {e}")


@mcp.tool()
def parse_bibtex(file_path: str) -> Dict[str, Any]:
    """Parse a local BibTeX file and return normalized entries.

    Args:
        file_path: path to .bib file
    """
    parser = BibtexParser()
    entries = parser.parse_file(file_path)

    EXPORT_PATH.mkdir(parents=True, exist_ok=True)
    parser.export_json(str(EXPORT_PATH / "parsed_entries.json"))

    return {
        "total_entries": len(entries),
        "entries": entries,
    }


@mcp.tool()
def google_scholar_search(
    title: str,
    author: str = "",
    year: str = "",
    max_results: int = 3,
) -> Dict[str, Any]:
    """Search Google Scholar and return normalized candidate matches.

    Args:
        title: reference title to query
        author: optional author string
        year: optional publication year
        max_results: number of result candidates to return
    """
    hits = search_google_scholar(
        title=title,
        author=author,
        year=year,
        max_results=max_results,
    )

    return {
        "query": {
            "title": title,
            "author": author,
            "year": year,
            "max_results": max_results,
        },
        "total_hits": len(hits),
        "results": hits,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")