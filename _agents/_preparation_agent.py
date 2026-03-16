"""
Preparation Agent
-----------------
Uses the bibtex_mcp parse_bibtex tool to ingest a BibTeX file and
normalise every entry into the flat schema expected by the validation agent.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_mcp_adapters.client import MultiServerMCPClient


MANDATORY_FIELDS: dict[str, list[str]] = {
    "article": ["title", "author", "year", "journal"],
    "inproceedings": ["title", "author", "year", "booktitle"],
    "book": ["title", "author", "year", "publisher"],
    "phdthesis": ["title", "author", "year", "school"],
    "techreport": ["title", "author", "year", "institution"],
    "misc": ["title", "author", "year"],
}
DEFAULT_MANDATORY = ["title", "author", "year"]


def normalise_entry(raw: dict) -> tuple[dict, list[str]]:
    """Convert parse_bibtex output into flat schema plus warnings."""
    key = raw.get("key", "unknown")
    entry_type = raw.get("entry_type", "misc").lower()
    fields = raw.get("fields", {})

    title = fields.get("title", "").strip()
    author = fields.get("author", "").strip()
    year = str(fields.get("year", "")).strip()
    journal = (fields.get("journal") or fields.get("booktitle") or "").strip()
    publisher = fields.get("publisher", "").strip()
    doi = fields.get("doi", "").strip()

    entry = {
        "id": key,
        "type": entry_type,
        "title": title,
        "author": author,
        "year": year,
        "journal": journal,
        "publisher": publisher,
        "doi": doi,
        "raw_fields": fields,
    }

    warnings: list[str] = []
    required = MANDATORY_FIELDS.get(entry_type, DEFAULT_MANDATORY)
    for field in required:
        value = fields.get(field, "").strip() if fields.get(field) else ""
        if not value:
            warnings.append(f"Missing mandatory field: '{field}' for type '{entry_type}'")

    if year:
        try:
            y = int(year)
            if y < 1900 or y > 2030:
                warnings.append(f"Suspicious year value: {year}")
        except ValueError:
            warnings.append(f"Non-numeric year: '{year}'")

    return entry, warnings


class PreparationAgent:
    """Deterministic preparation wrapper around parse_bibtex MCP tool."""

    def __init__(self, mcp_config_path: str):
        self.mcp_config_path = mcp_config_path

    async def prepare(self, file_path: str | None = None, url: str | None = None) -> dict[str, Any]:
        if not file_path and not url:
            raise ValueError("Provide either file_path or url")

        print(f"\n{'='*60}")
        print("PREPARATION AGENT")
        print(f"{'='*60}")
        source = file_path or url
        print(f"Source : {source}\n")

        with open(self.mcp_config_path, "r") as f:
            config = json.load(f)

        mcp_servers_config = config.get("mcpServers", config)
        client = MultiServerMCPClient(mcp_servers_config)

        tools = await client.get_tools(server_name="bibtex_mcp")
        tools_map = {t.name: t for t in tools}

        parse_tool = tools_map.get("parse_bibtex")
        if not parse_tool:
            raise RuntimeError(
                "parse_bibtex tool not found in bibtex_mcp server. "
                f"Available tools: {list(tools_map.keys())}"
            )

        kwargs: dict[str, Any] = {}
        if file_path:
            kwargs["file_path"] = file_path
        else:
            kwargs["url"] = url

        raw_result = await parse_tool.ainvoke(kwargs)

        if isinstance(raw_result, list) and len(raw_result) > 0:
            content = raw_result[0]
            if isinstance(content, dict) and "text" in content:
                raw_result = content["text"]

        if isinstance(raw_result, str):
            raw_result = json.loads(raw_result)

        if isinstance(raw_result, list):
            raw_entries = raw_result
            total_raw = len(raw_entries)
        elif isinstance(raw_result, dict):
            if raw_result.get("error"):
                raise RuntimeError(f"parse_bibtex error: {raw_result['error']}")
            raw_entries = raw_result.get("entries", [])
            total_raw = raw_result.get("total_entries", len(raw_entries))
        else:
            raise RuntimeError(f"Unexpected result type: {type(raw_result)}")

        prepared: list[dict] = []
        warnings_by_entry: dict[str, list[str]] = {}
        total_warnings = 0

        for raw in raw_entries:
            entry, warnings = normalise_entry(raw)
            prepared.append(entry)
            if warnings:
                warnings_by_entry[entry["id"]] = warnings
                total_warnings += len(warnings)

        type_counts: dict[str, int] = {}
        for e in prepared:
            type_counts[e["type"]] = type_counts.get(e["type"], 0) + 1

        report = {
            "source": source,
            "total_raw": total_raw,
            "total_prepared": len(prepared),
            "entry_types": type_counts,
            "entries_with_warnings": len(warnings_by_entry),
            "total_warnings": total_warnings,
            "warning_summary": warnings_by_entry,
        }

        return {
            "entries": prepared,
            "preparation_report": report,
            "total_entries": len(prepared),
            "warnings_by_entry": warnings_by_entry,
        }


async def main():
    project_root = Path(__file__).parent.parent
    mcp_config_path = str(project_root / "server" / "mcp.json")
    bibtex_file = str(project_root / "bibtex" / "bibtex_files" / "references.bib")

    agent = PreparationAgent(mcp_config_path)
    result = await agent.prepare(file_path=bibtex_file)
    print(json.dumps(result["preparation_report"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
