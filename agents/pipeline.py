"""
Pipeline — LangGraph Orchestration
------------------------------------
Sequential graph:

    START → prepare → validate → save_outputs → END

State carries data between nodes; each node is a thin wrapper around an agent.

Usage:
    python pipeline.py --file bibtex/bibtex_files/references.bib
    python pipeline.py --url  https://example.com/refs.bib
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, TypedDict, Optional

sys.path.insert(0, str(Path(__file__).parent))

from langgraph.graph import StateGraph, START, END

from preparation_agent import PreparationAgent
from validation_agent import DBLPValidationAgent

 # CONFIG FILE PATHS
# Get paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
mcp_config_path = str(project_root / "server" / "mcp.json")
# ── Shared pipeline state ─────────────────────────────────────────────────────

class PipelineState(TypedDict):
    # Inputs (set before running)
    mcp_config_path: str
    bibtex_source:   str          # file path or URL
    source_type:     str          # "file" | "url"
    output_dir:      str

    # Set by preparation node
    prepared_entries:    list[dict]
    preparation_report:  dict
    warnings_by_entry:   dict

    # Set by validation node
    all_results:     list[dict]
    grouped_results: dict         # {valid, partially_valid, invalid}
    statistics:      dict
    markdown_report: str

    # Set by save_outputs node
    saved_files:     list[str]

    # Error propagation
    error:           Optional[str]


# ── Node: prepare ─────────────────────────────────────────────────────────────

async def prepare_node(state: PipelineState) -> dict:
    """
    Calls the PreparationAgent which uses the parse_bibtex MCP tool to
    ingest the BibTeX file and normalise entries.
    """
    print("\n[PIPELINE] Node: prepare")

    if state.get("error"):
        return {}   # propagate error, skip node

    try:
        agent = PreparationAgent(mcp_config_path)

        kwargs: dict[str, Any] = {}
        if state["source_type"] == "file":
            kwargs["file_path"] = state["bibtex_source"]
        else:
            kwargs["url"] = state["bibtex_source"]

        result = await agent.prepare(**kwargs)

        return {
            "prepared_entries":   result["entries"],
            "preparation_report": result["preparation_report"],
            "warnings_by_entry":  result["warnings_by_entry"],
        }

    except Exception as e:
        print(f"[PIPELINE] prepare_node ERROR: {e}")
        return {"error": str(e)}


# ── Node: validate ────────────────────────────────────────────────────────────

async def validate_node(state: PipelineState) -> dict:
    """
    Calls the DBLPValidationAgent to validate every prepared entry.
    """
    print("\n[PIPELINE] Node: validate")

    if state.get("error"):
        return {}

    entries = state.get("prepared_entries", [])
    if not entries:
        return {"error": "No entries to validate — preparation produced empty list"}

    try:
        agent = DBLPValidationAgent(state["mcp_config_path"])
        await agent.initialize()

        report = await agent.validate_entries(entries)

        return {
            "all_results":     report["all_results"],
            "grouped_results": report["grouped_results"],
            "statistics":      report["statistics"],
            "markdown_report": report["markdown_report"],
        }

    except Exception as e:
        print(f"[PIPELINE] validate_node ERROR: {e}")
        return {"error": str(e)}


# ── Node: save_outputs ────────────────────────────────────────────────────────

async def save_outputs_node(state: PipelineState) -> dict:
    """
    Writes all outputs to disk:
      - validation_report.json    (grouped results + all_results)
      - validation_statistics.json
      - validation_report.md
      - preparation_report.json
    """
    print("\n[PIPELINE] Node: save_outputs")

    if state.get("error"):
        print(f"[PIPELINE] Skipping save — upstream error: {state['error']}")
        return {}

    output_dir = Path(state["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    # 1. Full validation JSON (what the task requires)
    val_json = output_dir / "validation_report.json"
    with open(val_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "statistics":      state["statistics"],
                "grouped_results": state["grouped_results"],
                "all_results":     state["all_results"],
            },
            f, indent=2, ensure_ascii=False,
        )
    saved.append(str(val_json))
    print(f"  ✓ {val_json}")

    # 2. Statistics JSON
    stats_json = output_dir / "validation_statistics.json"
    with open(stats_json, "w", encoding="utf-8") as f:
        json.dump(state["statistics"], f, indent=2, ensure_ascii=False)
    saved.append(str(stats_json))
    print(f"  ✓ {stats_json}")

    # 3. Markdown report
    md_file = output_dir / "validation_report.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(state["markdown_report"])
    saved.append(str(md_file))
    print(f"  ✓ {md_file}")

    # 4. Preparation report
    prep_json = output_dir / "preparation_report.json"
    with open(prep_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "preparation_report": state.get("preparation_report", {}),
                "warnings_by_entry":  state.get("warnings_by_entry", {}),
            },
            f, indent=2, ensure_ascii=False,
        )
    saved.append(str(prep_json))
    print(f"  ✓ {prep_json}")

    return {"saved_files": saved}


# ── Build LangGraph ───────────────────────────────────────────────────────────

def build_pipeline() -> Any:
    """Construct and compile the LangGraph pipeline."""
    builder = StateGraph(PipelineState)

    builder.add_node("prepare",      prepare_node)
    builder.add_node("validate",     validate_node)
    builder.add_node("save_outputs", save_outputs_node)

    builder.add_edge(START,         "prepare")
    builder.add_edge("prepare",     "validate")
    builder.add_edge("validate",    "save_outputs")
    builder.add_edge("save_outputs", END)

    return builder.compile()


# ── Runner ────────────────────────────────────────────────────────────────────

async def run_pipeline(
    bibtex_source:   str,
    source_type:     str = "file",
    mcp_config_path: str | None = None,
    output_dir:      str | None = None,
) -> PipelineState:

    # project_root is the parent of agents directory
    project_root = Path(__file__).parent.parent

    initial_state: PipelineState = {
        "mcp_config_path": mcp_config_path or str(project_root / "server" / "mcp.json"),
        "bibtex_source":   bibtex_source,
        "source_type":     source_type,
        "output_dir":      output_dir or str(project_root / "evaluation"),
        # remaining fields populated by nodes
        "prepared_entries":   [],
        "preparation_report": {},
        "warnings_by_entry":  {},
        "all_results":        [],
        "grouped_results":    {},
        "statistics":         {},
        "markdown_report":    "",
        "saved_files":        [],
        "error":              None,
    }

    pipeline = build_pipeline()

    print("\n" + "="*60)
    print("BIBTEX VALIDATION PIPELINE")
    print("="*60)

    final_state = await pipeline.ainvoke(initial_state)

    # ── Console summary ───────────────────────────────────────────────────
    if final_state.get("error"):
        print(f"\n✗ Pipeline failed: {final_state['error']}")
    else:
        stats = final_state.get("statistics", {})
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        print(f"  Total entries  : {stats.get('total_entries', 0)}")
        print(f"  Valid          : {stats.get('valid_count', 0)} ({stats.get('valid_percentage', 0)}%)")
        print(f"  Partially valid: {stats.get('partially_valid_count', 0)} ({stats.get('partially_valid_percentage', 0)}%)")
        print(f"  Invalid        : {stats.get('invalid_count', 0)} ({stats.get('invalid_percentage', 0)}%)")
        print(f"  DBLP match rate: {stats.get('dblp_match_rate', 0)}%")
        print(f"\n  Saved files:")
        for f in final_state.get("saved_files", []):
            print(f"    - {f}")

    return final_state


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BibTeX Validation Pipeline")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Path to a .bib file")
    group.add_argument("--url",  help="URL to a .bib file")
    parser.add_argument("--mcp-config", help="Path to mcp.json")
    parser.add_argument("--output-dir", help="Output directory for reports")
    args = parser.parse_args()

    if args.file:
        source, stype = args.file, "file"
    else:
        source, stype = args.url, "url"

    asyncio.run(run_pipeline(
        bibtex_source=source,
        source_type=stype,
        mcp_config_path=args.mcp_config,
        output_dir=args.output_dir,
    ))


if __name__ == "__main__":
    main()