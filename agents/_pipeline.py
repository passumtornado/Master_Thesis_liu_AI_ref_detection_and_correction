"""
LLM-Driven Pipeline — LangGraph Orchestration
-----------------------------------------------
Sequential graph with LLM-based validation:

    START → prepare → validate (LLM) → save_outputs → END

State carries data between nodes; each node is a thin wrapper around an agent.

Features:
  - PreparationAgent: Parses & normalizes BibTeX entries
  - LLMValidationAgent: AI-driven scoring and classification
  - Structured markdown reports and JSON statistics

Usage:
    python _pipeline.py --file bibtex/bibtex_files/references.bib
    python _pipeline.py --url  https://example.com/refs.bib
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
from _validate_agent import LLMValidationAgent
from correction_agent import CorrectionAgent
from evaluation_agent import EvaluationAgent


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

    # Set by validation node (LLM-driven)
    markdown_report: str
    raw_data:        list[dict]

    # Set by correction node
    corrected_entries:      list[dict]
    corrections:            list[dict]
    corrections_summary:    str

    # Set by evaluation node
    evaluation_metrics:     dict

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
        agent = PreparationAgent(state["mcp_config_path"])

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


# ── Node: validate (LLM-driven) ───────────────────────────────────────────────

async def validate_node(state: PipelineState) -> dict:
    """
    Calls the LLMValidationAgent for AI-driven scoring and classification.
    LLMValidationAgent returns: {markdown_report, raw_data, total_entries}
    """
    print("\n[PIPELINE] Node: validate (LLM-driven)")

    if state.get("error"):
        return {}

    entries = state.get("prepared_entries", [])
    if not entries:
        return {"error": "No entries to validate — preparation produced empty list"}

    try:
        agent = LLMValidationAgent(state["mcp_config_path"])

        report = await agent.validate_entries(entries)

        return {
            "markdown_report": report["markdown_report"],
            "raw_data": report["raw_data"],
        }

    except Exception as e:
        print(f"[PIPELINE] validate_node ERROR: {e}")
        return {"error": str(e)}


# ── Node: correction ──────────────────────────────────────────────────────────

async def correction_node(state: PipelineState) -> dict:
    """
    Calls the CorrectionAgent to auto-fix entries using validation/DBLP data.
    CorrectionAgent returns: {corrected_entries, corrections, correction_summary, saved_files}
    """
    print("\n[PIPELINE] Node: correction (Auto-fix entries)")

    if state.get("error"):
        return {}

    raw_data = state.get("raw_data", [])
    if not raw_data:
        return {"error": "No validation data to correct — validation produced empty list"}

    try:
        output_dir = Path(state["output_dir"]) / "corrections"
        agent = CorrectionAgent(str(output_dir))

        result = await agent.correct_entries(
            raw_data=raw_data,
            validation_markdown=state.get("markdown_report", "")
        )

        return {
            "corrected_entries": result["corrected_entries"],
            "corrections": result.get("corrections", []),
            "corrections_summary": result["correction_summary"],
        }

    except Exception as e:
        print(f"[PIPELINE] correction_node ERROR: {e}")
        return {"error": str(e)}


# ── Node: evaluation ──────────────────────────────────────────────────────────

async def evaluation_node(state: PipelineState) -> dict:
    """
    Calls the EvaluationAgent to calculate recall, precision, F1, and field accuracy.
    EvaluationAgent returns: {overall_metrics, field_accuracy, detailed_results}
    """
    print("\n[PIPELINE] Node: evaluation (Calculate metrics)")

    if state.get("error"):
        return {}

    raw_data = state.get("raw_data", [])
    corrections = state.get("corrections", [])
    if not raw_data or not corrections:
        print("  [PIPELINE] Skipping evaluation — no data from validation or correction")
        return {}

    try:
        output_dir = Path(state["output_dir"]) / "evaluation"
        agent = EvaluationAgent(str(output_dir))

        result = await agent.evaluate(
            raw_data=raw_data,
            corrections=corrections,
        )

        return {
            "evaluation_metrics": result["overall_metrics"],
        }

    except Exception as e:
        print(f"[PIPELINE] evaluation_node ERROR: {e}")
        return {"error": str(e)}


# ── Node: save_outputs ────────────────────────────────────────────────────────

async def save_outputs_node(state: PipelineState) -> dict:
    """
    Writes all outputs to disk:
      - validation_report.md
      - validation_report.json
      - preparation_report.json
      - corrections_report.md
      - corrections_metadata.json
      - evaluation_report.md
      - evaluation_metrics.json
    """
    print("\n[PIPELINE] Node: save_outputs")

    if state.get("error"):
        print(f"[PIPELINE] Skipping save — upstream error: {state['error']}")
        return {}

    output_dir = Path(state["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    # 1. Validation report (LLM-generated markdown)
    md_file = output_dir / "validation_report.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(state.get("markdown_report", ""))
    saved.append(str(md_file))
    print(f"  ✓ {md_file}")

    # 2. Validation raw data (DBLP hits)
    raw_json = output_dir / "validation_raw_data.json"
    with open(raw_json, "w", encoding="utf-8") as f:
        json.dump(state.get("raw_data", []), f, indent=2, ensure_ascii=False)
    saved.append(str(raw_json))
    print(f"  ✓ {raw_json}")

    # 3. Preparation report (from BibTeX parsing)
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

    # 4. Corrections summary (from CorrectionAgent)
    if state.get("corrections_summary"):
        corr_dir = output_dir / "corrections"
        corr_md = corr_dir / "corrections_summary.md"
        with open(corr_md, "w", encoding="utf-8") as f:
            f.write(state["corrections_summary"])
        saved.append(str(corr_md))
        print(f"  ✓ {corr_md}")

    # 5. Evaluation metrics (from EvaluationAgent)
    if state.get("evaluation_metrics"):
        eval_dir = output_dir / "evaluation"
        eval_json = eval_dir / "evaluation_metrics.json"
        with open(eval_json, "w", encoding="utf-8") as f:
            json.dump(state["evaluation_metrics"], f, indent=2, ensure_ascii=False)
        saved.append(str(eval_json))
        print(f"  ✓ {eval_json}")

    return {"saved_files": saved}


# ── Build LangGraph ───────────────────────────────────────────────────────────

def build_pipeline() -> Any:
    """Construct and compile the LangGraph pipeline."""
    builder = StateGraph(PipelineState)

    builder.add_node("prepare",      prepare_node)
    builder.add_node("validate",     validate_node)
    builder.add_node("correction",   correction_node)
    builder.add_node("evaluation",   evaluation_node)
    builder.add_node("save_outputs", save_outputs_node)

    builder.add_edge(START,         "prepare")
    builder.add_edge("prepare",     "validate")
    builder.add_edge("validate",    "correction")
    builder.add_edge("correction",  "evaluation")
    builder.add_edge("evaluation",  "save_outputs")
    builder.add_edge("save_outputs", END)

    return builder.compile()


# ── Runner ────────────────────────────────────────────────────────────────────

async def run_pipeline(
    bibtex_source:   str,
    source_type:     str = "file",
    mcp_config_path: str | None = None,
    output_dir:      str | None = None,
) -> PipelineState:
    """
    Execute the complete validation pipeline.
    """

    # project_root is the parent of agents directory
    project_root = Path(__file__).parent.parent

    initial_state: PipelineState = {
        "mcp_config_path": mcp_config_path or str(project_root / "server" / "mcp.json"),
        "bibtex_source":   bibtex_source,
        "source_type":     source_type,
        "output_dir":      output_dir or str(project_root / "evaluation"),
        # remaining fields populated by nodes
        "prepared_entries":     [],
        "preparation_report":   {},
        "warnings_by_entry":    {},
        "markdown_report":      "",
        "raw_data":             [],
        "corrected_entries":    [],
        "corrections":          [],
        "corrections_summary":  "",
        "evaluation_metrics":   {},
        "saved_files":          [],
        "error":                None,
    }

    pipeline = build_pipeline()

    print("\n" + "="*60)
    print("BIBTEX VALIDATION PIPELINE (LLM-DRIVEN)")
    print("="*60)

    final_state = await pipeline.ainvoke(initial_state)

    # ── Console summary ───────────────────────────────────────────────────
    if final_state.get("error"):
        print(f"\n✗ Pipeline failed: {final_state['error']}")
    else:
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        print(f"  Total entries processed: {len(final_state.get('raw_data', []))}")
        print(f"  Markdown report generated")
        print(f"\n  Saved files:")
        for f in final_state.get("saved_files", []):
            print(f"    - {f}")

    return final_state


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BibTeX Validation Pipeline (LLM-Driven)")
    group  = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--file", help="Path to a .bib file")
    group.add_argument("--url",  help="URL to a .bib file")
    parser.add_argument("--mcp-config", help="Path to mcp.json")
    parser.add_argument("--output-dir", help="Output directory for reports")
    args = parser.parse_args()

    if args.file:
        source, stype = args.file, "file"
    elif args.url:
        source, stype = args.url, "url"
    else:
        # Default to references.bib if no argument provided
        project_root = Path(__file__).parent.parent
        source = str(project_root / "bibtex" / "bibtex_files" / "references.bib")
        stype = "file"

    asyncio.run(run_pipeline(
        bibtex_source=source,
        source_type=stype,
        mcp_config_path=args.mcp_config,
        output_dir=args.output_dir,
    ))


if __name__ == "__main__":
    main()
