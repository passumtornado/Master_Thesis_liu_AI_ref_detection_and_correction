"""
LLM-Driven Prototype Pipeline
-----------------------------
Sequential LangGraph pipeline:
  START → prepare → validate → correction → evaluation → save_outputs → END

Two entry points:
  run_pipeline()   — single strategy run  (--strategy zero_shot|rag|cot)
  run_experiment() — all three strategies against same input (--experiment)
                     produces comparison.md / comparison.json

Usage:
  # Single run with RAG (default)
  python pipeline.py --file references.bib

  # Single run with specific strategy
  python pipeline.py --file references.bib --strategy cot

  # Full thesis experiment (all 3 strategies + comparison report)
  python pipeline.py --file references.bib --experiment
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Optional, TypedDict

sys.path.insert(0, str(Path(__file__).parent))

from langgraph.graph import END, START, StateGraph

from _correction_agent import CorrectionAgent
from _correction_agent import PromptStrategy as CPromptStrategy
from _evaluation_agent import EvaluationAgent
from _evaluation_agent import PromptStrategy as EPromptStrategy
from _evaluation_agent import build_comparison_report
from _preparation_agent import PreparationAgent
# from _validation_agent import LLMValidationAgent
# from _validation_agent import PromptStrategy as VPromptStrategy

# ── Manual agent (default) ────────────────────────────────────
# from _validation_agent import LLMValidationAgent
# from _validation_agent import PromptStrategy as VPromptStrategy
# from _validation_agent_react import LLMValidationAgent
# from _validation_agent_react import PromptStrategy as VPromptStrategy

# ── ReAct agent ───────────────────────────────────────────────
# from _validation_agent import LLMValidationAgent
# from _validation_agent import PromptStrategy as VPromptStrategy
from _validation_agent_react import LLMValidationAgent
from _validation_agent_react import PromptStrategy as VPromptStrategy



# ─────────────────────────────────────────────────────────────
# Pipeline State
# ─────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    # ── inputs ────────────────────────────────────────────────
    mcp_config_path: str
    bibtex_source:   str
    source_type:     str
    output_dir:      str
    strategy:        str          # "zero_shot" | "rag" | "cot"

    # ── preparation ───────────────────────────────────────────
    prepared_entries:   list[dict]
    preparation_report: dict
    warnings_by_entry:  dict

    # ── validation ────────────────────────────────────────────
    markdown_report:     str
    validation_structured: list[dict]   # per-entry structured results
    raw_data:            list[dict]

    # ── correction ────────────────────────────────────────────
    corrected_entries:  list[dict]
    corrections:        list[dict]
    corrections_summary: str

    # ── evaluation ────────────────────────────────────────────
    evaluation_metrics:        dict
    evaluation_field_accuracy: dict
    evaluation_error:          str

    # ── pipeline ──────────────────────────────────────────────
    saved_files: list[str]
    error:       Optional[str]


# ─────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────

async def prepare_node(state: PipelineState) -> dict:
    print("\n[PIPELINE] Node: prepare")
    if state.get("error"):
        return {}

    try:
        agent  = PreparationAgent(state["mcp_config_path"])
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
        return {"error": str(e)}


async def validate_node(state: PipelineState) -> dict:
    print(f"\n[PIPELINE] Node: validate [{state.get('strategy', 'rag')}]")
    if state.get("error"):
        return {}

    entries = state.get("prepared_entries", [])
    if not entries:
        return {"error": "No entries to validate — preparation produced empty list"}

    try:
        strategy = VPromptStrategy(state.get("strategy", "rag"))
        agent    = LLMValidationAgent(
            mcp_config_path=state["mcp_config_path"],
            strategy=strategy,
        )
        report = await agent.validate_entries(entries)
        return {
            "markdown_report":       report["markdown_report"],
            "validation_structured": report.get("structured", []),
            "raw_data":              report["raw_data"],
        }
    except Exception as e:
        return {"error": str(e)}


async def correction_node(state: PipelineState) -> dict:
    print(f"\n[PIPELINE] Node: correction [{state.get('strategy', 'rag')}]")
    if state.get("error"):
        return {}

    raw_data = state.get("raw_data", [])
    if not raw_data:
        return {"error": "No validation data — validation produced empty list"}

    try:
        strategy   = CPromptStrategy(state.get("strategy", "rag"))
        output_dir = Path(state["output_dir"]) / "corrections"
        agent      = CorrectionAgent(str(output_dir), strategy=strategy)

        result = await agent.correct_entries(
            raw_data=raw_data,
            validation_markdown=state.get("markdown_report", ""),
        )
        return {
            "corrected_entries":  result["corrected_entries"],
            "corrections":        result.get("corrections", []),
            "corrections_summary": result["correction_summary"],
        }
    except Exception as e:
        return {"error": str(e)}


async def evaluation_node(state: PipelineState) -> dict:
    print(f"\n[PIPELINE] Node: evaluation [{state.get('strategy', 'rag')}]")
    if state.get("error"):
        return {}

    raw_data    = state.get("raw_data", [])
    corrections = state.get("corrections", [])

    if not raw_data or not corrections:
        return {
            "evaluation_metrics":        {},
            "evaluation_field_accuracy": {},
            "evaluation_error":          "Skipped: missing raw_data or corrections.",
        }

    try:
        strategy   = EPromptStrategy(state.get("strategy", "rag"))
        output_dir = Path(state["output_dir"]) / "evaluation"
        agent      = EvaluationAgent(str(output_dir), strategy=strategy)

        result = await agent.evaluate(raw_data=raw_data, corrections=corrections)
        return {
            "evaluation_metrics":        result.get("overall_metrics", {}),
            "evaluation_field_accuracy": result.get("field_accuracy", {}),
            "evaluation_error":          "",
        }
    except Exception as e:
        return {
            "evaluation_metrics":        {},
            "evaluation_field_accuracy": {},
            "evaluation_error":          f"Evaluation failed: {e}",
        }


async def save_outputs_node(state: PipelineState) -> dict:
    """Save pipeline-level artefacts only — each agent saves its own files."""
    print("\n[PIPELINE] Node: save_outputs")
    output_dir = Path(state["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    # Preparation report (no individual agent owns this)
    prep_json = output_dir / "preparation_report.json"
    prep_json.write_text(
        json.dumps(
            {
                "preparation_report": state.get("preparation_report", {}),
                "warnings_by_entry":  state.get("warnings_by_entry", {}),
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    saved.append(str(prep_json))

    # Validation markdown (pipeline-level view)
    if state.get("markdown_report"):
        md_file = output_dir / "validation_report.md"
        md_file.write_text(state["markdown_report"], encoding="utf-8")
        saved.append(str(md_file))

    # Pipeline snapshot for debugging (excludes large lists)
    snapshot = {
        k: v for k, v in state.items()
        if k not in ("prepared_entries", "raw_data", "corrected_entries", "corrections")
    }
    snapshot_path = output_dir / "pipeline_snapshot.json"
    snapshot_path.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    saved.append(str(snapshot_path))

    # Error file
    if state.get("error"):
        err_file = output_dir / "pipeline_error.txt"
        err_file.write_text(str(state["error"]), encoding="utf-8")
        saved.append(str(err_file))

    if state.get("evaluation_error"):
        err_file = output_dir / "evaluation_error.txt"
        err_file.write_text(str(state["evaluation_error"]), encoding="utf-8")
        saved.append(str(err_file))

    print(f"  ✓ Saved {len(saved)} pipeline-level file(s)")
    return {"saved_files": saved}


# ─────────────────────────────────────────────────────────────
# Graph
# ─────────────────────────────────────────────────────────────

def build_pipeline() -> Any:
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


# ─────────────────────────────────────────────────────────────
# run_pipeline  — single strategy
# ─────────────────────────────────────────────────────────────

async def run_pipeline(
    bibtex_source:   str,
    source_type:     str = "file",
    mcp_config_path: str | None = None,
    output_dir:      str | None = None,
    strategy:        str = "rag",
) -> PipelineState:
    """Run the full pipeline with a single prompting strategy."""
    project_root = Path(__file__).parent.parent

    initial_state: PipelineState = {
        "mcp_config_path": mcp_config_path or str(project_root / "server" / "mcp.json"),
        "bibtex_source":   bibtex_source,
        "source_type":     source_type,
        "output_dir":      output_dir or str(project_root / "evaluation"),
        "strategy":        strategy,

        "prepared_entries":      [],
        "preparation_report":    {},
        "warnings_by_entry":     {},

        "markdown_report":         "",
        "validation_structured":   [],
        "raw_data":                [],

        "corrected_entries":     [],
        "corrections":           [],
        "corrections_summary":   "",

        "evaluation_metrics":        {},
        "evaluation_field_accuracy": {},
        "evaluation_error":          "",

        "saved_files": [],
        "error":       None,
    }

    pipeline    = build_pipeline()
    final_state = await pipeline.ainvoke(initial_state)

    if not isinstance(final_state, dict):
        final_state = {
            **initial_state,
            "error":       f"Unexpected pipeline return type: {type(final_state).__name__}",
            "saved_files": [],
        }

    _print_pipeline_result(final_state)
    return final_state


# ─────────────────────────────────────────────────────────────
# run_experiment  — all three strategies (thesis entry point)
# ─────────────────────────────────────────────────────────────

async def run_experiment(
    bibtex_source:   str,
    source_type:     str = "file",
    mcp_config_path: str | None = None,
    output_dir:      str | None = None,
) -> dict:
    """
    Run all three prompting strategies against the same .bib file
    and produce a side-by-side comparison report.

    Output structure:
      <output_dir>/
        comparison.md          ← thesis core result
        comparison.json        ← machine-readable comparison
        zero_shot/             ← per-strategy artefacts
        rag/
        cot/
    """
    strategies  = ["zero_shot", "rag", "cot"]
    all_results = []

    for strategy in strategies:
        print(f"\n{'#'*60}")
        print(f"  EXPERIMENT — Strategy: {strategy.upper()}")
        print(f"{'#'*60}")

        final_state = await run_pipeline(
            bibtex_source=bibtex_source,
            source_type=source_type,
            mcp_config_path=mcp_config_path,
            output_dir=output_dir,
            strategy=strategy,
        )

        all_results.append({
            "strategy":        strategy,
            "overall_metrics": final_state.get("evaluation_metrics", {}),
            "field_accuracy":  final_state.get("evaluation_field_accuracy", {}),
            "error":           final_state.get("error"),
        })

    # Build the comparison table — core thesis quantitative result
    print(f"\n{'='*60}")
    print("  Building cross-strategy comparison report …")
    print(f"{'='*60}")

    comparison_dir = str(Path(output_dir or "evaluation"))
    build_comparison_report(all_results, output_dir=comparison_dir)

    return {"strategies": all_results}


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _print_pipeline_result(state: dict) -> None:
    if state.get("error"):
        print(f"\n  ✗ PIPELINE ERROR    : {state['error']}")
    elif state.get("evaluation_error"):
        print(f"\n  ⚠ PIPELINE WARNING  : {state['evaluation_error']}")
    else:
        print("\n  ✓ PIPELINE COMPLETE")

    if state.get("saved_files"):
        print("  Saved files:")
        for path in state["saved_files"]:
            print(f"    - {path}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BibTeX Validation Pipeline (LLM-Driven)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single RAG run (default)
  python pipeline.py --file references.bib

  # Single zero-shot run
  python pipeline.py --file references.bib --strategy zero_shot

  # Full thesis experiment (all 3 strategies + comparison report)
  python pipeline.py --file references.bib --experiment

  # From a DBLP URL
  python pipeline.py --url https://dblp.org/pid/l/YannLeCun.bib --experiment
        """,
    )

    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--file", help="Path to a local .bib file")
    source_group.add_argument("--url",  help="URL to a remote .bib file")

    parser.add_argument("--mcp-config",  help="Path to mcp.json config")
    parser.add_argument("--output-dir",  help="Root output directory for reports")
    parser.add_argument(
        "--strategy",
        choices=["zero_shot", "rag", "cot"],
        default="rag",
        help="Prompting strategy for a single run (default: rag)",
    )
    parser.add_argument(
        "--experiment",
        action="store_true",
        help="Run all 3 strategies and produce a comparison report",
    )

    args = parser.parse_args()

    # Resolve source
    if args.file:
        source, stype = args.file, "file"
    elif args.url:
        source, stype = args.url, "url"
    else:
        project_root = Path(__file__).parent.parent
        source = str(project_root / "bibtex" / "bibtex_files" / "references.bib")
        stype  = "file"

    if args.experiment:
        asyncio.run(run_experiment(
            bibtex_source=source,
            source_type=stype,
            mcp_config_path=args.mcp_config,
            output_dir=args.output_dir,
        ))
    else:
        asyncio.run(run_pipeline(
            bibtex_source=source,
            source_type=stype,
            mcp_config_path=args.mcp_config,
            output_dir=args.output_dir,
            strategy=args.strategy,
        ))


if __name__ == "__main__":
    main()