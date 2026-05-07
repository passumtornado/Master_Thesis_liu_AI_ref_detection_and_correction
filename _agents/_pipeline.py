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
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional, TypedDict
from dotenv import load_dotenv
sys.path.insert(0, str(Path(__file__).parent))

from langgraph.graph import END, START, StateGraph

from _correction_agent import CorrectionAgent
from _correction_agent import PromptStrategy as CPromptStrategy
# from _evaluation_agent import EvaluationAgent
# from _evaluation_agent import PromptStrategy as EPromptStrategy
# from _evaluation_agent import build_comparison_report
from deterministic_eval_agent import EvaluationAgent
from deterministic_eval_agent import PromptStrategy as EPromptStrategy
from deterministic_eval_agent import build_comparison_report
from _preparation_agent import PreparationAgent

# ── Manual agent (default) ────────────────────────────────────
# from _validation_agent import LLMValidationAgent
# from _validation_agent import PromptStrategy as VPromptStrategy

# ── ReAct agent ───────────────────────────────────────────────
from _validation_agent_react import LLMValidationAgent
from _validation_agent_react import PromptStrategy as VPromptStrategy

load_dotenv('/Users/passum/Documents/SWEDEN/DSI/THESIS/AI_reference_agent_detection_correction/.env')

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
    batch_size:      int

    # ── preparation ───────────────────────────────────────────
    prepared_entries:   list[dict]
    preparation_report: dict
    warnings_by_entry:  dict

    # ── validation ────────────────────────────────────────────
    markdown_report:     str
    validation_structured: list[dict]   # per-entry structured results
    raw_data:            list[dict]
    validation_metrics:  dict
    validation_telemetry: dict
    validation_error:    str

    # ── correction ────────────────────────────────────────────
    corrected_entries:  list[dict]
    corrections:        list[dict]
    corrections_summary: str

    # ── evaluation ────────────────────────────────────────────
    ground_truth_path: str
    evaluation_metrics:        dict
    evaluation_field_accuracy: dict
    evaluation_entry_type_statistics: dict
    evaluation_error:          str

    # ── pipeline ──────────────────────────────────────────────
    saved_files: list[str]
    error:       Optional[str]


def _safe_div(n: float, d: float) -> float:
    return (n / d) if d else 0.0


def _compute_validation_metrics(
    validation_structured: list[dict],
    ground_truth_path: str,
) -> tuple[dict, str]:
    """Evaluate validation classification with one clear overall metric set."""
    try:
        gt_data = json.loads(Path(ground_truth_path).read_text(encoding="utf-8"))
    except Exception as e:
        return {}, f"Validation evaluation skipped: cannot read ground truth ({e})"

    gt_map: dict[str, str] = {}
    for item in gt_data:
        if isinstance(item, dict):
            entry_id = item.get("entry_id")
            status = item.get("expected_status")
            if isinstance(entry_id, str) and isinstance(status, str):
                gt_map[entry_id] = status

    pred_map: dict[str, str] = {}
    for item in validation_structured:
        if not isinstance(item, dict):
            continue
        entry_id = item.get("entry_id")
        status = item.get("status")
        if isinstance(entry_id, str) and isinstance(status, str):
            pred_map[entry_id] = status

    common_ids = sorted(set(gt_map.keys()) & set(pred_map.keys()))
    if not common_ids:
        return {}, "Validation evaluation skipped: no overlapping entry_id between predictions and ground truth"

    matched = len(common_ids)
    tp = sum(1 for entry_id in common_ids if gt_map[entry_id] == pred_map[entry_id])
    errors = matched - tp
    accuracy = _safe_div(tp, matched)
    precision = _safe_div(tp, tp + errors)
    recall = _safe_div(tp, tp + errors)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    gt_counter = Counter(gt_map[eid] for eid in common_ids)
    pred_counter = Counter(pred_map[eid] for eid in common_ids)

    metrics = {
        "support": {
            "ground_truth_entries": len(gt_map),
            "predicted_entries": len(pred_map),
            "matched_entries": matched,
            "coverage_vs_ground_truth": round(_safe_div(matched, len(gt_map)), 3),
            "class_distribution_truth": dict(gt_counter),
            "class_distribution_pred": dict(pred_counter),
        },
        "overall": {
            "accuracy": round(accuracy, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "true_positives": int(tp),
            "false_positives": int(errors),
            "false_negatives": int(errors),
        },
    }
    return metrics, ""


def _build_combined_markdown(state: PipelineState) -> str:
    """Create one markdown report that includes validation + correction performance."""
    v = state.get("validation_metrics", {}) or {}
    e = state.get("evaluation_metrics", {}) or {}
    fa = state.get("evaluation_field_accuracy", {}) or {}
    entry_types = state.get("evaluation_entry_type_statistics", {}) or {}

    lines = [
        "# End-to-End Performance Summary",
        "",
        "## Validation Agent (Classification)",
        "",
    ]

    if v.get("overall"):
        vo = v["overall"]
        vs = v.get("support", {})
        lines.extend([
            "| Metric | Value |",
            "|---|---|",
            f"| Accuracy | {vo.get('accuracy', 0):.3f} |",
            f"| Precision | {vo.get('precision', 0):.3f} |",
            f"| Recall | {vo.get('recall', 0):.3f} |",
            f"| F1 | {vo.get('f1', 0):.3f} |",
            f"| True Positives (correctly classified) | {int(vo.get('true_positives', 0))} |",
            f"| False Positives | {int(vo.get('false_positives', 0))} |",
            f"| False Negatives | {int(vo.get('false_negatives', 0))} |",
            f"| Ground-truth entries | {int(vs.get('ground_truth_entries', 0))} |",
            f"| Predicted entries | {int(vs.get('predicted_entries', 0))} |",
            f"| Matched entries | {int(vs.get('matched_entries', 0))} |",
            f"| Coverage vs ground truth | {float(vs.get('coverage_vs_ground_truth', 0.0)):.3f} |",
            "",
        ])
    else:
        lines.append("Validation metrics unavailable.")
        lines.append("")

    per_class = v.get("per_class", {}) if isinstance(v, dict) else {}
    if per_class:
        lines.extend([
            "### Validation Per-Class Metrics",
            "",
            "| Class | Count | Precision | Recall | F1 | TP | FP | FN |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for class_name in ["valid", "partially_valid", "invalid"]:
            metrics = per_class.get(class_name, {}) if isinstance(per_class, dict) else {}
            precision = metrics.get("precision")
            recall = metrics.get("recall")
            f1 = metrics.get("f1")
            lines.append(
                f"| {class_name} | {int(metrics.get('count', 0))} | "
                f"{('N/A' if precision is None else f'{float(precision):.3f}')} | "
                f"{('N/A' if recall is None else f'{float(recall):.3f}')} | "
                f"{('N/A' if f1 is None else f'{float(f1):.3f}')} | "
                f"{int(metrics.get('true_positives', 0))} | {int(metrics.get('false_positives', 0))} | {int(metrics.get('false_negatives', 0))} |"
            )
        lines.append("")

    if entry_types:
        lines.extend([
            "### BibTeX Entry Type Statistics",
            "",
            "| Entry Type | Count | Valid | Partially Valid | Invalid |",
            "|---|---:|---:|---:|---:|",
        ])
        for entry_type, stats in sorted(entry_types.items()):
            lines.append(
                f"| @{entry_type} | {int(stats.get('count', 0))} | {int(stats.get('valid', 0))} | {int(stats.get('partially_valid', 0))} | {int(stats.get('invalid', 0))} |"
            )
        lines.append("")

    lines.extend([
        "## Correction Agent",
        "",
    ])
    if e:
        scope = e.get("scope", {}) if isinstance(e, dict) else {}
        lines.extend([
            "| Metric | Value |",
            "|---|---|",
            f"| True Positives | {int(e.get('true_positives', 0))} |",
            f"| False Positives | {int(e.get('false_positives', 0))} |",
            f"| False Negatives | {int(e.get('false_negatives', 0))} |",
            f"| Precision | {float(e.get('precision', 0.0)):.3f} |",
            f"| Recall | {float(e.get('recall', 0.0)):.3f} |",
            f"| F1 | {float(e.get('f1', 0.0)):.3f} |",
            f"| Partially-valid in ground truth | {int(scope.get('total_partially_valid_ground_truth', 0))} |",
            f"| Correctly identified partially-valid | {int(scope.get('correctly_identified_partially_valid', 0))} |",
            "",
        ])
    else:
        lines.append("Correction metrics unavailable.")
        lines.append("")

    if fa:
        lines.extend([
            "### Field-Level Metrics (Partially Valid, Correctly Identified)",
            "",
            "| Field | Errors in Original | Errors Corrected | False Corrections | Accuracy |",
            "|---|---|---|---|---|",
        ])
        for field in sorted(fa.keys()):
            m = fa.get(field, {}) if isinstance(fa, dict) else {}
            lines.append(
                f"| {field} | {int(m.get('errors_in_original', 0))} | {int(m.get('errors_corrected', 0))} | {int(m.get('false_corrections', 0))} | {float(m.get('accuracy', 0.0)):.3f} |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


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

        total_entries = len(entries)
        requested_batch_size = int(state.get("batch_size", 0) or 0)

        # Policy:
        # - <= 50 entries: process in a single pass
        # - > 50 entries : force batched processing
        if total_entries <= 50:
            batch_size = total_entries
        else:
            if requested_batch_size <= 0:
                requested_batch_size = 25
            batch_size = min(max(1, requested_batch_size), 50)

        total_batches = (total_entries + batch_size - 1) // batch_size
        print(
            f"  [PIPELINE] Validation policy: total_entries={total_entries}, "
            f"batch_size={batch_size}, total_batches={total_batches}"
        )

        markdown_parts: list[str] = []
        all_structured: list[dict] = []
        all_raw_data: list[dict] = []
        all_validation_telemetry: dict[str, Any] = {
            "total_requests": 0,
            "cache_hits": 0,
            "total_retries": 0,
            "duration_seconds": 0.0,
            "retries_by_tool": {},
        }

        for batch_index, start in enumerate(range(0, total_entries, batch_size), start=1):
            batch_entries = entries[start:start + batch_size]
            print(
                f"  [PIPELINE] Validation batch {batch_index}/{total_batches} "
                f"({len(batch_entries)} entries)"
            )

            report = None
            last_err = None
            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                try:
                    report = await agent.validate_entries(batch_entries)
                    break
                except Exception as e:
                    last_err = e
                    print(
                        f"  [PIPELINE] Validation batch {batch_index} attempt {attempt}/{max_attempts} failed: {e}"
                    )
                    if attempt < max_attempts:
                        await asyncio.sleep(min(2 * attempt, 6))

            if report is None:
                return {
                    "error": (
                        f"Validation failed for batch {batch_index}/{total_batches} "
                        f"after {max_attempts} attempts: {last_err}"
                    )
                }

            all_raw_data.extend(report.get("raw_data", []))
            all_structured.extend(report.get("structured", []))
            telemetry = report.get("telemetry", {}) or {}
            all_validation_telemetry["total_requests"] += int(telemetry.get("total_requests", 0) or 0)
            all_validation_telemetry["cache_hits"] += int(telemetry.get("cache_hits", 0) or 0)
            all_validation_telemetry["total_retries"] += int(telemetry.get("total_retries", 0) or 0)
            all_validation_telemetry["duration_seconds"] += float(telemetry.get("duration_seconds", 0.0) or 0.0)
            retries_by_tool = telemetry.get("retries_by_tool", {}) or {}
            for tool_name, count in retries_by_tool.items():
                all_validation_telemetry["retries_by_tool"][tool_name] = (
                    all_validation_telemetry["retries_by_tool"].get(tool_name, 0) + int(count or 0)
                )

            batch_markdown = report.get("markdown_report", "").strip()
            if batch_markdown:
                # Keep model narrative, but normalize obvious count mismatches.
                batch_markdown = re.sub(
                    r"for the\s+\d+\s+BibTeX entries",
                    f"for the {len(batch_entries)} BibTeX entries",
                    batch_markdown,
                    flags=re.IGNORECASE,
                )
                if total_batches > 1:
                    markdown_parts.append(f"## Batch {batch_index}/{total_batches}\n\n{batch_markdown}")
                else:
                    markdown_parts.append(batch_markdown)

        combined_markdown = "\n\n---\n\n".join(markdown_parts).strip()

        # Save validation outputs immediately so users can inspect before full pipeline ends.
        validation_dir = Path(state["output_dir"]) / "validation" / strategy.value
        validation_dir.mkdir(parents=True, exist_ok=True)

        validation_md_path = validation_dir / "validation_report.md"
        validation_md_path.write_text(combined_markdown, encoding="utf-8")

        grouped = {
            "valid": [],
            "partially_valid": [],
            "invalid": [],
        }
        for row in all_structured:
            if not isinstance(row, dict):
                continue
            status = row.get("status")
            if status in grouped:
                grouped[status].append(row)

        validation_structured_path = validation_dir / "validation_structured.json"
        validation_structured_path.write_text(
            json.dumps({"results": all_structured}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        grouped_path = validation_dir / "validation_grouped.json"
        grouped_path.write_text(
            json.dumps(grouped, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        validation_metrics, validation_error = _compute_validation_metrics(
            validation_structured=all_structured,
            ground_truth_path=state.get("ground_truth_path", "bibtex/ground_truth/stefan_train2_truth.json"),
        )

        total_lookups = all_validation_telemetry["cache_hits"] + all_validation_telemetry["total_requests"]
        all_validation_telemetry["cache_hit_rate"] = round(
            (all_validation_telemetry["cache_hits"] / total_lookups * 100) if total_lookups > 0 else 0.0,
            2,
        )
        all_validation_telemetry["requests_per_second"] = round(
            (all_validation_telemetry["total_requests"] / all_validation_telemetry["duration_seconds"])
            if all_validation_telemetry["duration_seconds"] > 0 else 0.0,
            3,
        )

        validation_metrics_path = validation_dir / "validation_metrics.json"
        validation_metrics_path.write_text(
            json.dumps(validation_metrics or {}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        validation_telemetry_path = validation_dir / "validation_telemetry.json"
        validation_telemetry_path.write_text(
            json.dumps(all_validation_telemetry, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"  ✓ validation report saved       → {validation_md_path}")
        print(f"  ✓ validation structured saved   → {validation_structured_path}")
        print(f"  ✓ validation grouped saved      → {grouped_path}")
        print(f"  ✓ validation metrics saved      → {validation_metrics_path}")
        print(f"  ✓ validation telemetry saved    → {validation_telemetry_path}")
        if validation_error:
            print(f"  ⚠ {validation_error}")

        return {
            "markdown_report":       combined_markdown,
            "validation_structured": all_structured,
            "raw_data":              all_raw_data,
            "validation_metrics":    validation_metrics,
            "validation_telemetry":  all_validation_telemetry,
            "validation_error":      validation_error,
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
            validation_structured=state.get("validation_structured", []),
            validation_markdown=state.get("markdown_report", ""),
        )

        # Save corrected structured outputs immediately for early inspection.
        correction_dir = Path(state["output_dir"]) / "corrections" / strategy.value
        correction_dir.mkdir(parents=True, exist_ok=True)

        corrected_entries_path = correction_dir / "corrected_entries.json"
        corrected_entries_path.write_text(
            json.dumps({"corrected_entries": result.get("corrected_entries", [])}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        corrections_structured_path = correction_dir / "corrections_structured.json"
        corrections_structured_path.write_text(
            json.dumps({"corrections": result.get("corrections", [])}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"  ✓ corrected entries saved      → {corrected_entries_path}")
        print(f"  ✓ corrections structured saved → {corrections_structured_path}")

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
        agent      = EvaluationAgent(str(output_dir), strategy=strategy, ground_truth_path=state.get("ground_truth_path", "bibtex/ground_truth/stefan_train2_truth.json"))

        result = await agent.evaluate(
            raw_data=raw_data,
            corrections=corrections,
            validation_structured=state.get("validation_structured", []),
        )
        return {
            "validation_metrics": result.get("validation_metrics", state.get("validation_metrics", {})),
            "evaluation_metrics":        result.get("overall_metrics", {}),
            "evaluation_field_accuracy": result.get("field_accuracy", {}),
            "evaluation_entry_type_statistics": result.get("entry_type_statistics", {}),
            "evaluation_error":          "",
        }
    except Exception as e:
        return {
            "evaluation_metrics":        {},
            "evaluation_field_accuracy": {},
            "evaluation_entry_type_statistics": {},
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

    if state.get("validation_metrics"):
        validation_metrics_file = output_dir / "validation_metrics.json"
        validation_metrics_file.write_text(
            json.dumps(state.get("validation_metrics", {}), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        saved.append(str(validation_metrics_file))

    if state.get("validation_telemetry"):
        validation_telemetry_file = output_dir / "validation_telemetry.json"
        validation_telemetry_file.write_text(
            json.dumps(state.get("validation_telemetry", {}), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        saved.append(str(validation_telemetry_file))

    # Combined one-file summary for validation + correction performance.
    performance_json = output_dir / "performance_summary.json"
    performance_payload = {
        "validation": {
            "metrics": state.get("validation_metrics", {}),
            "telemetry": state.get("validation_telemetry", {}),
            "error": state.get("validation_error", ""),
        },
        "correction": {
            "overall_metrics": state.get("evaluation_metrics", {}),
            "field_accuracy": state.get("evaluation_field_accuracy", {}),
            "entry_type_statistics": state.get("evaluation_entry_type_statistics", {}),
            "error": state.get("evaluation_error", ""),
        },
    }
    performance_json.write_text(
        json.dumps(performance_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    saved.append(str(performance_json))

    performance_md = output_dir / "performance_summary.md"
    performance_md.write_text(_build_combined_markdown(state), encoding="utf-8")
    saved.append(str(performance_md))

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

    if state.get("validation_error"):
        err_file = output_dir / "validation_error.txt"
        err_file.write_text(str(state["validation_error"]), encoding="utf-8")
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
    ground_truth_path: str = "bibtex/ground_truth/stefan_train1_truth.json",
    batch_size: int = 25,
) -> PipelineState:
    """Run the full pipeline with a single prompting strategy."""
    project_root = Path(__file__).parent.parent

    initial_state: PipelineState = {
        "mcp_config_path": mcp_config_path or str(project_root / "server" / "mcp.json"),
        "bibtex_source":   bibtex_source,
        "source_type":     source_type,
        "output_dir":      output_dir or str(project_root / "evaluation"),
        "strategy":        strategy,
        "batch_size":      max(1, int(batch_size)),
        "ground_truth_path": ground_truth_path,
        "prepared_entries":      [],
        "preparation_report":    {},
        "warnings_by_entry":     {},

        "markdown_report":         "",
        "validation_structured":   [],
        "raw_data":                [],
        "validation_metrics":      {},
        "validation_error":        "",

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
    ground_truth_path: str = "bibtex/ground_truth/stefan_train1_truth.json",
    batch_size: int = 25,
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
            ground_truth_path=ground_truth_path,
            batch_size=batch_size,
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
        "--ground-truth",
        default="bibtex/ground_truth/stefan_train1_truth.json",
        help="Path to ground-truth JSON file used for evaluation",
    )
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Number of entries per validation batch (default: 25)",
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
            ground_truth_path=args.ground_truth,
            batch_size=args.batch_size,
        ))
    else:
        asyncio.run(run_pipeline(
            bibtex_source=source,
            source_type=stype,
            mcp_config_path=args.mcp_config,
            output_dir=args.output_dir,
            strategy=args.strategy,
            ground_truth_path=args.ground_truth,
            batch_size=args.batch_size,
        ))


if __name__ == "__main__":
    main()