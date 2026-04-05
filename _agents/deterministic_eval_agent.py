"""
BibTeX Evaluation Agent — Ground-Truth Edition
------------------------------------------------
Replaces LLM-based metric computation with deterministic Python comparison
against a known ground-truth file generated alongside the synthetic dataset.

The LLM is kept only for writing the narrative markdown report.

Two evaluation modes:
  1. ground_truth mode (recommended for synthetic dataset):
       Load ground_truth.json → deterministic TP/FP/FN → exact metrics
  2. legacy LLM mode (fallback when no ground truth is available):
       Send payload to LLM → LLM estimates metrics (inconsistent)

Usage:
  agent = EvaluationAgent(
      output_dir="evaluation",
      strategy=PromptStrategy.RAG,
      ground_truth_path="ground_truth.json",   # ← enables deterministic mode
  )
  result = await agent.evaluate(raw_data, corrections)
"""

import json
import os
import re
import sys
from collections import defaultdict
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import _extract_text

load_dotenv()


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _normalise(s: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for comparison."""
    s = str(s).lower().strip()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[{}]', '', s)   # strip BibTeX braces
    return s


def _fields_match(corrected: str, expected: str) -> bool:
    """
    Return True if corrected value is close enough to expected.
    Handles year (exact), short strings (exact after normalise),
    and longer strings (token overlap).
    """
    c = _normalise(corrected)
    e = _normalise(expected)

    if c == e:
        return True

    # For very short values (year, volume) require exact match
    if len(e) <= 6:
        return c == e

    # Token overlap >= 0.80 for longer strings (titles, author lists, venues)
    c_tokens = set(c.split())
    e_tokens = set(e.split())
    if not e_tokens:
        return False
    overlap = len(c_tokens & e_tokens) / len(e_tokens)
    return overlap >= 0.80


# ─────────────────────────────────────────────────────────────
# Prompt Strategy
# ─────────────────────────────────────────────────────────────

class PromptStrategy(Enum):
    ZERO_SHOT = "zero_shot"
    RAG       = "rag"
    COT       = "cot"


# ─────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────

class EvaluationAgent:
    """
    Hybrid evaluation agent.

    When ground_truth_path is provided (recommended):
      - TP / FP / FN computed deterministically in Python
      - LLM writes the narrative markdown report only
      - Results are reproducible across every run

    When ground_truth_path is None (legacy fallback):
      - Full evaluation delegated to LLM
      - Metrics may vary between runs
    """

    REPORT_SYSTEM_PROMPT = """You are an expert bibliographic data quality evaluator.

You will receive:
  - Pre-computed evaluation metrics (TP, FP, FN, Precision, Recall, F1)
  - Per-field accuracy breakdown
  - Detailed per-entry results showing what was wrong, what was corrected,
    and whether the correction was accurate

Your task is to write a clear, professional markdown evaluation report that:
  1. Opens with an overall metrics summary table
  2. Shows field-level accuracy in a table
  3. Groups entries by outcome: correctly fixed, missed errors, false corrections
  4. Provides key insights about what the correction agent did well and where it failed
  5. Concludes with actionable recommendations

Write in an academic style suitable for a Master's thesis.
Output only the markdown report — no JSON, no preamble, no extra text.
"""

    LEGACY_SYSTEM_PROMPT = """You are an expert bibliographic data quality evaluator.

You will receive correction records. Each record contains:
  - entry_id     : citation key
  - original     : original BibTeX fields before correction
  - corrected    : fields after the correction agent ran
  - ground_truth : best DBLP match (the authoritative reference)
  - changes      : list of changes the correction agent made

Your task — compute these metrics by comparing original → corrected → ground_truth
for each of these fields: title, author, year, journal, booktitle, venue.

Definitions:
  true_positive  (TP) : field was WRONG in original AND corrected to match ground_truth
  false_positive (FP) : field was CORRECT in original BUT was wrongly changed
  false_negative (FN) : field was WRONG in original AND was NOT fixed

Formulas:
  recall    = TP / (TP + FN)
  precision = TP / (TP + FP)
  f1        = 2 * precision * recall / (precision + recall)

You MUST respond with valid JSON in EXACTLY this structure — no extra text, no fences:
{
  "overall_metrics": {
    "true_positives":  <int>,
    "false_positives": <int>,
    "false_negatives": <int>,
    "recall":          <float 0-1, 3 decimal places>,
    "precision":       <float 0-1, 3 decimal places>,
    "f1":              <float 0-1, 3 decimal places>
  },
  "field_accuracy": {
    "<field_name>": {
      "errors_in_original": <int>,
      "errors_corrected":   <int>,
      "false_corrections":  <int>,
      "accuracy":           <float 0-1, 3 decimal places>
    }
  },
  "markdown_report": "<complete markdown string>"
}
"""

    def __init__(
        self,
        output_dir: str = "evaluation",
        strategy: PromptStrategy = PromptStrategy.RAG,
        ground_truth_path: str | None = None,
    ):
        self.strategy           = strategy
        self.ground_truth_path  = Path(ground_truth_path) if ground_truth_path else None
        self.output_dir         = Path(output_dir) / strategy.value
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Google Gemini backend (default)
        # self.llm = ChatGoogleGenerativeAI(
        #     model="gemini-2.0-flash",
        #     temperature=0.1,
        #     google_api_key=os.getenv("GOOGLE_API_KEY"),
        # )

        # Ollama backend — uncomment to switch
        self.llm = ChatOllama(
            model="qwen3-coder:480b-cloud",
            base_url="https://ollama.com",
            temperature=0.1,
            client_kwargs={
                "headers": {"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"}
            },
        )

    # ── public ────────────────────────────────────────────────

    async def evaluate(self, raw_data: list[dict], corrections: list[dict]) -> dict:
        """
        Main entry point.

        If ground_truth_path was provided at init:
          → deterministic evaluation (recommended)
        Else:
          → legacy LLM-based evaluation (fallback)
        """
        print(f"\n{'='*60}")
        print(f"EVALUATION AGENT [{self.strategy.value.upper()}]")
        mode = "DETERMINISTIC" if self.ground_truth_path else "LLM (legacy)"
        print(f"  Mode: {mode}")
        print(f"{'='*60}\n")

        if self.ground_truth_path:
            return await self._evaluate_deterministic(raw_data, corrections)
        else:
            return await self._evaluate_legacy(raw_data, corrections)

    # ─────────────────────────────────────────────────────────
    # Mode 1 — Deterministic evaluation
    # ─────────────────────────────────────────────────────────

    async def _evaluate_deterministic(
        self, raw_data: list[dict], corrections: list[dict]
    ) -> dict:
        """
        Compute TP/FP/FN in Python using ground_truth.json.
        Delegate only the narrative report to the LLM.
        """
        # ── Load ground truth ─────────────────────────────────
        with open(self.ground_truth_path, "r", encoding="utf-8") as f:
            gt_list = json.load(f)

        # Index by entry_id
        gt_map   = {r["entry_id"]: r for r in gt_list if isinstance(r, dict)}
        corr_map = {
            c["entry_id"]: c
            for c in corrections
            if isinstance(c, dict) and c.get("entry_id")
        }
        raw_map  = {
            item["entry"].get("id"): item
            for item in raw_data
            if isinstance(item, dict) and isinstance(item.get("entry"), dict)
        }

        print(f"  Ground truth entries : {len(gt_map)}")
        print(f"  Correction entries   : {len(corr_map)}")
        print(f"  Raw data entries     : {len(raw_map)}")

        # ── Deterministic metric computation ──────────────────
        TP = FP = FN = 0
        field_counts   = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
        detailed       = []

        for entry_id, gt in gt_map.items():
            corruption_list = gt.get("corruptions", [])
            correction      = corr_map.get(entry_id, {})
            corrected_fields = correction.get("corrected", {})
            original_fields  = raw_map.get(entry_id, {}).get("entry", {})

            entry_detail = {
                "entry_id":          entry_id,
                "expected_status":   gt.get("expected_status", "unknown"),
                "corruptions":       corruption_list,
                "field_outcomes":    [],
            }

            # ── Check each known corruption → TP or FN ───────
            for corruption in corruption_list:
                field          = corruption["field"]
                correct_value  = corruption["original_correct_value"]
                corrected_val  = str(corrected_fields.get(field, ""))

                was_fixed = _fields_match(corrected_val, correct_value)

                if was_fixed:
                    TP += 1
                    field_counts[field]["TP"] += 1
                    outcome = "TP"
                else:
                    FN += 1
                    field_counts[field]["FN"] += 1
                    outcome = "FN"

                entry_detail["field_outcomes"].append({
                    "field":          field,
                    "outcome":        outcome,
                    "correct_value":  correct_value,
                    "corrected_value": corrected_val,
                })

            # ── Check for false positives ─────────────────────
            # A FP is a field that was correct in the original but was changed
            corrupted_fields = {c["field"] for c in corruption_list}

            for field, corrected_val in corrected_fields.items():
                if field in corrupted_fields:
                    continue   # already counted above
                original_val = str(original_fields.get(field, ""))
                if not original_val:
                    continue   # field didn't exist in original, skip
                if not _fields_match(corrected_val, original_val):
                    # Field was correct but was changed → FP
                    FP += 1
                    field_counts[field]["FP"] += 1
                    entry_detail["field_outcomes"].append({
                        "field":          field,
                        "outcome":        "FP",
                        "correct_value":  original_val,
                        "corrected_value": corrected_val,
                    })

            detailed.append(entry_detail)

        # ── Compute aggregate metrics ─────────────────────────
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        overall_metrics = {
            "true_positives":  TP,
            "false_positives": FP,
            "false_negatives": FN,
            "recall":          round(recall, 3),
            "precision":       round(precision, 3),
            "f1":              round(f1, 3),
        }

        # ── Build field-level accuracy ────────────────────────
        field_accuracy = {}
        for field, counts in field_counts.items():
            tp_f = counts["TP"]
            fn_f = counts["FN"]
            fp_f = counts["FP"]
            errors    = tp_f + fn_f
            accuracy  = tp_f / errors if errors > 0 else 0.0
            field_accuracy[field] = {
                "errors_in_original": errors,
                "errors_corrected":   tp_f,
                "false_corrections":  fp_f,
                "accuracy":           round(accuracy, 3),
            }

        print(f"\n  ✓ TP={TP}  FP={FP}  FN={FN}")
        print(f"  ✓ Precision: {precision:.3f}")
        print(f"  ✓ Recall:    {recall:.3f}")
        print(f"  ✓ F1:        {f1:.3f}")

        # ── LLM writes the narrative report ───────────────────
        markdown_report = await self._generate_report(
            overall_metrics, field_accuracy, detailed
        )

        # ── Save everything ───────────────────────────────────
        saved_files = self._save_outputs(
            overall_metrics, field_accuracy, markdown_report, detailed
        )

        return {
            "strategy":        self.strategy.value,
            "overall_metrics": overall_metrics,
            "field_accuracy":  field_accuracy,
            "markdown_report": markdown_report,
            "saved_files":     saved_files,
        }

    async def _generate_report(
        self,
        overall_metrics: dict,
        field_accuracy: dict,
        detailed: list[dict],
    ) -> str:
        """Ask LLM to write a narrative markdown report from pre-computed numbers."""
        payload = {
            "overall_metrics": overall_metrics,
            "field_accuracy":  field_accuracy,
            "detailed_results": detailed[:30],  # limit to avoid token overflow
        }

        messages = [
            SystemMessage(content=self.REPORT_SYSTEM_PROMPT),
            HumanMessage(content=(
                "Here are the pre-computed evaluation results. "
                "Write the markdown report:\n\n"
                f"```json\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n```"
            )),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            return _extract_text(response)
        except Exception as e:
            print(f"  ⚠ LLM report generation failed: {e}")
            return self._fallback_report(overall_metrics, field_accuracy)

    @staticmethod
    def _fallback_report(overall_metrics: dict, field_accuracy: dict) -> str:
        """Generate a basic markdown report without the LLM."""
        m   = overall_metrics
        md  = "# Evaluation Report\n\n"
        md += "## Overall Metrics\n\n"
        md += "| Metric | Value |\n|---|---|\n"
        md += f"| True Positives  | {m.get('true_positives', 0)} |\n"
        md += f"| False Positives | {m.get('false_positives', 0)} |\n"
        md += f"| False Negatives | {m.get('false_negatives', 0)} |\n"
        md += f"| Precision       | {m.get('precision', 0):.3f} |\n"
        md += f"| Recall          | {m.get('recall', 0):.3f} |\n"
        md += f"| F1              | {m.get('f1', 0):.3f} |\n\n"

        if field_accuracy:
            md += "## Field-Level Accuracy\n\n"
            md += "| Field | Errors | Corrected | False Corrections | Accuracy |\n"
            md += "|---|---|---|---|---|\n"
            for field, fa in field_accuracy.items():
                md += (
                    f"| {field} "
                    f"| {fa.get('errors_in_original', 0)} "
                    f"| {fa.get('errors_corrected', 0)} "
                    f"| {fa.get('false_corrections', 0)} "
                    f"| {fa.get('accuracy', 0):.3f} |\n"
                )
        return md

    # ─────────────────────────────────────────────────────────
    # Mode 2 — Legacy LLM evaluation (fallback)
    # ─────────────────────────────────────────────────────────

    async def _evaluate_legacy(
        self, raw_data: list[dict], corrections: list[dict]
    ) -> dict:
        """Original LLM-only evaluation — kept as fallback."""
        payload     = self._build_legacy_payload(raw_data, corrections)
        llm_result  = await self._call_legacy_llm(payload)
        saved_files = self._save_outputs(
            llm_result.get("overall_metrics", {}),
            llm_result.get("field_accuracy", {}),
            llm_result.get("markdown_report", ""),
            payload,
        )

        m = llm_result.get("overall_metrics", {})
        print(f"  ✓ Recall:    {_to_float(m.get('recall', 0)):.3f}")
        print(f"  ✓ Precision: {_to_float(m.get('precision', 0)):.3f}")
        print(f"  ✓ F1:        {_to_float(m.get('f1', 0)):.3f}")

        return {
            "strategy":        self.strategy.value,
            "overall_metrics": m,
            "field_accuracy":  llm_result.get("field_accuracy", {}),
            "markdown_report": llm_result.get("markdown_report", ""),
            "saved_files":     saved_files,
        }

    def _build_legacy_payload(
        self, raw_data: list[dict], corrections: list[dict]
    ) -> list[dict]:
        raw_map  = {
            item["entry"].get("id"): item
            for item in raw_data
            if isinstance(item, dict) and isinstance(item.get("entry"), dict)
        }
        corr_map = {
            c["entry_id"]: c
            for c in corrections
            if isinstance(c, dict) and c.get("entry_id")
        }
        payload = []
        for entry_id, raw_item in raw_map.items():
            dblp_hits    = raw_item.get("dblp_hits", [])
            ground_truth = (dblp_hits[0]
                            if isinstance(dblp_hits, list) and dblp_hits
                            else {})
            correction      = corr_map.get(entry_id, {})
            corrected_entry = correction.get("corrected", raw_item["entry"])
            payload.append({
                "entry_id":     entry_id,
                "original":     raw_item["entry"],
                "corrected":    corrected_entry,
                "ground_truth": ground_truth,
                "changes":      correction.get("changes", []),
            })
        return payload

    async def _call_legacy_llm(self, payload: list[dict]) -> dict:
        messages = [
            SystemMessage(content=self.LEGACY_SYSTEM_PROMPT),
            HumanMessage(content=(
                "Evaluate these correction records:\n\n"
                f"```json\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n```"
            )),
        ]
        response = await self.llm.ainvoke(messages)
        raw_text = _extract_text(response)

        if "```json" in raw_text:
            raw_text = raw_text.split("```json", 1)[1].split("```")[0].strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            return {
                "overall_metrics": {
                    "true_positives": 0, "false_positives": 0,
                    "false_negatives": 0, "recall": 0.0,
                    "precision": 0.0, "f1": 0.0,
                },
                "field_accuracy":  {},
                "markdown_report": "Evaluation output could not be parsed.",
            }

    # ── file output (shared) ──────────────────────────────────

    def _save_outputs(
        self,
        overall_metrics: dict,
        field_accuracy: dict,
        markdown_report: str,
        detailed: list[dict],
    ) -> list[str]:
        saved = []

        # Normalise types
        overall_metrics = {
            "true_positives":  _to_int(overall_metrics.get("true_positives", 0)),
            "false_positives": _to_int(overall_metrics.get("false_positives", 0)),
            "false_negatives": _to_int(overall_metrics.get("false_negatives", 0)),
            "recall":          _to_float(overall_metrics.get("recall", 0.0)),
            "precision":       _to_float(overall_metrics.get("precision", 0.0)),
            "f1":              _to_float(overall_metrics.get("f1", 0.0)),
        }
        field_accuracy = {
            f: {
                "errors_in_original": _to_int(v.get("errors_in_original", 0)),
                "errors_corrected":   _to_int(v.get("errors_corrected", 0)),
                "false_corrections":  _to_int(v.get("false_corrections", 0)),
                "accuracy":           _to_float(v.get("accuracy", 0.0)),
            }
            for f, v in field_accuracy.items()
            if isinstance(v, dict)
        }

        p = self.output_dir / "evaluation_metrics.json"
        p.write_text(json.dumps({
            "overall_metrics": overall_metrics,
            "field_accuracy":  field_accuracy,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  ✓ evaluation_metrics.json  → {p}")
        saved.append(str(p))

        p = self.output_dir / "evaluation_report.md"
        p.write_text(markdown_report or "", encoding="utf-8")
        print(f"  ✓ evaluation_report.md     → {p}")
        saved.append(str(p))

        p = self.output_dir / "evaluation_details.json"
        p.write_text(json.dumps({"detailed_results": detailed},
                                indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  ✓ evaluation_details.json  → {p}")
        saved.append(str(p))

        return saved


# ─────────────────────────────────────────────────────────────
# Cross-strategy comparison  (unchanged interface)
# ─────────────────────────────────────────────────────────────

def build_comparison_report(
    results: list[dict],
    output_dir: str = "evaluation",
) -> str:
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)

    md  = "# Strategy Comparison Report\n\n"
    md += "## Overall Metrics\n\n"
    md += "| Strategy | Precision | Recall | F1 | TP | FP | FN |\n"
    md += "|---|---|---|---|---|---|---|\n"

    for r in results:
        m = r.get("overall_metrics", {})
        md += (
            f"| {r.get('strategy', '?'):12s} "
            f"| {_to_float(m.get('precision', 0)):.3f} "
            f"| {_to_float(m.get('recall', 0)):.3f} "
            f"| {_to_float(m.get('f1', 0)):.3f} "
            f"| {_to_int(m.get('true_positives', 0))} "
            f"| {_to_int(m.get('false_positives', 0))} "
            f"| {_to_int(m.get('false_negatives', 0))} |\n"
        )

    all_fields = sorted({
        f for r in results for f in r.get("field_accuracy", {})
    })

    if all_fields:
        md += "\n## Field-Level Accuracy by Strategy\n\n"
        for field in all_fields:
            md += f"### {field}\n\n"
            md += "| Strategy | Errors | Corrected | False Corrections | Accuracy |\n"
            md += "|---|---|---|---|---|\n"
            for r in results:
                fa = r.get("field_accuracy", {}).get(field, {})
                md += (
                    f"| {r.get('strategy', '?'):12s} "
                    f"| {_to_int(fa.get('errors_in_original', 0))} "
                    f"| {_to_int(fa.get('errors_corrected', 0))} "
                    f"| {_to_int(fa.get('false_corrections', 0))} "
                    f"| {_to_float(fa.get('accuracy', 0)):.3f} |\n"
                )
            md += "\n"

    if results:
        best_f1  = max(results, key=lambda r: _to_float(r.get("overall_metrics", {}).get("f1", 0)))
        best_pre = max(results, key=lambda r: _to_float(r.get("overall_metrics", {}).get("precision", 0)))
        best_rec = max(results, key=lambda r: _to_float(r.get("overall_metrics", {}).get("recall", 0)))

        md += "## Key Insights\n\n"
        md += f"- **Best F1**        : `{best_f1.get('strategy')}` ({_to_float(best_f1.get('overall_metrics',{}).get('f1',0)):.3f})\n"
        md += f"- **Best Precision** : `{best_pre.get('strategy')}` ({_to_float(best_pre.get('overall_metrics',{}).get('precision',0)):.3f})\n"
        md += f"- **Best Recall**    : `{best_rec.get('strategy')}` ({_to_float(best_rec.get('overall_metrics',{}).get('recall',0)):.3f})\n"

    md_path = folder / "comparison.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"\n  ✓ comparison.md   → {md_path}")

    json_path = folder / "comparison.json"
    json_path.write_text(
        json.dumps({"strategies": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  ✓ comparison.json → {json_path}")

    return md