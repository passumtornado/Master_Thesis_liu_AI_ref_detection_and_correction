"""
BibTeX Evaluation Agent - LLM-powered
-------------------------------------
Evaluates correction quality and writes metrics/report artefacts.

Supports three prompting strategies (mirrors CorrectionAgent):
  - zero_shot : baseline — evaluate corrections made without DBLP evidence
  - rag       : evaluate corrections made with DBLP evidence (default)
  - cot       : evaluate corrections made with CoT + DBLP reasoning

Fixes applied vs original:
  - invoke → ainvoke (async-safe)
  - PromptStrategy enum wired in
  - output_dir scoped per strategy (no overwrites across runs)
  - SYSTEM_PROMPT fully specifies JSON schema and metric definitions
  - build_comparison_report() added as module-level function
"""

import json
import os
import sys
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_ollama import ChatOllama

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import _extract_text

load_dotenv()


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


# ─────────────────────────────────────────────────────────────
# Prompt Strategy  (must match CorrectionAgent)
# ─────────────────────────────────────────────────────────────

class PromptStrategy(Enum):
    ZERO_SHOT = "zero_shot"
    RAG       = "rag"
    COT       = "cot"


# ─────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────

class EvaluationAgent:
    """LLM-powered evaluation of BibTeX correction quality."""

    MODEL = "qwen3-coder:480b-cloud"

    SYSTEM_PROMPT = """You are an expert bibliographic data quality evaluator.

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
  recall    = TP / (TP + FN)                          how many errors were caught
  precision = TP / (TP + FP)                          how many corrections were right
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
  "markdown_report": "<complete markdown string with summary table and per-entry details>"
}
"""

    def __init__(
        self,
        output_dir: str = "evaluation",
        strategy: PromptStrategy = PromptStrategy.RAG,
    ):
        self.strategy = strategy
        # Each strategy writes to its own subfolder — no overwrites across runs
        self.output_dir = Path(output_dir) / strategy.value
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.llm = ChatOllama(
            model=self.MODEL,
            base_url="https://ollama.com",
            temperature=0.1,
            client_kwargs={
                "headers": {"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"}
            },
        )
        
        #OpenAI backend (uncomment to switch)
        # self.llm = ChatOpenAI(
        #     model="gpt-5.2",
        #     temperature=0.1,
        #     openai_api_key=os.getenv("OPENAI_API_KEY"),
        # )
        
        
        # self.llm = ChatGoogleGenerativeAI(
        #     model="gemini-3.1-pro-preview",
        #     temperature=0.1,
        #     google_api_key=os.getenv("GOOGLE_API_KEY"),
        # )


        # Optional HuggingFace backend — uncomment to switch
        # self.llm = ChatHuggingFace(
        #     llm=HuggingFaceEndpoint(
        #         repo_id="openai/gpt-oss-20b",
        #         task="text-generation",
        #         huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        #         max_new_tokens=2048,
        #         temperature=0.1,
        #     )
        # )

    # ── public ────────────────────────────────────────────────

    async def evaluate(self, raw_data: list[dict], corrections: list[dict]) -> dict:
        """Run LLM evaluation and persist outputs."""
        print(f"\n{'='*60}")
        print(f"EVALUATION AGENT [{self.strategy.value.upper()}]")
        print(f"{'='*60}\n")

        evaluation_payload = self._build_payload(raw_data, corrections)
        print(f"  Evaluating {len(evaluation_payload)} entries …")

        llm_result  = await self._call_llm(evaluation_payload)
        saved_files = self._save_outputs(llm_result, evaluation_payload)

        m = llm_result.get("overall_metrics", {})
        print(f"  ✓ Recall:    {_to_float(m.get('recall', 0)):.3f}")
        print(f"  ✓ Precision: {_to_float(m.get('precision', 0)):.3f}")
        print(f"  ✓ F1:        {_to_float(m.get('f1', 0)):.3f}")

        return {
            "strategy":        self.strategy.value,
            "overall_metrics": llm_result.get("overall_metrics", {}),
            "field_accuracy":  llm_result.get("field_accuracy", {}),
            "markdown_report": llm_result.get("markdown_report", ""),
            "saved_files":     saved_files,
        }

    # ── private: payload building ─────────────────────────────

    def _build_payload(
        self, raw_data: list[dict], corrections: list[dict]
    ) -> list[dict]:
        """Merge raw validation evidence with correction outputs."""
        raw_map = {}
        for item in raw_data:
            if not isinstance(item, dict):
                continue
            entry    = item.get("entry", {})
            entry_id = entry.get("id") if isinstance(entry, dict) else None
            if entry_id:
                raw_map[entry_id] = item

        corr_map = {}
        for c in corrections:
            if not isinstance(c, dict):
                continue
            entry_id = c.get("entry_id")
            if entry_id:
                corr_map[entry_id] = c

        payload = []
        for entry_id, raw_item in raw_map.items():
            dblp_hits    = raw_item.get("dblp_hits", [])
            ground_truth = dblp_hits[0] if isinstance(dblp_hits, list) and dblp_hits else {}
            if not isinstance(ground_truth, dict):
                ground_truth = {}

            correction      = corr_map.get(entry_id, {})
            corrected_entry = correction.get("corrected", raw_item["entry"])
            if not isinstance(corrected_entry, dict):
                corrected_entry = raw_item["entry"]

            changes = correction.get("changes", [])
            if not isinstance(changes, list):
                changes = []

            payload.append({
                "entry_id":     entry_id,
                "original":     raw_item["entry"],
                "corrected":    corrected_entry,
                "ground_truth": ground_truth,
                "changes":      changes,
            })

        return payload

    # ── private: LLM call ─────────────────────────────────────

    async def _call_llm(self, payload: list[dict]) -> dict:
        """Send payload to LLM and parse JSON response."""
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=(
                "Here are the correction records to evaluate:\n\n"
                f"```json\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n```"
            )),
        ]

        response = await self.llm.ainvoke(messages)
        raw_text = _extract_text(response)

        # Strip ```json fences if present
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
            # Fallback keeps pipeline alive even when model returns malformed JSON
            return {
                "overall_metrics": {
                    "true_positives":  0,
                    "false_positives": 0,
                    "false_negatives": 0,
                    "recall":          0.0,
                    "precision":       0.0,
                    "f1":              0.0,
                },
                "field_accuracy":  {},
                "markdown_report": (
                    "# Evaluation Report\n\n"
                    "Evaluation output could not be parsed as JSON from the LLM.\n"
                    "Please inspect prompts/model behaviour and rerun."
                ),
            }

    # ── private: file output ──────────────────────────────────

    def _save_outputs(
        self, llm_result: dict, detailed_payload: list[dict]
    ) -> list[str]:
        """Write metrics / report / details files and return paths."""
        saved = []

        if not isinstance(llm_result, dict):
            llm_result = {}

        overall_metrics = llm_result.get("overall_metrics", {})
        field_accuracy  = llm_result.get("field_accuracy", {})
        markdown_report = llm_result.get("markdown_report", "")

        if not isinstance(overall_metrics, dict):
            overall_metrics = {}
        if not isinstance(field_accuracy, dict):
            field_accuracy = {}
        if not isinstance(markdown_report, str):
            markdown_report = str(markdown_report)

        # Normalize overall metric value types to avoid downstream formatting errors.
        overall_metrics = {
            "true_positives":  _to_int(overall_metrics.get("true_positives", 0)),
            "false_positives": _to_int(overall_metrics.get("false_positives", 0)),
            "false_negatives": _to_int(overall_metrics.get("false_negatives", 0)),
            "recall":          _to_float(overall_metrics.get("recall", 0.0)),
            "precision":       _to_float(overall_metrics.get("precision", 0.0)),
            "f1":              _to_float(overall_metrics.get("f1", 0.0)),
        }

        normalized_field_accuracy = {}
        for field, values in field_accuracy.items():
            if not isinstance(values, dict):
                continue
            normalized_field_accuracy[field] = {
                "errors_in_original": _to_int(values.get("errors_in_original", 0)),
                "errors_corrected":   _to_int(values.get("errors_corrected", 0)),
                "false_corrections":  _to_int(values.get("false_corrections", 0)),
                "accuracy":           _to_float(values.get("accuracy", 0.0)),
            }
        field_accuracy = normalized_field_accuracy

        metrics_path = self.output_dir / "evaluation_metrics.json"
        metrics_path.write_text(
            json.dumps(
                {"overall_metrics": overall_metrics, "field_accuracy": field_accuracy},
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"  ✓ evaluation_metrics.json  → {metrics_path}")
        saved.append(str(metrics_path))

        report_path = self.output_dir / "evaluation_report.md"
        report_path.write_text(markdown_report, encoding="utf-8")
        print(f"  ✓ evaluation_report.md     → {report_path}")
        saved.append(str(report_path))

        details_path = self.output_dir / "evaluation_details.json"
        details_path.write_text(
            json.dumps({"detailed_results": detailed_payload}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  ✓ evaluation_details.json  → {details_path}")
        saved.append(str(details_path))

        return saved


# ─────────────────────────────────────────────────────────────
# Module-level: cross-strategy comparison
# ─────────────────────────────────────────────────────────────

def build_comparison_report(
    results: list[dict],
    output_dir: str = "evaluation",
) -> str:
    """
    Build a side-by-side comparison of all three strategy runs.

    Parameters
    ----------
    results    : list of dicts returned by EvaluationAgent.evaluate()
                 each must contain 'strategy', 'overall_metrics', 'field_accuracy'
    output_dir : root evaluation folder (comparison files written here)

    Returns
    -------
    str  — the markdown comparison text
    """
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)

    # ── Overall metrics table ─────────────────────────────────
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

    # ── Field-level accuracy per strategy ────────────────────
    all_fields = sorted({
        field
        for r in results
        for field in r.get("field_accuracy", {}).keys()
    })

    if all_fields:
        md += "\n## Field-Level Accuracy by Strategy\n\n"

        for field in all_fields:
            md += f"### {field}\n\n"
            md += "| Strategy | Errors in Original | Corrected | False Corrections | Accuracy |\n"
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

    # ── Key insights ──────────────────────────────────────────
    if results:
        best_f1        = max(results, key=lambda r: _to_float(r.get("overall_metrics", {}).get("f1", 0)))
        best_precision = max(results, key=lambda r: _to_float(r.get("overall_metrics", {}).get("precision", 0)))
        best_recall    = max(results, key=lambda r: _to_float(r.get("overall_metrics", {}).get("recall", 0)))

        md += "## Key Insights\n\n"
        md += f"- **Best F1**        : `{best_f1.get('strategy')}` "
        md += f"({_to_float(best_f1.get('overall_metrics', {}).get('f1', 0)):.3f})\n"
        md += f"- **Best Precision** : `{best_precision.get('strategy')}` "
        md += f"({_to_float(best_precision.get('overall_metrics', {}).get('precision', 0)):.3f})\n"
        md += f"- **Best Recall**    : `{best_recall.get('strategy')}` "
        md += f"({_to_float(best_recall.get('overall_metrics', {}).get('recall', 0)):.3f})\n"

    # ── Save ──────────────────────────────────────────────────
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