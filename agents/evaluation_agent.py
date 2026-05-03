"""
BibTeX Evaluation Agent  —  LLM-powered
-----------------------------------------
Rather than manually computing metrics and hardcoding report templates,
we pass the corrections data directly to an LLM and ask it to:
  1. Compute precision, recall, F1, and field-level accuracy
  2. Interpret the results
  3. Return a structured JSON metrics block
  4. Write a human-readable Markdown report

The LLM response is parsed and saved as:
  - evaluation_metrics.json
  - evaluation_report.md
  - evaluation_details.json
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

class EvaluationAgent:
    """LLM-powered evaluation of BibTeX correction quality."""

    MODEL = "qwen3-coder:480b-cloud"

    SYSTEM_PROMPT = """\
You are an expert bibliographic data quality evaluator.

You will receive:
  - A list of BibTeX correction records produced by a correction agent.
  - Each record contains: entry_id, original entry fields, corrected entry fields,
    DBLP ground-truth fields, validation_status (valid/partially_valid/invalid), 
    and a list of changes made.

Your job:
1. Compute the following OVERALL metrics by comparing original → corrected → ground_truth:
   - true_positives  : fields that had an error AND were correctly fixed
   - false_positives : fields that were correct but got wrongly changed
   - false_negatives : fields that had an error but were NOT fixed
   - recall    = TP / (TP + FN)
   - precision = TP / (TP + FP)
   - f1        = 2 * precision * recall / (precision + recall)

2. Compute PER-CLASS metrics GROUPED BY VALIDATION STATUS (valid, partially_valid, invalid):
   For each validation status class, calculate:
   - count: number of entries in this class
   - class_true_positives, class_false_positives, class_false_negatives
   - class_recall, class_precision, class_f1
   - class_field_accuracy (percentage of fields in this class that match ground truth)
   Note: If a class has no records where ground_truth exists, set recall/precision/f1 to null

3. Break down accuracy per FIELD (title, author, year, journal/venue) both:
   - Overall field accuracy across all entries
   - Per-class field accuracy (accuracy within each valid/partially_valid/invalid subset)

4. Write a concise but insightful Markdown evaluation report that includes:
   - An overall metrics table
   - A per-class metrics table (valid, partially_valid, invalid) with entry counts
   - A field-level accuracy table (overall + breakdown by class)
   - Key insights and recommendations (e.g., "Field X had higher accuracy in partially_valid class")

You MUST respond with valid JSON in exactly this structure — no extra text:
{
  "overall_metrics": {
    "true_positives": <int>,
    "false_positives": <int>,
    "false_negatives": <int>,
    "recall": <float 0-1>,
    "precision": <float 0-1>,
    "f1": <float 0-1>
  },
  "per_class_metrics": {
    "valid": {
      "count": <int>,
      "true_positives": <int>,
      "false_positives": <int>,
      "false_negatives": <int>,
      "recall": <float 0-1 or null if no ground truth>,
      "precision": <float 0-1 or null if no ground truth>,
      "f1": <float 0-1 or null if no ground truth>,
      "field_accuracy": <float 0-1>
    },
    "partially_valid": {
      "count": <int>,
      "true_positives": <int>,
      "false_positives": <int>,
      "false_negatives": <int>,
      "recall": <float 0-1 or null if no ground truth>,
      "precision": <float 0-1 or null if no ground truth>,
      "f1": <float 0-1 or null if no ground truth>,
      "field_accuracy": <float 0-1>
    },
    "invalid": {
      "count": <int>,
      "true_positives": <int>,
      "false_positives": <int>,
      "false_negatives": <int>,
      "recall": <float 0-1 or null if no ground truth>,
      "precision": <float 0-1 or null if no ground truth>,
      "f1": <float 0-1 or null if no ground truth>,
      "field_accuracy": <float 0-1>
    }
  },
  "field_accuracy": {
    "<field_name>": {
      "overall_accuracy": <float 0-1>,
      "errors_in_original": <int>,
      "errors_corrected": <int>,
      "false_corrections": <int>,
      "per_class": {
        "valid": <float 0-1>,
        "partially_valid": <float 0-1>,
        "invalid": <float 0-1>
      }
    }
  },
  "markdown_report": "<full markdown string>"
}
"""

    def __init__(self, output_dir: str = "evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.llm = ChatOllama(
            model=self.MODEL,
            base_url="https://ollama.com",
            temperature=0.1,
            client_kwargs={
                "headers": {"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"}
            }
        )

    # ── public ────────────────────────────────────────────────

    async def evaluate(
        self,
        raw_data: list[dict],
        corrections: list[dict],
        validation_structured: list[dict] | None = None,
    ) -> dict:
        """
        Evaluate correction quality using an LLM.

        Parameters
        ----------
        raw_data    : [{entry, dblp_hits}, ...]   — original entries + DBLP matches
        corrections : [{entry_id, original,
                        corrected, changes}, ...]  — output from correction agent
        validation_structured : [{entry_id, status, ...}, ...]  — validation results with status

        Returns
        -------
        dict with overall_metrics, per_class_metrics, field_accuracy, markdown_report, saved_files
        """
        print(f"\n{'='*60}")
        print("EVALUATION AGENT  —  LLM-powered")
        print(f"{'='*60}\n")

        # ── 1. Build the payload the LLM will reason over ─────
        evaluation_payload = self._build_payload(raw_data, corrections, validation_structured or [])
        print(f"  → Built payload: {len(evaluation_payload)} entries to evaluate")

        # ── 2. Call the LLM ───────────────────────────────────
        print("  → Sending to LLM for evaluation …")
        llm_result = self._call_llm(evaluation_payload)

        # ── 3. Save outputs ───────────────────────────────────
        saved_files = self._save_outputs(llm_result, evaluation_payload)

        print(f"\n  ✓ Overall Recall:    {llm_result['overall_metrics']['recall']:.1%}")
        print(f"  ✓ Overall Precision: {llm_result['overall_metrics']['precision']:.1%}")
        print(f"  ✓ Overall F1:        {llm_result['overall_metrics']['f1']:.3f}")
        
        # Print per-class metrics if available
        if "per_class_metrics" in llm_result:
            print(f"\n  Per-Class Metrics:")
            for class_name, metrics in llm_result["per_class_metrics"].items():
                count = metrics.get("count", 0)
                recall = metrics.get("recall")
                precision = metrics.get("precision")
                f1 = metrics.get("f1")
                acc = metrics.get("field_accuracy", 0)
                
                if recall is not None:
                    print(f"    {class_name:20s}: {count:3d} entries | R:{recall:.1%} P:{precision:.1%} F1:{f1:.3f} | Acc:{acc:.1%}")
                else:
                    print(f"    {class_name:20s}: {count:3d} entries | (N/A - no ground truth positives)")

        return {
            "overall_metrics":    llm_result["overall_metrics"],
            "per_class_metrics":  llm_result.get("per_class_metrics", {}),
            "field_accuracy":     llm_result.get("field_accuracy", {}),
            "markdown_report":    llm_result["markdown_report"],
            "saved_files":        saved_files,
        }

    # ── private ───────────────────────────────────────────────

    def _build_payload(self, raw_data: list[dict], corrections: list[dict], validation_structured: list[dict]) -> list[dict]:
        """
        Merge raw_data, corrections, and validation results into a single list of records
        that gives the LLM everything it needs to evaluate each entry.

        Each record:
        {
          "entry_id"    : str,
          "original"    : {field: value, ...},
          "corrected"   : {field: value, ...},
          "ground_truth": {field: value, ...},   ← best DBLP hit
          "changes"     : [{field, from, to}, ...],
          "validation_status": "valid|partially_valid|invalid"  ← from validation agent
        }
        """
        raw_map  = {item["entry"].get("id"): item for item in raw_data}
        corr_map = {c["entry_id"]: c for c in corrections}
        val_map  = {v.get("entry_id"): v.get("status", "unknown") for v in validation_structured if isinstance(v, dict)}

        payload = []
        for entry_id, raw_item in raw_map.items():
            dblp_hits = raw_item.get("dblp_hits", [])
            ground_truth = dblp_hits[0] if dblp_hits else {}

            correction = corr_map.get(entry_id, {})
            validation_status = val_map.get(entry_id, "unknown")

            payload.append({
                "entry_id":     entry_id,
                "original":     raw_item["entry"],
                "corrected":    correction.get("corrected", raw_item["entry"]),
                "ground_truth": ground_truth,
                "changes":      correction.get("changes", []),
                "validation_status": validation_status,
            })

        return payload

    def _call_llm(self, payload: list[dict]) -> dict:
        """
        Send the payload to Ollama and parse its JSON response.

        Returns the parsed dict with keys:
          overall_metrics, field_accuracy, markdown_report
        """
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=(
                "Here are the BibTeX correction records to evaluate:\n\n"
                f"```json\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n```\n\n"
                "Please evaluate them and return the JSON result as specified."
            )),
        ]

        response = self.llm.invoke(messages)
        raw_text = response.content.strip()

        # Strip ```json fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        return json.loads(raw_text)

    def _save_outputs(self, llm_result: dict, detailed_payload: list[dict]) -> list[str]:
        """Write the three output files and return their paths."""
        saved = []

        # evaluation_metrics.json  — just the numbers
        metrics_path = self.output_dir / "evaluation_metrics.json"
        metrics_path.write_text(
            json.dumps({
                "overall_metrics": llm_result["overall_metrics"],
                "per_class_metrics": llm_result.get("per_class_metrics", {}),
                "field_accuracy":  llm_result.get("field_accuracy", {}),
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  ✓ Saved: {metrics_path}")
        saved.append(str(metrics_path))

        # evaluation_report.md  — LLM-written narrative
        report_path = self.output_dir / "evaluation_report.md"
        report_path.write_text(llm_result["markdown_report"], encoding="utf-8")
        print(f"  ✓ Saved: {report_path}")
        saved.append(str(report_path))

        # evaluation_details.json  — full per-entry breakdown
        details_path = self.output_dir / "evaluation_details.json"
        details_path.write_text(
            json.dumps({"detailed_results": detailed_payload}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  ✓ Saved: {details_path}")
        saved.append(str(details_path))

        return saved