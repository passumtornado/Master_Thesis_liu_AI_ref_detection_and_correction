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
    DBLP ground-truth fields, and a list of changes made.

Your job:
1. Compute the following metrics by comparing original → corrected → ground_truth:
   - true_positives  : fields that had an error AND were correctly fixed
   - false_positives : fields that were correct but got wrongly changed
   - false_negatives : fields that had an error but were NOT fixed
   - recall    = TP / (TP + FN)
   - precision = TP / (TP + FP)
   - f1        = 2 * precision * recall / (precision + recall)

2. Break down accuracy per field (title, author, year, journal/venue).

3. Write a concise but insightful Markdown evaluation report that includes:
   - An overall metrics table
   - A field-level accuracy table
   - Key insights and recommendations

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
  "field_accuracy": {
    "<field_name>": {
      "errors_in_original": <int>,
      "errors_corrected": <int>,
      "false_corrections": <int>,
      "accuracy": <float 0-1>
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
    ) -> dict:
        """
        Evaluate correction quality using an LLM.

        Parameters
        ----------
        raw_data    : [{entry, dblp_hits}, ...]   — original entries + DBLP matches
        corrections : [{entry_id, original,
                        corrected, changes}, ...]  — output from correction agent

        Returns
        -------
        dict with overall_metrics, field_accuracy, markdown_report, saved_files
        """
        print(f"\n{'='*60}")
        print("EVALUATION AGENT  —  LLM-powered")
        print(f"{'='*60}\n")

        # ── 1. Build the payload the LLM will reason over ─────
        evaluation_payload = self._build_payload(raw_data, corrections)
        print(f"  → Built payload: {len(evaluation_payload)} entries to evaluate")

        # ── 2. Call the LLM ───────────────────────────────────
        print("  → Sending to LLM for evaluation …")
        llm_result = self._call_llm(evaluation_payload)

        # ── 3. Save outputs ───────────────────────────────────
        saved_files = self._save_outputs(llm_result, evaluation_payload)

        print(f"\n  ✓ Recall:    {llm_result['overall_metrics']['recall']:.1%}")
        print(f"  ✓ Precision: {llm_result['overall_metrics']['precision']:.1%}")
        print(f"  ✓ F1:        {llm_result['overall_metrics']['f1']:.3f}")

        return {
            "overall_metrics":  llm_result["overall_metrics"],
            "field_accuracy":   llm_result["field_accuracy"],
            "markdown_report":  llm_result["markdown_report"],
            "saved_files":      saved_files,
        }

    # ── private ───────────────────────────────────────────────

    def _build_payload(self, raw_data: list[dict], corrections: list[dict]) -> list[dict]:
        """
        Merge raw_data and corrections into a single list of records
        that gives the LLM everything it needs to evaluate each entry.

        Each record:
        {
          "entry_id"    : str,
          "original"    : {field: value, ...},
          "corrected"   : {field: value, ...},
          "ground_truth": {field: value, ...},   ← best DBLP hit
          "changes"     : [{field, from, to}, ...]
        }
        """
        raw_map  = {item["entry"].get("id"): item for item in raw_data}
        corr_map = {c["entry_id"]: c for c in corrections}

        payload = []
        for entry_id, raw_item in raw_map.items():
            dblp_hits = raw_item.get("dblp_hits", [])
            ground_truth = dblp_hits[0] if dblp_hits else {}

            correction = corr_map.get(entry_id, {})

            payload.append({
                "entry_id":     entry_id,
                "original":     raw_item["entry"],
                "corrected":    correction.get("corrected", raw_item["entry"]),
                "ground_truth": ground_truth,
                "changes":      correction.get("changes", []),
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
                "field_accuracy":  llm_result["field_accuracy"],
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