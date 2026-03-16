"""
BibTeX Evaluation Agent
-----------------------
Calculates precision, recall, F1, and field-level accuracy for corrections.

Metrics:
  - Recall: % of errors detected and corrected
  - Precision: % of corrections that were accurate
  - F1: Harmonic mean of recall and precision
  - Field accuracy: Per-field error rates (title, author, year, venue)
"""

import json
from collections import defaultdict
from pathlib import Path


class EvaluationAgent:
    """Evaluate validation and correction quality."""

    def __init__(self, output_dir: str = "evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def evaluate(
        self,
        raw_data: list[dict],
        corrections: list[dict],
    ) -> dict:
        """
        Evaluate validation and corrections against DBLP ground truth.
        
        Input:
          - raw_data: [{entry, dblp_hits}, ...]
          - corrections: [{entry_id, original, corrected, changes}, ...]
        
        Returns:
          {
            "overall_metrics": {recall, precision, f1},
            "field_accuracy": {field: {errors, corrected, accuracy}},
            "detailed_results": [...],
          }
        """
        print(f"\n{'='*60}")
        print(f"EVALUATION AGENT — Evaluating corrections")
        print(f"{'='*60}\n")

        # Build lookup for raw data
        raw_map = {item["entry"].get("id"): item for item in raw_data}
        
        # Build lookup for corrections
        corr_map = {c["entry_id"]: c for c in corrections}

        # Metrics tracking
        total_errors_in_original = 0
        errors_corrected = 0
        false_corrections = 0

        field_metrics = defaultdict(lambda: {
            "errors_in_original": 0,
            "errors_corrected": 0,
            "false_corrections": 0,
        })

        detailed_results = []

        # Evaluate each entry
        for entry_id, raw_item in raw_map.items():
            original = raw_item["entry"]
            dblp_matches = raw_item["dblp_hits"]

            if not dblp_matches:
                continue

            # Ground truth from best DBLP match
            ground_truth = dblp_matches[0]

            # Check if this entry was corrected
            corrected = corr_map.get(entry_id, {}).get("corrected", original)

            # Compare fields
            field_results = {}
            for field in ["title", "author", "year", "journal"]:
                orig_val = str(original.get(field, "")).strip().lower()
                corr_val = str(corrected.get(field, "")).strip().lower()
                truth_val = str(ground_truth.get(field, "") or ground_truth.get("venue", "")).strip().lower()

                if not truth_val:
                    continue

                had_error = orig_val != truth_val
                is_corrected = corr_val == truth_val
                had_false_correction = orig_val == truth_val and corr_val != truth_val

                if had_error:
                    total_errors_in_original += 1
                    field_metrics[field]["errors_in_original"] += 1

                    if is_corrected:
                        errors_corrected += 1
                        field_metrics[field]["errors_corrected"] += 1

                if had_false_correction:
                    false_corrections += 1
                    field_metrics[field]["false_corrections"] += 1

                field_results[field] = {
                    "original": original.get(field, ""),
                    "corrected": corrected.get(field, ""),
                    "ground_truth": ground_truth.get(field, "") or ground_truth.get("venue", ""),
                    "had_error": had_error,
                    "is_corrected": is_corrected,
                }

            detailed_results.append({
                "entry_id": entry_id,
                "field_results": field_results,
            })

        # Calculate overall metrics
        recall = errors_corrected / total_errors_in_original if total_errors_in_original > 0 else 0.0
        precision = errors_corrected / (errors_corrected + false_corrections) if (errors_corrected + false_corrections) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Calculate field-level metrics
        field_accuracy = {}
        for field, metrics in field_metrics.items():
            errors = metrics["errors_in_original"]
            corrected = metrics["errors_corrected"]
            false_corr = metrics["false_corrections"]
            accuracy = corrected / errors if errors > 0 else 0.0

            field_accuracy[field] = {
                "errors_in_original": errors,
                "errors_corrected": corrected,
                "false_corrections": false_corr,
                "accuracy": round(accuracy, 3),
            }

        # Generate reports
        summary_md = self._generate_summary_md(
            recall, precision, f1, field_accuracy, total_errors_in_original
        )
        summary_json = self._generate_summary_json(
            recall, precision, f1, field_accuracy
        )

        # Save files
        md_path = self.output_dir / "evaluation_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(summary_md)
        print(f"✓ Saved: {md_path}")

        json_path = self.output_dir / "evaluation_metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_json, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved: {json_path}\n")

        details_path = self.output_dir / "evaluation_details.json"
        with open(details_path, "w", encoding="utf-8") as f:
            json.dump({"detailed_results": detailed_results}, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved: {details_path}\n")

        return {
            "overall_metrics": {
                "recall": round(recall, 3),
                "precision": round(precision, 3),
                "f1": round(f1, 3),
                "total_errors_in_original": total_errors_in_original,
                "errors_corrected": errors_corrected,
                "false_corrections": false_corrections,
            },
            "field_accuracy": field_accuracy,
            "detailed_results": detailed_results,
            "summary": summary_md,
            "saved_files": [str(md_path), str(json_path), str(details_path)],
        }

    @staticmethod
    def _generate_summary_md(recall: float, precision: float, f1: float, field_accuracy: dict, total_errors: int) -> str:
        """Generate markdown evaluation report."""
        md = "# BibTeX Correction Evaluation Report\n\n"
        md += "## Overall Metrics\n\n"
        md += "| Metric | Value |\n|---|---|\n"
        md += f"| Recall | {recall:.1%} |\n"
        md += f"| Precision | {precision:.1%} |\n"
        md += f"| F1 Score | {f1:.3f} |\n"
        md += f"| Total Errors in Original | {total_errors} |\n"
        md += "\n"

        md += "**Interpretation:**\n"
        md += f"- **Recall {recall:.1%}**: Out of {total_errors} errors, {int(recall * total_errors)} were detected and corrected\n"
        md += f"- **Precision {precision:.1%}**: Of the corrections made, {precision:.1%} were accurate\n"
        md += f"- **F1 {f1:.3f}**: Harmonic mean balancing recall and precision\n\n"

        # Field-level accuracy
        if field_accuracy:
            md += "## Field-Level Accuracy\n\n"
            md += "| Field | Errors | Corrected | Accuracy |\n|---|---|---|---|\n"
            for field in sorted(field_accuracy.keys()):
                metrics = field_accuracy[field]
                md += (
                    f"| {field} | {metrics['errors_in_original']} | "
                    f"{metrics['errors_corrected']} | {metrics['accuracy']:.1%} |\n"
                )
            md += "\n"

        md += "## Key Insights\n\n"
        if recall > 0.8:
            md += "✓ **Strong recall**: Most errors were detected\n"
        elif recall > 0.5:
            md += "⚠ **Moderate recall**: About half of errors were caught\n"
        else:
            md += "✗ **Low recall**: Most errors were missed\n"

        if precision > 0.8:
            md += "✓ **Strong precision**: Corrections are mostly accurate\n"
        elif precision > 0.5:
            md += "⚠ **Moderate precision**: Some corrections created new errors\n"
        else:
            md += "✗ **Low precision**: Many corrections were inaccurate\n"

        return md

    @staticmethod
    def _generate_summary_json(recall: float, precision: float, f1: float, field_accuracy: dict) -> dict:
        """Generate JSON evaluation metrics."""
        return {
            "overall_metrics": {
                "recall": round(recall, 3),
                "precision": round(precision, 3),
                "f1": round(f1, 3),
            },
            "field_accuracy": {
                field: {
                    "errors_in_original": metrics["errors_in_original"],
                    "errors_corrected": metrics["errors_corrected"],
                    "false_corrections": metrics["false_corrections"],
                    "accuracy": metrics["accuracy"],
                }
                for field, metrics in field_accuracy.items()
            },
        }
