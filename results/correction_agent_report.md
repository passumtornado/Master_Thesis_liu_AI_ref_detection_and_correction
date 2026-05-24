# Correction Agent — Summary Report

Date: 22 May 2026

## Executive Summary

The correction agent demonstrates high precision but low recall on field-level correction tasks. When the agent proposes a change it is usually correct, however many corrupted fields remain uncorrected.

## Aggregate Metrics (Field-Level)

| Metric                               | Value |
| ------------------------------------ | ----: |
| True Positives (TP)                  |   158 |
| False Positives (FP)                 |     6 |
| False Negatives (FN)                 |   847 |
| Precision                            | 0.963 |
| Recall                               | 0.157 |
| F1 Score                             | 0.270 |
| Total Fields Changed (agent outputs) |   225 |

Source: aggregated from `results/rag_correction_analysis.md` (Table 1) and per-run `evaluation/*/evaluation/rag` artifacts.

## Best-performing Model

- `gemini-3.1-pro-prev` showed the best balance of precision and recall (F1 ≈ 0.558) and recovered the largest share of corrupted fields among tested models.

## Observations

- High precision (≈96%) indicates the agent rarely makes incorrect changes.
- Very low recall (≈15.7%) shows the agent misses the majority of corrupted fields.
- The exported BibTeX may look visually good despite low recall because many entries have few corrupted fields (entry-level visual correctness can be higher than aggregated field recall).
- A pipeline ground-truth misalignment was identified earlier; affected runs may under-report recall and should be re-evaluated.

## Recommendations

1. Re-run evaluations for runs with incorrect or missing ground-truth propagation to obtain definitive recall numbers.
2. Improve recall via:
   - richer retrieval context in RAG, prompt engineering, and multi-turn correction attempts;
   - model ensembling or voting across multiple candidate corrections;
   - targeted heuristics for common bibliographic fields (author, year, title).
3. Add a small human-in-the-loop validation sampling step to calibrate entry-level visual quality vs. field-level metrics.
4. Consider loosening or improving the deterministic matching heuristic (semantic or embedding-based matching) to reduce false negatives due to minor string differences.

## Next Steps I can take

- List runs missing `evaluation_details.json` and those with ground-truth mismatch (so you can decide which to re-run).
- Re-run selected model/dataset evaluations with correct `--ground-truth` paths and update the report.

---

If you want me to re-run affected runs now, tell me which models/datasets to prioritize (or I can list them first).
