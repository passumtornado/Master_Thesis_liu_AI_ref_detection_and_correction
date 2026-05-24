## Evaluation Metrics

This subsection summarizes the metrics used to evaluate both the validation agent and the correction agent, and documents the exact formulas used for reproducibility.

- Validation agent (classification)
  - Metrics: Accuracy, Precision, Recall, F1, per-class counts (TP, FP, FN), coverage vs ground-truth. These were computed from the confusion counts per class and aggregated using micro-averaging for the overall numbers.
  - Formulas (standard):
    - Precision = $\mathrm{TP}/(\mathrm{TP}+\mathrm{FP})$
    - Recall = $\mathrm{TP}/(\mathrm{TP}+\mathrm{FN})$
    - F1 = $2\cdot(\mathrm{Precision}\cdot\mathrm{Recall})/(\mathrm{Precision}+\mathrm{Recall})$
  - Notes: reported "Coverage vs ground truth" = matched entries / ground-truth entries. When true negatives (TN) are not directly available, accuracy is reported relative to the total ground-truth entries (i.e., TP/Total), and this is documented where used.

- Correction agent (field-level and entry-level)
  - Field-level metrics are computed per-field and aggregated across the dataset. Primary counts are True Positives (TP — corrupted field fixed), False Positives (FP — correct field changed incorrectly), and False Negatives (FN — corrupted field left unchanged).
  - Field-level formulas:
    - Field Precision = $\mathrm{TP}/(\mathrm{TP}+\mathrm{FP})$
    - Field Recall = $\mathrm{TP}/(\mathrm{TP}+\mathrm{FN})$
    - Field F1 = $2\cdot(\mathrm{Precision}\cdot\mathrm{Recall})/(\mathrm{Precision}+\mathrm{Recall})$
    - Field Accuracy (per-field) = $\mathrm{TP}/(\mathrm{TP}+\mathrm{FN})$ (where appropriate)
  - Entry-level (visual) correctness: counts whether an entire BibTeX entry is fully corrected (no remaining corrupted fields) or only partially corrected. This is a stricter, human-oriented measure and can diverge from field-level recall because a single uncorrected field makes an entry "partially correct" even if most fields are fixed.
  - Implementation detail: the deterministic evaluator uses an explicit ground-truth map and a field-matching routine that supports exact and token-overlap matching; this affects TP/FN counting and is documented alongside the evaluator code.

## Limitations

- Ground-truth alignment: some pipeline runs fell back to a default ground-truth file when the intended path was not propagated, producing mismatches between corrections and evaluation; this can reduce measured recall artificially.
- Small dataset and class imbalance: a limited number of partially-valid entries reduces statistical power and inflates variance of per-model estimates.
- Evaluation strictness: field-level scoring treats any remaining corrupted field as a failure for that field and makes entry-level correctness binary; human perception of exported BibTeX quality may therefore differ from measured recall.
- Matching heuristics: the deterministic matching strategy (exact/prefix/token-overlap thresholds) may treat near-equivalent strings as mismatches or matches depending on threshold choices.
- Domain scope: experiments focus on BibTeX bibliographic entries; transferability to other structured-data domains (JSON records, CSV tables) is not validated here.

## Future Work

- Re-run affected evaluations with corrected ground-truth propagation and re-compute metrics to obtain definitive recall estimates.
- Improve recall via: prompt engineering, multi-turn corrections, ensemble of models, or expanded retrieval context in the RAG pipeline.
- Human-in-the-loop evaluation: sample manual inspection to calibrate entry-level visual correctness vs. field-level metrics and to derive task-specific tolerance thresholds.
- Robust matching: adopt semantic matching (embedding distances) for field values to reduce false negatives when strings differ slightly.
- Broader datasets: scale to larger, more diverse bibliographic corpora and replicate across domains to confirm generalization.

## Author Contribution

- Pipeline and orchestration: implemented `run_pipeline` orchestration, integrated validation, correction, and evaluation stages.
- Deterministic evaluator: authored the field-level deterministic evaluation code and the matching heuristics used to compute TP/FP/FN.
- Analysis and reporting: aggregated model-level and dataset-level metrics, diagnosed ground-truth mismatch issues, and produced the analysis reports in `results/`.

## PhD Research Directions

- Evaluation theory for LLM-based cleaning: design metrics that reconcile human-perceived entry quality with field-level correctness, and propose composite or hierarchical metrics.
- Grounded correction methods: study RAG augmentation strategies for robust factual repair under noisy inputs.
- Human-AI collaboration: optimize mixed-initiative correction workflows where models propose fixes and humans confirm or edit the top-k suggestions.
- Metric-aware learning: train models to optimize evaluation metrics aligned with downstream human tasks (e.g., minimizing critical-field FN rate for citation extraction).

## Conclusion

This project demonstrates a practical pipeline for detecting and correcting bibliographic errors using LLM-based correction agents and deterministic evaluation. Results show high precision but limited recall: the agents are conservative and avoid incorrect changes at the cost of leaving many corrupted fields unchanged. The analysis surfaced a key operational issue (ground-truth misalignment) that must be resolved to obtain definitive recall estimates. Future work should focus on improving recall, aligning human and automated evaluation, and scaling the approach across datasets.

---

If you want, I can (1) incorporate these sections into your thesis document (`results/thesis_result.md` or another target), (2) expand any subsection into more detail, or (3) run the re-evaluation for runs missing correct ground-truth paths.
