# Comprehensive Report: How Evaluation Works in This Project

## 1. Scope and Goal

This report explains, in implementation terms, how your project evaluates:

1. Validation agent performance (classification quality)
2. Correction agent performance (field-level fix quality)

It is based on the current prototype pipeline implementation in:

- [ \_agents/\_pipeline.py ](_agents/_pipeline.py)
- [ \_agents/deterministic_eval_agent.py ](_agents/deterministic_eval_agent.py)
- [ \_agents/\_validation_agent.py ](_agents/_validation_agent.py)
- [ \_agents/\_validation_agent_react.py ](_agents/_validation_agent_react.py)
- [ \_agents/\_correction_agent.py ](_agents/_correction_agent.py)

## 2. End-to-End Evaluation Flow

The prototype orchestration runs this sequence:

Preparation -> Validation -> Correction -> Evaluation -> Save outputs

The controlling graph and state handoff are defined in [ \_agents/\_pipeline.py ](_agents/_pipeline.py).

At a high level:

1. Preparation parses BibTeX entries into normalized structured records.
2. Validation assigns status labels for each entry.
3. Correction applies field edits using validation evidence.
4. Evaluation computes correction quality metrics, and separately the pipeline computes validation classification metrics.
5. Save outputs writes consolidated JSON and markdown summaries.

## 3. Ground Truth Sources Used During Evaluation

Two different ground-truth signals are used for the two tasks:

1. Validation evaluation ground truth:

- expected_status per entry from the ground truth JSON file
- used by the function that computes validation metrics in [ \_agents/\_pipeline.py ](_agents/_pipeline.py)

2. Correction evaluation ground truth:

- per-entry corruption list in ground truth JSON, including original correct field value
- used by deterministic evaluation in [ \_agents/deterministic_eval_agent.py ](_agents/deterministic_eval_agent.py)

This separation is important:

- Validation is a classification problem.
- Correction is a field-level error-repair problem.

## 4. How Validation Results Are Evaluated

Validation outputs are first produced by the validation agent as structured predictions containing:

- entry_id
- predicted status in {valid, partially_valid, invalid}

Then [ \_agents/\_pipeline.py ](_agents/_pipeline.py) computes metrics against expected_status.

### 4.1 Label Set

Fixed classes are:

- valid
- partially_valid
- invalid

### 4.2 Matching and Coverage

The evaluator intersects IDs between:

- ground-truth entries
- predicted validation entries

Only overlapping IDs are evaluated. This gives:

- matched_entries
- coverage_vs_ground_truth

So the reported performance is explicitly tied to overlap, not assumed full coverage.

### 4.3 Confusion Matrix and Per-Class Counts

A full confusion matrix is accumulated for the 3 classes.

Per class, one-vs-rest counts are computed:

- TP: predicted class c and true class c
- FP: predicted class c but true class not c
- FN: true class c but predicted not c

### 4.4 Validation Metrics Reported

From those counts, the following are computed:

Per class:

$$
Precision_c = \frac{TP_c}{TP_c + FP_c},\quad
Recall_c = \frac{TP_c}{TP_c + FN_c},\quad
F1_c = \frac{2 Precision_c Recall_c}{Precision_c + Recall_c}
$$

Micro:

$$
Precision_{micro} = \frac{\sum TP}{\sum TP + \sum FP},\quad
Recall_{micro} = \frac{\sum TP}{\sum TP + \sum FN},\quad
F1_{micro} = \frac{2 Precision_{micro} Recall_{micro}}{Precision_{micro}+Recall_{micro}}
$$

Macro:

$$
Precision_{macro} = \frac{1}{K}\sum_c Precision_c,
\quad Recall_{macro} = \frac{1}{K}\sum_c Recall_c,
\quad F1_{macro} = \frac{1}{K}\sum_c F1_c
$$

Accuracy:

$$
Accuracy = \frac{\# correct\ predictions}{\# matched\ entries}
$$

### 4.5 Validation Files Produced

Per strategy, validation outputs are written under the validation folder tree, including:

- validation_report.md
- validation_structured.json
- validation_grouped.json
- validation_metrics.json

A pipeline-level copy of validation_metrics.json is also saved.

## 5. How Correction Results Are Evaluated

Correction evaluation is performed by the deterministic evaluator in:

- [ \_agents/deterministic_eval_agent.py ](_agents/deterministic_eval_agent.py)

In your current prototype, this deterministic mode is the primary mode in the pipeline.

### 5.1 Inputs Used

The evaluator combines:

- ground truth entry corruption specification
- correction outputs from correction agent
- original raw entry values

### 5.2 Field-Level Outcome Logic

For each known corruption field in ground truth:

1. Read expected correct value from ground truth
2. Read corrected value from correction output
3. Compare with a matching function

If it matches -> TP
If it does not match -> FN

Then FP detection is performed on fields that were not corrupted:

- if a non-corrupted original field was changed incorrectly, it is counted as FP

### 5.3 Matching Function Behavior

The matching logic is not strict string equality only. It uses normalized comparison:

- lowercase
- whitespace normalization
- brace stripping

For short values, it requires near exact match.
For longer strings, it allows token-overlap matching above a threshold.

This design makes evaluation less brittle to harmless formatting differences.

### 5.4 Correction Metrics Reported

Overall correction metrics:

- true_positives
- false_positives
- false_negatives
- precision
- recall
- f1

Formulas:

$$
Precision = \frac{TP}{TP+FP},\quad
Recall = \frac{TP}{TP+FN},\quad
F1 = \frac{2PR}{P+R}
$$

Field-level accuracy table entries are computed as:

- errors_in_original = TP_field + FN_field
- errors_corrected = TP_field
- false_corrections = FP_field
- accuracy = errors_corrected / errors_in_original

So field accuracy measures how many known original errors in that field were successfully repaired.

### 5.5 Evaluation Files Produced

Per strategy evaluation outputs include:

- evaluation_metrics.json
- evaluation_report.md
- evaluation_details.json

## 6. Role of the LLM During Evaluation

There are two evaluation modes in the code:

1. Deterministic mode (recommended and currently used in pipeline):

- metrics are computed in Python
- LLM is used only to write narrative markdown report text

2. Legacy LLM-only mode:

- LLM computes metrics and report together
- this mode is less reproducible and is retained as fallback

This architecture gives stronger reproducibility for thesis experiments while retaining readable narrative reports.

## 7. How Validation and Correction Metrics Are Unified in Final Summary

The pipeline writes a combined summary payload and markdown:

- performance_summary.json
- performance_summary.md

The combined JSON groups results into:

- validation.metrics and validation.error
- correction.overall_metrics, correction.field_accuracy, correction.error

The combined markdown presents:

1. Validation metrics and per-class table
2. Correction metrics table

This is produced by the summary builder in [ \_agents/\_pipeline.py ](_agents/_pipeline.py).

## 8. Interpretation Guidance for Results

When reading your results:

1. Validation quality determines how well entries are routed into valid, partially_valid, invalid.
2. Correction quality measures whether wrong fields were actually fixed without harming correct fields.

Typical interpretation patterns:

- High validation recall for partially_valid means more fixable entries are surfaced for correction.
- High correction precision with low recall means conservative edits: safe changes but many missed fixes.
- Strong invalid class performance with weaker valid class performance usually indicates conservative classification boundaries.

## 9. Strengths of Current Evaluation Design

1. Reproducible correction metrics through deterministic computation.
2. Explicit classwise validation metrics with confusion matrix support.
3. Field-level correction diagnostics, not only one aggregate score.
4. Clear separation between classification quality and repair quality.
5. Strategy-wise comparability through consistent output schemas.

## 10. Known Limitations and Caveats

1. Validation evaluation depends on overlap of IDs between predictions and ground truth; missing IDs reduce evaluable coverage.
2. Correction scoring sensitivity depends on matching rules; token-overlap thresholds can affect borderline judgments.
3. If a model output is malformed or missing structure, fallback behaviors may reduce metric completeness.
4. Some reports may include NR values because certain model runs only output correction metrics or only partial artifacts.

## 11. Practical Conclusion

Your current evaluation setup is methodologically strong for thesis-grade comparison:

- Validation is evaluated as a proper multi-class classification task.
- Correction is evaluated as deterministic field-level error repair.
- Combined performance summaries support direct model-to-model comparison.

This structure is suitable for analyzing trade-offs such as precision-versus-recall behavior, conservative versus aggressive correction patterns, and evidence-grounded versus zero-shot prompting strategies.
