# End-to-End Performance Summary

## Validation Agent (Classification)

| Metric            | Value |
| ----------------- | ----- |
| Accuracy          | 0.760 |
| Precision (micro) | 0.760 |
| Recall (micro)    | 0.760 |
| F1 (micro)        | 0.760 |
| Precision (macro) | 0.782 |
| Recall (macro)    | 0.820 |
| F1 (macro)        | 0.763 |

**Micro vs macro:** micro scores aggregate all predictions before computing the metric, so they reflect overall dataset-level performance and are influenced more by larger classes. Macro scores compute the metric separately for each class and then average them, so they show how consistently the model performs across classes regardless of class size.

### Per-Class Metrics

| Class           | Precision | Recall | F1    | TP  | FP  | FN  |
| --------------- | --------- | ------ | ----- | --- | --- | --- |
| valid           | 0.474     | 0.900  | 0.621 | 9   | 10  | 1   |
| partially_valid | 0.933     | 0.560  | 0.700 | 14  | 1   | 11  |
| invalid         | 0.938     | 1.000  | 0.968 | 15  | 1   | 0   |

## Correction Agent

| Metric          | Value |
| --------------- | ----- |
| True Positives  | 12    |
| False Positives | 1     |
| False Negatives | 19    |
| Precision       | 0.923 |
| Recall          | 0.387 |
| F1              | 0.545 |

## Short Interpretation

- Validation performance is solid overall (accuracy/F1 around 0.76), with very strong detection of invalid references (recall = 1.000).
- The main classification weakness is under-detection of partially valid entries (recall = 0.560), which means many fixable records are not flagged strongly enough.
- Correction behavior is high precision but low recall: most applied edits are correct (precision = 0.923), but many needed fixes are still missed (FN = 19, recall = 0.387).
- Overall, the pipeline is conservative and reliable when it edits, but it needs better error coverage to improve end-to-end correction completeness.


# End-to-End Performance Summary

## Validation Agent (Classification)

| Metric | Value |
|---|---|
| Accuracy | 0.765 |
| Precision | 0.765 |
| Recall | 0.765 |
| F1 | 0.765 |
| True Positives (correctly classified) | 39 |
| False Positives | 12 |
| False Negatives | 12 |
| Ground-truth entries | 51 |
| Predicted entries | 51 |
| Matched entries | 51 |
| Coverage vs ground truth | 1.000 |

## Correction Agent

| Metric | Value |
|---|---|
| True Positives | 2 |
| False Positives | 0 |
| False Negatives | 48 |
| Precision | 1.000 |
| Recall | 0.040 |
| F1 | 0.077 |
| Partially-valid in ground truth | 17 |
| Correctly identified partially-valid | 8 |

### Field-Level Metrics (Partially Valid, Correctly Identified)

| Field | Errors in Original | Errors Corrected | False Corrections | Accuracy |
|---|---|---|---|---|
| archiveprefix | 1 | 0 | 0 | 0.000 |
| author | 8 | 0 | 0 | 0.000 |
| booktitle | 2 | 0 | 0 | 0.000 |
| doi | 3 | 0 | 0 | 0.000 |
| eprint | 1 | 0 | 0 | 0.000 |
| journal | 5 | 1 | 0 | 0.200 |
| number | 3 | 0 | 0 | 0.000 |
| pages | 5 | 0 | 0 | 0.000 |
| primaryclass | 1 | 0 | 0 | 0.000 |
| title | 8 | 0 | 0 | 0.000 |
| volume | 5 | 1 | 0 | 0.200 |
| year | 8 | 0 | 0 | 0.000 |