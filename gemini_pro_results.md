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
