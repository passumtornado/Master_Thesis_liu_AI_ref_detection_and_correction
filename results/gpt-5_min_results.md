# End-to-End Performance Summary

## Executive Summary

The validation agent is performing reasonably well overall, especially on the `invalid` class, but it is still missing a large share of `partially_valid` cases. That weakens the downstream correction stage, because the correction agent can only fix what it receives as actionable evidence. The result is a correction system with very high precision, but very low recall: when it edits, it is usually correct, but it is not editing enough cases.

## Validation Agent (Classification)

| Metric            | Value |
| ----------------- | ----- |
| Accuracy          | 0.720 |
| Precision (micro) | 0.720 |
| Recall (micro)    | 0.720 |
| F1 (micro)        | 0.720 |
| Precision (macro) | 0.735 |
| Recall (macro)    | 0.793 |
| F1 (macro)        | 0.721 |

### Per-Class Metrics

| Class           | Precision | Recall | F1    | TP  | FP  | FN  |
| --------------- | --------- | ------ | ----- | --- | --- | --- |
| valid           | 0.600     | 0.900  | 0.720 | 9   | 6   | 1   |
| partially_valid | 0.923     | 0.480  | 0.632 | 12  | 1   | 13  |
| invalid         | 0.682     | 1.000  | 0.811 | 15  | 7   | 0   |

### Interpretation

- `invalid` detection is strong: recall is 1.000, so the model is not missing invalid references, but precision is only 0.682 because it sometimes labels valid items as invalid.
- `partially_valid` is the weakest class for validation recall (0.480), which means many correctable entries are not being surfaced reliably for correction.
- `valid` has high recall (0.900), so most correct entries are recognized, but precision is only 0.600, which suggests the model is too willing to call uncertain entries valid.
- Overall, the validation agent is usable, but it is still uneven across classes, especially for `partially_valid`.

## Correction Agent

| Metric          | Value |
| --------------- | ----- |
| True Positives  | 4     |
| False Positives | 0     |
| False Negatives | 27    |
| Precision       | 1.000 |
| Recall          | 0.129 |
| F1              | 0.229 |

### Interpretation

- Precision is perfect (1.000), which means every applied correction in this run was correct.
- Recall is low (0.129): only 4 out of 31 field-level errors were fixed, while 27 remained unresolved.
- F1 is therefore low (0.229), reflecting strong correction reliability but weak correction coverage.

### Why Recall Is Still Low

The main reason is coverage, not correctness.

- The validation stage still misses many `partially_valid` cases, so the correction stage does not receive enough actionable correction opportunities.
- Even when validation flags an entry, the correction agent may only fix obvious fields and leave secondary errors unchanged.
- Field-level evaluation is strict: each uncorrected wrong field is counted as a false negative.

### Overall Takeaway

This run shows a pipeline that is **highly precise but under-correcting**. The next improvement should target correction coverage (recall) while preserving the current high precision.
