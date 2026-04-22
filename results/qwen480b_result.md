# End-to-End Performance Summary

This report summarizes the validation and correction stages of the pipeline. The validation agent decides whether a bibliographic entry is `valid`, `partially_valid`, or `invalid`. The correction agent then attempts to repair incorrect or incomplete entries.

## Validation Agent Summary

The validation stage evaluated 50 ground-truth entries and produced 50 predictions, so coverage against the labeled set is complete.

| Metric                   | Value |
| ------------------------ | ----- |
| Ground truth entries     | 50    |
| Predicted entries        | 50    |
| Matched entries          | 50    |
| Coverage vs ground truth | 1.000 |
| Accuracy                 | 0.900 |
| Precision (micro)        | 0.900 |
| Recall (micro)           | 0.900 |
| F1 (micro)               | 0.900 |
| Precision (macro)        | 0.905 |
| Recall (macro)           | 0.853 |
| F1 (macro)               | 0.871 |

### Class Distribution

| Class           | Ground Truth | Predicted |
| --------------- | -----------: | --------: |
| valid           |           10 |         7 |
| partially_valid |           25 |        28 |
| invalid         |           15 |        15 |

### Per-Class Metrics

| Class           | Precision | Recall |    F1 |  TP |  FP |  FN |
| --------------- | --------: | -----: | ----: | --: | --: | --: |
| valid           |     0.857 |  0.600 | 0.706 |   6 |   1 |   4 |
| partially_valid |     0.857 |  0.960 | 0.906 |  24 |   4 |   1 |
| invalid         |     1.000 |  1.000 | 1.000 |  15 |   0 |   0 |

### Interpretation

The validation agent performs well overall, with 90% accuracy and identical micro precision, recall, and F1. That means the total number of correct label decisions is strong, and there is no obvious asymmetry between over-predicting and under-predicting when all classes are aggregated.

The macro scores are more informative about class balance. Macro recall is lower than micro recall, which shows that performance is uneven across the three labels. The clearest weakness is the `valid` class: recall is only 0.600, meaning the model misses 40% of the truly valid entries and often reassigns them to `partially_valid`. In contrast, `invalid` is classified perfectly, with precision, recall, and F1 all at 1.000. The `partially_valid` class is the strongest in terms of coverage, with very high recall at 0.960, but its precision is lower because some non-partially-valid entries are absorbed into that class.

In practical terms, the validation agent is reliable at detecting clearly invalid entries, but it is more conservative when deciding that an entry is fully valid. The main improvement opportunity is reducing confusion between `valid` and `partially_valid`.

## Correction Agent Summary

| Metric          | Value |
| --------------- | ----: |
| True positives  |    27 |
| False positives |     2 |
| False negatives |     4 |
| Precision       | 0.931 |
| Recall          | 0.871 |
| F1              | 0.900 |

### Field Accuracy Table

| Field     | Errors in Original | Errors Corrected | False Corrections | Accuracy |
| --------- | -----------------: | ---------------: | ----------------: | -------: |
| author    |                  8 |                8 |                 0 |    1.000 |
| booktitle |                  2 |                2 |                 0 |    1.000 |
| year      |                  5 |                5 |                 2 |    1.000 |
| doi       |                  3 |                3 |                 0 |    1.000 |
| title     |                  3 |                3 |                 0 |    1.000 |
| publisher |                  2 |                1 |                 0 |    0.500 |
| volume    |                  3 |                2 |                 0 |    0.667 |
| number    |                  2 |                1 |                 0 |    0.500 |
| journal   |                  3 |                2 |                 0 |    0.667 |

### Interpretation

The correction stage is also strong overall. Precision of 0.931 means most suggested corrections are correct, while recall of 0.871 means the system recovers most of the true issues but still misses some. The resulting F1 score of 0.900 indicates a good balance between making useful corrections and avoiding excessive bad edits.

The field-level table shows a clear pattern. `author`, `booktitle`, `doi`, `title`, and `year` are fully recovered according to the reported accuracy metric. These are the most reliable fields in the correction pipeline. The weaker fields are `publisher`, `volume`, `number`, and `journal`, which have lower accuracy and therefore represent the main correction bottlenecks.

The `year` row is worth reading carefully: the model corrected all true year errors, but it also produced two false corrections. That means the field was fixed whenever it truly needed repair, but it still introduced some unnecessary changes. For the lower-accuracy fields, the issue is more direct: the agent does not consistently recover every true correction opportunity.

## Overall Takeaway

The pipeline is working well end to end. Validation is strong enough for reliable downstream use, and correction produces high-quality fixes for the most important bibliographic fields. The main weakness is not global performance, but class- and field-specific imbalance: `valid` labels are under-recalled, and a small group of citation fields still needs better correction coverage.

The most useful next step is to focus model or prompt improvements on the ambiguous `valid` versus `partially_valid` boundary and on the lower-performing fields in the correction stage, especially `publisher`, `volume`, `number`, and `journal`.
