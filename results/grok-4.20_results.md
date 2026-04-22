# End-to-End Performance Summary

## Executive Interpretation

The pipeline is currently strong at detecting clearly invalid references, moderate at broad validation classification, and weak at correction coverage. In practical terms: the system is safe when it edits, but it is still missing many fixable field-level errors.

## Validation Agent (Classification)

| Metric            | Value |
| ----------------- | ----- |
| Accuracy          | 0.660 |
| Precision (micro) | 0.660 |
| Recall (micro)    | 0.660 |
| F1 (micro)        | 0.660 |
| Precision (macro) | 0.672 |
| Recall (macro)    | 0.673 |
| F1 (macro)        | 0.658 |

### Per-Class Metrics

| Class           | Precision | Recall | F1    | TP  | FP  | FN  |
| --------------- | --------- | ------ | ----- | --- | --- | --- |
| valid           | 0.294     | 0.500  | 0.370 | 5   | 12  | 5   |
| partially_valid | 0.722     | 0.520  | 0.605 | 13  | 5   | 12  |
| invalid         | 1.000     | 1.000  | 1.000 | 15  | 0   | 0   |

### Validation Interpretation

- Overall validation quality is fair (micro F1 = 0.660), but not yet robust for a high-reliability correction pipeline.
- The invalid class is excellent (precision/recall/F1 = 1.000), so fabricated/non-existent references are being caught reliably.
- The valid class is the weakest (precision = 0.294), meaning many entries predicted as valid are actually not valid.
- The partially_valid class has moderate recall (0.520), so many correctable references are still missed at the validation stage.
- The gap between strong invalid detection and weaker valid/partially_valid discrimination suggests thresholding or class-balance issues in boundary cases.

## Correction Agent

| Metric          | Value |
| --------------- | ----- |
| True Positives  | 16    |
| False Positives | 1     |
| False Negatives | 15    |
| Precision       | 0.941 |
| Recall          | 0.516 |
| F1              | 0.667 |

### Interpretation

The correction agent shows strong correction quality with moderate coverage. It correctly fixed 16 field-level errors while introducing only 1 false correction, resulting in high precision (0.941). Recall (0.516) indicates that about half of the remaining errors were fixed, so the agent is now reliable but still misses a meaningful portion of fixable issues. The F1 score (0.667) reflects this balance between high accuracy of applied edits and incomplete correction coverage.

- **Strength**: Corrections are highly trustworthy when applied (very low false-positive rate).
- **Current limitation**: 15 errors remain unfixed (false negatives), so coverage should be improved in the next iteration.
- **Overall**: The correction stage has moved from overly conservative behavior to a more effective and practically usable profile.

## End-to-End Diagnosis

1. Validation is not yet surfacing enough actionable partially_valid cases with high confidence.
2. Correction behavior favors precision over coverage, so many known issues are left unchanged.
3. The combination produces stable but low correction recall and therefore low correction F1.

## Recommendations

1. Increase correction coverage by enforcing per-entry structured output completeness and validating that each input entry yields a correction record.
2. Improve valid vs partially_valid separation in validation to reduce missed fix opportunities.
3. Add a post-correction coverage check: expected corrupted fields vs corrected fields, and fail the run if coverage drops below a minimum target.

## Conclusion

Current performance indicates a safe but under-correcting pipeline. The next milestone should prioritize recall gains in correction while preserving the current high precision.
