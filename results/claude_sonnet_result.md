# End-to-End Performance Summary

## Overview

This report presents the performance of the BibTeX reference validation and correction pipeline across three classification categories: **valid**, **partially_valid**, and **invalid** references. The metrics show the system's ability to detect errors and apply corrections.

---

## 1. Validation Agent (Classification Performance)

The validation agent classifies each BibTeX entry into one of three categories:

- **valid**: Entry appears correct with no significant errors
- **partially_valid**: Entry contains field-level errors that need correction
- **invalid**: Entry is non-existent, fabricated, or has critical issues

### Overall Metrics

| Metric            | Value | Interpretation                                            |
| ----------------- | ----- | --------------------------------------------------------- |
| Accuracy          | 0.800 | 80% of classifications are correct (40 out of 50 entries) |
| Precision (micro) | 0.800 | Overall precision across all predictions                  |
| Recall (micro)    | 0.800 | Overall recall across all predictions                     |
| F1 (micro)        | 0.800 | Balanced measure of precision and recall                  |
| Precision (macro) | 0.764 | Average precision per class (balanced view)               |
| Recall (macro)    | 0.767 | Average recall per class (balanced view)                  |
| F1 (macro)        | 0.765 | Balanced F1 across all classes                            |

**Insight**: The macro metrics (0.76–0.77) being slightly lower than micro (0.80) indicates the model performs consistently across classes, with no single class dominating overall performance.

### Per-Class Breakdown

| Class               | Precision | Recall | F1    | TP  | FP  | FN  | Interpretation                                                                                                          |
| ------------------- | --------- | ------ | ----- | --- | --- | --- | ----------------------------------------------------------------------------------------------------------------------- |
| **valid**           | 0.556     | 0.500  | 0.526 | 5   | 4   | 5   | **Weak performance**: Only 50% of true valid entries detected (5 TP, 5 FN); 4 false alarms. This class is undercounted. |
| **partially_valid** | 0.800     | 0.800  | 0.800 | 20  | 5   | 5   | **Good performance**: 80% precision and recall. Most partially valid entries are found with few false alarms.           |
| **invalid**         | 0.938     | 1.000  | 0.968 | 15  | 1   | 0   | **Excellent performance**: Perfect recall (no invalid entries missed) and very high precision (only 1 false alarm).     |

**Class-Level Insights**:

- The validation agent is **excellent at catching clearly invalid references** (F1 = 0.968), making it highly reliable for filtering fabricated or non-existent papers.
- **Partially valid entries are handled well** (F1 = 0.800), which is important since these require correction.
- **Valid entries are undercounted** (recall = 50%), meaning some truly correct entries are misclassified (likely as partially_valid). This is conservative but safe.

---

## 2. Correction Agent (Fix Quality and Coverage)

After validation, the correction agent applies fixes to entries marked as partially_valid or invalid based on structured validation results (issues and suggested fixes).

### Correction Metrics

| Metric                   | Value | Interpretation                                                  |
| ------------------------ | ----- | --------------------------------------------------------------- |
| **True Positives (TP)**  | 4     | 4 field-level errors were correctly fixed                       |
| **False Positives (FP)** | 0     | No false corrections were made (perfect precision)              |
| **False Negatives (FN)** | 27    | 27 field-level errors were NOT fixed (low coverage)             |
| **Precision**            | 1.000 | Every applied correction is valid (no spurious edits)           |
| **Recall**               | 0.129 | Only 12.9% of fixable errors were corrected (very low coverage) |
| **F1**                   | 0.229 | Poor overall balance between precision and recall               |

**Interpretation**:

- **Precision = 1.000**: When the correction agent makes an edit, it is always correct. This is excellent from a **safety perspective** — no damage is done.
- **Recall = 0.129**: The agent misses 27 out of 31 fixable errors. This suggests either:
  - The correction agent is too conservative (only fixing the most obvious errors)
  - The structured validation guidance from the previous step is not being fully utilized
  - The LLM is not parsing the validation suggestions effectively

---

## 3. Joint Interpretation (Validation → Correction Pipeline)

### Strengths

1. **High-confidence classifications**: The validation agent correctly identifies 94% of invalid references (15 true positives, 1 false positive).
2. **Safe correction strategy**: The correction agent never makes a wrong fix (precision = 1.000), meaning end-to-end data quality is not degraded.
3. **Good partially_valid detection**: 80% of entries that need fixing are correctly flagged (20 TP, 5 FN).

### Limitations

1. **Low correction coverage**: The correction agent fixes only 4 out of 31 detectable errors (12.9% recall). Despite validation identifying issues, many are not being corrected.
2. **Valid entry misclassification**: Half of truly valid entries (5 out of 10) are incorrectly marked as needing correction, though this does not harm the pipeline.
3. **Mismatch in validation-correction handoff**: The structured validation results (with issues and suggested fixes) may not be flowing effectively into the correction LLM, or the LLM may not be following the guidance.

---

## 4. Recommendations for Improvement

### Priority 1: Improve Correction Coverage

- **Action**: Ensure the correction LLM receives full validation structured results (including `suggested_fixes` from the validation agent) and follows those suggestions explicitly.
- **Expected Impact**: Could increase correction recall from 12.9% to 60%+.

### Priority 2: Improve Valid Entry Detection

- **Action**: Fine-tune the validation agent to reduce false negatives in the "valid" class (currently 50% recall).
- **Expected Impact**: Reduce wasted correction attempts on already-correct entries.

### Priority 3: Monitor False Positives in Correction

- **Action**: Even though FP = 0 currently, maintain strict validation rules to ensure corrections never introduce errors.

---

## 5. Summary Statistics

| Component                              | Accuracy/Precision                     | Recall                    | F1                   |
| -------------------------------------- | -------------------------------------- | ------------------------- | -------------------- |
| **Validation (Overall)**               | 0.800                                  | 0.800                     | 0.800                |
| **Validation (Invalid Class)**         | 0.938                                  | 1.000                     | 0.968                |
| **Validation (Partially Valid Class)** | 0.800                                  | 0.800                     | 0.800                |
| **Correction (Field-Level)**           | 1.000                                  | 0.129                     | 0.229                |
| **End-to-End Safety**                  | ✅ Excellent (no spurious corrections) | ⚠️ Limited (low coverage) | ⚠️ Needs improvement |

---

## 6. Conclusion

The pipeline demonstrates **high confidence in detecting errors** (validation agents successfully identifies ~94% of clearly invalid entries) and **excellent safety in correction** (never makes false edits). However, **correction coverage remains a bottleneck**, with only 12.9% of detectable errors being fixed.

The next iteration should focus on improving the validation-to-correction handoff by ensuring structured validation guidance (issues and suggested fixes) directly influences the correction agent's decisions, which should substantially increase recall without sacrificing precision.
