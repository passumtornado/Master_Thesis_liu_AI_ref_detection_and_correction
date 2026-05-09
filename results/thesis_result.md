# Results

## 1. Dataset and Ground Truth

This study uses one BibTeX dataset split into 4 sets.
Each set has a matching ground-truth file.
The ground truth gives the expected label for each reference:

- valid
- partially_valid
- invalid

### 1.1 Dataset Split

| Set       | BibTeX File       | Ground-Truth File        | Total Entries |   Valid | Partially Valid | Invalid |
| --------- | ----------------- | ------------------------ | ------------: | ------: | --------------: | ------: |
| Set 1     | stefan_train1.bib | stefan_train1_truth.json |            51 |      26 |              17 |       8 |
| Set 2     | stefan_train2.bib | stefan_train2_truth.json |            51 |      26 |              17 |       8 |
| Set 3     | stefan_train3.bib | stefan_train3_truth.json |            50 |      26 |              17 |       7 |
| Set 4     | stefan_train4.bib | stefan_train4_truth.json |            50 |      26 |              17 |       7 |
| **Total** | -                 | -                        |       **202** | **104** |          **68** |  **30** |

### 1.2 What Ground Truth Contains

For each entry, ground truth stores:

- entry_id
- entry_type
- expected_status
- corruption notes (for wrong or fake entries)

### 1.3 Why This Dataset Design Is Good

- It is balanced and easy to repeat.
- All 3 labels are present in every set.
- It tests multiple BibTeX entry types.
- It supports fair model comparison across runs.

---

## 2. Evaluation Setup

Results were collected from model folders in `evaluation/`.
Folder names include both model and set id.

Models included in this report:

- Claude Sonnet 4.6
- Gemini 2.5 Pro
- GPT-5.4
- Grok 4.20

### 2.1 Available Runs

| Model             |     Set 2 |     Set 3 |     Set 4 |
| ----------------- | --------: | --------: | --------: |
| Claude Sonnet 4.6 |   Missing | Available | Available |
| Gemini 2.5 Pro    | Available |   Missing | Available |
| GPT-5.4           | Available | Available | Available |
| Grok 4.20         | Available |   Missing | Available |

Important note:
Some runs are missing, so averages are based only on available runs.

---

## 3. Validation Results (Run-Level)

| Model             | Set | Accuracy | Precision | Recall |    F1 |  TP |  FP |  FN | Coverage |
| ----------------- | --: | -------: | --------: | -----: | ----: | --: | --: | --: | -------: |
| Claude Sonnet 4.6 |   3 |    0.915 |     0.915 |  0.915 | 0.915 |  43 |   4 |   4 |     0.94 |
| Claude Sonnet 4.6 |   4 |    0.760 |     0.760 |  0.760 | 0.760 |  38 |  12 |  12 |     1.00 |
| Gemini 2.5 Pro    |   2 |    0.627 |     0.627 |  0.627 | 0.627 |  32 |  19 |  19 |     1.00 |
| Gemini 2.5 Pro    |   4 |    0.900 |     0.900 |  0.900 | 0.900 |  45 |   5 |   5 |     1.00 |
| GPT-5.4           |   2 |    0.843 |     0.843 |  0.843 | 0.843 |  43 |   8 |   8 |     1.00 |
| GPT-5.4           |   3 |    0.840 |     0.840 |  0.840 | 0.840 |  42 |   8 |   8 |     1.00 |
| GPT-5.4           |   4 |    0.560 |     0.560 |  0.560 | 0.560 |  28 |  22 |  22 |     1.00 |
| Grok 4.20         |   2 |    0.922 |     0.922 |  0.922 | 0.922 |  47 |   4 |   4 |     1.00 |
| Grok 4.20         |   4 |    0.940 |     0.940 |  0.940 | 0.940 |  47 |   3 |   3 |     1.00 |

### 3.1 Main Validation Discussion

- Grok 4.20 gives the best validation scores in available runs.
- Claude Sonnet 4.6 reaches strong performance in Set 3, but drops in Set 4.
- Gemini 2.5 Pro changes a lot between Set 2 and Set 4.
- GPT-5.4 is steady in Set 2 and Set 3, but weak in Set 4.
- Coverage is mostly 1.00, so most entries were matched.

---

## 4. Class-Wise Validation Behavior

| Model             | Set | Recall (Valid) | Recall (Partially Valid) | Recall (Invalid) | Predicted Unverifiable |
| ----------------- | --: | -------------: | -----------------------: | ---------------: | ---------------------: |
| Claude Sonnet 4.6 |   3 |          1.000 |                    0.867 |            0.667 |                      0 |
| Claude Sonnet 4.6 |   4 |          0.654 |                    0.824 |            1.000 |                     11 |
| Gemini 2.5 Pro    |   2 |          0.500 |                    0.647 |            1.000 |                     15 |
| Gemini 2.5 Pro    |   4 |          0.962 |                    0.824 |            0.857 |                      0 |
| GPT-5.4           |   2 |          0.808 |                    0.824 |            1.000 |                      0 |
| GPT-5.4           |   3 |          0.769 |                    1.000 |            0.714 |                      0 |
| GPT-5.4           |   4 |          0.269 |                    0.824 |            1.000 |                      0 |
| Grok 4.20         |   2 |          0.962 |                    0.824 |            1.000 |                      1 |
| Grok 4.20         |   4 |          1.000 |                    0.941 |            0.714 |                      0 |

### 4.1 Class-Wise Discussion

- Grok is strongest on valid references, especially in Set 4.
- GPT-5.4 reaches perfect partially-valid recall in Set 3.
- Claude and Gemini show more instability across sets.
- Some runs create many unverifiable labels, which reduces total score.

---

## 5. Correction Results

| Model             | Set | Correction Precision | Correction Recall | Correction F1 | Correctly Identified Partially Valid |
| ----------------- | --: | -------------------: | ----------------: | ------------: | -----------------------------------: |
| Claude Sonnet 4.6 |   3 |                0.900 |             0.118 |         0.209 |                                   13 |
| Claude Sonnet 4.6 |   4 |                0.667 |             0.026 |         0.050 |                                   14 |
| Gemini 2.5 Pro    |   2 |                1.000 |             0.028 |         0.055 |                                   11 |
| Gemini 2.5 Pro    |   4 |                0.875 |             0.091 |         0.165 |                                   14 |
| GPT-5.4           |   2 |                1.000 |             0.091 |         0.167 |                                   14 |
| GPT-5.4           |   3 |                1.000 |             0.358 |         0.527 |                                   17 |
| GPT-5.4           |   4 |                0.929 |             0.169 |         0.286 |                                   14 |
| Grok 4.20         |   2 |                1.000 |             0.138 |         0.242 |                                   14 |
| Grok 4.20         |   4 |                1.000 |             0.023 |         0.045 |                                   16 |

### 5.1 Correction Discussion

- Precision is often high, which means edits are mostly safe.
- Recall is low for most runs, which means many needed fixes are still missed.
- GPT-5.4 gives the strongest correction F1 in the current data.
- Strong validation does not always mean strong correction.

---

## 6. Model-Level Summary (Available Runs)

| Model             | Completed Runs | Avg Validation F1 | Best Validation F1 | Avg Correction F1 |
| ----------------- | -------------: | ----------------: | -----------------: | ----------------: |
| Claude Sonnet 4.6 |              2 |             0.838 |              0.915 |             0.130 |
| Gemini 2.5 Pro    |              2 |             0.764 |              0.900 |             0.110 |
| GPT-5.4           |              3 |             0.748 |              0.843 |             0.327 |
| Grok 4.20         |              2 |             0.931 |              0.940 |             0.144 |

Interpretation:

- Best average validation: Grok 4.20.
- Best average correction: GPT-5.4.
- No model is best on both tasks at the same time.

---

## 7. Weighted Final Ranking

To provide one overall score, we use:

`weighted_score = 0.6 * avg_validation_f1 + 0.4 * avg_correction_f1`

And a reliability-aware version:

`weighted_score_with_reliability = weighted_score * (1 - avg_unverifiable_rate)`

### 7.1 Ranking Table

| Rank | Model             | Runs | Avg Val F1 | Avg Corr F1 | Avg Unverifiable Rate | Weighted Score (60/40) | Reliability-Aware Score |
| ---: | ----------------- | ---: | ---------: | ----------: | --------------------: | ---------------------: | ----------------------: |
|    1 | Grok 4.20         |    2 |      0.931 |       0.143 |                 0.010 |                  0.616 |                   0.610 |
|    2 | GPT-5.4           |    3 |      0.748 |       0.327 |                 0.000 |                  0.579 |                   0.579 |
|    3 | Claude Sonnet 4.6 |    2 |      0.838 |       0.130 |                 0.110 |                  0.554 |                   0.493 |
|    4 | Gemini 2.5 Pro    |    2 |      0.764 |       0.110 |                 0.147 |                  0.502 |                   0.428 |

### 7.2 Ranking Discussion

- Grok ranks first because of very strong validation and low unverifiable rate.
- GPT-5.4 ranks second and is best on correction quality.
- Claude and Gemini lose points due to weaker correction and higher unreliable outcomes in some runs.

---

## 8. Reliability and System Effects

Model scores depend on both model behavior and tool availability.
In this project, some runs include many `unverifiable` outputs due to lookup failures.

Practical takeaway:

- Report model metrics and reliability metrics together.
- Do not judge model quality from one set only.
- Keep repeated runs to reduce the effect of one bad run.

---

## 9. Threats to Validity and Limitations

### 9.1 Missing Runs

Not all model-set pairs are available.
This can affect averages and ranking fairness.

### 9.2 External Tool Failures

DBLP or Scholar access issues can produce unverifiable labels.
This can lower validation scores without reflecting true model quality.

### 9.3 Limited Number of Sets

Only 4 sets are used.
More sets would give stronger confidence in ranking stability.

### 9.4 Correction Recall Gap

All models show low correction recall.
So the correction module may still miss many real errors.

### 9.5 Distribution Shift Across Sets

Some models change a lot between sets.
This indicates sensitivity to set composition or runtime conditions.

---

## 10. Key Findings (Simple Summary)

1. The dataset split is clear and balanced enough for fair testing.
2. Grok 4.20 performs best on validation in available runs.
3. GPT-5.4 performs best on correction quality.
4. No model is best for both validation and correction at once.
5. Reliability issues (unverifiable cases) strongly affect final scores.
6. Correction recall is the main weakness across all models.

---

## 11. Suggested Extra Reports to Add in Thesis

### Report A: Stability Report

- Show one line chart per model across Set 1 to Set 4.
- Metric: Validation F1 and Correction F1.
- Goal: show consistency.

### Report B: Error Type Report

- Break errors into:
  - class confusion
  - unverifiable due to tool access
  - missed corrections
- Goal: show where each model fails.

### Report C: Reliability Report

- Include:
  - total requests
  - retries
  - runtime
  - request rate
  - unverifiable rate
- Goal: separate model issues from system issues.

### Report D: Per-Class Performance Report

- Show precision, recall, F1 for each class.
- Goal: show class-level strengths and weaknesses.

### Report E: Case Study Report

- Add short examples:
  - one valid success
  - one partially-valid correction
  - one invalid/fabricated case
  - one unverifiable case
- Goal: make results easier to trust and explain.

---

## 12. Final Conclusion for the Results Chapter

This benchmark clearly separates two tasks: reference validation and reference correction. The current results show that model behavior changes by task. Grok 4.20 is strongest for validation, while GPT-5.4 is strongest for correction in the available runs. However, correction recall is still low for all models, so many needed fixes are missed. Also, tool access failures can change measured performance, which means reliability must be tracked with model quality. Therefore, the final system should be judged by both prediction metrics and runtime reliability metrics, not by one score alone.
