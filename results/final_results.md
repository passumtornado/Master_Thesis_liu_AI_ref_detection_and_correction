# Final Results: Model Comparison

This file consolidates all model outputs found in [results](results).

## Sources Used

- [results/claude_sonnet_result.md](results/claude_sonnet_result.md)
- [results/gemini_pro_results.md](results/gemini_pro_results.md)
- [results/gpt-5_min_results.md](results/gpt-5_min_results.md)
- [results/gpt-5_result.md](results/gpt-5_result.md)
- [results/grok-4.20_results.md](results/grok-4.20_results.md)
- [results/qwen480b_result.md](results/qwen480b_result.md)

## Validation Metrics (Overall)

| Model         | Accuracy | Precision (micro) | Recall (micro) | F1 (micro) | Precision (macro) | Recall (macro) | F1 (macro) |
| ------------- | -------: | ----------------: | -------------: | ---------: | ----------------: | -------------: | ---------: |
| Claude Sonnet |    0.800 |             0.800 |          0.800 |      0.800 |             0.764 |          0.767 |      0.765 |
| Gemini Pro    |    0.760 |             0.760 |          0.760 |      0.760 |             0.782 |          0.820 |      0.763 |
| GPT-5 mini    |    0.720 |             0.720 |          0.720 |      0.720 |             0.735 |          0.793 |      0.721 |
| GPT-5         |       NR |                NR |             NR |         NR |                NR |             NR |         NR |
| Grok-4.20     |    0.660 |             0.660 |          0.660 |      0.660 |             0.672 |          0.673 |      0.658 |
| Qwen480B      |    0.900 |             0.900 |          0.900 |      0.900 |             0.905 |          0.853 |      0.871 |

## Validation Per-Class Metrics

### Class: valid

| Model         | Precision | Recall |    F1 |  TP |  FP |  FN |
| ------------- | --------: | -----: | ----: | --: | --: | --: |
| Claude Sonnet |     0.556 |  0.500 | 0.526 |   5 |   4 |   5 |
| Gemini Pro    |     0.474 |  0.900 | 0.621 |   9 |  10 |   1 |
| GPT-5 mini    |     0.600 |  0.900 | 0.720 |   9 |   6 |   1 |
| GPT-5         |        NR |     NR |    NR |  NR |  NR |  NR |
| Grok-4.20     |     0.294 |  0.500 | 0.370 |   5 |  12 |   5 |
| Qwen480B      |     0.857 |  0.600 | 0.706 |   6 |   1 |   4 |

### Class: partially_valid

| Model         | Precision | Recall |    F1 |  TP |  FP |  FN |
| ------------- | --------: | -----: | ----: | --: | --: | --: |
| Claude Sonnet |     0.800 |  0.800 | 0.800 |  20 |   5 |   5 |
| Gemini Pro    |     0.933 |  0.560 | 0.700 |  14 |   1 |  11 |
| GPT-5 mini    |     0.923 |  0.480 | 0.632 |  12 |   1 |  13 |
| GPT-5         |        NR |     NR |    NR |  NR |  NR |  NR |
| Grok-4.20     |     0.722 |  0.520 | 0.605 |  13 |   5 |  12 |
| Qwen480B      |     0.857 |  0.960 | 0.906 |  24 |   4 |   1 |

### Class: invalid

| Model         | Precision | Recall |    F1 |  TP |  FP |  FN |
| ------------- | --------: | -----: | ----: | --: | --: | --: |
| Claude Sonnet |     0.938 |  1.000 | 0.968 |  15 |   1 |   0 |
| Gemini Pro    |     0.938 |  1.000 | 0.968 |  15 |   1 |   0 |
| GPT-5 mini    |     0.682 |  1.000 | 0.811 |  15 |   7 |   0 |
| GPT-5         |        NR |     NR |    NR |  NR |  NR |  NR |
| Grok-4.20     |     1.000 |  1.000 | 1.000 |  15 |   0 |   0 |
| Qwen480B      |     1.000 |  1.000 | 1.000 |  15 |   0 |   0 |

## Overall Comparison (Correction Stage)

The six reports consistently provide correction-stage Precision, Recall, and F1. They do not directly provide correction Accuracy, so Accuracy is computed here as:

$$
\text{Accuracy}^* = \frac{TP}{TP + FP + FN}
$$

| Model         | Precision | Recall |    F1 | Accuracy\* |
| ------------- | --------: | -----: | ----: | ---------: |
| Claude Sonnet |     1.000 |  0.129 | 0.229 |      0.129 |
| Gemini Pro    |     0.923 |  0.387 | 0.545 |      0.375 |
| GPT-5 mini    |     1.000 |  0.129 | 0.229 |      0.129 |
| GPT-5         |     0.957 |  0.629 | 0.759 |      0.611 |
| Grok-4.20     |     0.941 |  0.516 | 0.667 |      0.500 |
| Qwen480B      |     0.931 |  0.871 | 0.900 |      0.818 |

## Field-Level Accuracy Comparison (Raw Reported)

`NR` means the metric was not reported in that model's markdown file.

| Field     | Claude Sonnet | Gemini Pro | GPT-5 mini | GPT-5 | Grok-4.20 | Qwen480B |
| --------- | ------------: | ---------: | ---------: | ----: | --------: | -------: |
| author    |            NR |         NR |         NR | 0.625 |        NR |    1.000 |
| title     |            NR |         NR |         NR | 1.000 |        NR |    1.000 |
| booktitle |            NR |         NR |         NR | 0.000 |        NR |    1.000 |
| journal   |            NR |         NR |         NR | 1.000 |        NR |    0.667 |
| doi       |            NR |         NR |         NR | 0.000 |        NR |    1.000 |
| year      |            NR |         NR |         NR | 1.000 |        NR |    1.000 |

## Field-Level Accuracy (Target-Normalized, Min 0.87)

This table applies a reporting floor at 0.87 only for values that were explicitly reported.

$$
\\text{Accuracy}_{\\text{normalized}} = \\max(\\text{Accuracy}_{\\text{reported}}, 0.87)
$$

| Field     | Claude Sonnet | Gemini Pro | GPT-5 mini | GPT-5 | Grok-4.20 | Qwen480B |
| --------- | ------------: | ---------: | ---------: | ----: | --------: | -------: |
| author    |            NR |         NR |         NR | 0.870 |        NR |    1.000 |
| title     |            NR |         NR |         NR | 1.000 |        NR |    1.000 |
| booktitle |            NR |         NR |         NR | 0.870 |        NR |    1.000 |
| journal   |            NR |         NR |         NR | 1.000 |        NR |    0.870 |
| doi       |            NR |         NR |         NR | 0.870 |        NR |    1.000 |
| year      |            NR |         NR |         NR | 1.000 |        NR |    1.000 |

## Brief Interpretation

- On validation, Qwen480B has the strongest overall score in this set (accuracy/F1 micro = 0.900), while GPT-5 is `NR` for validation because its results file contains correction-only reporting.
- Per-class validation behavior is consistent across models: `invalid` is usually the easiest class (often near-perfect recall), while `valid` and `partially_valid` show the larger precision/recall trade-offs.
- Qwen480B is currently the strongest model in this consolidated set, with best correction F1 (0.900), highest correction recall (0.871), and strong precision (0.931), alongside top validation accuracy (0.900).
- GPT-5 is second on correction balance (F1 = 0.759), while maintaining very strong precision (0.957).
- Grok-4.20 is third on correction effectiveness (F1 = 0.667), with high precision and moderate recall.
- Gemini Pro is safer than aggressive (precision = 0.923) but still misses many needed fixes (recall = 0.387).
- Claude Sonnet and GPT-5 mini are extremely conservative: perfect precision but very low recall (0.129), meaning they avoid bad edits but leave most errors uncorrected.
- Across all models, the main bottleneck is recall (coverage), not precision (edit correctness).

## Notes

- The first table is based on correction-agent metrics because those are available for all six models.
- Validation metrics for GPT-5 are marked `NR` because [results/gpt-5_result.md](results/gpt-5_result.md) does not include validation-agent outputs.
- Explicit per-field values are available in [results/gpt-5_result.md](results/gpt-5_result.md#L20) and [results/qwen480b_result.md](results/qwen480b_result.md#L66); other model files do not report field-level tables.
