# Section A: Validation Agent Model & Strategy Comparison

This section compares the performance of different models across three prompt strategies
(zero-shot, RAG, and chain-of-thought) using datasets Stefan2 and Stefan3 combined.

## ZERO-SHOT Strategy

| Model | Precision | Recall | F1 | Accuracy |
| --- | --- | --- | --- | --- |
| claude-sonnet-4.6 | 0.878 | 0.878 | 0.878 | 0.878 |
| gpt-5.4 | 0.861 | 0.861 | 0.861 | 0.861 |
| gemini-3.1-pro-preview | 0.811 | 0.811 | 0.811 | 0.811 |
| grok-4.20 | 0.801 | 0.801 | 0.801 | 0.801 |
| qwen3.6-35b-a3b | 0.732 | 0.732 | 0.732 | 0.732 |

**Commentary:** claude-sonnet-4.6 achieved the highest accuracy (87.8%) using the zero-shot strategy on combined Stefan2+3 datasets.

## RAG Strategy

| Model | Precision | Recall | F1 | Accuracy |
| --- | --- | --- | --- | --- |
| grok-4.20 | 0.951 | 0.951 | 0.951 | 0.951 |
| claude-sonnet-4.6 | 0.919 | 0.919 | 0.919 | 0.919 |
| gemini-3.1-pro-prev | 0.901 | 0.901 | 0.901 | 0.901 |
| gpt-5.4 | 0.881 | 0.881 | 0.881 | 0.881 |
| gemini-2.5-pro | 0.627 | 0.627 | 0.627 | 0.627 |

**Commentary:** grok-4.20 achieved the highest accuracy (95.1%) using the rag strategy on combined Stefan2+3 datasets.

## COT Strategy

| Model | Precision | Recall | F1 | Accuracy |
| --- | --- | --- | --- | --- |
| gemini-3.1-pro-prev | 0.922 | 0.922 | 0.922 | 0.922 |
| gpt-5.4 | 0.921 | 0.921 | 0.921 | 0.921 |
| claude-sonnet-4.6 | 0.840 | 0.840 | 0.840 | 0.840 |
| grok-4.20 | 0.744 | 0.744 | 0.744 | 0.744 |

**Commentary:** gemini-3.1-pro-prev achieved the highest accuracy (92.2%) using the cot strategy on combined Stefan2+3 datasets.


# Section B: Validation Per-Class Metrics (RAG Strategy)

This section focuses exclusively on the RAG (Retrieval-Augmented Generation) strategy,
analyzing per-class classification performance across all available datasets (Stefan1-4).
Models are ranked by F1 score for each class.

## VALID Class Metrics

| Model | Set | Precision | Recall | F1 | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-3.1-pro-prev | 3 | 1.000 | 1.000 | 1.000 | 26 | 0 | 0 |
| gemini-3.1-pro-prev | 2 | 0.963 | 1.000 | 0.981 | 26 | 1 | 0 |
| grok-4.20 | 4 | 0.963 | 1.000 | 0.981 | 26 | 1 | 0 |
| gemini-3.1-pro-prev | 4 | 1.000 | 0.962 | 0.980 | 27 | 2 | 1 |
| grok-4.20 | 2 | 1.000 | 0.962 | 0.980 | 25 | 0 | 1 |
| grok-4.20 | 3 | 1.000 | 0.962 | 0.980 | 25 | 0 | 1 |
| claude-sonnet-4.6 | 2 | 0.929 | 1.000 | 0.963 | 26 | 2 | 0 |
| claude-sonnet-4.6 | 3 | 0.929 | 1.000 | 0.963 | 26 | 2 | 0 |
| claude-sonnet-4.6 | 4 | 0.958 | 0.920 | 0.939 | 23 | 1 | 2 |
| gpt-5.4 | 2 | 0.867 | 1.000 | 0.929 | 26 | 4 | 0 |
| gpt-5.4 | 3 | 1.000 | 0.769 | 0.870 | 20 | 0 | 6 |
| gemini-2.5-pro | 2 | 0.867 | 0.500 | 0.634 | 13 | 2 | 13 |
| gpt-5.4 | 4 | 1.000 | 0.269 | 0.424 | 7 | 0 | 19 |

## PARTIALLY_VALID Class Metrics

| Model | Set | Precision | Recall | F1 | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| grok-4.20 | 3 | 0.944 | 1.000 | 0.971 | 17 | 1 | 0 |
| gemini-3.1-pro-prev | 4 | 1.000 | 0.934 | 0.956 | 15 | 1 | 1 |
| gpt-5.4 | 3 | 0.895 | 1.000 | 0.944 | 17 | 2 | 0 |
| gemini-3.1-pro-prev | 3 | 0.850 | 1.000 | 0.919 | 17 | 3 | 0 |
| grok-4.20 | 4 | 0.889 | 0.941 | 0.914 | 16 | 2 | 1 |
| claude-sonnet-4.6 | 4 | 1.000 | 0.824 | 0.903 | 14 | 0 | 3 |
| grok-4.20 | 2 | 1.000 | 0.824 | 0.903 | 14 | 0 | 3 |
| claude-sonnet-4.6 | 2 | 1.000 | 0.765 | 0.867 | 13 | 0 | 4 |
| claude-sonnet-4.6 | 3 | 0.867 | 0.867 | 0.867 | 13 | 2 | 2 |
| gpt-5.4 | 2 | 1.000 | 0.765 | 0.867 | 13 | 0 | 4 |
| gemini-3.1-pro-prev | 2 | 1.000 | 0.706 | 0.828 | 12 | 0 | 5 |
| gemini-2.5-pro | 2 | 1.000 | 0.647 | 0.786 | 11 | 0 | 6 |
| gpt-5.4 | 4 | 0.424 | 0.824 | 0.560 | 14 | 19 | 3 |

## INVALID Class Metrics

| Model | Set | Precision | Recall | F1 | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| claude-sonnet-4.6 | 2 | 1.000 | 1.000 | 1.000 | 8 | 0 | 0 |
| claude-sonnet-4.6 | 3 | 1.000 | 1.000 | 1.000 | 6 | 0 | 0 |
| claude-sonnet-4.6 | 4 | 1.000 | 1.000 | 1.000 | 7 | 0 | 0 |
| gemini-3.1-pro-prev | 4 | 1.000 | 1.000 | 1.000 | 7 | 0 | 0 |
| gpt-5.4 | 2 | 1.000 | 1.000 | 1.000 | 8 | 0 | 0 |
| grok-4.20 | 3 | 1.000 | 1.000 | 1.000 | 7 | 0 | 0 |
| grok-4.20 | 4 | 1.000 | 1.000 | 1.000 | 7 | 0 | 0 |
| gemini-2.5-pro | 2 | 0.800 | 1.000 | 0.889 | 8 | 2 | 0 |
| gemini-3.1-pro-prev | 2 | 1.000 | 0.750 | 0.857 | 6 | 0 | 2 |
| grok-4.20 | 2 | 0.727 | 1.000 | 0.842 | 8 | 3 | 0 |
| gpt-5.4 | 4 | 0.700 | 1.000 | 0.824 | 7 | 3 | 0 |
| gemini-3.1-pro-prev | 3 | 1.000 | 0.571 | 0.727 | 4 | 0 | 3 |
| gpt-5.4 | 3 | 0.455 | 0.714 | 0.556 | 5 | 6 | 2 |

## Average Metrics Across Datasets (RAG Strategy)

| Model | Avg Precision | Avg Recall | Avg F1 |
| --- | --- | --- | --- |
| grok-4.20 | 0.947 | 0.965 | 0.952 |
| claude-sonnet-4.6 | 0.965 | 0.931 | 0.945 |
| gemini-3.1-pro-prev | 0.979 | 0.880 | 0.916 |
| gpt-5.4 | 0.816 | 0.816 | 0.775 |
| gemini-2.5-pro | 0.889 | 0.716 | 0.770 |

## Overall Confusion Matrix (RAG Strategy, All Datasets Combined)

| Metric | Value |
| --- | --- |
| True Positives (TP) | 570 |
| False Positives (FP) | 59 |
| False Negatives (FN) | 82 |


# Section C: Correction Agent Performance Analysis

This section analyzes the correction agent performance across different strategies,
evaluating the effectiveness of field-level corrections on partially-valid entries.

## Correction Agent Performance by Strategy

### COT Strategy

| Model | Precision | Recall | F1 |
| --- | --- | --- | --- |
| gpt-5.4 | 0.980 | 0.312 | 0.473 |
| gemini-3.1-pro-prev | 1.000 | 0.272 | 0.427 |
| grok-4.20 | 1.000 | 0.045 | 0.084 |
| claude-sonnet-4.6 | 0.334 | 0.038 | 0.069 |

### RAG Strategy

| Model | Precision | Recall | F1 |
| --- | --- | --- | --- |
| grok-4.20 | 0.958 | 0.392 | 0.453 |
| gpt-5.4 | 0.976 | 0.207 | 0.328 |
| gemini-3.1-pro-prev | 0.792 | 0.246 | 0.322 |
| claude-sonnet-4.6 | 0.745 | 0.056 | 0.102 |
| gemini-2.5-pro | 1.000 | 0.028 | 0.055 |

### ZERO-SHOT Strategy

| Model | Precision | Recall | F1 |
| --- | --- | --- | --- |
| claude-sonnet-4.6 | 0.900 | 0.164 | 0.277 |
| grok-4.20 | 1.000 | 0.076 | 0.141 |
| gpt-5.4 | 1.000 | 0.048 | 0.091 |
| gemini-3.1-pro-preview | 0.250 | 0.015 | 0.029 |
| qwen3.6-35b-a3b | 0.000 | 0.000 | 0.000 |

## Key Observations

- **Low Recall Challenge:** Correction agents across all strategies show consistently low recall,
  indicating difficulty in identifying all partially-valid entries that require correction.
- **High Precision:** When corrections are attempted, most are accurate (high precision),
  suggesting careful prediction when corrections are made.
- **Strategy Effectiveness:** RAG strategy generally outperforms other strategies in correction tasks,
  likely due to access to reference data for validation and correction decisions.

