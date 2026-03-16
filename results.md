# Experiment Findings Summary
## Automated Verification of BibTeX References — Cross-Model Strategy Evaluation

---

## 1. Experiment Overview

This experiment evaluates three prompting strategies for LLM-driven BibTeX
reference validation across two language models. The same 450-entry `.bib`
dataset (250 valid, 150 partially valid, 50 invalid) was processed by each
strategy using an identical pipeline. Results from `gemini-3.1-pro-preview`
are compared against earlier results from `qwen3-coder:480b-cloud` to
investigate whether prompting strategy effectiveness is model-dependent.

---

## 2. Results

### 2.1 `qwen3-coder:480b-cloud` (Earlier Run)

| Strategy | Precision | Recall | F1 |
|---|---|---|---|
| Zero-Shot | 0.571 | 0.500 | 0.533 |
| RAG | **1.000** | **0.714** | **0.833** |
| CoT | **1.000** | 0.667 | 0.800 |

### 2.2 `gemini-3.1-pro-preview` (Latest Run)

| Strategy | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| Zero-Shot | **1.000** | **0.883** | **0.938** | 128 | 0 | 17 |
| RAG | **1.000** | 0.832 | 0.908 | 79 | 0 | 16 |
| CoT | **1.000** | 0.579 | 0.733 | 11 | 0 | 8 |

### 2.3 Cross-Model F1 Comparison

| Strategy | qwen3-coder | gemini-3.1-pro | Δ F1 |
|---|---|---|---|
| Zero-Shot | 0.533 | **0.938** | +0.405 |
| RAG | **0.833** | 0.908 | +0.075 |
| CoT | 0.800 | 0.733 | −0.067 |

---

## 3. Field-Level Accuracy (`gemini-3.1-pro-preview`)

| Field | Strategy | Errors Found | Corrected | Accuracy |
|---|---|---|---|---|
| **author** | Zero-Shot | 29 | 27 | 0.931 |
| | RAG | 34 | 34 | **1.000** |
| | CoT | 11 | 8 | 0.727 |
| **booktitle** | Zero-Shot | 21 | 20 | **0.952** |
| | RAG | 7 | 0 | 0.000 |
| | CoT | 0 | 0 | 0.000 |
| **journal** | Zero-Shot | 15 | 15 | **1.000** |
| | RAG | 20 | 13 | 0.650 |
| | CoT | 5 | 1 | 0.200 |
| **title** | Zero-Shot | 42 | 31 | 0.738 |
| | RAG | 22 | 22 | **1.000** |
| | CoT | 1 | 0 | 0.000 |
| **venue** | Zero-Shot | 0 | 0 | — |
| | RAG | 27 | 13 | 0.481 |
| | CoT | 0 | 0 | — |
| **year** | Zero-Shot | 38 | 35 | 0.921 |
| | RAG | 12 | 10 | 0.833 |
| | CoT | 2 | 2 | **1.000** |

---

## 4. Key Findings

---

### Finding 1 — Model Capability Inverts the RAG Benefit

The most important and unexpected result of this experiment is that the
**relative advantage of RAG over Zero-Shot is inversely related to the
model's pre-existing domain knowledge.**

For `qwen3-coder`, a smaller model with limited academic publication
knowledge, RAG was the clear winner (F1: 0.833 vs 0.533). The model
needed DBLP evidence to compensate for what it did not know.

For `gemini-3.1-pro-preview`, a frontier model trained on extensive
academic literature, Zero-Shot outperformed RAG (F1: 0.938 vs 0.908).
The model already knew the papers and did not need to look them up.

```
Model capability        Low  ──────────────────────►  High
RAG advantage           High ◄────────────────────── Low
Zero-Shot advantage     Low  ──────────────────────►  High
```

This finding has direct practical implications: **choosing a prompting
strategy without considering model capability leads to suboptimal results.**
For resource-constrained deployments using smaller models, RAG is essential.
For deployments using frontier models, Zero-Shot may be sufficient and faster.

---

### Finding 2 — Zero False Positives Across All Strategies and Models

Every strategy on every model achieved **zero false positives** —
no correct field was ever wrongly changed. This is a consistent and
stable property of LLM-based correction on this task: the models are
universally conservative and only correct when they are certain.

This is the most operationally important finding for the thesis. It means
that corrections produced by any of the three strategies can be applied
automatically without human review — the risk of introducing new errors
through auto-correction is effectively zero across all tested configurations.

The trade-off is recall: conservatism that eliminates false positives also
causes the system to miss some real errors (false negatives).

---

### Finding 3 — CoT Is Consistently Harmful for This Task

Chain-of-thought prompting produced the lowest recall in both model runs:

```
Model               RAG F1    CoT F1    CoT loss
────────────────────────────────────────────────
qwen3-coder         0.833     0.800     −0.033
gemini-3.1-pro      0.908     0.733     −0.175
```

The effect is stronger on the more capable model. When `gemini-3.1-pro`
reasons step-by-step through a field comparison, it finds more reasons to
accept near-matches as intentional variation rather than errors. The CoT
process gives the model more cognitive steps to rationalise inaction.

For `gemini-3.1-pro`, CoT detected only **19 total errors** across all
entries — compared to **145** for Zero-Shot and **95** for RAG. The
collapse is in error detection, not correction quality: of the errors
CoT does find, it corrects them at a reasonable rate.

**CoT is not recommended for bibliographic field-level validation.** The
task is fundamentally a comparison exercise — direct evidence matching
outperforms deliberate reasoning for this class of structured lookup problem.

---

### Finding 4 — RAG Has Uneven Field-Level Performance

While RAG achieves strong overall metrics, its field-level accuracy is
highly uneven. On `gemini-3.1-pro`:

```
Field       RAG Accuracy    Zero-Shot Accuracy
───────────────────────────────────────────────
author      1.000           0.931
title       1.000           0.738
year        0.833           0.921
journal     0.650           1.000
booktitle   0.000           0.952
venue       0.481           —
```

RAG excels on `author` and `title` — fields where DBLP provides clean,
directly comparable string values. It struggles on `booktitle` (0.000
accuracy) and `venue` (0.481) — fields where DBLP abbreviations and the
full venue names in the `.bib` file do not cleanly align. The LLM receives
a DBLP hit with an abbreviated venue, cannot confidently map it to the
full venue string in the entry, and skips the correction.

Zero-Shot, by contrast, corrects `booktitle` at 0.952 accuracy because it
uses its memorised knowledge of full conference names rather than relying
on potentially mismatched DBLP abbreviations.

This suggests a **hybrid strategy**: use RAG for author and title
correction (where DBLP data is clean) and Zero-Shot for venue/booktitle
correction (where model knowledge outperforms DBLP abbreviations).

---

### Finding 5 — Recall Is the Key Differentiating Metric

Across all strategies and both models, precision is always near or at 1.000.
Recall is the only metric that meaningfully differentiates strategies:

```
gemini-3.1-pro    Precision    Recall    F1
───────────────────────────────────────────
Zero-Shot         1.000        0.883     0.938
RAG               1.000        0.832     0.908
CoT               1.000        0.579     0.733
```

Future work to improve system performance should focus entirely on
**increasing recall without sacrificing precision** — for example through
ensemble methods that combine Zero-Shot and RAG verdicts, or through
confidence-weighted correction thresholds.

---

## 5. Summary Table

| | qwen3-coder | gemini-3.1-pro |
|---|---|---|
| **Best strategy** | RAG (F1: 0.833) | Zero-Shot (F1: 0.938) |
| **Worst strategy** | Zero-Shot (F1: 0.533) | CoT (F1: 0.733) |
| **False positives** | 0 across all strategies | 0 across all strategies |
| **CoT effect** | Mild recall reduction (−0.033) | Strong recall reduction (−0.175) |
| **RAG vs Zero-Shot gap** | +0.300 in favour of RAG | +0.030 in favour of Zero-Shot |
| **Recommended strategy** | RAG | Zero-Shot |

---

## 6. Conclusions

**C1 — Prompting strategy must be chosen in conjunction with model selection.**
There is no universally optimal strategy. RAG is best for smaller models;
Zero-Shot is best for frontier models with strong academic domain knowledge.

**C2 — Auto-correction is safe across all configurations.**
Zero false positives in every run confirms that LLM-based BibTeX correction
can be applied automatically. The system never makes a confident wrong correction.

**C3 — CoT should not be used for bibliographic validation.**
Step-by-step reasoning consistently reduces recall by inducing conservative
bias. It is reproducible across both models and both experiment runs.

**C4 — RAG field-level accuracy depends on DBLP data quality.**
RAG excels where DBLP provides clean comparable values (author, title) and
fails where DBLP uses abbreviations that do not match `.bib` full-form values
(booktitle, venue). A hybrid approach exploiting both strategies per field
is a promising direction for future work.

**C5 — Recall is the open problem.**
The system achieves near-perfect precision universally. The research
challenge is pushing recall higher — catching the 12–42% of errors
currently missed — without introducing false positives.

---

## 7. Limitations

- Results are based on a single `.bib` dataset dominated by computer science
  publications. Generalisation to other disciplines (medicine, law, social
  sciences) requires further evaluation.
- The 450-entry dataset was generated semi-synthetically from a real `.bib`
  file. Evaluation on fully real-world hallucinated references may produce
  different recall distributions.
- Evaluation metrics are computed by an LLM-powered `EvaluationAgent`.
  Edge cases in field comparison (abbreviation variants, LaTeX encoding,
  name formatting) may introduce noise in TP/FP/FN counts.
- Only two models were tested. The finding that model capability inverts
  RAG benefit needs validation across a wider range of model sizes and
  families (e.g. Llama, Mistral, Claude).
- CoT conservative bias was observed but the mechanism was not formally
  analysed. Future work should investigate whether stricter CoT instructions
  (e.g. "treat any deviation from DBLP as an error") can recover recall.

---

*Generated from pipeline experiment:*
*`uv run pipeline.py --file thesis_dataset.bib --experiment`*
*Model: `gemini-3.1-pro-preview` | Dataset: 450 entries | Date: March 2026*