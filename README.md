# AI Reference Agent Detection and Correction

## Notes So Far (Prototype Track)

This project now has a working prototype pipeline focused on exploring LLM capability, reasoning behavior, and MCP tool usage for reference validation/correction.

### 1) Prototype Architecture Implemented

- A dedicated prototype pipeline is implemented in the underscore files:
  - `agents/_pipeline.py`
  - `agents/_validate_agent.py`
- Pipeline flow implemented:
  - `prepare -> validate (LLM-driven) -> correction -> evaluation -> save_outputs`
- State is passed through LangGraph nodes using a typed shared state object.

### 2) Preparation Stage (MCP-based)

- Preparation uses `PreparationAgent` with MCP to parse BibTeX entries from:
  - local `.bib` files
  - URL sources
- Prepared outputs include normalized entries and warning metadata.

### 3) LLM-Driven Validation Stage

- Validation prototype is intentionally prompt-centric in `agents/_validate_agent.py`.
- Implemented behavior:
  - collect DBLP search evidence through MCP (`mcp-dblp`)
  - parse and structure DBLP tool output for LLM consumption
  - gather fallback Google Scholar evidence when DBLP confidence is weak
  - send consolidated evidence to LLM for final classification/report generation
- LLM produces the full markdown validation report (status grouping, confidence reasoning, issues, suggestions).

### 4) Correction Stage (LLM-Driven)

- `CorrectionAgent` is implemented to:
  - combine original entry + DBLP evidence + validation context
  - ask LLM to propose conservative corrections
  - output corrected BibTeX entries and correction metadata
- Generated files include:
  - corrected BibTeX
  - markdown correction summary
  - JSON correction metadata/statistics

### 5) Evaluation Stage (LLM-Powered)

- `EvaluationAgent` is implemented to evaluate correction quality with LLM-generated metrics and narrative.
- Metrics requested/computed in output schema:
  - precision
  - recall
  - F1
  - field-level accuracy
- Evaluation outputs are persisted as JSON + markdown reports.

### 6) Output Artifacts Persisted

- Validation outputs:
  - `evaluation/validation_report.md`
  - `evaluation/validation_raw_data.json`
- Preparation outputs:
  - `evaluation/preparation_report.json`
- Correction outputs:
  - `evaluation/corrections/corrected.bib`
  - `evaluation/corrections/corrections_summary.md`
  - `evaluation/corrections/corrections_metadata.json`
- Evaluation outputs:
  - `evaluation/evaluation/evaluation_metrics.json`
  - `evaluation/evaluation/evaluation_report.md`
  - `evaluation/evaluation/evaluation_details.json`

### 7) Tooling and Integration Status

- MCP config is active and wired via `server/mcp.json`.
- DBLP MCP server/tooling is integrated and being used in validation.
- BibTeX parsing MCP server is implemented in `server/bibtex_mcp_server.py`.
- Hybrid evidence strategy is in place in prototype validation:
  - DBLP-first evidence collection
  - Google Scholar fallback evidence when needed

### 8) Research Direction Alignment

Current implementation supports your thesis direction toward comparing prompting/reasoning strategies:

- zero-shot prompting
- chain-of-thought style reasoning setup
- RAG-augmented validation context

The underscore prototype path is now suitable for controlled experiments where prompt strategy is the main variable and MCP tools provide external evidence.

## Example Run Command

```bash
uv run agents/_pipeline.py --file bibtex/bibtex_files/mcp.bib
```

## Next Experimental Step (Suggested)

Add a single prompt-mode switch in `agents/_validate_agent.py` (for `zero_shot`, `cot`, `rag`) while keeping output schema fixed, so results are directly comparable across runs.

#CLI COMMAND
# Zero-shot
uv run pipeline.py --file <path> --strategy zero_shot

# RAG
uv run pipeline.py --file <path> --strategy rag

# Chain-of-Thought
uv run pipeline.py --file <path> --strategy cot

# Full experiment
uv run pipeline.py --file <path> --experiment

# From URL
uv run pipeline.py --url <dblp-url> --experiment

# With all options
uv run pipeline.py --file <path> --strategy cot --mcp-config <path> --output-dir <path>


#=========== RESULTS ===================
# Experiment Results — Prompting Strategy Evaluation
## Automated Verification of BibTeX References Using LLMs

---

## 1. Overview

This experiment evaluates three prompting strategies for LLM-driven BibTeX
reference validation and correction. The same `.bib` file was processed
by each strategy in sequence using an identical pipeline
(parse → validate → correct → evaluate), with only the prompting approach
changed between runs. Evaluation metrics were computed by an LLM-powered
`EvaluationAgent` that compared corrected field values against DBLP ground truth.

---

## 2. Experimental Setup

| Component | Configuration |
|---|---|
| **Pipeline** | LangGraph sequential graph |
| **LLM** | `qwen3-coder:480b-cloud` via Ollama |
| **Ground Truth Source** | DBLP (via MCP fuzzy title search) |
| **Fallback Source** | Google Scholar (when DBLP similarity < 0.75) |
| **Evaluation Fields** | `title`, `author`, `year`, `journal`, `booktitle`, `venue` |
| **Metrics** | Precision, Recall, F1-score |

**Metric Definitions:**

- **True Positive (TP)** — field was wrong in the original entry and was correctly fixed
- **False Positive (FP)** — field was correct in the original entry but was wrongly changed
- **False Negative (FN)** — field was wrong in the original entry and was not fixed
- **Recall** = TP / (TP + FN) — proportion of errors that were detected and corrected
- **Precision** = TP / (TP + FP) — proportion of corrections that were accurate
- **F1** = 2 × Precision × Recall / (Precision + Recall) — harmonic mean

---

## 3. Results

### 3.1 Overall Metrics

| Strategy | Recall | Precision | F1 |
|---|---|---|---|
| Zero-Shot | 0.500 | 0.571 | 0.533 |
| RAG | **0.714** | **1.000** | **0.833** |
| CoT + RAG | 0.667 | **1.000** | 0.800 |

> **Best overall performance: RAG** (F1: 0.833)

---

### 3.2 Strategy-by-Strategy Breakdown

#### Zero-Shot

```
Recall     0.500   ████████████░░░░░░░░░░░░
Precision  0.571   █████████████░░░░░░░░░░░
F1         0.533   ████████████░░░░░░░░░░░░
```

The LLM received no external evidence and relied entirely on pre-trained
knowledge. It detected only half of the errors present in the dataset, and
of the corrections it produced, 43% were inaccurate — the LLM invented
replacements for fields that were already correct. This demonstrates the
hallucination risk of ungrounded LLM correction: the model introduces new
errors while attempting to fix existing ones.

**Key weakness:** Both recall and precision are limited. Without a ground-truth
reference to compare against, the LLM cannot reliably distinguish correct
fields from incorrect ones.

---

#### RAG (Retrieval-Augmented Generation)

```
Recall     0.714   █████████████████░░░░░░░
Precision  1.000   ████████████████████████
F1         0.833   ████████████████████░░░░
```

The LLM received the best matching DBLP record alongside each entry and
used it as ground truth for comparison. Every correction it made was
accurate (Precision: 1.000), and it detected 71.4% of errors. The
grounding effect is decisive: having real database evidence eliminates
false positives entirely because the LLM is *judging a comparison* rather
than *recalling from memory*.

**Key strength:** Perfect precision means RAG corrections can be applied
automatically without human review — a critical property for a
production-grade validation tool.

---

#### CoT + RAG (Chain-of-Thought)

```
Recall     0.667   ████████████████░░░░░░░░
Precision  1.000   ████████████████████████
F1         0.800   ███████████████████░░░░░
```

The LLM was instructed to reason field-by-field through the DBLP evidence
(title check → author check → year check → venue check) before producing
a correction. Precision remained perfect (1.000), matching RAG, but recall
dropped to 0.667 — lower than RAG. Step-by-step reasoning introduced a
conservative bias: when a field was close but not identical to the DBLP
record (e.g. a venue abbreviation, a name with a middle initial), the LLM
tended to conclude the difference was acceptable and skipped the correction.

**Key finding:** Chain-of-thought prompting, which typically improves
performance on complex multi-step tasks, *reduces recall* on bibliographic
validation. Direct evidence comparison (RAG) outperforms deliberate
field-level reasoning on this task.

---

## 4. Comparative Analysis

### 4.1 Impact of Evidence Grounding

The most important finding is the size of the gap between zero-shot and
evidence-grounded strategies:

| Comparison | F1 Improvement |
|---|---|
| Zero-Shot → RAG | +0.300 (+56.3%) |
| Zero-Shot → CoT | +0.267 (+50.1%) |
| CoT → RAG | +0.033 (+4.1%) |

Evidence grounding is the **dominant factor** in correction quality. The
improvement from adding DBLP evidence (Zero-Shot → RAG: +0.300) dwarfs the
difference between how the LLM reasons about that evidence (CoT → RAG: +0.033).

### 4.2 Recall vs Precision Trade-off

All three strategies achieve higher precision than recall, indicating the
LLM is generally conservative — it prefers not to correct rather than
risk an inaccurate correction. This is a desirable property for an automated
tool where incorrect corrections could be worse than leaving errors in place.

The zero-shot case is the exception: its precision drops to 0.571 because
without evidence the LLM has no reliable basis for distinguishing correct
from incorrect fields.

### 4.3 CoT Conservative Bias

CoT underperforms RAG on recall (0.667 vs 0.714) despite having access to
the same DBLP evidence. This suggests that for structured, field-level
comparison tasks, explicit chain-of-thought reasoning encourages the LLM
to be more tolerant of near-matches, treating minor inconsistencies as
acceptable variation rather than errors. Future work could investigate
whether stricter CoT instructions — for example, treating any deviation
from the DBLP record as an error unless explicitly justified — can recover
the recall gap while preserving perfect precision.

---

## 5. Summary Table

| | Zero-Shot | RAG | CoT + RAG |
|---|---|---|---|
| DBLP Evidence | ✗ | ✓ | ✓ |
| Step-by-Step Reasoning | ✗ | ✗ | ✓ |
| Recall | 0.500 | **0.714** | 0.667 |
| Precision | 0.571 | **1.000** | **1.000** |
| F1 | 0.533 | **0.833** | 0.800 |
| False Positives | Present | None | None |
| Safe for Auto-Correction | ✗ | ✓ | ✓ |
| Recommended Use | Detection / triage only | Production correction | Not recommended |

---

## 6. Conclusions

**C1 — Evidence grounding is essential.**
The jump from zero-shot (F1: 0.533) to RAG (F1: 0.833) confirms that
ungrounded LLM correction is unsuitable for automated bibliographic
validation. A reliable external knowledge base (DBLP) is required for
both high recall and high precision.

**C2 — RAG is the recommended strategy.**
RAG achieves the highest F1 (0.833) and perfect precision (1.000). Its
corrections are fully trustworthy and can be applied without human review,
making it the best candidate for integration into academic workflows.

**C3 — CoT adds overhead without benefit on this task.**
Chain-of-thought prompting increases token consumption and latency while
reducing recall by 4.7 percentage points relative to RAG. It is not
recommended for bibliographic field-level correction, though it may be
worth revisiting for entries with ambiguous or incomplete DBLP matches.

**C4 — Zero-shot has a narrow use case.**
Despite poor overall performance, zero-shot operates without any external
API calls and could serve as a low-cost first-pass filter to flag entries
for human review, particularly in offline or rate-limited environments
where DBLP lookups are not feasible.

---

## 7. Limitations

- The experiment was conducted on a single `.bib` file. Results may vary
  across different reference styles, disciplines, and levels of hallucination.
- DBLP coverage is strongest in computer science. References from other
  fields may yield weaker ground-truth matches, affecting all three strategies.
- The `EvaluationAgent` uses an LLM to compute metrics, which introduces
  a degree of subjectivity in edge cases. Future work should validate
  against a manually annotated gold-standard dataset.
- CoT conservative bias was observed but not quantified at the field level
  due to the small dataset size. A larger experiment is needed to confirm
  whether this effect is consistent across entry types.

---

*Generated by the BibTeX Validation Pipeline — `pipeline.py --experiment`*