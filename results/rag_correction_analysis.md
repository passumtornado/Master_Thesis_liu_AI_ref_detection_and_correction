# Correction Agent Performance Analysis (RAG Strategy)

## Overview

This report analyzes the performance of the correction agent across 5 LLM models using the RAG strategy. The correction agent attempts to fix bibliographic errors in entries classified as "partially_valid" by the validation agent.

The metrics below are field-level and aggregated per model from the scored datasets. They are not the same as whole-entry visual correctness in the exported BibTeX, which can look higher because many entries contain only a few corrupted fields.

---

## Table 1: Aggregate Correction Metrics by Model

| Rank | Model               | Precision | Recall | F1 Score | Accuracy | TP  | FP  | FN  | Total Fields Corrected |
| ---- | ------------------- | --------- | ------ | -------- | -------- | --- | --- | --- | ---------------------- |
| 1    | gemini-3.1-pro-prev | 0.985     | 0.390  | 0.558    | 0.387    | 67  | 1   | 105 | 30                     |
| 2    | gpt-5.4             | 0.982     | 0.213  | 0.349    | 0.212    | 55  | 1   | 203 | 47                     |
| 3    | grok-4.20           | 0.955     | 0.077  | 0.144    | 0.077    | 21  | 1   | 248 | 59                     |
| 4    | claude-sonnet-4.6   | 0.813     | 0.055  | 0.103    | 0.055    | 13  | 3   | 222 | 76                     |
| 5    | gemini-2.5-pro      | 1.000     | 0.028  | 0.055    | 0.028    | 2   | 0   | 69  | 13                     |

| **TOTAL ACROSS MODELS** | **5 Models** | **0.963** | **0.157** | **0.270** | **0.156** | **158** | **6** | **847** | **225** |

### Table 1 Legend

- **TP (True Positives):** Field errors successfully corrected
- **FP (False Positives):** Correct fields incorrectly changed by the correction agent
- **FN (False Negatives):** Field errors not fixed by the correction agent
- **Precision:** Of all corrections made, how many were correct?
- **Recall:** Of all errors that existed, how many were fixed?
- **F1:** Harmonic mean of precision and recall
- **Accuracy:** Field-level correctness ratio computed from TP, FP, and FN

---

## Table 2: Field-Level Corrections by Model

| Model               | Dataset 2 Fields | Dataset 3 Fields | Dataset 4 Fields | **Total Fields Corrected** |
| ------------------- | ---------------- | ---------------- | ---------------- | -------------------------- |
| claude-sonnet-4.6   | 18               | 25               | 33               | **76**                     |
| grok-4.20           | 15               | 25               | 19               | **59**                     |
| gpt-5.4             | 20               | 0                | 27               | **47**                     |
| gemini-3.1-pro-prev | 0                | 8                | 22               | **30**                     |
| gemini-2.5-pro      | 13               | 0                | 0                | **13**                     |
| **TOTAL**           |                  |                  |                  | **225**                    |

## Table 3: Entry-Level Visual Correctness (per-model)

This table counts how many eligible `partially_valid` entries were fully corrected (no remaining corrupted fields) versus partially corrected. "Correctness Rate" shows the percent of eligible entries that became fully correct after correction.

| Model (dataset)               | Eligible Entries | Fully Correct | Partially Correct | Correctness Rate |
| ----------------------------- | ---------------: | ------------: | ----------------: | ---------------: |
| claude-sonnet-4.6 (stefan2)   |               13 |             0 |                 2 |             0.0% |
| claude-sonnet-4.6 (stefan3)   |               13 |             1 |                 3 |             7.7% |
| claude-sonnet-4.6 (stefan4)   |               14 |             0 |                 2 |             0.0% |
| gemini-2.5-pro (stefan2)      |               16 |             0 |                 5 |             0.0% |
| gemini-3.1-pro-prev (stefan3) |               17 |            10 |                 0 |            58.8% |
| gemini-3.1-pro-prev (stefan4) |               14 |             1 |                 2 |             7.1% |
| gpt-5.4 (stefan2)             |               13 |             1 |                 1 |             7.7% |
| gpt-5.4 (stefan3)             |               17 |             6 |                 2 |            35.3% |
| gpt-5.4 (stefan4)             |               14 |             1 |                 4 |             7.1% |
| grok-4.20 (stefan2)           |               14 |             2 |                 0 |            14.3% |
| grok-4.20 (stefan3)           |               17 |             1 |                 1 |             5.9% |
| grok-4.20 (stefan4)           |               16 |             0 |                 1 |             0.0% |

**Note:** A few model/dataset folders were missing evaluation details; those are omitted. Entry-level correctness is stricter than visual inspection and explains why exported BibTeX may _appear_ ~90% correct while field-level recall is lower.

---

## Key Findings & Discussion

### 1. Precision vs. Recall Tradeoff

The models show a clear precision-recall tradeoff.

- **High Precision (0.813-1.000):** When the correction agent makes a change, it is usually correct.
- **Low Recall (0.028-0.390):** Most corrupted fields are still left unchanged.

This means the correction agent is conservative. It avoids wrong fixes, but it also misses many fields that should have been corrected.

### 2. Best Performing Models

1. **gemini-3.1-pro-prev**
   - F1: 0.558
   - Precision: 0.985 | Recall: 0.390
   - Fields corrected: 30
   - Best balance of precision and recall.

2. **gpt-5.4**
   - F1: 0.349
   - Precision: 0.982 | Recall: 0.213
   - Fields corrected: 47
   - Highest correction volume, but still moderate recall.

3. **grok-4.20**
   - F1: 0.144
   - Precision: 0.955 | Recall: 0.077
   - Fields corrected: 59
   - Many attempts, but few of the corrupted fields were recovered.

### 3. Model-Specific Observations

#### claude-sonnet-4.6

- Metrics: P=0.813, R=0.055, F1=0.103
- Lowest precision among the five models.
- The model still corrected 76 fields overall, but most corrupted fields remained unfixed.

#### gemini-2.5-pro

- Metrics: P=1.000, R=0.028, F1=0.055
- Perfect precision, but it corrected very few fields.
- This is the most conservative model in the set.

#### gemini-3.1-pro-prev

- Metrics: P=0.985, R=0.390, F1=0.558
- Best overall model by F1.
- It recovered the largest share of errors while keeping precision extremely high.

#### gpt-5.4

- Metrics: P=0.982, R=0.213, F1=0.349
- Strong precision and better recall than most models.
- Good candidate when balanced correction quality matters.

#### grok-4.20

- Metrics: P=0.955, R=0.077, F1=0.144
- Very high correction count, but low recovery rate.

### 4. Why the Export May Look Better Than Recall

The exported BibTeX can look close to correct at a glance because many entries only have one or two corrupted fields. Field-level recall is stricter: it counts every corrupted field that remains unresolved. So a file can look 90-95% clean visually and still show low recall if several corrupted fields remain per entry.

### 5. Overall Assessment

The recheck confirms that the report should be read as a **field-level correction evaluation**, not an entry-level visual quality score.

- The best performer is **gemini-3.1-pro-prev**.
- The correction system is conservative across all models.
- The main weakness is recall, not precision.

---

## Conclusion

The correction agent is reliable when it makes a change, but it still misses many corrupted fields. The aggregate numbers do not support a recall of 0.8-0.9 for this run. Instead, they show a strong precision bias with limited field recovery, especially for the lower-performing models.

If you want, the next step is to produce a second table that separates entry-level visual correctness from field-level recall, so the report matches both how the BibTeX looks and how the evaluator scores it.
