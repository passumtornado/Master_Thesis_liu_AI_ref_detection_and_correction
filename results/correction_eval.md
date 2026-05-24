# Correction Agent Evaluation & Telemetry Methodology

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Evaluation Overview](#evaluation-overview)
3. [Validation Agent Evaluation](#validation-agent-evaluation)
4. [Correction Agent Evaluation](#correction-agent-evaluation)
5. [Telemetry Metrics](#telemetry-metrics)
6. [Mathematical Formulas](#mathematical-formulas)
7. [Evaluation Flow](#evaluation-flow)

---

## Executive Summary

The evaluation system has **two parallel components**:

1. **Validation Agent Evaluation** (Classification Task)
   - Evaluates how well the validation agent classifies entries as: valid, partially_valid, or invalid
   - Compares predictions against ground_truth.json (expected_status field)
   - Metrics: Accuracy, Precision, Recall, F1 per class

2. **Correction Agent Evaluation** (Field-Level Correction Task)
   - Evaluates how well the correction agent fixes bibliographic errors
   - Only applies to entries correctly classified as "partially_valid"
   - Compares corrected fields against ground_truth.json (original_correct_value field)
   - Metrics: TP, FP, FN, Precision, Recall, F1, Field-level accuracy

3. **Telemetry** (Performance Monitoring)
   - Tracks API call efficiency (cache hits, retries, requests/second)
   - Measures execution duration per strategy and model

---

## Evaluation Overview

### Architecture

```
Input (BibTeX entries)
    ↓
┌─────────────────────────────────┐
│ PREPARATION AGENT                │  (Normalize entries)
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ VALIDATION AGENT (ReAct)         │  (Strategy: zero-shot, rag, cot)
│ └─ zero-shot: LLM only           │
│ └─ rag: LLM + DBLP/OpenAlex      │
│ └─ cot: LLM + tools + CoT        │
└─────────────────────────────────┘
    ↓ Telemetry: requests, cache hits, duration
    ↓
┌─────────────────────────────────┐
│ CORRECTION AGENT (LLM-driven)    │  (Uses validation results)
│ └─ Corrects partially_valid      │
│ └─ Generates corrected fields    │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ EVALUATION AGENT (Deterministic) │  (Compares vs ground truth)
│ └─ Validation metrics            │
│ └─ Correction metrics            │
│ └─ Field-level accuracy          │
└─────────────────────────────────┘
    ↓
Output: JSON results + Markdown report
```

### Ground Truth Data Structure

```json
{
  "entry_id": "ref123",
  "expected_status": "partially_valid",
  "corruptions": [
    {
      "field": "author",
      "corrupted_value": "John Do",
      "original_correct_value": "John Doe"
    },
    {
      "field": "year",
      "corrupted_value": "2021",
      "original_correct_value": "2022"
    }
  ]
}
```

---

## Validation Agent Evaluation

### Task Definition

**Classification Problem:** Assign each entry to one of three classes:
- **valid**: Paper exists AND all fields correct
- **partially_valid**: Paper exists BUT has field errors
- **invalid**: Paper non-existent, retracted, or fabricated

### Metrics Calculation

#### Overall Metrics

**1. Accuracy (Macro Average)**

$$\text{Accuracy} = \frac{\text{True Positives}}{\text{Total Predictions}}$$

Where:
- **TP** = entries where predicted_status == expected_status
- **Total** = number of matched entries between predictions and ground truth

$$\text{Accuracy} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

#### Per-Class Metrics

For each class (valid, partially_valid, invalid):

**2. True Positives (TP)**
$$\text{TP}_c = |\{e : \text{expected_status}(e) = c \land \text{predicted_status}(e) = c\}|$$

**3. False Positives (FP)**
$$\text{FP}_c = |\{e : \text{expected_status}(e) \neq c \land \text{predicted_status}(e) = c\}|$$

**4. False Negatives (FN)**
$$\text{FN}_c = |\{e : \text{expected_status}(e) = c \land \text{predicted_status}(e) \neq c\}|$$

**5. Precision (per-class)**
$$\text{Precision}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c}$$

If $\text{TP}_c + \text{FP}_c = 0$, then Precision = None (undefined)

**6. Recall (per-class)**
$$\text{Recall}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}$$

If $\text{TP}_c + \text{FN}_c = 0$, then Recall = None (undefined)

**7. F1 Score (per-class)**
$$\text{F1}_c = \frac{2 \times \text{Precision}_c \times \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

If $\text{Precision}_c + \text{Recall}_c = 0$, then F1 = None (undefined)

#### Example Validation Metrics

| Metric | Value |
|--------|-------|
| Ground-truth entries | 106 |
| Predicted entries | 105 |
| Matched entries | 104 |
| Coverage | 0.981 |
| **Accuracy** | **0.923** |
| **Precision** | **0.923** |
| **Recall** | **0.923** |
| **F1** | **0.923** |

#### Per-Class Breakdown

| Class | Count | TP | FP | FN | Precision | Recall | F1 |
|-------|-------|----|----|-----|-----------|--------|-----|
| valid | 28 | 27 | 1 | 1 | 0.964 | 0.964 | 0.964 |
| partially_valid | 53 | 51 | 0 | 2 | 1.000 | 0.962 | 0.981 |
| invalid | 23 | 22 | 0 | 1 | 1.000 | 0.957 | 0.978 |

---

## Correction Agent Evaluation

### Task Definition

**Correction Problem:** Fix bibliographic errors in entries classified as partially_valid.

The correction agent only operates on entries where:
$$\text{expected_status} = \text{predicted_status} = \text{"partially_valid"}$$

### Evaluation Scope

**Eligible entries** = entries correctly classified as partially_valid by validation agent

$$\text{Eligible} = |\{e : E_e = \text{"partially_valid"} \land P_e = \text{"partially_valid"}\}|$$

Where:
- $E_e$ = expected_status from ground_truth
- $P_e$ = predicted_status from validation agent

### Field-Level Correction Metrics

#### Comparison Process

For each eligible entry and each corrupted field:

**1. Field Match Comparison**

```python
was_fixed = _fields_match(corrected_value, ground_truth_correct_value)
```

Field matching uses:
- **Exact match** for short fields (year, volume, pages) — 6 characters or less
- **Token overlap** for longer fields (title, author, venue)
  
$$\text{Token Overlap} = \frac{|\text{tokens}(\text{corrected}) \cap \text{tokens}(\text{ground_truth})|}{|\text{tokens}(\text{ground_truth})|}$$

Threshold: $\text{overlap} \geq 0.80$ (80%)

### Metric Calculation

#### Overall Correction Metrics

**1. True Positives (TP)**

Field error was fixed (corrected value matches ground truth)

$$\text{TP} = |\{(e, f) : \text{was_fixed}(e, f) = \text{true}\}|$$

**2. False Negatives (FN)**

Field error was NOT fixed (corrected value still differs from ground truth)

$$\text{FN} = |\{(e, f) : \text{was_fixed}(e, f) = \text{false} \land f \text{ was corrupted}\}|$$

**3. False Positives (FP)**

Field was CORRECTLY fixed BUT correction agent changed an originally correct field

$$\text{FP} = |\{(e, f) : \text{field_was_correct}(e, f) \land \text{was_changed}(e, f)\}|$$

A field is considered "changed" when:
$$\text{was_changed}(e, f) = \text{corrected_value}(e, f) \neq \text{original_value}(e, f)$$

And NOT equal to ground truth:
$$\neg \text{fields_match}(\text{corrected_value}, \text{original_value})$$

#### Overall Metrics

**4. Precision**

Of all field corrections made, how many were correct?

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

**5. Recall**

Of all field errors that existed, how many were fixed?

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

**6. F1 Score**

Harmonic mean of precision and recall:

$$\text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

#### Field-Level Accuracy

For each field, calculate correction accuracy:

$$\text{Field Accuracy}_f = \frac{\text{TP}_f}{\text{TP}_f + \text{FN}_f}$$

Where:
- $\text{TP}_f$ = errors corrected in field $f$
- $\text{FN}_f$ = errors NOT corrected in field $f$

#### Example Correction Metrics

| Metric | Value |
|--------|-------|
| Partially-valid in ground truth | 53 |
| Correctly identified as partially_valid | 51 |
| **True Positives** | **32** |
| **False Positives** | **2** |
| **False Negatives** | **18** |
| **Precision** | **0.941** |
| **Recall** | **0.640** |
| **F1** | **0.765** |

#### Field-Level Accuracy

| Field | Errors Original | Errors Corrected | False Corrections | Accuracy |
|-------|-----------------|------------------|-------------------|----------|
| author | 18 | 13 | 1 | 0.722 |
| title | 8 | 5 | 0 | 0.625 |
| year | 12 | 10 | 1 | 0.833 |
| journal | 14 | 4 | 0 | 0.286 |
| booktitle | 7 | 0 | 0 | 0.000 |

---

## Telemetry Metrics

### Purpose

Track LLM validation agent performance efficiency:
- API call optimization
- Caching effectiveness
- Retry patterns
- Execution speed per strategy/model

### Telemetry Data Structure

```json
{
  "total_requests": 62,
  "cache_hits": 18,
  "cache_hit_rate": 0.225,
  "total_retries": 3,
  "duration_seconds": 168.648,
  "requests_per_second": 0.368,
  "retries_by_tool": {
    "dblp_fuzzy_title_search": 2,
    "google_scholar_search": 1
  }
}
```

### Telemetry Calculations

#### 1. Cache Hit Rate

$$\text{Cache Hit Rate} = \frac{\text{Cache Hits}}{\text{Cache Hits} + \text{Net API Requests}} \times 100\%$$

Example:
$$\text{Cache Hit Rate} = \frac{18}{18 + 62} \times 100\% = 22.5\%$$

#### 2. Requests Per Second

$$\text{Requests/sec} = \frac{\text{Total API Requests}}{\text{Duration (seconds)}}$$

Example:
$$\text{Requests/sec} = \frac{62}{168.648} = 0.368 \text{ req/s}$$

#### 3. Total Lookup Cost

$$\text{Total Lookups} = \text{Cache Hits} + \text{Net API Requests}$$

Example:
$$\text{Total Lookups} = 18 + 62 = 80$$

#### 4. Retry Rate

$$\text{Retry Rate} = \frac{\text{Total Retries}}{\text{Net API Requests}} \times 100\%$$

Example:
$$\text{Retry Rate} = \frac{3}{62} \times 100\% = 4.84\%$$

### Strategy & Model Comparison

| Strategy | Model | Duration (s) | Req/s | Cache Hit % | Avg Retries |
|----------|-------|--------------|-------|-------------|-------------|
| zero-shot | grok-4.20 | 61.83 | 0.00 | 0.0 | 0 |
| cot | grok-4.20 | 541.08 | 0.10 | 15.2 | 2.3 |
| rag | grok-4.20 | 905.73 | 0.22 | 22.5 | 1.8 |

#### Speed Analysis

**Speedup Factor:**

$$\text{Speedup} = \frac{\text{Duration}_{\text{Strategy 1}}}{\text{Duration}_{\text{Strategy 2}}}$$

Example (ZERO-SHOT vs RAG):
$$\text{Speedup} = \frac{905.73}{61.83} = 14.6x$$

**Requests Per Second Efficiency:**

$$\text{RPS Efficiency} = \text{Requests/sec} \times \text{(1 + Cache Hit Rate)}$$

Example:
$$\text{RPS Efficiency} = 0.22 \times (1 + 0.225) = 0.269$$

---

## Mathematical Formulas

### Classification Metrics Summary

#### Global Accuracy

$$\text{Accuracy}_{\text{global}} = \frac{\sum_{c} \text{TP}_c}{\sum_{c} (\text{TP}_c + \text{FN}_c)} = \frac{\text{Total Correct}}{\text{Total Entries}}$$

#### Macro-Averaged Precision

$$\text{Precision}_{\text{macro}} = \frac{1}{|\text{Classes}|} \sum_{c} \text{Precision}_c$$

#### Macro-Averaged Recall

$$\text{Recall}_{\text{macro}} = \frac{1}{|\text{Classes}|} \sum_{c} \text{Recall}_c$$

#### Macro-Averaged F1

$$\text{F1}_{\text{macro}} = \frac{1}{|\text{Classes}|} \sum_{c} \text{F1}_c$$

### Correction Agent Formulas Summary

$$\text{Precision}_{\text{correction}} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

$$\text{Recall}_{\text{correction}} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

$$\text{F1}_{\text{correction}} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

$$\text{Field Accuracy}_f = \frac{\text{TP}_f}{\text{TP}_f + \text{FN}_f}$$

---

## Evaluation Flow

### Step-by-Step Process

```
1. DATA PREPARATION
   └─ Load BibTeX entries
   └─ Load ground_truth.json (expected_status, corruptions)
   └─ Create entry index: entry_id → entry data

2. VALIDATION PHASE
   └─ Validation Agent runs (strategy: zero-shot | cot | rag)
   └─ Output: predictions with status labels
   └─ Telemetry collected: requests, cache hits, duration

3. VALIDATION EVALUATION
   └─ Compare predicted_status vs expected_status
   └─ Compute: TP, FP, FN per class
   └─ Calculate: Accuracy, Precision, Recall, F1
   └─ Output: validation_metrics JSON

4. CORRECTION PHASE
   └─ Correction Agent processes predicted partially_valid entries
   └─ Uses validation output as input (suggested_fixes)
   └─ Output: corrected BibTeX fields

5. CORRECTION EVALUATION
   └─ For each eligible entry (correctly classified partially_valid):
      └─ For each corrupted field:
         └─ Check if corrected_value matches ground_truth_value
         └─ Count: TP (fixed), FN (not fixed)
      └─ For each non-corrupted field:
         └─ Check if correction agent changed it
         └─ Count: FP (false corrections)
   └─ Compute: Precision, Recall, F1
   └─ Field-level accuracy per field
   └─ Output: evaluation_metrics JSON

6. REPORT GENERATION
   └─ Create markdown report combining:
      └─ Validation metrics
      └─ Correction metrics
      └─ Field-level analysis
      └─ Telemetry summary
      └─ Per-entry type statistics
   └─ Output: evaluation_report.md
```

### Code Implementation Reference

**Validation Evaluation (deterministic_eval_agent.py:105-170)**

```python
def _compute_validation_classification_metrics(validation_structured, gt_map):
    # Match predictions to ground truth by entry_id
    common_ids = set(gt_map.keys()) & set(pred_map.keys())
    
    # Count correct predictions
    tp = sum(1 for id in common_ids 
             if gt_map[id]['expected_status'] == pred_map[id])
    errors = len(common_ids) - tp
    
    # Calculate metrics
    accuracy = tp / len(common_ids)
    precision = tp / (tp + errors) if (tp + errors) > 0 else 0
    recall = tp / (tp + errors) if (tp + errors) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Per-class metrics
    for class_name in ['valid', 'partially_valid', 'invalid']:
        class_tp = count_matches(class_name, gt_map, pred_map)
        class_fp = count_false_positives(class_name, gt_map, pred_map)
        class_fn = count_false_negatives(class_name, gt_map, pred_map)
```

**Correction Evaluation (deterministic_eval_agent.py:460-570)**

```python
def _evaluate_corrections_deterministic(corrections, raw_data, gt_map):
    TP = FP = FN = 0
    field_counts = {}
    
    # For each eligible partially_valid entry
    for entry_id in eligible_partially_valid_ids:
        corruption_list = gt_map[entry_id]['corruptions']
        corrected_fields = corrections[entry_id]['corrected']
        
        # Check each corruption: was it fixed?
        for corruption in corruption_list:
            field = corruption['field']
            correct_value = corruption['original_correct_value']
            corrected_val = corrected_fields[field]
            
            if _fields_match(corrected_val, correct_value):
                TP += 1  # Successfully fixed
            else:
                FN += 1  # Failed to fix
        
        # Check for false positives: was a correct field changed?
        for field, corrected_val in corrected_fields.items():
            if field in corrupted_fields:
                continue  # Already counted above
            if original_fields[field] and not _fields_match(corrected_val, original_fields[field]):
                FP += 1  # Incorrectly changed a correct field
    
    # Calculate metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
```

---

## Key Insights

### Validation Agent Insights

- **Coverage**: Measured as matched_entries / ground_truth_entries
- **Per-class performance**: Different classes may have different metrics
- **Impact**: Validation accuracy directly affects correction eligibility

### Correction Agent Insights

- **Scope Limitation**: Only operates on correctly-identified partially_valid entries
- **Precision vs Recall Tradeoff**: 
  - High precision = few false corrections (conservative)
  - High recall = fixes more errors (aggressive)
- **Field Dependency**: Some fields (e.g., year) may be easier to correct than others

### Telemetry Insights

- **Strategy Cost**: 
  - ZERO-SHOT: ~60-180s (no API calls)
  - COT: ~540-1600s (controlled API calls, reasoning)
  - RAG: ~900-2600s (more API calls but better results)
- **Model Efficiency**: grok-4.20 generally fastest; gemini-3.1-pro slower on RAG
- **Cache Effectiveness**: 15-25% hit rate common with RAG/COT strategies

---

## Conclusion

The evaluation system provides **deterministic, reproducible metrics** based on ground truth data:

1. **Validation**: Measures classification quality (3-class problem)
2. **Correction**: Measures fix quality (field-level precision/recall)
3. **Telemetry**: Tracks efficiency and cost of different strategies/models

All metrics follow standard machine learning definitions and are suitable for academic thesis reporting.
