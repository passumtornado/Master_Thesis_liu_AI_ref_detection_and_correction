# Summary of Evaluation and Validation Agent Updates

## Date: 3 May 2026

### Overview

Updated the evaluation and validation agents to:

1. **Fix validation classification logic** — distinguish between field errors and non-existent papers
2. **Add per-class metrics** — evaluate correction quality separately for valid, partially_valid, and invalid entries
3. **Add field-level accuracy by class** — understand which fields corrected best in each validation category

---

## 1. Validation Agent Changes

### File: `_agents/_validation_agent_react.py`

#### Updated System Prompts (ZERO_SHOT_SYSTEM, RAG_SYSTEM, COT_SYSTEM)

**Problem Fixed:**

- Previously, entries were marked as "invalid" when they had field errors (typos, misspellings, wrong year, etc.)
- Correct behavior: mark as "partially_valid" if the paper EXISTS but has field errors, only "invalid" if the paper doesn't exist

**Changes:**

1. **ZERO_SHOT_SYSTEM (lines ~77-109)**
   - Added explicit CRITICAL DISTINCTION section
   - Clarified: Field errors → PARTIALLY_VALID if paper exists
   - Clarified: Paper non-existent → INVALID
   - Added note: "Do NOT mark as invalid just because a field is wrong"

2. **RAG_SYSTEM (lines ~111-163)**
   - Added handling rule: "If DBLP/Scholar confirms paper EXISTS (similarity >= 0.70) but has field errors → PARTIALLY_VALID"
   - Added only mark INVALID if:
     - No database match found at all
     - Paper provably non-existent (author died before pub year)
     - Completely fabricated (impossible author combinations, fictional venues)
   - Clarified efficiency rules for mirror switching vs. retry logic

3. **COT_SYSTEM (lines ~165-226)**
   - Added same critical rules as RAG
   - Emphasized: "If you confirmed the paper EXISTS (similarity >= 0.70) but found field errors → PARTIALLY_VALID"
   - Added: "Do NOT confuse 'field errors' with 'paper doesn't exist'"

---

## 2. Evaluation Agent Changes

### Files Updated:

- `agents/evaluation_agent.py` (for external use)
- `_agents/_evaluation_agent.py` (used by pipeline)

### Changes Made:

#### A. System Prompt Enhancement (SYSTEM_PROMPT)

**Extended JSON schema to include per-class metrics:**

```json
{
  "overall_metrics": { ... },
  "per_class_metrics": {
    "valid": {
      "count": <int>,
      "true_positives": <int>,
      "false_positives": <int>,
      "false_negatives": <int>,
      "recall": <float or null>,
      "precision": <float or null>,
      "f1": <float or null>,
      "field_accuracy": <float>
    },
    "partially_valid": { ... },
    "invalid": { ... }
  },
  "field_accuracy": {
    "<field_name>": {
      "overall_accuracy": <float>,
      "errors_in_original": <int>,
      "errors_corrected": <int>,
      "false_corrections": <int>,
      "per_class": {
        "valid": <float>,
        "partially_valid": <float>,
        "invalid": <float>
      }
    }
  },
  "markdown_report": "..."
}
```

#### B. Method Signature Updates

**evaluate() method now accepts validation_structured:**

```python
async def evaluate(
    self,
    raw_data: list[dict],
    corrections: list[dict],
    validation_structured: list[dict] | None = None,
) -> dict:
```

#### C. \_build_payload() Enhancement

**Now includes validation_status from validation agent:**

```python
def _build_payload(
    self, raw_data: list[dict], corrections: list[dict], validation_structured: list[dict]
) -> list[dict]:
    # ... builds payload with validation_status field
    payload.append({
        "entry_id": entry_id,
        "original": raw_item["entry"],
        "corrected": corrected_entry,
        "ground_truth": ground_truth,
        "changes": changes,
        "validation_status": validation_status,  # NEW
    })
```

#### D. Output Saving (\_save_outputs())

**Now normalizes and saves per_class_metrics:**

```python
metrics_path.write_text(
    json.dumps({
        "overall_metrics": overall_metrics,
        "per_class_metrics": per_class_metrics,  # NEW
        "field_accuracy": field_accuracy
    }, ...),
    encoding="utf-8",
)
```

#### E. Terminal Output Enhancement

**evaluate() method now prints per-class summary:**

```
  Per-Class Metrics:
    valid               : 20 entries | R:85.0% P:90.0% F1:0.875 | Acc:92.3%
    partially_valid     :  8 entries | R:62.5% P:75.0% F1:0.682 | Acc:78.0%
    invalid             :  3 entries | (N/A - no ground truth)
```

#### F. Return Value Enhancement

**evaluate() now returns per_class_metrics:**

```python
return {
    "strategy": self.strategy.value,
    "overall_metrics": llm_result.get("overall_metrics", {}),
    "per_class_metrics": llm_result.get("per_class_metrics", {}),  # NEW
    "field_accuracy": llm_result.get("field_accuracy", {}),
    "markdown_report": llm_result.get("markdown_report", ""),
    "saved_files": saved_files,
}
```

---

## 3. Output Artifacts

### New/Updated Files Generated:

**evaluation/evaluation_metrics.json** — Now includes:

```json
{
  "overall_metrics": { ... },
  "per_class_metrics": {
    "valid": { count, TP, FP, FN, recall, precision, f1, field_accuracy },
    "partially_valid": { ... },
    "invalid": { ... }
  },
  "field_accuracy": { ... with per_class breakdown }
}
```

**evaluation/evaluation_report.md** — Updated markdown includes:

- Overall metrics table
- Per-class metrics table (valid/partially_valid/invalid)
- Field-level accuracy table (overall + per-class)
- Insights section comparing performance across classes

---

## 4. Interpretation Guide

### What the Metrics Tell You:

#### Overall Metrics

- **Recall**: % of errors in original entries that were successfully corrected
- **Precision**: % of corrections made that were actually correct
- **F1**: Harmonic mean balancing recall and precision

#### Per-Class Metrics

- **valid entries**: Did corrections help entries that were already mostly correct?
  - Low F1 → corrections may be introducing errors in already-valid entries
  - High F1 → improvements to correct minor errors in valid papers

- **partially_valid entries**: Did corrections help entries with known field errors?
  - High recall → agent found and fixed most field errors
  - High precision → corrections didn't introduce new errors

- **invalid entries**: Did corrections "fix" non-existent papers?
  - Should have low or null metrics (no ground truth for fabricated papers)
  - If metrics exist, suggests agent attempted corrections on fabricated data

#### Field-Level Accuracy by Class

- **title accuracy in valid vs. partially_valid**: Compare correction impact
- **author accuracy**: Often hardest field; check if accuracy differs by class
- **year accuracy**: Should be highest (deterministic matching)
- **venue accuracy**: Often confused (journal vs. conference vs. arXiv abbreviations)

---

## 5. Pipeline Integration

### Where Changes Take Effect:

**\_agents/\_pipeline.py** — `evaluate_node()` function:

```python
result = await agent.evaluate(
    raw_data=raw_data,
    corrections=corrections,
    validation_structured=state.get("validation_structured", []),  # NEW parameter
)

return {
    "validation_metrics": result.get("validation_metrics", ...),
    "evaluation_metrics": result.get("overall_metrics", {}),
    "evaluation_per_class": result.get("per_class_metrics", {}),  # NEW
    "evaluation_field_accuracy": result.get("field_accuracy", {}),
    "evaluation_error": "",
}
```

### Performance Summary Output:

**evaluation/performance_summary.json** now includes:

```json
{
  "validation": { "metrics": {...} },
  "correction": {
    "overall_metrics": {...},
    "per_class_metrics": {...},  // NEW
    "field_accuracy": {...}
  }
}
```

---

## 6. Running the Updated Pipeline

### Command to execute with new updates:

```bash
cd /Users/passum/Documents/SWEDEN/DSI/THESIS/AI_reference_agent_detection_correction

# Run with specific shard and ground truth
uv run _agents/_pipeline.py \
  --file bibtex/bibtex_files/stefan_train2.bib \
  --strategy rag \
  --ground-truth bibtex/ground_truth/stefan_train2_truth.json
```

### Expected Output Changes:

1. Validation agent will produce more "partially_valid" entries (fewer false "invalid" due to field errors)
2. Evaluation metrics will show per-class breakdowns
3. Field-level accuracy will show performance differences across validation classes
4. evaluation_metrics.json will be larger due to additional per_class_metrics section

---

## 7. Files Modified

| File                                 | Lines Modified                    | Change Type             |
| ------------------------------------ | --------------------------------- | ----------------------- |
| `_agents/_validation_agent_react.py` | 77-226                            | System prompts enhanced |
| `_agents/_evaluation_agent.py`       | 68-149, 249-315, 379-460          | Per-class metrics added |
| `agents/evaluation_agent.py`         | 35-106, 130-180, 153-220, 249-315 | Per-class metrics added |

---

## 8. Testing Recommendations

### Verify the changes work correctly:

1. **Check validation status clarity:**
   - Papers with typos should be "partially_valid" not "invalid"
   - Only non-existent papers should be "invalid"

2. **Check per-class metrics:**
   - All three classes (valid, partially_valid, invalid) should have entries
   - Recall/precision should make sense per class
   - Field accuracy should vary by class

3. **Check markdown report:**
   - Report should include per-class table
   - Insights should mention class-specific patterns

### Example diagnostic command:

```bash
python -c "
import json
from pathlib import Path

metrics = json.loads(Path('evaluation/evaluation_metrics.json').read_text())
for class_name, metrics_dict in metrics.get('per_class_metrics', {}).items():
    print(f'{class_name}: {metrics_dict.get(\"count\")} entries, F1={metrics_dict.get(\"f1\"):.3f}')
"
```

---

## 9. Key Improvements Summary

✅ **Validation Logic Fix**

- Field errors no longer cause false "invalid" classification
- Only truly non-existent papers marked as invalid

✅ **Per-Class Metrics**

- Understand correction quality varies by paper validity status
- Identify which classes benefit most from corrections

✅ **Field-Level Accuracy by Class**

- See if certain fields corrected better in specific categories
- Tailor correction strategy per class

✅ **Comprehensive Reporting**

- Markdown reports now include all three metrics dimensions
- JSON output suitable for downstream analysis

---

## Next Steps (Optional Enhancements)

1. **Add telemetry counters** to validation/correction agents:
   - Count tool calls, cache hits, mirror switches
   - Surface in evaluation report

2. **Add input validation** for google_scholar_search:
   - Require title parameter before tool invocation
   - Return structured error if missing

3. **Run full shard suite** with updated agents:
   - Execute all 4 shards (stefan_train1-4)
   - Compare per-class metrics across shards
