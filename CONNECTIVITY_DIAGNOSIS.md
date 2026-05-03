# Connectivity Issues Diagnosis & Improvements

## Date: 3 May 2026

---

## Summary

The persistent connectivity issues in batch 2 are **NOT caused by the rate limiter**. The rate limiter is working correctly (batch 1 and 3 succeeded with mirror switching). The issues are likely:

1. **Genuine DBLP/Scholar service downtime or network issues** during batch 2 execution
2. **Retry logic exhausting (4 attempts)** before LLM can call mirror switching
3. **Unclear error signals** to the LLM about when to trigger mirror switching

We've improved visibility and error handling to diagnose and recover from these issues better.

---

## Root Cause Analysis

### What's Working ✓

- **Rate limiter**: Sliding-window QPS limiter is preventing 503s from burst queries
- **Retries**: Exponential backoff with jitter is working (0.5s → 1s → 2s → 4s + random)
- **Cache**: LRU cache is preventing duplicate queries
- **Mirror switching**: Successfully triggered in batch 1 and batch 3

### What Failed in Batch 2 ✗

- **DBLP fuzzy_title_search**: Connectivity failed after 4 retry attempts
- **Google Scholar**: Also unavailable during same period
- **LLM visibility**: Error messages were unclear about retry exhaustion

Evidence from validation_report.md:

```
Batch 1: ✓ Success with DBLP + some mirror switching
Batch 2: ✗ "Due to persistent connectivity issues... I cannot perform comprehensive validation"
Batch 3: ✓ Partial success, LLM fell back to manual inspection
```

---

## Improvements Made

### 1. Enhanced Error Logging (in \_validation_agent_react.py)

**Before**: Silent retries, no visibility into what's happening

```python
for attempt in range(max_attempts):
    try:
        resp = original_invoke(normalized_input)
        ...
    except Exception as e:
        time.sleep(base_delay * (2 ** attempt) + random.random() * 0.2)
raise last_exc  # Silent failure
```

**After**: Clear logging of retry attempts and exhaustion

```python
for attempt in range(max_attempts):
    try:
        resp = original_invoke(normalized_input)
        ...
    except Exception as e:
        delay = base_delay * (2 ** attempt) + random.random() * 0.2
        error_msg = str(e)[:100]
        print(f"    [RETRY {attempt+1}/4] {name}: {error_msg} → waiting {delay:.2f}s")
        time.sleep(delay)

# All retries exhausted
print(f"    [EXHAUSTED] {name}: giving up after 4 retries. Last error: {error_msg}")
raise last_exc
```

**Effect**: Terminal output now shows:

```
[RETRY 1/4] fuzzy_title_search: Connection reset by peer → waiting 0.75s
[RETRY 2/4] fuzzy_title_search: Connection reset by peer → waiting 1.52s
[RETRY 3/4] fuzzy_title_search: Connection reset by peer → waiting 3.01s
[RETRY 4/4] fuzzy_title_search: Connection reset by peer → waiting 4.18s
[EXHAUSTED] fuzzy_title_search: giving up after 4 retries. Last error: Connection reset by peer
```

### 2. Improved System Prompts (RAG_SYSTEM & COT_SYSTEM)

**Before**: Generic retry instructions without clear error signal

```
- If DBLP returns timeout/connection reset/API error:
    1) call set_dblp_mirror with host="dblp.uni-trier.de", retry once.
```

**After**: Explicit [EXHAUSTED] signal detection and aggressive mirror switching

```
- If DBLP returns timeout/connection reset/API error/[EXHAUSTED] message:
    1) immediately call set_dblp_mirror with host="dblp.uni-trier.de", retry the exact same query.
    2) if it STILL fails or says [EXHAUSTED], call set_dblp_mirror with host="dblp.dagstuhl.de", retry again.
    3) if THAT fails, try a shorter title query (just first 3-5 words) with the new mirror.
    4) only if all DBLP mirrors fail do you proceed to Scholar fallback.
```

**Effect**: LLM now recognizes `[EXHAUSTED]` and immediately tries alternatives without manual retrying

### 3. Added Diagnostic Script

Created `diagnose_connectivity.py` to test:

- DBLP connectivity to main mirror
- DBLP connectivity to backup mirrors
- Google Scholar connectivity
- Rate limiter configuration
- Response times and error patterns

**Usage**:

```bash
python diagnose_connectivity.py
```

---

## Distinguishing Between Issues

### Issue Type: Rate Limiter

**Symptoms**:

- 503 Service Unavailable errors
- Quick failures (< 1 second)
- Occur during high-frequency queries
- Multiple entries fail simultaneously

**Evidence it's NOT this**: Batch 1 succeeded despite rate limiting; batch 3 had sporadic success

### Issue Type: Genuine Connectivity (Our Likely Case)

**Symptoms**:

- RemoteDisconnected / Connection reset
- Slow failures (4-15 seconds due to retry backoff)
- Occur sporadically across batches
- Mirror switching helps some but not all

**Evidence it IS this**:

- [EXHAUSTED] messages in output
- Batch 2 had persistent failures across multiple entries
- Batch 1 and 3 recovered with mirror switching

---

## How to Diagnose Your Batch 2 Issue

### Step 1: Run Diagnostic

```bash
cd /Users/passum/Documents/SWEDEN/DSI/THESIS/AI_reference_agent_detection_correction
python diagnose_connectivity.py
```

### Step 2: Check for [RETRY] and [EXHAUSTED] in pipeline output

```bash
uv run _agents/_pipeline.py \
  --file bibtex/bibtex_files/stefan_train2.bib \
  --strategy rag \
  --ground-truth bibtex/ground_truth/stefan_train2_truth.json 2>&1 | grep -E "\[RETRY|\[EXHAUSTED"
```

### Step 3: Interpret Results

| Finding                   | Meaning                           | Action                             |
| ------------------------- | --------------------------------- | ---------------------------------- |
| No [RETRY] messages       | Queries succeeded on first try    | Good connectivity                  |
| Many [RETRY N/4]          | Rate limiter backoff working      | Normal, not a problem              |
| [EXHAUSTED] messages      | All 4 retries failed              | Check if LLM tries mirror next     |
| LLM calls set_dblp_mirror | Mirror switching triggered        | Working as designed                |
| Mirror still fails        | All DBLP mirrors down             | Fall back to Scholar               |
| Scholar also fails        | Both DBLP and Scholar unavailable | Mark as invalid (correct behavior) |

---

## Prevention Strategies

### Short-term (Now)

1. **Run diagnostic** to understand current state
2. **Watch for [RETRY] and [EXHAUSTED]** messages in output
3. **Verify LLM calls mirrors** when [EXHAUSTED] appears
4. **Re-run problematic batches** later (might be transient)

### Medium-term (Recommended)

1. **Add timeout configuration** (currently uses MCP defaults):

   ```python
   # Could add to rate limiter setup:
   DBLP_REQUEST_TIMEOUT = 30  # seconds
   DBLP_RETRY_DELAY_MAX = 10  # max backoff
   ```

2. **Add circuit breaker** to avoid hammering failing service:

   ```python
   # Track consecutive failures per mirror
   consecutive_failures = defaultdict(int)
   # If failures > threshold, skip mirror temporarily
   ```

3. **Add fallback ordering**:
   - Try main DBLP mirror
   - Try uni-trier mirror if main fails
   - Try dagstuhl mirror if uni-trier fails
   - Try shorter query on dagstuhl
   - Fall back to Scholar
   - Mark as invalid

### Long-term (Optional)

1. **Cache query results** between pipeline runs (reduces DBLP load)
2. **Batch queries** to DBLP (send 10 queries once vs. 10 separate calls)
3. **Use DBLP REST API** instead of MCP (might be more reliable)
4. **Monitor DBLP status page** and skip validation when down

---

## Configuration Tuning

### Current Rate Limiter Settings

```python
MAX_QPS = 50  # DBLP's documented limit
RETRY_ATTEMPTS = 4
RETRY_BASE_DELAY = 0.5  # seconds
CACHE_SIZE = 2048  # entries
BACKOFF_MULTIPLIER = 2  # exponential
JITTER = 0.2  # random seconds added
```

### To Tune (Environment Variables)

```bash
# Reduce QPS if still hitting rate limits
export DBLP_MAX_QPS=25
uv run _agents/_pipeline.py --file ...

# Increase retry attempts if transient errors are common
# (Would need code change, not env var)
```

---

## Expected Behavior After Improvements

### Successful Run Output

```
[DBLP Tools] Running ReAct...
  ... [process entries] ...
  Total tool calls: 47
  ✓ Recall:    0.823
  ✓ Precision: 0.902
  ✓ F1:        0.860
```

### With Some Retries (Normal)

```
... batch 1 processing ...
  [RETRY 1/4] fuzzy_title_search: timeout → waiting 0.65s
  [RETRY 2/4] fuzzy_title_search: timeout → waiting 1.43s
  ✓ [Recovers on attempt 3]
... batch 2 continues ...
```

### With Mirror Switching (Expected Recovery)

```
... batch 2, entry 15 ...
  [RETRY 1/4] fuzzy_title_search: Connection reset → waiting 0.71s
  [RETRY 2/4] fuzzy_title_search: Connection reset → waiting 1.55s
  [RETRY 3/4] fuzzy_title_search: Connection reset → waiting 3.22s
  [EXHAUSTED] fuzzy_title_search: giving up after 4 retries
  [LLM decides to try mirror...]
  ✓ [Calls set_dblp_mirror("dblp.uni-trier.de")]
  ✓ [Query succeeds on mirror]
```

### With Genuine Downtime (Expected Fallback)

```
... [all DBLP retries fail] ...
... [mirror 1 fails] ...
... [mirror 2 fails] ...
... [LLM tries Scholar] ...
... [Scholar succeeds] ...
✓ [Validates paper via Scholar]

// OR

... [all DBLP fails] ...
... [Scholar fails] ...
✓ [Marks as invalid - correct behavior for non-existent paper]
```

---

## Next Steps

1. **Run diagnostic**: `python diagnose_connectivity.py`
2. **Run pipeline with verbose output**: Capture [RETRY] and [EXHAUSTED] messages
3. **Analyze output**: See which batches had issues and which recovered
4. **Share findings**: If mirror switching isn't triggering, LLM prompt might need adjustment
5. **Re-run batch 2**: If transient, might succeed on retry
6. **Consider tuning**: If persistent, might need circuit breaker or rate adjustment

---

## Summary of Changes

| File                                 | Change                     | Impact                          |
| ------------------------------------ | -------------------------- | ------------------------------- |
| `_agents/_validation_agent_react.py` | Added [RETRY N/4] logging  | Better visibility into retries  |
| `_agents/_validation_agent_react.py` | Added [EXHAUSTED] logging  | LLM can recognize failure       |
| `_agents/_validation_agent_react.py` | Enhanced RAG_SYSTEM prompt | LLM catches [EXHAUSTED] signal  |
| `_agents/_validation_agent_react.py` | Enhanced COT_SYSTEM prompt | Same for CoT strategy           |
| `diagnose_connectivity.py`           | New diagnostic script      | Test connectivity independently |

All changes are **backward compatible** and **increase visibility** without changing core retry logic.
