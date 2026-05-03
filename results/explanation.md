# Validation Agent Update: Robust Rate-Limit Handling

## Scope of the update

This document explains the recent update in `_agents/_validation_agent_react.py` focused on making tool calls to DBLP (and scholar fallback tools) robust under request pressure.

The key objective was to prevent transient API failures (especially `503 Service Unavailable`, connection resets, and timeouts) from breaking validation batches.

## Why this was needed

The validation agent executes ReAct tool calls per entry and can trigger many DBLP requests in a short period.

Observed failure modes included:

- `503 Service Unavailable`
- `Connection reset by peer`
- batch retries exhausting after repeated transient transport errors
- wrapper compatibility errors while integrating custom throttling logic

## Final implemented solution

The final design keeps full compatibility with LangChain/LangGraph tool contracts while adding three protections:

1. Rate limiting
2. Retry with exponential backoff + jitter
3. In-memory LRU caching

These protections are applied by wrapping MCP tools into new `StructuredTool` instances via `StructuredTool.from_function(...)`.

---

## 1) Rate limiting design

### Configuration

- Environment variable: `DBLP_MAX_QPS`
- Default value: `50`

This matches the DBLP query budget target and prevents bursts above allowed request rate.

### Implementation details

A sliding-window limiter is used:

- `_recent_requests: deque[float]` stores timestamps
- `_acquire_slot()` for async path
- `_acquire_slot_sync()` for sync path

Flow:

1. Remove timestamps older than 1 second.
2. If request count in current 1-second window is below `MAX_QPS`, allow request.
3. Otherwise sleep briefly (`0.01s`) and retry.

Why this approach:

- minimal dependencies
- predictable limit behavior
- simple to reason about and tune

---

## 2) Retry/backoff strategy

Transient failures are detected by `_looks_transient_error(...)` using:

- exception text match: `503`, `service unavailable`, `connection reset`, `timeout`, `temporarily unavailable`
- response text match: same patterns and `error:` prefixes

Retry policy:

- `max_attempts = 4`
- `base_delay = 0.5`
- delay formula: `base_delay * (2 ** attempt) + random_jitter`
- jitter range: `0.0 .. 0.2s`

Behavior:

- transient errors are retried
- non-transient errors are re-raised immediately

This avoids unnecessary retries for validation/model errors while stabilizing transport-level failures.

---

## 3) LRU cache

A lightweight `SimpleLRU` cache is used with:

- max size: `2048`
- key from tool name + normalized input payload

Cache function:

- repeated identical tool calls return cached results immediately
- reduces DBLP pressure and duplicate network traffic

Impact:

- lower effective QPS
- fewer repeated queries from similar tool calls
- better stability during long ReAct loops

---

## Tool wrapping architecture (critical compatibility fix)

Earlier attempts failed because direct wrapper objects and monkey-patching `run/arun` on existing tool objects were incompatible with `ToolNode`/Pydantic constraints.

Final robust approach:

- take each MCP tool
- keep original metadata (`name`, `description`, `args_schema`)
- create a new `StructuredTool` using `StructuredTool.from_function(...)`
- delegate calls to original `invoke/ainvoke`
- apply limiter + cache + retry in the delegated wrapper functions

This preserves:

- expected tool schema and signature
- proper `tool_input` handling
- compatibility with `self.llm.bind_tools(...)` and `ToolNode(...)`

---

## Input normalization

`_normalize_tool_input(tool_input=None, **kwargs)` handles two invocation styles:

- single payload passed as `tool_input`
- parsed keyword arguments passed by structured tools

Normalization ensures consistent cache keys and invocation payloads.

---

## How this interacts with ReAct validation

No change to high-level ReAct flow:

- tools still selected by LLM
- same strategy modes (`zero_shot`, `rag`, `cot`)
- same batching behavior in pipeline

Only tool execution layer changed to be resilient and rate-aware.

---

## Reliability improvements achieved

With the current update:

- previous interface errors (`tool_input`, callable/docstring, pydantic field assignment) are removed
- validation proceeds with real DBLP calls instead of failing at wrapper integration
- query bursts are controlled against configured QPS
- transient transport failures are retried safely

---

## Remaining enhancement (not yet added)

The remaining planned improvement is explicit telemetry/summary counters, such as:

- total tool calls
- cache hits/misses
- retry count
- transient error count
- final QPS estimate

This is useful for benchmarking and proving the effectiveness of throttling in result reports.

---

## Operational tuning guidance

Use these knobs based on API behavior:

- If 503 still appears:
  - reduce `DBLP_MAX_QPS` to `30-40`
- If throughput is too slow but stable:
  - increase gradually toward `50`
- If repeated network instability appears:
  - keep retries at 4 but increase base delay (e.g., `0.7`)

Suggested default for stable production-like runs:

- `DBLP_MAX_QPS=40`

---

## Summary

The validation agent now uses a robust, simplified, and compatible rate-limit control layer by wrapping MCP tools as `StructuredTool` delegates with:

- sliding-window QPS limiting
- transient-error retries with exponential backoff + jitter
- in-memory LRU response caching

This addresses DBLP pressure-related failures while preserving existing ReAct behavior and tool contracts.
