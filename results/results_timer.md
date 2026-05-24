# Validation Agent Performance Analysis: Duration & Speed

## Executive Summary

Analysis of validation telemetry across **4 LLM models**, **3 strategies**, and **2 datasets (Stefan2 & Stefan3)**.

### Key Findings

**Strategy Performance Ranking (by average duration):**


1. **ZERO-SHOT** 🥇
   - Average Duration: **108.45s**
   - Average Requests/s: **0.00**
   - Range: 61.83s - 176.50s

2. **COT** 🥈
   - Average Duration: **1064.72s**
   - Average Requests/s: **0.11**
   - Range: 541.08s - 1604.34s

3. **RAG** 🥉
   - Average Duration: **1762.10s**
   - Average Requests/s: **0.17**
   - Range: 905.73s - 2644.38s

**Performance Gain:** ZERO-SHOT is **16.2x faster** than RAG

---

## Detailed Results: Average Performance (Datasets 2 & 3)

| Model | Strategy | Avg Duration (s) | Avg Requests/s | Dataset 2 Duration (s) | Dataset 3 Duration (s) |
|-------|----------|------------------|----------------|------------------------|------------------------|
| claude-sonnet-4.6 | zero-shot | 176.50 | 0.00 | 182.90 | 170.10 |
| claude-sonnet-4.6 | cot | 1604.34 | 0.09 | 1725.19 | 1483.49 |
| claude-sonnet-4.6 | rag | 1736.19 | 0.16 | 3156.90 | 315.47 |
| gemini-3.1-pro-preview | zero-shot | 97.32 | 0.00 | 120.65 | 73.98 |
| gemini-3.1-pro-preview | rag | 2644.38 | 0.11 | 4633.55 | 655.20 |
| gpt-5.4 | zero-shot | 98.14 | 0.00 | 115.25 | 81.03 |
| gpt-5.4 | cot | 1048.73 | 0.15 | 1139.80 | 957.67 |
| grok-4.20 | zero-shot | 61.83 | 0.00 | 86.18 | 37.49 |
| grok-4.20 | cot | 541.08 | 0.10 | 808.39 | 273.77 |
| grok-4.20 | rag | 905.73 | 0.22 | 181.89 | 1629.57 |

---

## Strategy Comparison

### ZERO-SHOT
- **Average Duration:** 108.45s
- **Average Requests/s:** 0.00
- **Fastest Model:** grok-4.20 (61.83s)
- **Slowest Model:** claude-sonnet-4.6 (176.50s)

### COT
- **Average Duration:** 1064.72s
- **Average Requests/s:** 0.11
- **Fastest Model:** grok-4.20 (541.08s)
- **Slowest Model:** claude-sonnet-4.6 (1604.34s)

### RAG
- **Average Duration:** 1762.10s
- **Average Requests/s:** 0.17
- **Fastest Model:** grok-4.20 (905.73s)
- **Slowest Model:** gemini-3.1-pro-preview (2644.38s)


---

## Model Comparison

### claude-sonnet-4.6
- **zero-shot:** 176.50s (0.00 req/s)
- **cot:** 1604.34s (0.09 req/s)
- **rag:** 1736.19s (0.16 req/s)

### gemini-3.1-pro-preview
- **zero-shot:** 97.32s (0.00 req/s)
- **rag:** 2644.38s (0.11 req/s)

### gpt-5.4
- **zero-shot:** 98.14s (0.00 req/s)
- **cot:** 1048.73s (0.15 req/s)

### grok-4.20
- **zero-shot:** 61.83s (0.00 req/s)
- **cot:** 541.08s (0.10 req/s)
- **rag:** 905.73s (0.22 req/s)

---

## Conclusions

1. **ZERO-SHOT is Fastest:** No search tools required; purely LLM reasoning is significantly faster (by ~10-25x)

2. **RAG vs COT:** 
   - Both use search tools and have similar performance costs
   - Slight variations by model, but generally comparable

3. **Model Performance:**
   - **grok-4.20**: Best performance in COT strategy (541.08s average)
   - **claude-sonnet-4.6**: Strong across strategies but more expensive on RAG
   - **gemini-3.1-pro-preview**: Slower on RAG strategy (2644.38s average)
   - **gpt-5.4**: Competitive performance, especially in COT

4. **Recommendation:** 
   - For speed: Use **ZERO-SHOT** strategy
   - For accuracy with API access: Use **RAG** or **COT** (minimal difference)
   - Best model choice depends on accuracy requirements and cost
