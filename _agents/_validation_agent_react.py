"""
LLM-Driven Validation Agent  —  ReAct edition
-----------------------------------------------
All three strategies (zero_shot, rag, cot) are implemented as ReAct agents.
The LLM autonomously decides which tools to call, when, and with what
arguments — Python never manually orchestrates tool calls.

To use this agent instead of the manual one, update pipeline.py:

    # Comment out the manual agent
    # from _validation_agent import LLMValidationAgent, PromptStrategy as VPromptStrategy

    # Uncomment the ReAct agent
    from _validation_agent_react import LLMValidationAgent, PromptStrategy as VPromptStrategy

Strategy behaviour:
  - zero_shot : ReAct agent gets NO tools bound — LLM reasons from pre-trained
                knowledge only, cannot call DBLP or Scholar
  - rag       : ReAct agent gets DBLP + Scholar tools — LLM calls them as
                needed, stops as soon as it has strong evidence (similarity >= 0.75)
  - cot       : Same tools as RAG but system prompt forces field-by-field
                reasoning (title → authors → year → venue) before each verdict

Speed advantage over manual pipeline:
  - Manual: Python calls DBLP for every entry, then Scholar for weak ones
  - ReAct:  LLM skips Scholar when DBLP already gives a strong match
  - Result: ~2-4x fewer total API calls on a typical .bib dataset

Dependencies:
  uv add langchain langgraph langchain-google-genai langchain-mcp-adapters
"""

import json
import os
import sys
import asyncio
import time
import random
from collections import deque, OrderedDict
from enum import Enum
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import _ainvoke_with_retry, _extract_text, parse_validation_results

load_dotenv('/Users/passum/Documents/SWEDEN/DSI/THESIS/AI_reference_agent_detection_correction/.env')

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_openrouter import ChatOpenRouter

MAX_REACT_ITERATIONS = 120
REACT_RECURSION_BUFFER = 20
DBLP_MIRRORS = ["dblp.org", "dblp.uni-trier.de", "dblp.dagstuhl.de"]


# ----------------------------------------------------------------------
# Telemetry
# ----------------------------------------------------------------------
class ValidationTelemetry:
    def __init__(self):
        self.total_requests = 0
        self.cache_hits = 0
        self.total_retries = 0
        self.retry_by_tool = {}
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.monotonic()

    def end(self):
        self.end_time = time.monotonic()

    def record_cache_hit(self):
        self.cache_hits += 1

    def record_request(self):
        self.total_requests += 1

    def record_retry(self, tool_name: str):
        self.total_retries += 1
        self.retry_by_tool[tool_name] = self.retry_by_tool.get(tool_name, 0) + 1

    def get_duration(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

    def get_requests_per_second(self) -> float:
        duration = self.get_duration()
        return self.total_requests / duration if duration > 0 else 0.0

    def print_summary(self):
        duration = self.get_duration()
        total_lookups = self.cache_hits + self.total_requests
        cache_hit_rate = (self.cache_hits / total_lookups * 100) if total_lookups > 0 else 0
        print(f"\n{'─'*60}")
        print("VALIDATION TELEMETRY")
        print(f"{'─'*60}")
        print(f"  Total API calls (net)    : {self.total_requests}")
        print(f"  Cache hits               : {self.cache_hits}")
        print(f"  Cache hit rate           : {cache_hit_rate:.1f}%")
        print(f"  Total retry attempts     : {self.total_retries}")
        print(f"  Total duration           : {duration:.2f}s")
        print(f"  Effective request rate   : {self.get_requests_per_second():.1f} req/s")
        if self.retry_by_tool:
            print("  Retries by tool:")
            for name, cnt in sorted(self.retry_by_tool.items()):
                print(f"    • {name}: {cnt}")
        print(f"{'─'*60}\n")

    def to_dict(self) -> dict:
        duration = self.get_duration()
        total_lookups = self.cache_hits + self.total_requests
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round((self.cache_hits / total_lookups * 100) if total_lookups > 0 else 0.0, 2),
            "total_retries": self.total_retries,
            "retries_by_tool": dict(self.retry_by_tool),
            "duration_seconds": round(duration, 3),
            "requests_per_second": round(self.get_requests_per_second(), 3),
        }


# ----------------------------------------------------------------------
# Prompt Strategy
# ----------------------------------------------------------------------
class PromptStrategy(Enum):
    ZERO_SHOT = "zero_shot"
    RAG = "rag"
    COT = "cot"


# ----------------------------------------------------------------------
# System Prompts (abridged – replace with your full prompts)
# ----------------------------------------------------------------------
ZERO_SHOT_SYSTEM = """You are an expert BibTeX validation assistant.

You have NO search tools available. You must rely entirely on your
pre-trained knowledge to validate each entry.

For EACH entry:
1. Assess whether the paper is likely to exist based on your knowledge.
2. Check if the title, authors, year, and venue look plausible and consistent.
3. Assign a verdict based on what you know.

Assign:
  - status          : valid | partially_valid | invalid
  - confidence      : float [0.0, 1.0]
  - issues          : list of field-level problems detected
  - suggested_fixes : dict of {field: corrected_value}

Rules:
  - valid          : paper exists AND all major fields appear correct
    - partially_valid: paper likely EXISTS but 1+ bibliographic fields are wrong (typo, misspelling, incomplete author list, wrong year, wrong venue, etc.)
    - invalid        : paper is retracted, fabricated, impossible, or cannot be found in any academic database
  
CRITICAL DISTINCTION:
  - Field errors (typos, misspellings, capitalization, incomplete data) → PARTIALLY_VALID if paper exists
    - Paper retracted / fraudulent / fabricated / impossible / anachronistic author combination / impossible date → INVALID
    - Paper non-existent (completely fabricated, impossible combination, cannot be verified) → INVALID
  - Do NOT mark as invalid just because a field is wrong; only if paper doesn't actually exist

CONFIDENCE GUIDANCE:
    - valid          : 0.90-0.99 when the match is exact and all major fields align
    - partially_valid: 0.70-0.90 when the paper exists and field errors are confirmed
    - invalid        : 0.90-0.99 when the paper is retracted, fabricated, or logically impossible
    - unverifiable   : 0.00-0.60 when API or search failures prevent verification

Produce a complete markdown report grouped by status with a summary
statistics table and per-entry details.

Markdown format requirements (MANDATORY):
1. Include these exact section headers in this exact order:
  - ## 🟢 Valid References
  - ## 🟡 Partially Valid References
  - ## 🔴 Invalid References
2. Every reference must appear in exactly one of those three sections.
3. Do not create any other status section names.
4. Keep entry IDs in each section so they are easy to trace.

Then append a JSON block in EXACTLY this format:
```json
{
  "results": [
    {
      "entry_id": "...",
      "status": "valid|partially_valid|invalid",
      "confidence": 0.0,
      "issues": ["field: description"],
      "suggested_fixes": {"field": "corrected value"}
    }
  ]
}
```
Output markdown first, then the JSON block. No other wrapper text.
"""
# RAG_SYSTEM = """You are an expert BibTeX validation assistant with access to
# DBLP and Google Scholar search tools.

# You will receive a list of BibTeX entries. For EACH entry:

# ════════════════════════════════════════
# STEP 1 — SEARCH
# ════════════════════════════════════════
# Call fuzzy_title_search with the entry's title and authors.

#   • similarity >= 0.75  → strong match found. Skip Scholar; proceed to STEP 2.
#   • similarity < 0.75   → call google_scholar_search as fallback.
#       – Scholar finds a match  → proceed to STEP 2 using Scholar's result.
#       – Scholar also fails     → mark UNVERIFIABLE (see STEP 3).
#   • Tool returns [EXHAUSTED] or any error marker → stop; mark UNVERIFIABLE.
#     Do NOT treat an API failure as evidence that the paper does not exist.

# ════════════════════════════════════════
# STEP 2 — COMPARE (only if a match was found)
# ════════════════════════════════════════
# Check every bibliographic field against the best match:
#   title, author(s), year, journal / booktitle / venue.

# Note every discrepancy, no matter how minor:
#   • typos or misspellings in any field
#   • author name variants or missing co-authors
#   • wrong or off-by-one year
#   • wrong, abbreviated, or incomplete venue name
#   • capitalisation errors

# ════════════════════════════════════════
# STEP 3 — CLASSIFY (follow this decision tree top-to-bottom)
# ════════════════════════════════════════

#   [A] Did ALL searches fail with API/network errors?
#       → UNVERIFIABLE. Stop. (Never penalise the entry for an infrastructure failure.)

#   [B] Did searches complete successfully but return NO match at all?
#       AND is the paper provably non-existent
#       (e.g., author died before the stated year, impossible author/venue
#        combination, title is clearly fabricated)?
#       → INVALID.

#   [C] Did searches complete successfully but return NO match at all,
#       and the paper cannot be confirmed non-existent?
#       → UNVERIFIABLE (benefit of the doubt; the database may lack coverage).

#   [D] Was a match found (similarity >= 0.75 OR Scholar confirmed)?
#       → Now look at the field comparison from STEP 2:

#         • Zero field discrepancies → VALID.

#         • One or more field discrepancies (ANY of: wrong year, misspelled
#           author, wrong venue, incomplete author list, capitalisation error,
#           title word swap, etc.) → PARTIALLY_VALID.
#           ↳ This is true even if the paper is well-known and obviously real.
#           ↳ The match proves the paper EXISTS; the discrepancy is a field error.

#         • Paper is retracted, fraudulent, or otherwise invalid as a citation
#           (regardless of whether a record exists) → INVALID.

# ⚠️  CRITICAL RULE — read before classifying every entry:
#     A confirmed match (step D) can NEVER produce an INVALID classification
#     solely because of field errors. Field errors on a confirmed match → PARTIALLY_VALID.
#     INVALID is reserved for (B) above or for retracted/fabricated papers.

# ════════════════════════════════════════
# EXAMPLE (partially_valid case)
# ════════════════════════════════════════
#   Entry:   author={John Doe and Jane Smith}, year={2021}, venue={ICML}
#   Match:   author={John Doe and Jane Smith and Bob Lee}, year={2022}, venue={ICML}
#   Result:  PARTIALLY_VALID
#   Issues:  ["author: missing co-author Bob Lee", "year: 2021 should be 2022"]

# ════════════════════════════════════════
# ⚠️ CRITICAL RULE #2 — access_error and suggested_fixes usage
# ════════════════════════════════════════
#   • IF you have ANY suggested_fixes AND access_error=false:
#     → MUST classify as PARTIALLY_VALID, NEVER UNVERIFIABLE.
#     Example: "venue abbreviated; actual IEEE TNN" + suggested_fixes={journal: "..."}
#     → This is PARTIALLY_VALID (paper found, field error detected).

#   • IF access_error=true (API error occurred):
#     → ONLY use UNVERIFIABLE. Do not use VALID or PARTIALLY_VALID.
#     Set suggested_fixes={} (empty dict).

#   • IF you have NO suggested_fixes AND access_error=false:
#     → Use UNVERIFIABLE ONLY if no match was found AND paper cannot be confirmed non-existent.
#     Otherwise use INVALID if provably non-existent.

# COMMON MISTAKE (DO NOT DO):
#   ❌ status=unverifiable, access_error=false, suggested_fixes={journal: "..."}
#   ✓ status=partially_valid, access_error=false, suggested_fixes={journal: "..."}

# ════════════════════════════════════════
# EFFICIENCY RULES
# ════════════════════════════════════════
#   • Do NOT call Scholar if DBLP similarity is already >= 0.75.
#   • One DBLP call per entry is usually sufficient.
#   • Only retry with a shorter query (dblp_search) if the full-title search
#     returns nothing — not if it returns a low-similarity match.
#   • If you see [EXHAUSTED] or an error marker, stop; do not retry further.

# ════════════════════════════════════════
# OUTPUT FORMAT
# ════════════════════════════════════════
# Produce a complete markdown report with a summary table and per-entry
# details showing which evidence source was used and what issues were found.

# Mandatory section headers (use exactly these, in this order):

#   ## 🟢 Valid References
#   ## 🟡 Partially Valid References
#   ## 🔴 Invalid References
#   ## 🟠 Unverifiable References

# Every reference must appear in exactly one section. Keep entry IDs visible.

# Before writing the JSON block, perform this self-check for every entry:
#   "Did I find a match for this entry? If yes, is my status VALID or PARTIALLY_VALID?
#    If I wrote INVALID for a matched entry, I must correct it to PARTIALLY_VALID
#    unless the paper is retracted or fabricated."

# Then append a JSON block in EXACTLY this format:

# ```json
# {
#   "results": [
#     {
#       "entry_id": "...",
#       "status": "valid|partially_valid|invalid|unverifiable",
#       "confidence": 0.0,
#       "access_error": false,
#       "issues": ["field: description"],
#       "suggested_fixes": {"field": "corrected value"}
#     }
#   ]
# }
# ```

# Output the markdown first, then the JSON block. No other wrapper text.
# """

RAG_SYSTEM = """You are an expert BibTeX validation assistant with access to
DBLP, OpenAlex, and Google Scholar search tools.

You will receive a list of BibTeX entries. For EACH entry:

════════════════════════════════════════
STEP 1 — SEARCH (Primary: DBLP)
════════════════════════════════════════
Call fuzzy_title_search with the entry's title and authors.
  • Priotize DBLP API Error over similarity score and switch to OpenAlex immediately if you see an error (e.g., 500 Server Error).
  • similarity >= 0.75  → strong match found. Skip OpenAlex & Scholar; proceed to STEP 2.
  • similarity < 0.75   → proceed to STEP 1B (OpenAlex fallback).
  • Tool returns [EXHAUSTED] or any error marker (500, timeout, connection reset) →
    IMMEDIATELY proceed to STEP 1B (OpenAlex fallback).
    ⚠️ CRITICAL: API errors mean "backend unavailable", NOT "paper not found".
       Always try the next source.

════════════════════════════════════════
STEP 1B — FALLBACK: OpenAlex
════════════════════════════════════════
Call openalex_search with the entry's title, author(s), and year.

  • OpenAlex similarity >= 0.70 → good match. Proceed to STEP 2 using OpenAlex result.
  • OpenAlex similarity < 0.70 → proceed to STEP 1C (Scholar).
  • Tool returns [EXHAUSTED] or any error marker → proceed to STEP 1C (Scholar).
    ⚠️ CRITICAL: OpenAlex error means "try next source", NOT "paper not found".

════════════════════════════════════════
STEP 1C — FINAL FALLBACK: Google Scholar
════════════════════════════════════════
Call google_scholar_search.

  • Scholar finds a match → proceed to STEP 2 using Scholar's result.
  • Scholar also fails → mark UNVERIFIABLE (see STEP 3).

════════════════════════════════════════
STEP 2 — COMPARE (only if a match was found)
════════════════════════════════════════
Check every bibliographic field against the best match:
  title, author(s), year, journal / booktitle / venue.

Note every discrepancy, no matter how minor:
  • typos or misspellings in any field
  • author name variants or missing co-authors
  • wrong or off-by-one year
  • wrong, abbreviated, or incomplete venue name
  • capitalisation errors

════════════════════════════════════════
STEP 3 — CLASSIFY (follow this decision tree top-to-bottom)
════════════════════════════════════════

  [A] Did ALL THREE searches (DBLP, OpenAlex, Scholar) fail with API/network errors?
      → UNVERIFIABLE. Set access_error=true. Stop.
      (Never penalise the entry for an infrastructure failure.)

  [B] Did at least ONE source complete successfully (no error)?
      → Continue to step [D]. (We have enough information to classify.)

  [D] Was a match found (similarity >= 0.75 from DBLP, OR >= 0.70 from OpenAlex,
      OR Scholar confirmed)?
      → NOW look at the field comparison from STEP 2:

        • Zero field discrepancies → VALID. Set access_error=false.

        • One or more field discrepancies (ANY of: wrong year, misspelled
          author, wrong venue, incomplete author list, capitalisation error,
          title word swap, etc.) → PARTIALLY_VALID. Set access_error=false.
          ↳ This is true even if DBLP returned a 500 error earlier.
          ↳ If OpenAlex or Scholar found the paper, we have confirmation.
          ↳ The match proves the paper EXISTS; the discrepancy is a field error.

        • Paper is retracted, fraudulent, or otherwise invalid as a citation
          (regardless of whether a record exists) → INVALID. Set access_error=false.

  [C] Did all sources complete successfully (no errors) but return NO match at all?
      → Determine if the paper is provably non-existent
      (e.g., author died before the stated year, impossible author/venue
       combination, title is clearly fabricated)?
      → If provably non-existent: INVALID. Set access_error=false.
      → If cannot be confirmed non-existent: UNVERIFIABLE. Set access_error=false.
         (Benefit of the doubt; the database may lack coverage.)

⚠️  CRITICAL RULE — read before classifying every entry:
    A confirmed match (step D) can NEVER produce an INVALID classification
    solely because of field errors. Field errors on a confirmed match → PARTIALLY_VALID.
    INVALID is reserved for (B) above or for retracted/fabricated papers.

════════════════════════════════════════
EXAMPLE (partially_valid case)
════════════════════════════════════════
  Entry:   author={John Doe and Jane Smith}, year={2021}, venue={ICML}
  Match:   author={John Doe and Jane Smith and Bob Lee}, year={2022}, venue={ICML}
  Result:  PARTIALLY_VALID
  Issues:  ["author: missing co-author Bob Lee", "year: 2021 should be 2022"]

════════════════════════════════════════
⚠️ CRITICAL RULE #2 — access_error and suggested_fixes usage
════════════════════════════════════════
WHEN TO SET access_error:
  • access_error=TRUE: ONLY if ALL THREE sources (DBLP, OpenAlex, Scholar) failed with errors.
    → In this case, ONLY use UNVERIFIABLE. Set suggested_fixes={} (empty dict).
  
  • access_error=FALSE: If AT LEAST ONE source succeeded (returned a match OR completed with no match).
    → Use VALID, PARTIALLY_VALID, or INVALID based on the best result from any source.

SUGGESTED_FIXES RULE:
  • IF you have ANY suggested_fixes AND access_error=false:
    → MUST classify as PARTIALLY_VALID, NEVER UNVERIFIABLE.
    Example: "venue abbreviated; actual IEEE TNN" + access_error=false
           + suggested_fixes={journal: "IEEE Transactions on Neural Networks"}
    → This is PARTIALLY_VALID (paper found by OpenAlex or Scholar, field error detected).

COMMON MISTAKES (DO NOT DO):
  ❌ status=unverifiable, access_error=false, suggested_fixes={journal: "..."}
     (You have a match from OpenAlex/Scholar; it's PARTIALLY_VALID, not unverifiable.)
  
  ❌ status=partially_valid, access_error=true, suggested_fixes={journal: "..."}
     (If access_error=true, DBLP/OpenAlex/Scholar all failed; it's UNVERIFIABLE, not partially_valid.)

✓ status=partially_valid, access_error=false, suggested_fixes={journal: "..."}
  (Best match found by OpenAlex or Scholar; field errors detected; not an API failure.)

════════════════════════════════════════
EFFICIENCY RULES
════════════════════════════════════════
  • Do NOT call OpenAlex or Scholar if DBLP similarity is already >= 0.75.
  • Do NOT call Scholar if OpenAlex similarity is >= 0.70.
  • One DBLP call per entry is usually sufficient.
  • Only retry with a shorter query (dblp_search) if the full-title search
    returns nothing — not if it returns a low-similarity match.
  • If you see [EXHAUSTED] or an error marker, stop; do not retry further.

════════════════════════════════════════
OUTPUT FORMAT
════════════════════════════════════════
Produce a complete markdown report with a summary table and per-entry
details showing which evidence source was used and what issues were found.

Mandatory section headers (use exactly these, in this order):

  ## 🟢 Valid References
  ## 🟡 Partially Valid References
  ## 🔴 Invalid References
  ## 🟠 Unverifiable References

Every reference must appear in exactly one section. Keep entry IDs visible.

Before writing the JSON block, perform this self-check for every entry:
  "Did I find a match for this entry? If yes, is my status VALID or PARTIALLY_VALID?
   If I wrote INVALID for a matched entry, I must correct it to PARTIALLY_VALID
   unless the paper is retracted or fabricated."

Then append a JSON block in EXACTLY this format:

```json
{
  "results": [
    {
      "entry_id": "...",
      "status": "valid|partially_valid|invalid|unverifiable",
      "confidence": 0.0,
      "access_error": false,
      "issues": ["field: description"],
      "suggested_fixes": {"field": "corrected value"}
    }
  ]
}
```
Output the markdown first, then the JSON block. No other wrapper text.
"""


COT_SYSTEM = """You are an expert BibTeX validation assistant with access to
DBLP, OpenAlex, and Google Scholar search tools.

You will receive a list of BibTeX entries. For EACH entry you must
reason step by step before assigning a verdict:

Step 1 — SEARCH (DBLP primary)
 Call fuzzy_title_search with the entry's title and authors.
  • similarity >= 0.75  → strong match. Skip OpenAlex & Scholar.
  • similarity < 0.75   → proceed to STEP 1B (OpenAlex).
  • Tool returns [FALLBACK_TO_OPENALEX] or any error marker → 
    IMMEDIATELY call openalex_search. Do NOT retry DBLP again.
    ⚠️ CRITICAL: [FALLBACK_TO_OPENALEX] means DBLP is unavailable.
       You must use OpenAlex for this entry.
       
Step 1B — FALLBACK (OpenAlex)
  Call openalex_search with title, author(s), and year.
  - If similarity >= 0.70: good match. Proceed to Step 2.
  - If similarity < 0.70 or error marker: proceed to Step 1C (Scholar).

Step 1C — FINAL FALLBACK (Google Scholar)
  Call google_scholar_search.
  - If Scholar finds a match: Proceed to Step 2 using Scholar's result.
  - If Scholar also fails: All sources exhausted. Proceed to Step 6.

Step 2 — Title check
  Does the title closely match the best hit?
  Note typos, truncated words, or word-order differences.

Step 3 — Author check
  Are all author names spelled correctly and present?
  Note missing authors, misspellings, or swapped name order.

Step 4 — Year check
  Does the year match the evidence?
  Flag if off by more than 1 year.

Step 5 — Venue check
  Does the journal / booktitle / venue match?
  Note abbreviation mismatches or entirely wrong venue.

Step 6 — Verdict
  Based on steps 2–5 assign:
  - valid          : strong match from DBLP/OpenAlex/Scholar AND all fields correct
  - partially_valid: paper EXISTS (match found) BUT has field errors (typo, misspelling, author variant, year mismatch, venue error, incomplete author list, etc.)
  - invalid        : paper is retracted, fabricated, impossible, or searches succeeded but no match found AND paper provably non-existent
  - unverifiable   : ALL sources (DBLP, OpenAlex, Scholar) failed with API/network errors

CRITICAL RULE:
  - If ANY source found the paper (DBLP >= 0.75, OpenAlex >= 0.70, Scholar confirmed):
    AND you found field errors → PARTIALLY_VALID (set access_error=false)
  - If the paper is retracted, fraudulent, fabricated, anachronistic, or logically impossible → INVALID (even if a record exists)
  - Only INVALID if searches completed successfully (no errors) AND no match found AND paper is provably fabricated
  - If ALL sources failed with errors → UNVERIFIABLE (set access_error=true)

⚠️ CRITICAL RULE #2 — access_error and suggested_fixes usage:
  • IF you have ANY suggested_fixes AND access_error=false:
    → MUST classify as PARTIALLY_VALID, NEVER UNVERIFIABLE.
    (OpenAlex or Scholar found the paper; field errors detected; not an API failure.)
  
  • IF access_error=true (ALL sources failed with errors):
    → ONLY use UNVERIFIABLE. Set suggested_fixes={} (empty dict).
  
  • COMMON MISTAKE (DO NOT DO):
    ❌ status=unverifiable, access_error=false, suggested_fixes={journal: "IEEE TNN"}
    ✓ status=partially_valid, access_error=false, suggested_fixes={journal: "IEEE TNN"}

EFFICIENCY RULES:
  - Do NOT call Scholar if DBLP similarity is already >= 0.75
  - One DBLP call per entry is usually enough (but use mirrors if initial call fails)
  - If you see [EXHAUSTED] in error, it means retry already happened 4x → use mirrors immediately

Show your step-by-step reasoning for each entry, then produce a full
markdown report with summary table and per-entry details.

Markdown format requirements (MANDATORY):
1. Include these exact section headers in this exact order:
  - ## 🟢 Valid References
  - ## 🟡 Partially Valid References
  - ## 🔴 Invalid References
  - ## 🟠 Unverifiable References
2. Every reference must appear in exactly one of those three sections.
3. Do not create any other status section names.
4. Keep entry IDs in each section so they are easy to trace.

Then append a JSON block in EXACTLY this format:
```json
{
  "results": [
    {
      "entry_id": "...",
      "status": "valid|partially_valid|invalid|unverifiable",
      "confidence": 0.0,
      "access_error": false,
      "issues": ["field: description"],
      "suggested_fixes": {"field": "corrected value"}
    }
  ]
}
```
Output markdown first, then the JSON block. No other wrapper text.
"""

STRATEGY_SYSTEM_PROMPTS = {
    PromptStrategy.ZERO_SHOT: ZERO_SHOT_SYSTEM,
    PromptStrategy.RAG:       RAG_SYSTEM,
    PromptStrategy.COT:       COT_SYSTEM,
}


# ----------------------------------------------------------------------
# React state
# ----------------------------------------------------------------------
class ReactState(TypedDict):
    messages: Annotated[list, add_messages]


# ----------------------------------------------------------------------
# Sequential tool node as an async function (LangGraph compatible)
# ----------------------------------------------------------------------
def create_sequential_tool_node(tools: list):
    """Return an async function that processes tool calls one by one."""
    tools_by_name = {tool.name: tool for tool in tools if hasattr(tool, "name")}

    async def sequential_tool_node(state: ReactState) -> dict:
        messages = state["messages"]
        if not messages:
            return {"messages": []}
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not getattr(last_message, "tool_calls", None):
            return {"messages": []}

        tool_calls = last_message.tool_calls
        responses = []
        for i, tc in enumerate(tool_calls):
            if i > 0:
                await asyncio.sleep(0.5)  # 500 ms gap between calls
            tool_name = tc.get("name")
            tool_args = tc.get("args", {})
            tool_call_id = tc.get("id")
            tool = tools_by_name.get(tool_name)
            if not tool:
                responses.append(ToolMessage(content=f"Error: tool '{tool_name}' not found", tool_call_id=tool_call_id))
                continue
            try:
                result = await tool.ainvoke(tool_args)
                responses.append(ToolMessage(content=str(result), tool_call_id=tool_call_id))
            except Exception as e:
                responses.append(ToolMessage(content=f"Tool error: {e}", tool_call_id=tool_call_id))
        return {"messages": responses}

    return sequential_tool_node


# ----------------------------------------------------------------------
# Main Agent
# ----------------------------------------------------------------------
class LLMValidationAgent:
    """
        ReAct-based validation agent.

        All three strategies use a LangGraph ReAct loop internally:
          START -> llm_node -> (tool_node -> llm_node)* -> END

        The key difference between strategies is:
          - zero_shot : no tools bound to the LLM — pure knowledge reasoning
          - rag       : tools bound, LLM calls them as needed, stops early when
                        DBLP gives a strong match
          - cot       : same tools as RAG, but system prompt enforces explicit
                        field-by-field reasoning before each verdict

        Interface is identical to the manual _validation_agent.py so the
        pipeline can swap between them with a single import change.
      """

    def __init__(
        self,
        mcp_config_path: str,
        strategy: PromptStrategy = PromptStrategy.RAG,
        ):
        self.mcp_config_path = mcp_config_path
        self.strategy        = strategy

        # Google Gemini backend (default)
        # self.llm = ChatGoogleGenerativeAI(
        #     model="gemini-2.0-flash",
        #     temperature=0.1,
        #     google_api_key=os.getenv("GOOGLE_API_KEY"),
        # )
        
        # OpenAI fallback — uncomment to switch
        # self.llm = ChatOpenAI(
        #     model="gpt-5.4",
        #     temperature=0.1,
        #     openai_api_key=os.getenv("OPENAI_API_KEY"),
        # )
        
        # self.llm = ChatOpenRouter(
        #     #model="anthropic/claude-sonnet-4.6",
        #     #model="google/gemini-2.5-pro",
        #     #model="google/gemini-3.1-pro-preview",
        #     #model="mistralai/mistral-medium-3-5",
        #     # model ="openai/gpt-5.4",
        #     #model="qwen/qwen3.6-35b-a3b",
        #     #model="x-ai/grok-4.20",
        #     #temperature=0.1,
        #     # max_tokens=30000,
        #     #openrouter_api_key=os.getenv("OPENROUTER_API_KEY")
        # )

        #Ollama backend — uncomment to switch
        self.llm = ChatOllama(
            model="qwen3-coder:480b-cloud",
            # model="deepseek-v3.2:cloud",
            base_url="https://ollama.com",
            temperature=0.1,
            client_kwargs={
                "headers": {"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"}
            },
        )

        #HuggingFace backend — uncomment to switch
        # self.llm = ChatHuggingFace(
        #     llm=HuggingFaceEndpoint(
        #         repo_id="deepseek-ai/DeepSeek-R1",
        #         task="text-generation",
        #         huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        #         max_new_tokens=4096,
        #         temperature=0.1,
        #     )
        # )
    async def validate_entries(self, entries: list[dict]) -> dict:
        print(f"\n{'='*60}")
        print(f"LLM VALIDATION AGENT [REACT/{self.strategy.value.upper()}] — {len(entries)} entries")
        print(f"{'='*60}\n")

        telemetry = ValidationTelemetry()
        telemetry.start()

        # Load MCP tools
        with open(self.mcp_config_path) as f:
            config = json.load(f)
        mcp_servers_config = config.get("mcpServers", config)
        client = MultiServerMCPClient(mcp_servers_config)

        dblp_tools_list = await client.get_tools(server_name="mcp-dblp")
        dblp_tools = list(dblp_tools_list)
        scholar_tools = []
        try:
            scholar_tools_list = await client.get_tools(server_name="mcp-scholar")
            scholar_tools = list(scholar_tools_list)
            
        except Exception as e:
            print(f"  Scholar MCP tools unavailable: {e}")
            print(f"  OpenAlex MCP tools unavailable: {e}\n")

        print(f"  DBLP tools   : {[t.name for t in dblp_tools]}")
        print(f"  Scholar tools: {[t.name for t in scholar_tools]}\n")
        
        # Load DBLP tools (external MCP)
        # dblp_tools_list = await client.get_tools(server_name="mcp-dblp")
        # dblp_tools = list(dblp_tools_list)

        # # Load your own bibtex_mcp (provides openalex_search and google_scholar_search)
        # bibtex_tools_list = await client.get_tools(server_name="bibtex_mcp")
        # bibtex_tools = list(bibtex_tools_list)

        # print(f"  DBLP tools   : {[t.name for t in dblp_tools]}")
        # print(f"  BibTeX MCP tools (OpenAlex + Scholar): {[t.name for t in bibtex_tools]}\n")
 
        # Rate limiter (1 request per second)
        MAX_QPS = 1
        recent_requests = deque()

        async def acquire_slot():
            while True:
                now = time.monotonic()
                while recent_requests and now - recent_requests[0] > 1.0:
                    recent_requests.popleft()
                if len(recent_requests) < MAX_QPS:
                    recent_requests.append(now)
                    return
                await asyncio.sleep(0.1)

        # Cache
        class SimpleLRU:
            def __init__(self, maxsize=1024):
                self.maxsize = maxsize
                self.data = OrderedDict()
            def get(self, key):
                if key in self.data:
                    self.data.move_to_end(key)
                    telemetry.record_cache_hit()
                    return self.data[key]
                return None
            def set(self, key, value):
                if key in self.data:
                    self.data.move_to_end(key)
                elif len(self.data) >= self.maxsize:
                    self.data.popitem(last=False)
                self.data[key] = value

        cache = SimpleLRU()

        def make_cache_key(tool_name, *args, **kwargs):
            try:
                args_key = json.dumps(args, sort_keys=True, default=str)
                kw_key = json.dumps(kwargs, sort_keys=True, default=str)
            except Exception:
                args_key = str(args)
                kw_key = str(kwargs)
            return (tool_name, args_key, kw_key)

        def is_cacheable(resp):
            if isinstance(resp, str):
                txt = resp.strip().lower()
                if not txt or (txt.startswith("[") and "error" in txt) or "[exhausted]" in txt:
                    return False
            return True

        def looks_transient_error(resp, exc=None):
            keywords = ["503", "connection reset", "timeout", "exhausted", "unavailable"]
            text = str(exc or resp).lower()
            return any(k in text for k in keywords)

        def is_remote_reset(exc):
            return "remotedisconnected" in str(exc).lower() or "connection reset" in str(exc).lower()

        def set_dblp_mirror(host):
            # Placeholder – actual mirror switching handled by MCP server
            return None

        # Wrap tools with retry and mirror fallback
        def make_wrapped_tool(tool):
            name = tool.name
            orig_invoke = tool.invoke
            orig_ainvoke = tool.ainvoke

            def sync_wrapped(tool_input=None, **kwargs):
                norm_input = kwargs if kwargs else tool_input
                key = make_cache_key(name, norm_input)
                cached = cache.get(key)
                if cached is not None:
                    return cached

                max_attempts = 2
                base_delay = 2.0
                last_exc = None
                search_tools = {"dblp_fuzzy_title_search", "dblp_search", "openalex_search"}
                mirrors = DBLP_MIRRORS if name in search_tools else [None]

                for mirror in mirrors:
                    if mirror and name in search_tools:
                        set_dblp_mirror(mirror)
                    for attempt in range(max_attempts):
                        # Rate limit (sync)
                        while True:
                            now = time.monotonic()
                            while recent_requests and now - recent_requests[0] > 1.0:
                                recent_requests.popleft()
                            if len(recent_requests) < MAX_QPS:
                                recent_requests.append(now)
                                break
                            time.sleep(0.1)
                        try:
                            telemetry.record_request()
                            resp = orig_invoke(norm_input)
                            if isinstance(resp, list):
                                for item in resp:
                                    if isinstance(item, dict) and 'text' in item:
                                        text = item['text']
                                        # The search result typically starts with "Found X publications"
                                        if text.strip().startswith("Found") and "publications" in text:
                                            resp = text
                                            break
                                else:
                                    # No valid search result found – treat as error
                                    resp = "[ERROR] No valid DBLP result"
                            if looks_transient_error(resp):
                                raise RuntimeError(f"Transient: {resp}")
                            # --- NEW: Force fallback on DBLP API errors ---
                            if isinstance(resp, str) and ("ERROR: DBLP API error" in resp or "500 Server Error" in resp):
                                return "[FALLBACK_TO_OPENALEX] DBLP failed with HTTP 500. Use openalex_search."
                            if is_cacheable(resp):
                                cache.set(key, resp)
                            return resp
                        except Exception as e:
                            last_exc = e
                            if not looks_transient_error(None, e) and not is_remote_reset(e):
                                return f"[ERROR] {name}: {e}"
                            telemetry.record_retry(name)
                            delay = base_delay * (2**attempt) + random.uniform(0, 0.5)
                            if is_remote_reset(e):
                                delay = 5.0
                            print(f"    [RETRY {attempt+1}] {name}: {str(e)[:100]} → waiting {delay:.2f}s")
                            time.sleep(delay)
                exhausted = f"[EXHAUSTED] All mirrors failed: {last_exc}"
                print(f"    {exhausted}")
                return exhausted

            async def async_wrapped(tool_input=None, **kwargs):
                norm_input = kwargs if kwargs else tool_input
                key = make_cache_key(name, norm_input)
                cached = cache.get(key)
                if cached is not None:
                    return cached

                max_attempts = 2
                base_delay = 2.0
                last_exc = None
                search_tools = {"dblp_fuzzy_title_search", "dblp_search", "openalex_search"}
                mirrors = DBLP_MIRRORS if name in search_tools else [None]

                for mirror in mirrors:
                    if mirror and name in search_tools:
                        set_dblp_mirror(mirror)
                    for attempt in range(max_attempts):
                        await acquire_slot()
                        try:
                            telemetry.record_request()
                            if orig_ainvoke:
                                resp = await orig_ainvoke(norm_input)
                            else:
                                resp = orig_invoke(norm_input)
                                if isinstance(resp, list):
                                    for item in resp:
                                        if isinstance(item, dict) and 'text' in item:
                                            text = item['text']
                                            if text.strip().startswith("Found") and "publications" in text:
                                                resp = text
                                                break
                                    else:
                                        resp = "[ERROR] No valid DBLP result"
                            if looks_transient_error(resp):
                                raise RuntimeError(f"Transient: {resp}")
                            if isinstance(resp, str) and ("ERROR: DBLP API error" in resp or "500 Server Error" in resp):
                                return "[FALLBACK_TO_OPENALEX] DBLP failed with HTTP 500. Use openalex_search."
                            if is_cacheable(resp):
                                cache.set(key, resp)
                            return resp
                        except Exception as e:
                            last_exc = e
                            if not looks_transient_error(None, e) and not is_remote_reset(e):
                                return f"[ERROR] {name}: {e}"
                            telemetry.record_retry(name)
                            delay = base_delay * (2**attempt) + random.uniform(0, 0.5)
                            if is_remote_reset(e):
                                delay = 5.0
                            print(f"    [RETRY {attempt+1}] {name}: {str(e)[:100]} → waiting {delay:.2f}s")
                            await asyncio.sleep(delay)
                exhausted = f"[EXHAUSTED] All mirrors failed: {last_exc}"
                print(f"    {exhausted}")
                return exhausted

            return StructuredTool.from_function(
                func=sync_wrapped,
                coroutine=async_wrapped,
                name=name,
                description=tool.description,
                args_schema=tool.args_schema,
            )

        wrapped_dblp = [make_wrapped_tool(t) for t in dblp_tools]
        wrapped_scholar = [make_wrapped_tool(t) for t in scholar_tools]
       
        all_tools = wrapped_dblp + wrapped_scholar 
        # instrumented_dblp_tools = [make_wrapped_tool(t) for t in dblp_tools]
        # instrumented_bibtex_tools = [make_wrapped_tool(t) for t in bibtex_tools]   
        # _tool_map = {t.name: t for t in dblp_tools + bibtex_tools}

        result = await self._run_react(entries, all_tools, telemetry)
        # result = await self._run_react(entries, instrumented_dblp_tools + instrumented_bibtex_tools, telemetry)

        telemetry.end()
        telemetry.print_summary()
        result["telemetry"] = telemetry.to_dict()
        return result

    async def _run_react(self, entries: list[dict], tools: list, telemetry) -> dict:
        if self.strategy == PromptStrategy.ZERO_SHOT:
            llm_with_tools = self.llm
            all_tools = []
        else:
            llm_with_tools = self.llm.bind_tools(tools)
            all_tools = tools

        tool_budget = max(MAX_REACT_ITERATIONS, len(entries) * 3) if all_tools else 0

        async def llm_node(state: ReactState) -> dict:
            response = await _ainvoke_with_retry(llm_with_tools, state["messages"], attempts=4, base_delay=1.0)
            return {"messages": [response]}

        # Create the sequential tool node function
        tool_node = create_sequential_tool_node(all_tools) if all_tools else None

        def should_continue(state: ReactState) -> str:
            last = state["messages"][-1]
            tool_msgs = sum(1 for m in state["messages"] if isinstance(m, ToolMessage))
            if tool_msgs >= tool_budget:
                return END
            if all_tools and isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                return "tools"
            return END

        graph = StateGraph(ReactState)
        graph.add_node("llm", llm_node)
        graph.add_edge(START, "llm")
        if all_tools and tool_node:
            graph.add_node("tools", tool_node)
            graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
            graph.add_edge("tools", "llm")
        else:
            graph.add_edge("llm", END)

        app = graph.compile()

        system_prompt = STRATEGY_SYSTEM_PROMPTS[self.strategy]
        user_content = (
            f"Validate these {len(entries)} BibTeX entries.\n\n"
            f"```json\n{json.dumps(entries, indent=2, ensure_ascii=False)}\n```"
        )
        initial_messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]

        print(f"  Running [REACT/{self.strategy.value}] – max {tool_budget} tool calls")
        tool_call_count = 0
        final_messages = []
        recursion_limit = (tool_budget * 2) + REACT_RECURSION_BUFFER
        async for chunk in app.astream(
            {"messages": initial_messages},
            config={"recursion_limit": recursion_limit, "max_concurrency": 1}
        ):
            for node_name, node_output in chunk.items():
                msgs = node_output.get("messages", [])
                if node_name == "tools":
                    tool_call_count += len(msgs)
                    if tool_call_count % 10 == 0:
                        print(f"    ... {tool_call_count} tool calls completed")
                final_messages = msgs

        print(f"  Total tool calls: {tool_call_count}")
        raw_text = ""
        for msg in reversed(final_messages):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                raw_text = _extract_text(msg)
                break
        if not raw_text:
            for msg in reversed(final_messages):
                if isinstance(msg, AIMessage):
                    raw_text = _extract_text(msg)
                    if raw_text:
                        break
        if not raw_text:
            print("  Warning: could not extract final response from ReAct graph")

        markdown, structured = self._split_response(raw_text)
        raw_data = self._rebuild_raw_data(entries, structured)
        return {
            "markdown_report": markdown,
            "structured": structured,
            "raw_data": raw_data,
            "total_entries": len(entries),
            "tool_calls": tool_call_count,
        }

    @staticmethod
    def _split_response(raw_text: str):
        if "<think>" in raw_text:
            raw_text = raw_text.split("</think>")[-1].strip()
        markdown, structured = raw_text, []
        if "```json" in raw_text:
            parts = raw_text.split("```json", 1)
            markdown = parts[0].strip()
            json_block = parts[1].split("```")[0].strip()
            try:
                data = json.loads(json_block)
                structured, _ = parse_validation_results(data)
            except json.JSONDecodeError:
                pass
        return markdown, structured

    @staticmethod
    def _rebuild_raw_data(entries: list[dict], structured: list[dict]) -> list[dict]:
        result_map = {r.get("entry_id"): r for r in structured if isinstance(r, dict)}
        raw_data = []
        for entry in entries:
            entry_id = entry.get("id", "")
            result = result_map.get(entry_id, {})
            dblp_hit = {}
            if result.get("suggested_fixes"):
                fixes = result["suggested_fixes"]
                dblp_hit = {
                    "title": fixes.get("title", entry.get("title", "")),
                    "authors": fixes.get("author", entry.get("author", "")),
                    "year": fixes.get("year", entry.get("year", "")),
                    "venue": fixes.get("journal", entry.get("journal", entry.get("booktitle", ""))),
                    "similarity_score": result.get("confidence", 0.0),
                }
            raw_data.append({
                "entry": entry,
                "dblp_hits": [dblp_hit] if dblp_hit else [],
                "scholar_hits": [],
                "react_result": result,
            })
        return raw_data