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

# Base cap used as a lower bound. Actual per-run cap is scaled by entry count.
MAX_REACT_ITERATIONS = 120
REACT_RECURSION_BUFFER = 20  # extra graph steps for START/END + finalization overhead
DBLP_MIRRORS = [
  "dblp.org",
  "dblp.uni-trier.de",
  "dblp.dagstuhl.de",
]


# ─────────────────────────────────────────────────────────────
# Telemetry Tracking
# ─────────────────────────────────────────────────────────────

class ValidationTelemetry:
    """Track API usage metrics: requests, cache hits, retries, and request rate."""
    
    def __init__(self):
        self.total_requests = 0  # actual API calls made
        self.cache_hits = 0      # requests served from cache
        self.total_retries = 0   # total retry attempts
        self.retry_by_tool = {}  # retry count by tool name
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
        """Get elapsed time in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    def get_requests_per_second(self) -> float:
        """Calculate actual requests per second."""
        duration = self.get_duration()
        if duration > 0:
            return self.total_requests / duration
        return 0.0
    
    def print_summary(self):
        """Print comprehensive telemetry summary."""
        duration = self.get_duration()
        total_lookups = self.cache_hits + self.total_requests
        total_with_retries = self.total_requests + self.total_retries
        qps = self.get_requests_per_second()
        cache_hit_rate = (self.cache_hits / total_lookups * 100) if total_lookups > 0 else 0
        
        print(f"\n{'─'*60}")
        print(f"VALIDATION TELEMETRY")
        print(f"{'─'*60}")
        print(f"  Total API calls (net)    : {self.total_requests}")
        print(f"  Cache hits               : {self.cache_hits}")
        print(f"  Cache hit rate           : {cache_hit_rate:.1f}%")
        print(f"  Total retry attempts     : {self.total_retries}")
        print(f"  Requests with ≥1 retry  : {total_with_retries if self.total_retries > 0 else 'none'}")
        
        if self.retry_by_tool:
            print(f"  Retries by tool:")
            for tool_name in sorted(self.retry_by_tool.keys()):
                count = self.retry_by_tool[tool_name]
                print(f"    • {tool_name}: {count}")
        
        print(f"  Total duration           : {duration:.2f}s")
        print(f"  Effective request rate   : {qps:.1f} req/s")
        print(f"{'─'*60}\n")

    def to_dict(self) -> dict:
        """Serialize telemetry for persistence in pipeline artefacts."""
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


# ─────────────────────────────────────────────────────────────
# Prompt Strategy
# ─────────────────────────────────────────────────────────────

class PromptStrategy(Enum):
    ZERO_SHOT = "zero_shot"
    RAG       = "rag"
    COT       = "cot"


# ─────────────────────────────────────────────────────────────
# System Prompts
# ─────────────────────────────────────────────────────────────

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
  - partially_valid: paper likely EXISTS but 1+ fields are wrong (typo, misspelling, incomplete author list, wrong year, wrong venue, etc.)
  - invalid        : paper does NOT exist / appears fabricated / cannot be found in any academic database
  
CRITICAL DISTINCTION:
  - Field errors (typos, misspellings, capitalization, incomplete data) → PARTIALLY_VALID if paper exists
  - Paper non-existent (completely fabricated, anachronistic authors, future dates, wrong author combinations) → INVALID
  - Do NOT mark as invalid just because a field is wrong; only if paper doesn't actually exist

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

RAG_SYSTEM = """You are an expert BibTeX validation assistant with access to
DBLP and Google Scholar search tools.

You will receive a list of BibTeX entries. For EACH entry:

1. SEARCH — call dblp_fuzzy_title_search with the entry title and authors.
   - similarity >= 0.75  -> strong evidence, no need for Scholar.
   - similarity < 0.75   -> call google_scholar_search as fallback.
  - Scholar also fails  -> mark as unverifiable if the returned tool output contains an error marker.
  - All searches fail   -> mark as invalid only if the searches completed successfully and the paper is provably fabricated.
  - If a tool returns [EXHAUSTED] or another error marker, assume the backend could not verify the entry.

2. COMPARE — check each field against the best match:
   title, author, year, journal / booktitle / venue.

3. ASSIGN:
   - valid          : strong match AND all major fields are correct
   - partially_valid: paper EXISTS in database BUT 1+ fields are wrong (typo, misspelling, author name variant, wrong year, wrong venue, incomplete author list, capitalization error, etc.)
   - invalid        : searches succeeded (no API errors) but no match found AND paper provably doesn't exist / is fabricated
   - unverifiable   : could not verify due to API/network failures (DBLP mirrors + Scholar all failed with errors)
  
CRITICAL FIELD ERROR HANDLING:
  - If DBLP/Scholar confirms the paper EXISTS (similarity >= 0.70) but has field errors → PARTIALLY_VALID
  - Only mark as INVALID if:
    * Searches succeeded (no API errors) and returned no match
    * AND paper is provably non-existent (e.g., author died before year, impossible combination)
  - Do NOT mark as INVALID when API/network errors prevented verification → use UNVERIFIABLE instead
  - Do NOT confuse "field errors" with "paper doesn't exist"
  - Do NOT confuse "API access failure" with "paper doesn't exist"

EFFICIENCY RULES:
  - Do NOT call Scholar if DBLP similarity is already >= 0.75
  - One DBLP call per entry is usually enough (but retry with mirrors if it fails)
  - Only retry with a shorter query search with the dblp_search tool if the full-title search returns nothing
  - If you see [EXHAUSTED] or an error marker in a tool response, stop treating the entry as verifiable

Produce a complete markdown report with summary table and per-entry
details showing which evidence source was used and what issues were found.

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

COT_SYSTEM = """You are an expert BibTeX validation assistant with access to
DBLP and Google Scholar search tools.

You will receive a list of BibTeX entries. For EACH entry you must
reason step by step before assigning a verdict:

Step 1 — SEARCH
  Call dblp_fuzzy_title_search with the entry title.
  - If the tool response contains [EXHAUSTED] or another error marker, treat the backend as unavailable and move to Scholar fallback.
  - If similarity < 0.75, call google_scholar_search as fallback.
  - If all searches fail with no results and no error markers, proceed to step 6.

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
  - valid          : strong match AND all fields correct
  - partially_valid: paper EXISTS in database BUT has field errors (typo, misspelling, author variant, year mismatch, venue error, incomplete author list, etc.)
  - invalid        : searches succeeded (no API errors) but no match found AND paper provably non-existent
  - unverifiable   : could not verify due to API/network errors (set access_error: true)

CRITICAL RULE:
  - If you confirmed the paper EXISTS (similarity >= 0.70) but found field errors → PARTIALLY_VALID
  - Only INVALID if searches completed successfully with no results AND paper is provably fabricated
  - If searches failed due to API errors → UNVERIFIABLE (not INVALID)
  - Do NOT confuse "field errors" with "paper doesn't exist"
  - Do NOT confuse "API access failure" with "paper doesn't exist"

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


# ─────────────────────────────────────────────────────────────
# ReAct Graph State
# ─────────────────────────────────────────────────────────────

class ReactState(TypedDict):
    messages: Annotated[list, add_messages]


# ─────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────

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
        
        self.llm = ChatOpenRouter(
            #model="anthropic/claude-sonnet-4.6",
            model="google/gemini-2.5-pro",
            #model ="openai/gpt-5.4",
            #model="x-ai/grok-4.20",
            temperature=0.1,
            max_tokens=30000,
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY")
        )

        #Ollama backend — uncomment to switch
        # self.llm = ChatOllama(
        #     model="qwen3-coder:480b-cloud",
        #     # model="deepseek-v3.2:cloud",
        #     base_url="https://ollama.com",
        #     temperature=0.1,
        #     client_kwargs={
        #         "headers": {"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"}
        #     },
        # )

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

    # ── public entry point ────────────────────────────────────

    async def validate_entries(self, entries: list[dict]) -> dict:
        """
        Main entry point — identical interface to manual _validation_agent.py.
        Internally always runs a ReAct loop regardless of strategy.
        """
        print(f"\n{'='*60}")
        print(f"LLM VALIDATION AGENT [REACT/{self.strategy.value.upper()}] — {len(entries)} entries")
        print(f"{'='*60}\n")

        # Initialize telemetry
        telemetry = ValidationTelemetry()
        telemetry.start()

        # Load MCP tools
        with open(self.mcp_config_path, "r") as f:
            config = json.load(f)

        mcp_servers_config = config.get("mcpServers", config)
        client             = MultiServerMCPClient(mcp_servers_config)

        dblp_tools_list = await client.get_tools(server_name="mcp-dblp")
        dblp_tools      = list(dblp_tools_list)

        scholar_tools = []
        try:
            scholar_tools_list = await client.get_tools(server_name="mcp-scholar")
            scholar_tools      = list(scholar_tools_list)
        except Exception as e:
            print(f"  Scholar MCP tools unavailable: {e}")

        print(f"  DBLP tools   : {[t.name for t in dblp_tools]}")
        print(f"  Scholar tools: {[t.name for t in scholar_tools]}\n")

        # --- Rate limiter + retries + cache setup -----------------
        # DBLP rate limit: reduced to 2 req/s to avoid connection reset
        # DBLP is rejecting connections at higher rates
        MAX_QPS = int(os.getenv("DBLP_MAX_QPS", "2"))
        _recent_requests: deque[float] = deque()
        _tool_map: dict[str, object] = {}

        async def _acquire_slot():
          # simple sliding-window rate limiter
          while True:
            now = time.monotonic()
            # drop timestamps older than 1s
            while _recent_requests and now - _recent_requests[0] > 1.0:
              _recent_requests.popleft()
            if len(_recent_requests) < MAX_QPS:
              _recent_requests.append(now)
              return
            await asyncio.sleep(1.0)  # long delay to prevent connection reset

        class SimpleLRU:
          def __init__(self, maxsize: int = 1024):
            self.maxsize = maxsize
            self.data = OrderedDict()

          def get(self, key):
            try:
              val = self.data.pop(key)
              self.data[key] = val
              telemetry.record_cache_hit()
              return val
            except KeyError:
              return None

          def set(self, key, value):
            if key in self.data:
              self.data.pop(key)
            elif len(self.data) >= self.maxsize:
              self.data.popitem(last=False)
            self.data[key] = value

        _cache = SimpleLRU(maxsize=2048)

        def _make_cache_key(tool_name, *args, **kwargs):
          try:
            args_key = json.dumps(args, sort_keys=True, default=str)
            kw_key = json.dumps(kwargs, sort_keys=True, default=str)
          except Exception:
            args_key = str(args)
            kw_key = str(kwargs)
          return (tool_name, args_key, kw_key)

        def _is_cacheable_response(resp) -> bool:
          if isinstance(resp, str):
            txt = resp.strip().lower()
            if not txt:
              return False
            if txt.startswith("[") and "error" in txt:
              return False
            if "[exhausted]" in txt:
              return False
            if txt.startswith("error:"):
              return False
          return True

        def _select_mirror_tool():
          return _tool_map.get("set_dblp_mirror")

        def _set_dblp_mirror(host: str):
          mirror_tool = _select_mirror_tool()
          if mirror_tool is None:
            return None
          try:
            return mirror_tool.invoke({"host": host})
          except Exception:
            return None

        def _cacheable_or_error(resp):
          if _is_cacheable_response(resp):
            return resp
          return None

        def _acquire_slot_sync():
          while True:
            now = time.monotonic()
            while _recent_requests and now - _recent_requests[0] > 1.0:
              _recent_requests.popleft()
            if len(_recent_requests) < MAX_QPS:
              _recent_requests.append(now)
              return
            time.sleep(0.01)

        def _looks_transient_error(resp, exc: Exception | None = None) -> bool:
          """Heuristic detector for transient transport/API errors.

          Extended to catch a wider range of real-world network messages
          returned by different MCP adapters and HTTP clients.
          """
          keywords = [
            "503",
            "service unavailable",
            "connection reset",
            "connection refused",
            "could not connect",
            "failed to establish",
            "temporary failure",
            "temporarily unavailable",
            "timeout",
            "timed out",
            "name or service not known",
            "could not resolve host",
            "exhausted",
            "error:",
            "mirror",
            "503 service",
          ]

          def contains_any(text: str) -> bool:
            t = text.lower()
            return any(k in t for k in keywords)

          if exc is not None:
            return contains_any(str(exc))
          if isinstance(resp, str):
            return contains_any(resp)
          return False

        def _normalize_tool_input(tool_input=None, **kwargs):
          if kwargs:
            # Structured tools typically pass parsed kwargs
            return kwargs
          return tool_input

        def _make_wrapped_tool(tool):
          name = getattr(tool, "name", "tool")
          description = getattr(tool, "description", None) or f"Wrapped MCP tool: {name}"
          args_schema = getattr(tool, "args_schema", None)
          original_invoke = getattr(tool, "invoke", None)
          original_ainvoke = getattr(tool, "ainvoke", None)

          if not callable(original_invoke):
            return tool

          def _sync_wrapped(tool_input=None, **kwargs):
            """Rate-limited and retried sync wrapper around MCP tool invocation."""
            normalized_input = _normalize_tool_input(tool_input, **kwargs)
            key = _make_cache_key(name, normalized_input)
            cached = _cache.get(key)
            if cached is not None:
              return cached

            max_attempts = int(os.getenv("TOOL_MAX_RETRIES", "4"))
            base_delay = float(os.getenv("TOOL_BASE_DELAY", "2.0"))  # increased from 0.5 to 2.0
            last_exc = None
            search_tool_names = {"dblp_fuzzy_title_search", "dblp_search"}

            mirror_sequence = DBLP_MIRRORS if name in search_tool_names else [None]
            for mirror in mirror_sequence:
              if mirror and name in search_tool_names:
                _set_dblp_mirror(mirror)
              for attempt in range(max_attempts):
                _acquire_slot_sync()
                try:
                  telemetry.record_request()
                  resp = original_invoke(normalized_input)
                  if _looks_transient_error(resp):
                    raise RuntimeError(f"Transient tool error: {resp}")
                  if _is_cacheable_response(resp):
                    _cache.set(key, resp)
                  return resp
                except Exception as e:
                  last_exc = e
                  if not _looks_transient_error(None, e):
                    error_msg = str(e)[:200]
                    return f"[ERROR] {name}: {error_msg}"
                  telemetry.record_retry(name)
                  delay = base_delay * (2 ** attempt) + random.random() * 0.2
                  error_msg = str(e)[:100]  # truncate long error messages
                  print(f"    [RETRY {attempt+1}/{max_attempts}] {name}: {error_msg} → waiting {delay:.2f}s")
                  time.sleep(delay)

            # All retries / mirrors exhausted
            error_msg = str(last_exc)[:200] if last_exc is not None else "unknown failure"
            exhausted_msg = f"[EXHAUSTED] All DBLP mirrors failed. Last error: {error_msg}"
            print(f"    {exhausted_msg}")
            return exhausted_msg

          async def _async_wrapped(tool_input=None, **kwargs):
            """Rate-limited and retried async wrapper around MCP tool invocation."""
            normalized_input = _normalize_tool_input(tool_input, **kwargs)
            key = _make_cache_key(name, normalized_input)
            cached = _cache.get(key)
            if cached is not None:
              return cached

            max_attempts = int(os.getenv("TOOL_MAX_RETRIES", "4"))
            base_delay = float(os.getenv("TOOL_BASE_DELAY", "2.0"))  # increased from 0.5 to 2.0
            last_exc = None
            search_tool_names = {"fuzzy_title_search", "search"}

            mirror_sequence = DBLP_MIRRORS if name in search_tool_names else [None]
            for mirror in mirror_sequence:
              if mirror and name in search_tool_names:
                _set_dblp_mirror(mirror)
              for attempt in range(max_attempts):
                await _acquire_slot()
                try:
                  telemetry.record_request()
                  if callable(original_ainvoke):
                    resp = await original_ainvoke(normalized_input)
                  else:
                    resp = original_invoke(normalized_input)
                  if _looks_transient_error(resp):
                    raise RuntimeError(f"Transient tool error: {resp}")
                  if _is_cacheable_response(resp):
                    _cache.set(key, resp)
                  return resp
                except Exception as e:
                  last_exc = e
                  if not _looks_transient_error(None, e):
                    error_msg = str(e)[:200]
                    return f"[ERROR] {name}: {error_msg}"
                  telemetry.record_retry(name)
                  delay = base_delay * (2 ** attempt) + random.random() * 0.2
                  error_msg = str(e)[:100]  # truncate long error messages
                  print(f"    [RETRY {attempt+1}/{max_attempts}] {name}: {error_msg} → waiting {delay:.2f}s")
                  await asyncio.sleep(delay)

            # All retries / mirrors exhausted
            error_msg = str(last_exc)[:200] if last_exc is not None else "unknown failure"
            exhausted_msg = f"[EXHAUSTED] All DBLP mirrors failed. Last error: {error_msg}"
            print(f"    {exhausted_msg}")
            return exhausted_msg

          return StructuredTool.from_function(
            func=_sync_wrapped,
            coroutine=_async_wrapped,
            name=name,
            description=description,
            args_schema=args_schema,
            infer_schema=args_schema is None,
          )

        instrumented_dblp_tools = [_make_wrapped_tool(t) for t in dblp_tools]
        instrumented_scholar_tools = [_make_wrapped_tool(t) for t in scholar_tools]
        _tool_map = {t.name: t for t in dblp_tools + scholar_tools}

        result = await self._run_react(entries, instrumented_dblp_tools, instrumented_scholar_tools)
        
        telemetry.end()
        telemetry.print_summary()
        
        result["telemetry"] = telemetry.to_dict()
        return result

    # ─────────────────────────────────────────────────────────
    # ReAct graph
    # ─────────────────────────────────────────────────────────

    async def _run_react(
        self,
        entries: list[dict],
        dblp_tools: list,
        scholar_tools: list,
    ) -> dict:
        """
        Build and run the LangGraph ReAct loop.

        Graph structure:
          START → llm_node → should_continue?
                                 ├── "tools" → tool_node → llm_node
                                 └── END

        For zero_shot: no tools are bound so the graph goes directly
          START → llm_node → END  (single pass, no tool calls)

        For rag / cot: tools are bound and the graph loops until the
          LLM produces a final response with no remaining tool calls.
        """

        # ── Bind tools based on strategy ─────────────────────
        # zero_shot: no tools — LLM cannot search, pure knowledge
        # rag / cot: full DBLP + Scholar access
        if self.strategy == PromptStrategy.ZERO_SHOT:
            all_tools      = []
            llm_with_tools = self.llm   # no tool binding
        else:
            all_tools      = dblp_tools + scholar_tools
            llm_with_tools = self.llm.bind_tools(all_tools)

        # Scale tool budget so medium-sized runs (e.g., 50 entries) do not
        # terminate early before the agent has covered all entries.
        tool_budget = MAX_REACT_ITERATIONS
        if all_tools:
          # Empirically: ~1-3 tool calls per entry on healthy runs.
          # Use generous headroom for retries and Scholar fallbacks.
          tool_budget = max(MAX_REACT_ITERATIONS, len(entries) * 6)

        # ── Node: LLM reasons and decides next action ─────────
        async def llm_node(state: ReactState) -> dict:
            response = await _ainvoke_with_retry(
                llm_with_tools,
                state["messages"],
                attempts=4,
                base_delay=1.0,
            )
            return {"messages": [response]}

        # ── Node: execute the tool calls the LLM requested ────
        tool_node = ToolNode(all_tools) if all_tools else None

        # ── Edge: loop or finish ──────────────────────────────
        def should_continue(state: ReactState) -> str:
            last = state["messages"][-1]

            # Stop deterministically once the tool-call cap is reached.
            # LangGraph recursion_limit counts node transitions, not tool calls,
            # so we enforce the tool budget explicitly here.
            tool_messages_so_far = sum(
                1 for msg in state["messages"] if isinstance(msg, ToolMessage)
            )
            if tool_messages_so_far >= tool_budget:
                return END

            exhausted_messages = sum(
                1
                for msg in state["messages"]
                if isinstance(msg, ToolMessage) and "[EXHAUSTED]" in str(msg.content)
            )
            if exhausted_messages >= 2:
                return END

            if (
                all_tools
                and isinstance(last, AIMessage)
                and getattr(last, "tool_calls", None)
            ):
                return "tools"
            return END

        # ── Build graph ───────────────────────────────────────
        graph = StateGraph(ReactState)
        graph.add_node("llm", llm_node)
        graph.add_edge(START, "llm")

        if all_tools and tool_node:
            graph.add_node("tools", tool_node)
            graph.add_conditional_edges(
                "llm",
                should_continue,
                {"tools": "tools", END: END},
            )
            graph.add_edge("tools", "llm")
        else:
            # zero_shot: single pass, no tool loop
            graph.add_edge("llm", END)

        app = graph.compile()

        # ── Initial prompt ────────────────────────────────────
        system_prompt = STRATEGY_SYSTEM_PROMPTS[self.strategy]

        if self.strategy == PromptStrategy.ZERO_SHOT:
            user_content = (
                f"Validate these {len(entries)} BibTeX entries using your "
                f"pre-trained knowledge only. You have no search tools.\n\n"
                f"```json\n{json.dumps(entries, indent=2, ensure_ascii=False)}\n```"
            )
        else:
            user_content = (
                f"Validate these {len(entries)} BibTeX entries. "
                f"Use your tools efficiently — one DBLP call per entry is "
                f"usually enough. Only call Scholar when DBLP gives weak results "
                f"(similarity < 0.75).\n\n"
                f"```json\n{json.dumps(entries, indent=2, ensure_ascii=False)}\n```"
            )

        initial_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]

        # ── Run and stream ────────────────────────────────────
        strategy_label = f"REACT/{self.strategy.value}"
        if self.strategy == PromptStrategy.ZERO_SHOT:
            print(f"  Running [{strategy_label}] — no tools, single pass ...")
        else:
          print(f"  Running [{strategy_label}] — max {tool_budget} tool calls ...")

        tool_call_count = 0
        final_messages  = []

        recursion_limit = (tool_budget * 2) + REACT_RECURSION_BUFFER

        async for chunk in app.astream(
            {"messages": initial_messages},
            config={"recursion_limit": recursion_limit},
        ):
            for node_name, node_output in chunk.items():
                msgs = node_output.get("messages", [])
                if node_name == "tools":
                    tool_call_count += len(msgs)
                    if tool_call_count % 10 == 0:
                        print(f"    ... {tool_call_count} tool calls completed")
                final_messages = msgs

        print(f"  Total tool calls: {tool_call_count}")
        if all_tools and tool_call_count >= tool_budget:
          print(
            "  Warning: ReAct run reached the tool-call budget; "
            "results may be partial for this batch."
          )

        # ── Extract final LLM text ────────────────────────────
        raw_text = ""
        for msg in reversed(final_messages):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                raw_text = _extract_text(msg)
                break

        # If we stopped exactly at the tool budget, the last AI message may still
        # include tool_calls. Use its text as a best-effort fallback.
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
            "structured":      structured,
            "raw_data":        raw_data,
            "total_entries":   len(entries),
            "tool_calls":      tool_call_count,
        }

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _rebuild_raw_data(
        entries: list[dict],
        structured: list[dict],
    ) -> list[dict]:
        """
        Rebuild raw_data in the shape CorrectionAgent and EvaluationAgent
        expect: [{entry, dblp_hits, scholar_hits}, ...]

        Since the ReAct LLM managed its own tool calls internally, we
        reconstruct a minimal dblp_hits entry from the suggested_fixes
        the LLM produced so downstream agents have something to compare.
        """
        result_map = {r.get("entry_id"): r for r in structured if isinstance(r, dict)}
        raw_data   = []

        for entry in entries:
            entry_id = entry.get("id", "")
            result   = result_map.get(entry_id, {})

            dblp_hit = {}
            if result.get("suggested_fixes"):
                fixes    = result["suggested_fixes"]
                dblp_hit = {
                    "title":            fixes.get("title",   entry.get("title", "")),
                    "authors":          fixes.get("author",  entry.get("author", "")),
                    "year":             fixes.get("year",    entry.get("year", "")),
                    "venue":            fixes.get("journal", entry.get(
                                            "journal", entry.get("booktitle", ""))),
                    "similarity_score": result.get("confidence", 0.0),
                }

            raw_data.append({
                "entry":        entry,
                "dblp_hits":    [dblp_hit] if dblp_hit else [],
                "scholar_hits": [],
                "react_result": result,
            })

        return raw_data

    @staticmethod
    def _split_response(raw_text: str) -> tuple[str, list[dict]]:
        """
        Split LLM response into (markdown_part, structured_results).
        Handles <think> blocks and ```json fences.
        """
        # Strip <think>...</think> reasoning blocks (some models emit these)
        if "<think>" in raw_text:
            raw_text = raw_text.split("</think>")[-1].strip()

        structured = []

        if "```json" in raw_text:
          parts      = raw_text.split("```json", 1)
          markdown   = parts[0].strip()
          json_block = parts[1].split("```")[0].strip()
          try:
            data = json.loads(json_block)
            structured, schema_errors = parse_validation_results(data)
            if schema_errors:
              print(f"\n  [DEBUG] Validation schema warnings: {len(schema_errors)}")
          except json.JSONDecodeError:
            pass
        else:
            markdown = raw_text

        return markdown, structured