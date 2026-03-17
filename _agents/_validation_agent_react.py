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
from enum import Enum
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import _extract_text

load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


MAX_REACT_ITERATIONS = 120   # hard cap on tool calls in a single validation run
REACT_RECURSION_BUFFER = 20  # extra graph steps for START/END + finalization overhead


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
  - valid          : all major fields appear correct based on your knowledge
  - partially_valid: paper likely exists but one or more fields are wrong
  - invalid        : paper appears fabricated / non-existent

Produce a complete markdown report grouped by status with a summary
statistics table and per-entry details.

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

1. SEARCH — call dblp_fuzzy_title_search with the entry title.
   - similarity >= 0.75  -> strong evidence, no need for Scholar.
   - similarity < 0.75   -> call google_scholar_search as fallback.
   - Scholar also fails  -> try DBLP again with a shorter title (3-5 keywords).
   - All searches fail   -> mark as invalid.

2. COMPARE — check each field against the best match:
   title, author, year, journal / booktitle / venue.

3. ASSIGN:
   - valid          : strong match and all major fields are correct
   - partially_valid: paper exists but 1+ fields are wrong
   - invalid        : no credible match found anywhere

EFFICIENCY RULES:
  - Do NOT call Scholar if DBLP similarity is already >= 0.75
  - One DBLP call per entry is usually enough
  - Only retry with a shorter query if the full-title search returns nothing

Produce a complete markdown report with summary table and per-entry
details showing which evidence source was used and what issues were found.

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

COT_SYSTEM = """You are an expert BibTeX validation assistant with access to
DBLP and Google Scholar search tools.

You will receive a list of BibTeX entries. For EACH entry you must
reason step by step before assigning a verdict:

Step 1 — SEARCH
  Call dblp_fuzzy_title_search with the entry title.
  If similarity < 0.75, call google_scholar_search as fallback.
  If all searches fail, the entry is invalid — stop here.

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
  - valid          : strong match, all fields correct
  - partially_valid: paper exists, 1+ fields are wrong
  - invalid        : no credible match found

EFFICIENCY RULES:
  - Do NOT call Scholar if DBLP similarity is already >= 0.75
  - One DBLP call per entry is usually enough

Show your step-by-step reasoning for each entry, then produce a full
markdown report with summary table and per-entry details.

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

        # Ollama backend — uncomment to switch
        self.llm = ChatOllama(
            model="qwen3-coder:480b-cloud",
            base_url="https://ollama.com",
            temperature=0.1,
            client_kwargs={
                "headers": {"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"}
            },
        )

        # HuggingFace backend — uncomment to switch
        # self.llm = ChatHuggingFace(
        #     llm=HuggingFaceEndpoint(
        #         repo_id="openai/gpt-oss-20b",
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

        return await self._run_react(entries, dblp_tools, scholar_tools)

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

        # ── Node: LLM reasons and decides next action ─────────
        async def llm_node(state: ReactState) -> dict:
            response = await llm_with_tools.ainvoke(state["messages"])
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
            if tool_messages_so_far >= MAX_REACT_ITERATIONS:
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
            print(f"  Running [{strategy_label}] — max {MAX_REACT_ITERATIONS} tool calls ...")

        tool_call_count = 0
        final_messages  = []

        recursion_limit = (MAX_REACT_ITERATIONS * 2) + REACT_RECURSION_BUFFER

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
                data       = json.loads(json_block)
                structured = data.get("results", [])
            except json.JSONDecodeError:
                pass
        else:
            markdown = raw_text

        return markdown, structured