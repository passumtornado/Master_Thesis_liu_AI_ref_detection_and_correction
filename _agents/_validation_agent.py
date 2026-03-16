"""
LLM-Driven Validation Agent
-----------------------------
Collects DBLP and fallback Google Scholar evidence, then delegates
validation classification/report generation to the LLM.

Supports three prompting strategies:
  - zero_shot : LLM relies on pre-trained knowledge only (no DBLP evidence)
  - rag       : LLM receives DBLP/Scholar hits as grounding evidence (default)
  - cot       : LLM reasons field-by-field before producing verdict (CoT + RAG)
"""

import json
import os
import sys
from pathlib import Path
from enum import Enum

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import _extract_text

load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

#model = ChatOpenAI(model="gpt-5.2")

MIN_ACCEPT_SCORE = 0.75
SCHOLAR_ACCEPT_SCORE = 0.85


# ─────────────────────────────────────────────────────────────
# Prompt Strategy
# ─────────────────────────────────────────────────────────────

class PromptStrategy(Enum):
    ZERO_SHOT = "zero_shot"
    RAG       = "rag"
    COT       = "cot"


ZERO_SHOT_SYSTEM = """You are an expert BibTeX validation assistant.

You will receive a list of BibTeX entries with NO external database evidence.
Rely only on your pre-trained knowledge to assess each entry.

For each entry assign:
  - status     : valid | partially_valid | invalid
  - confidence : float [0.0, 1.0]
  - issues     : list of field-level problems detected
  - suggested_fixes : dict of {field: corrected_value}

Rules:
  - valid         : all major fields appear correct based on your knowledge
  - partially_valid: paper likely exists but one or more fields are wrong
  - invalid       : paper appears fabricated / non-existent

After your analysis produce a complete markdown report grouped by status
with a summary statistics table.

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

RAG_SYSTEM = """You are an expert BibTeX validation assistant.

You will receive BibTeX entries together with matching records retrieved
from DBLP and optionally Google Scholar. Use the retrieved evidence as
ground truth to validate each entry.

For each entry:
1. Compare title, authors, year, and venue against the DBLP/Scholar hits.
2. Use Scholar hits only when DBLP confidence is weak (similarity < 0.75).
3. Assign status:
   - valid         : strong match found and all major fields are correct
   - partially_valid: paper exists but one or more fields are wrong
   - invalid       : paper appears fabricated — no credible match anywhere
4. Generate confidence score [0.0, 1.0].
5. List field-level issues and propose concrete fixes.

Critical rule: if credible evidence the paper exists is found, do NOT mark
it invalid. Use partially_valid and describe corrections needed.

After your analysis produce a complete markdown report with a summary
statistics table, grouped sections by status, and per-entry details
including the DBLP/Scholar evidence used.

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

COT_SYSTEM = """You are an expert BibTeX validation assistant.

You will receive BibTeX entries together with DBLP and Google Scholar hits.
For EACH entry reason step by step through the evidence before deciding:

Step 1 — Title check:
  Does the title closely match any DBLP/Scholar hit?
  Note typos, truncation, or word order differences.

Step 2 — Author check:
  Are author names spelled correctly and complete?
  Note missing authors, spelling errors, or name order issues.

Step 3 — Year check:
  Does the publication year match the evidence?
  Flag if off by more than 1.

Step 4 — Venue check:
  Does the journal/booktitle match the evidence?
  Note abbreviation mismatches or entirely wrong venue.

Step 5 — Final verdict:
  Based on steps 1–4 assign:
  - status     : valid | partially_valid | invalid
  - confidence : float [0.0, 1.0]
  - issues     : list of specific field errors found
  - suggested_fixes : dict of {field: corrected_value}

Show your step-by-step reasoning for each entry, then produce a full
markdown report with a summary statistics table and per-entry details.

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
# Agent
# ─────────────────────────────────────────────────────────────

class LLMValidationAgent:
    """Minimal agent: collect evidence and let LLM do reasoning/reporting."""

    def __init__(
        self,
        mcp_config_path: str,
        strategy: PromptStrategy = PromptStrategy.RAG,
        model: str = "gpt-oss:120b-cloud" #"qwen3-coder:480b-cloud",
    ):
        self.mcp_config_path = mcp_config_path
        self.strategy        = strategy

        # self.llm = ChatOllama(
        #     model=model,
        #     base_url="https://ollama.com",
        #     temperature=0.1,
        #     client_kwargs={
        #         "headers": {"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"}
        #     },
        # )
        #OpenAI backend (uncomment to switch)
        # self.llm = ChatOpenAI(
        #     model="gpt-5.2",
        #     temperature=0.1,
        #     openai_api_key=os.getenv("OPENAI_API_KEY"),
        # )
        
        # GOOGLE GEMINI BACKEND (uncomment to switch)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-pro-preview",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )   
         
        # Optional HuggingFace backend — uncomment to switch
        # self.llm = ChatHuggingFace(
        #     llm=HuggingFaceEndpoint(
        #         repo_id="openai/gpt-oss-20b",
        #         task="text-generation",
        #         huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        #         max_new_tokens=2048,
        #         temperature=0.1,
        #     )
        # )

    # ── public ────────────────────────────────────────────────

    async def validate_entries(self, entries: list[dict]) -> dict:
        """Main entry point for LLM-driven validation."""
        print(f"\n{'='*60}")
        print(f"LLM VALIDATION AGENT [{self.strategy.value.upper()}] - {len(entries)} entries")
        print(f"{'='*60}\n")

        with open(self.mcp_config_path, "r") as f:
            config = json.load(f)

        mcp_servers_config = config.get("mcpServers", config)
        client = MultiServerMCPClient(mcp_servers_config)

        dblp_tools_list = await client.get_tools(server_name="mcp-dblp")
        dblp_tools = {t.name: t for t in dblp_tools_list}

        scholar_tools = {}
        try:
            scholar_tools_list = await client.get_tools(server_name="mcp-scholar")
            scholar_tools = {t.name: t for t in scholar_tools_list}
        except Exception as e:
            print(f"  Scholar MCP tools unavailable: {e}")

        print(f"  DBLP tools   : {list(dblp_tools.keys())}")
        print(f"  Scholar tools: {list(scholar_tools.keys())}\n")

        print("  Gathering DBLP/Scholar evidence …")
        gathered = await self._gather_dblp_data(dblp_tools, scholar_tools, entries)

        print(f"  Generating validation report [{self.strategy.value}] …")
        markdown, structured = await self._generate_report(gathered)

        return {
            "markdown_report": markdown,
            "structured":      structured,   # list of per-entry result dicts
            "raw_data":        gathered,
            "total_entries":   len(entries),
        }

    # ── evidence gathering ────────────────────────────────────

    async def _gather_dblp_data(
        self,
        dblp_tools: dict,
        scholar_tools: dict,
        entries: list[dict],
    ) -> list[dict]:
        """Collect DBLP hits and optional Scholar fallback per entry."""
        gathered = []

        for entry in entries:
            title = entry.get("title", "")
            year  = str(entry.get("year", ""))

            kwargs = {
                "title":               title,
                "similarity_threshold": 0.20,
                "max_results":          3,
                "include_bibtex":       False,
            }
            if year:
                try:
                    y = int(year)
                    kwargs["year_from"] = y - 1
                    kwargs["year_to"]   = y + 1
                except ValueError:
                    pass

            dblp_hits = await self._dblp_search(
                dblp_tools, kwargs, entry_id=entry.get("id", "?")
            )

            # Retry without year constraints to recover entries with wrong year
            if not dblp_hits:
                retry_kwargs = {
                    k: v for k, v in kwargs.items()
                    if k not in ("year_from", "year_to")
                }
                dblp_hits = await self._dblp_search(
                    dblp_tools, retry_kwargs, entry_id=entry.get("id", "?")
                )

            scholar_hits    = []
            best_dblp_score = 0.0
            if dblp_hits:
                try:
                    best_dblp_score = max(
                        float(hit.get("similarity_score", hit.get("similarity", 0.0)))
                        for hit in dblp_hits
                        if isinstance(hit, dict)
                    )
                except (TypeError, ValueError):
                    best_dblp_score = 0.0

            if best_dblp_score < MIN_ACCEPT_SCORE:
                scholar_hits = await self._scholar_search(
                    scholar_tools=scholar_tools,
                    title=title,
                    author=entry.get("author", ""),
                    year=year,
                )
                if not scholar_hits:
                    scholar_hits = await self._scholar_search(
                        scholar_tools=scholar_tools,
                        title=title,
                        author=entry.get("author", ""),
                        year="",
                    )

            gathered.append({
                "entry":        entry,
                "dblp_hits":    dblp_hits,
                "scholar_hits": scholar_hits,
            })
            print(
                f"    {entry.get('id', '?')} — "
                f"{len(dblp_hits)} DBLP hit(s), {len(scholar_hits)} Scholar hit(s)"
            )

        return gathered

    async def _dblp_search(
        self, dblp_tools: dict, kwargs: dict, entry_id: str
    ) -> list[dict]:
        """Call DBLP fuzzy_title_search and normalise response."""
        try:
            tool = dblp_tools.get("fuzzy_title_search")
            if not tool:
                return []

            result = await tool.ainvoke(kwargs)

            if isinstance(result, list) and result and isinstance(result[0], dict):
                if "text" in result[0]:
                    result = result[0]["text"]

            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except (json.JSONDecodeError, TypeError):
                    result = self._parse_dblp_text_response(result)

            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                hits = result.get("results", result.get("publications", []))
                return hits if isinstance(hits, list) else []
            return []
        except Exception as e:
            print(f"    [DBLP error] {entry_id}: {e}")
            return []

    async def _scholar_search(
        self,
        scholar_tools: dict,
        title: str,
        author: str,
        year: str,
        max_results: int = 3,
    ) -> list[dict]:
        """Call google_scholar_search via MCP and normalise response."""
        tool = scholar_tools.get("google_scholar_search")
        if not tool or not title:
            return []
        try:
            result = await tool.ainvoke({
                "title":       title,
                "author":      author,
                "year":        year,
                "max_results": max_results,
            })

            if isinstance(result, list) and result and isinstance(result[0], dict):
                if "text" in result[0]:
                    result = result[0]["text"]

            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except (json.JSONDecodeError, TypeError):
                    return []

            if isinstance(result, dict):
                results = result.get("results", [])
                return results if isinstance(results, list) else []
            if isinstance(result, list):
                return result
            return []
        except Exception as e:
            print(f"    [Scholar error] {e}")
            return []

    # ── report generation ─────────────────────────────────────

    async def _generate_report(self, gathered: list[dict]) -> tuple[str, list[dict]]:
        """
        Pass evidence to LLM using the configured strategy.
        Returns (markdown_string, structured_results_list).
        """
        # Zero-shot: strip DBLP/Scholar hits so LLM cannot see them
        if self.strategy == PromptStrategy.ZERO_SHOT:
            payload = [{"entry": item["entry"]} for item in gathered]
        else:
            payload = gathered

        messages = [
            SystemMessage(content=STRATEGY_SYSTEM_PROMPTS[self.strategy]),
            HumanMessage(content=(
                "Please validate these BibTeX entries:\n\n"
                f"```json\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n```"
            )),
        ]

        response  = await self.llm.ainvoke(messages)
        raw_text  = _extract_text(response)
        # raw_text  = response.content.strip()

        # Split markdown narrative from appended JSON block
        markdown, structured = self._split_response(raw_text)
        return markdown, structured

    @staticmethod
    def _split_response(raw_text: str) -> tuple[str, list[dict]]:
        """
        Split LLM response into (markdown_part, structured_results).
        The LLM is instructed to append a ```json block at the end.
        """
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

    @staticmethod
    def _parse_dblp_text_response(text: str) -> list[dict]:
        """Parse formatted DBLP text response into structured hits."""
        if not text or "Found" not in text:
            return []

        results      = []
        lines        = text.split("\n")
        current_item = None

        for line in lines:
            line = line.strip()
            if not line:
                if current_item:
                    results.append(current_item)
                    current_item = None
                continue

            if line and line[0].isdigit() and "[Similarity:" in line:
                if current_item:
                    results.append(current_item)

                parts      = line.split("[Similarity:")
                title_part = parts[0].split(". ", 1)
                title      = title_part[1].strip() if len(title_part) > 1 else line
                sim_str    = parts[1].split("]")[0].strip()
                try:
                    similarity = float(sim_str)
                except Exception:
                    similarity = 0.0

                current_item = {
                    "title":            title,
                    "similarity_score": similarity,
                    "authors":          "",
                    "venue":            "",
                    "year":             "",
                }
            elif current_item and line.startswith("Authors:"):
                current_item["authors"] = line.replace("Authors:", "").strip()
            elif current_item and line.startswith("Venue:"):
                venue_str = line.replace("Venue:", "").strip()
                if "(" in venue_str and ")" in venue_str:
                    venue_part, year_part       = venue_str.rsplit("(", 1)
                    current_item["venue"]       = venue_part.strip()
                    current_item["year"]        = year_part.rstrip(")").strip()
                else:
                    current_item["venue"] = venue_str

        if current_item:
            results.append(current_item)

        return results