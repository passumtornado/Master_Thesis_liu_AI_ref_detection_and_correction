"""
LLM-Driven Validation Agent
-----------------------------
Minimal, focused agent with maximum delegation to LLM.

Two core responsibilities:
  1. _gather_dblp_data()  — Collect raw DBLP data only (no scoring/classification)
  2. _generate_report()   — Pass all data to LLM; LLM handles everything else

The LLM receives:
  - All BibTeX entries with their metadata
  - Raw DBLP search results for each entry
  - Classification rules and output format

The LLM produces:
  - Complete markdown validation report
  - Scoring and classification
  - Issue detection and suggestions
  - Statistics table
"""

import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from utils.google_scholar import search_google_scholar


MIN_ACCEPT_SCORE = 0.75
SCHOLAR_ACCEPT_SCORE = 0.85

class LLMValidationAgent:
    """Minimal agent: collect DBLP data, hand to LLM for everything else."""

    # def __init__(self, mcp_config_path: str, model: str = "qwen3-coder:480b-cloud"):
    #     self.mcp_config_path = mcp_config_path
    #     self.llm = ChatOllama(
    #         model=model,
    #         base_url="https://ollama.com",
    #         temperature=0.1,
    #         #add the API key for authentication
    #         client_kwargs={
    #             "headers": {"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"}
    #         }
    #     )
    
    # let try with Model from langchain-HuggingFaceHub
    def __init__(self, mcp_config_path: str, model: str = "google/flan-t5-xxl"):
        self.mcp_config_path = mcp_config_path
        self.llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
            repo_id="openai/gpt-oss-20b",
            task="text-generation",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            max_new_tokens=2048,
            temperature=0.1,
            )
        )

    # ── Core Responsibility 1: Gather DBLP data only ───────────────────────

    async def _gather_dblp_data(self, tools: dict, entries: list[dict]) -> list[dict]:
        """
        For each entry, call fuzzy_title_search and collect raw results.
        NO scoring, NO classification — just raw data collection.
        
        Returns: [{entry, dblp_hits, scholar_hits}, ...]
        """
        gathered = []

        for entry in entries:
            title = entry.get("title", "")
            year = str(entry.get("year", ""))

            kwargs = {
                "title": title,
                "similarity_threshold": 0.20,
                "max_results": 3,
                "include_bibtex": False,
            }
            if year:
                try:
                    y = int(year)
                    kwargs["year_from"] = y - 1
                    kwargs["year_to"] = y + 1
                except ValueError:
                    pass

            dblp_hits = []
            try:
                tool = tools.get("fuzzy_title_search")
                result = await tool.ainvoke(kwargs)
                
                # Unwrap LangChain message structure: [{"type": "text", "text": "..."}]
                if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                    if "text" in result[0]:
                        result = result[0]["text"]
                
                # Try JSON parsing first
                if isinstance(result, str):
                    try:
                        result = json.loads(result)
                    except (json.JSONDecodeError, TypeError):
                        # Fall back to parsing formatted text
                        result = self._parse_dblp_text_response(result)
                
                # Extract results from various structures
                if isinstance(result, list):
                    dblp_hits = result
                elif isinstance(result, dict):
                    dblp_hits = result.get("results", result.get("publications", []))
                    
            except Exception as e:
                print(f"  [DBLP error] {entry.get('id', '?')}: {e}")

            scholar_hits = []
            best_dblp_score = 0.0
            if dblp_hits:
                try:
                    best_dblp_score = max(float(hit.get("similarity_score", hit.get("similarity", 0.0))) for hit in dblp_hits)
                except (TypeError, ValueError):
                    best_dblp_score = 0.0

            if best_dblp_score < MIN_ACCEPT_SCORE:
                scholar_hits = search_google_scholar(
                    title=title,
                    author=entry.get("author", ""),
                    year=year,
                    max_results=3,
                )

            gathered.append({"entry": entry, "dblp_hits": dblp_hits, "scholar_hits": scholar_hits})
            print(
                f"  fetched: {entry.get('id', '?')} — {len(dblp_hits)} DBLP hit(s), "
                f"{len(scholar_hits)} Google Scholar hit(s)"
            )

        return gathered

    @staticmethod
    def _parse_dblp_text_response(text: str) -> list[dict]:
        """Parse DBLP MCP tool's formatted text response into structured data."""
        if not text or "Found" not in text:
            return []
        
        results = []
        lines = text.split("\n")
        current_item = None
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_item:
                    results.append(current_item)
                    current_item = None
                continue
            
            # Parse title line: "N. Title [Similarity: X.XX]"
            if line and line[0].isdigit() and "[Similarity:" in line:
                if current_item:
                    results.append(current_item)
                
                parts = line.split("[Similarity:")
                title_part = parts[0].split(". ", 1)
                title = title_part[1].strip() if len(title_part) > 1 else line
                
                sim_str = parts[1].split("]")[0].strip()
                try:
                    similarity = float(sim_str)
                except:
                    similarity = 0.0
                
                current_item = {
                    "title": title,
                    "similarity_score": similarity,
                    "authors": "",
                    "venue": "",
                    "year": "",
                }
            elif current_item and line.startswith("Authors:"):
                current_item["authors"] = line.replace("Authors:", "").strip()
            elif current_item and line.startswith("Venue:"):
                venue_str = line.replace("Venue:", "").strip()
                if "(" in venue_str and ")" in venue_str:
                    venue_part, year_part = venue_str.rsplit("(", 1)
                    current_item["venue"] = venue_part.strip()
                    current_item["year"] = year_part.rstrip(")").strip()
                else:
                    current_item["venue"] = venue_str
        
        if current_item:
            results.append(current_item)
        
        return results

    # ── Core Responsibility 2: Generate report via LLM ──────────────────────

    async def _generate_report(self, gathered: list[dict]) -> str:
        """
        Pass all entries + raw DBLP results to LLM in one shot.
        LLM handles: scoring, classification, issue detection, markdown generation.
        """
        
        # Serialize all data into JSON block
        data_block = json.dumps(gathered, indent=2, ensure_ascii=False)

        system = SystemMessage(content="""
You are an expert BibTeX validation assistant.

You will receive BibTeX entries with supporting evidence from:

- DBLP hits (`dblp_hits`)
- Google Scholar fallback hits (`scholar_hits`)

Your task for each entry:

1. Compare title, authors, year, and venue across entry metadata and available evidence.
2. Prefer DBLP evidence when strong and consistent.
3. Use Google Scholar evidence when DBLP evidence is weak or absent.
4. Assign one status:
   - valid
   - partially_valid
   - invalid
5. Provide a confidence score from 0.0 to 1.0.
6. List concrete issues and practical fix suggestions.

Decision guidance:

- Treat a high-similarity match with aligned title/authors/year as strong evidence.
- If evidence exists but some fields conflict, mark partially_valid.
- If no reliable evidence is found, mark invalid.
- Accept common venue abbreviations when clearly equivalent.

Output requirements:

- Return a complete markdown validation report only.
- Include:
  - summary table (total, valid, partially valid, invalid, percentages)
  - grouped sections by status
  - per-entry details: entry id, key fields, best evidence, issues, suggestions, confidence
- Do not output JSON.
- Do not include any text before or after the markdown report.
""")

        human = HumanMessage(content=f"""
Please validate these BibTeX entries against DBLP and generate the full markdown report:

```json
{data_block}
```

Produce the markdown report now.
""")

        response = await self.llm.ainvoke([system, human])
        return response.content.strip()

    # ── Main orchestrator ─────────────────────────────────────────────────

    async def validate_entries(self, entries: list[dict]) -> dict:
        """
        Main entry point: orchestrate data gathering and LLM report generation.
        """
        print(f"\n{'='*60}")
        print(f"LLM VALIDATION AGENT — {len(entries)} entries")
        print(f"{'='*60}\n")

        # Load MCP config
        with open(self.mcp_config_path, "r") as f:
            config = json.load(f)
        
        mcp_servers_config = config.get("mcpServers", config)
        client = MultiServerMCPClient(mcp_servers_config)
        
        # Get DBLP tools
        tools_list = await client.get_tools(server_name="mcp-dblp")
        tools = {t.name: t for t in tools_list}
        print(f"✓ DBLP tools loaded: {list(tools.keys())}\n")

        # Step 1: Gather DBLP data
        print("Gathering DBLP data...")
        gathered = await self._gather_dblp_data(tools, entries)

        # Step 2: Generate markdown report via LLM
        print("\nGenerating validation report with LLM...")
        markdown = await self._generate_report(gathered)

        print("\n✓ Report generation complete\n")

        return {
            "markdown_report": markdown,
            "raw_data": gathered,
            "total_entries": len(entries),
        }
