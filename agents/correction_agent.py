"""
BibTeX Correction Agent (LLM-Driven)
-------------------------------------
Minimal Python wrapper for LLM-driven corrections.

Architecture:
  1. Fetch corrected BibTeX from DBLP for each entry (minimal Python)
  2. Pass to LLM: original entry + issues + DBLP corrected version
  3. LLM decides what to correct and generates corrected entries + report

Input:
  - Raw validation data with DBLP matches
  - Validation markdown report for context

Output:
  - corrections_summary.md (markdown report from LLM)
  - corrections_metadata.json (correction details)
  - corrected.bib (corrected BibTeX file)
"""

import json
import os
from pathlib import Path
from typing import Optional

# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient


class CorrectionAgent:
    """Minimal Python + LLM-driven BibTeX correction."""

    def __init__(self, output_dir: str = "corrections"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize LLM (local Ollama)
        self.llm = ChatOllama(
            model="qwen3-coder:480b-cloud",
            base_url="https://ollama.com",
            temperature=0.1,
            client_kwargs={
                "headers": {"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"}
            }
        )
        
        #HuggingFaceHub alternative
    #     self.llm = ChatHuggingFace(
    #     llm=HuggingFaceEndpoint(
    #         repo_id="openai/gpt-oss-20b",
    #         task="text-generation",
    #         huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    #         max_new_tokens=2048,
    #         temperature=0.1,
    #      )
    #   )
    async def correct_entries(
        self,
        raw_data: list[dict],
        validation_markdown: str,
    ) -> dict:
        """
        Correct entries using LLM intelligence with DBLP corrected BibTeX as reference.
        
        Flow:
        1. Fetch corrected BibTeX from DBLP for each entry
        2. Pass original + DBLP corrected + issues to LLM
        3. LLM decides what to correct and generates corrected entries + report
        
        Returns:
          {
            "corrected_entries": [...],
            "corrections": [...],
            "correction_summary": str (markdown from LLM),
            "saved_files": [...]
          }
        """
        print(f"\n{'='*60}")
        print(f"CORRECTION AGENT — Fetching DBLP corrections")
        print(f"{'='*60}\n")

        # Fetch DBLP corrected BibTeX for each entry
        correction_data = self._prepare_correction_data(raw_data)

        print(f"\n  [LLM] Generating intelligent corrections...")

        # Send to LLM with all context
        llm_response = await self._generate_corrections_with_llm(
            correction_data=correction_data,
            validation_context=validation_markdown,
        )

        # Parse LLM response
        corrected_entries, corrections_list, markdown_report = self._parse_llm_response(
            llm_response,
            raw_data
        )

        # Generate BibTeX file
        corrected_bib = self._generate_bib(corrected_entries)

        # Save files
        bib_path = self.output_dir / "corrected.bib"
        with open(bib_path, "w", encoding="utf-8") as f:
            f.write(corrected_bib)
        print(f"✓ Saved: {bib_path}")

        md_path = self.output_dir / "corrections_summary.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_report)
        print(f"✓ Saved: {md_path}")

        meta_path = self.output_dir / "corrections_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "corrections": corrections_list,
                "statistics": self._compute_stats(corrections_list),
            }, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved: {meta_path}\n")

        return {
            "corrected_entries": corrected_entries,
            "corrections": corrections_list,
            "correction_summary": markdown_report,
            "saved_files": [str(bib_path), str(md_path), str(meta_path)],
        }

    @staticmethod
    def _prepare_correction_data(raw_data: list[dict]) -> list[dict]:
        """
        Prepare minimal Python: just organize original + DBLP corrected pairs.
        For now, use DBLP best hit as "corrected" reference.
        In production, could fetch full BibTeX from DBLP.
        """
        correction_data = []

        for item in raw_data:
            entry = item["entry"]
            dblp_hits = item["dblp_hits"]

            if not dblp_hits:
                continue

            best_hit = dblp_hits[0]

            correction_data.append({
                "entry_id": entry.get("id"),
                "original_entry": entry,
                "dblp_corrected": best_hit,  # This would be fetched from DBLP in production
                "similarity_score": best_hit.get("similarity_score", 0),
            })

        return correction_data

    async def _generate_corrections_with_llm(
        self,
        correction_data: list[dict],
        validation_context: str,
    ) -> str:
        """
        Pass original + DBLP corrected + issues to LLM.
        LLM decides what to correct and generates output.
        """
        correction_json = json.dumps(correction_data, indent=2, ensure_ascii=False)

        system_prompt = """You are an expert BibTeX reference correction system.

Your task:
1. Compare original BibTeX entries vs DBLP corrected versions
2. Decide intelligently which fields should be corrected (NOT all)
3. Generate corrected entries in BibTeX format
4. Create detailed markdown report

CORRECTION LOGIC:
- Only correct if DBLP value is clearly better (e.g., fixes typos, capitalization, incomplete authors)
- Preserve original if already correct or ambiguous
- Trust DBLP year as authoritative
- Be conservative: confidence must be high

Output Format (MUST be valid JSON):
{
  "markdown_summary": "# Corrections Report\\n\\n## Statistics\\n...",
  "corrected_entries": [
    {
      "entry_id": "key1",
      "corrected_bibtex": "@article{key1,\\n  author = {...}\\n}",
      "corrections_applied": [
        {"field": "title", "old": "...", "new": "...", "reason": "Fix typo"}
      ]
    }
  ]
}

Generate comprehensive markdown_summary with:
- Total entries processed and corrected
- Statistics by field
- Detailed corrections with reasoning"""

        user_message = f"""Analyze and correct these BibTeX entries:

{correction_json}

Validation Context:
{validation_context[:800]}

Generate JSON with corrected entries and detailed report."""

        response = self.llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ])

        response_text = response.content if hasattr(response, 'content') else str(response)
        return response_text

    @staticmethod
    def _parse_llm_response(
        llm_response: str,
        raw_data: list[dict],
    ) -> tuple[list[dict], list[dict], str]:
        """
        Parse LLM JSON response into corrected entries, corrections list, and markdown.
        """
        try:
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = llm_response[json_start:json_end]
                llm_data = json.loads(json_str)
            else:
                llm_data = {"corrected_entries": [], "markdown_summary": ""}
        except (json.JSONDecodeError, ValueError):
            llm_data = {"corrected_entries": [], "markdown_summary": ""}

        corrected_entries = []
        corrections_list = []
        markdown_report = llm_data.get("markdown_summary", "")

        # Build lookup for raw data
        raw_map = {item["entry"].get("id"): item["entry"] for item in raw_data}

        for llm_entry in llm_data.get("corrected_entries", []):
            entry_id = llm_entry.get("entry_id")
            original = raw_map.get(entry_id, {})
            corrections = llm_entry.get("corrections_applied", [])

            corrected_entries.append({
                "entry_id": entry_id,
                "original": original,
                "corrected_bibtex": llm_entry.get("corrected_bibtex", ""),
            })

            if corrections:
                corrections_list.append({
                    "entry_id": entry_id,
                    "corrections": corrections,
                    "num_corrections": len(corrections),
                })

        return corrected_entries, corrections_list, markdown_report

    @staticmethod
    def _compute_stats(corrections_list: list[dict]) -> dict:
        """Compute correction statistics."""
        total_entries = len(corrections_list)
        total_corrections = sum(c.get("num_corrections", 0) for c in corrections_list)
        
        field_counts = {}
        for corr in corrections_list:
            for fix in corr.get("corrections", []):
                field = fix.get("field", "unknown")
                field_counts[field] = field_counts.get(field, 0) + 1

        return {
            "total_corrected": total_entries,
            "total_corrections": total_corrections,
            "corrections_by_field": field_counts,
        }

    @staticmethod
    def _generate_bib(corrected_entries: list[dict]) -> str:
        """Generate corrected BibTeX file from LLM output."""
        lines = []

        for item in corrected_entries:
            # If LLM provided corrected_bibtex, use it directly
            if item.get("corrected_bibtex"):
                lines.append(item["corrected_bibtex"])
            else:
                # Fallback: generate from dict
                entry = item.get("corrected", item.get("original", {}))
                entry_type = entry.get("type", "article").lower()
                entry_id = entry.get("id", "unknown")

                lines.append(f"@{entry_type}{{{entry_id},")
                for field in ["author", "title", "year", "journal", "booktitle", 
                             "publisher", "doi", "url", "pages", "volume", "number", "note"]:
                    if field in entry and entry[field]:
                        value = str(entry[field]).replace("{", "{{").replace("}", "}}")
                        lines.append(f'\t{field} = {{{value}}},')

                if lines[-1].endswith(","):
                    lines[-1] = lines[-1][:-1]
                lines.append("}\n")

        return "\n".join(lines)
