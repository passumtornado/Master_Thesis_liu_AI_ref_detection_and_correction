"""
BibTeX Correction Agent (LLM-Driven)
-------------------------------------
Supports three prompting strategies:
  - zero_shot : LLM corrects from pre-trained knowledge only (no DBLP evidence)
  - rag       : LLM uses DBLP hits as ground truth (default)
  - cot       : LLM reasons field-by-field before correcting (CoT + RAG)

Fixes applied vs original:
  - invoke → ainvoke (async-safe)
  - PromptStrategy enum wired in
  - _extract_json handles ```json fences (needed for CoT)
  - corrections_list["corrected"] now populated from corrected BibTeX fields
  - output_dir scoped per strategy
"""

import json
import os
import sys
from enum import Enum
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_ollama import ChatOllama
from utils import _extract_text

# ─────────────────────────────────────────────────────────────
# Prompt Strategy
# ─────────────────────────────────────────────────────────────

class PromptStrategy(Enum):
    ZERO_SHOT = "zero_shot"
    RAG       = "rag"
    COT       = "cot"


ZERO_SHOT_PROMPT = """You are an expert BibTeX reference correction system.

You will receive original BibTeX entries with NO external database evidence.
Rely only on your pre-trained knowledge to identify and correct wrong fields.

For each entry:
1. Identify which fields are likely incorrect.
2. Propose corrected values based on your knowledge.
3. Generate the corrected BibTeX string.

You MUST respond with valid JSON in EXACTLY this structure — no extra text:
```json
{
  "markdown_summary": "<markdown correction report>",
  "corrected_entries": [
    {
      "entry_id": "...",
      "corrected_bibtex": "@article{...}",
      "corrections_applied": [
        {"field": "...", "original": "...", "corrected": "..."}
      ]
    }
  ]
}
```
"""

RAG_PROMPT = """You are an expert BibTeX reference correction system.

You will receive original BibTeX entries together with the best matching
record from DBLP. Use the DBLP record as ground truth.

For each entry:
1. Compare each field against the DBLP record.
2. Correct only fields that genuinely differ — do NOT blindly copy all DBLP fields.
3. Preserve fields that are already correct.
4. Generate the corrected BibTeX string.

You MUST respond with valid JSON in EXACTLY this structure — no extra text:
```json
{
  "markdown_summary": "<markdown correction report>",
  "corrected_entries": [
    {
      "entry_id": "...",
      "corrected_bibtex": "@article{...}",
      "corrections_applied": [
        {"field": "...", "original": "...", "corrected": "..."}
      ]
    }
  ]
}
```
"""

COT_PROMPT = """You are an expert BibTeX reference correction system.

For EACH entry reason step by step through the DBLP evidence before correcting:

Step 1 — Title: Does it match the DBLP hit? Note any differences.
Step 2 — Authors: Are names spelled correctly and complete?
Step 3 — Year: Does it match DBLP? Flag if off by more than 1.
Step 4 — Venue: Does journal/booktitle match DBLP?
Step 5 — Decision: Which fields need correction based on steps 1–4?
           Only correct fields with clear evidence of error.

Show your reasoning, then output valid JSON in EXACTLY this structure:
```json
{
  "markdown_summary": "<markdown correction report including your reasoning>",
  "corrected_entries": [
    {
      "entry_id": "...",
      "corrected_bibtex": "@article{...}",
      "corrections_applied": [
        {"field": "...", "original": "...", "corrected": "..."}
      ]
    }
  ]
}
```
"""

STRATEGY_PROMPTS = {
    PromptStrategy.ZERO_SHOT: ZERO_SHOT_PROMPT,
    PromptStrategy.RAG:       RAG_PROMPT,
    PromptStrategy.COT:       COT_PROMPT,
}


# ─────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────

class CorrectionAgent:
    """LLM-driven BibTeX correction with swappable prompting strategies."""

    def __init__(
        self,
        output_dir: str = "corrections",
        strategy: PromptStrategy = PromptStrategy.RAG,
    ):
        self.strategy   = strategy
        # Each strategy writes to its own subfolder — no overwrites across runs
        self.output_dir = Path(output_dir) / strategy.value
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.llm = ChatOllama(
                model= "qwen3-coder:480b-cloud",
                base_url="https://ollama.com",
                temperature=0.1,
                client_kwargs={
                    "headers": {"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"}
                },
            )
        
        #OpenAI fallback — uncomment to switch
        # self.llm = ChatOpenAI(
        #     model="gpt-5.2",
        #     temperature=0.1,
        #     openai_api_key=os.getenv("OPENAI_API_KEY"),
        # )
        
         # GOOGLE GEMINI BACKEND (uncomment to switch)
        # self.llm = ChatGoogleGenerativeAI(
        #     model="gemini-3.1-pro-preview",
        #     temperature=0.1,
        #     google_api_key=os.getenv("GOOGLE_API_KEY"),
        # )
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

    async def correct_entries(
        self,
        raw_data: list[dict],
        validation_markdown: str = "",
    ) -> dict:
        """Generate corrected entries and save correction artefacts."""
        print(f"\n{'='*60}")
        print(f"CORRECTION AGENT [{self.strategy.value.upper()}]")
        print(f"{'='*60}\n")

        correction_data = self._prepare_correction_data(raw_data)

        llm_response = await self._generate_corrections_with_llm(
            correction_data=correction_data,
            validation_context=validation_markdown,
        )

        corrected_entries, corrections_list, markdown_report = (
            self._parse_llm_response(llm_response, raw_data)
        )

        corrected_bib = self._generate_bib(corrected_entries)

        # Save artefacts
        bib_path = self.output_dir / "corrected.bib"
        bib_path.write_text(corrected_bib, encoding="utf-8")

        md_path = self.output_dir / "corrections_summary.md"
        md_path.write_text(markdown_report, encoding="utf-8")

        meta_path = self.output_dir / "corrections_metadata.json"
        meta_path.write_text(
            json.dumps(
                {
                    "strategy":   self.strategy.value,
                    "corrections": corrections_list,
                    "statistics":  self._compute_stats(corrections_list),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print(f"  ✓ corrected.bib            → {bib_path}")
        print(f"  ✓ corrections_summary.md   → {md_path}")
        print(f"  ✓ corrections_metadata.json → {meta_path}")

        return {
            "corrected_entries":  corrected_entries,
            "corrections":        corrections_list,
            "correction_summary": markdown_report,
            "saved_files":        [str(bib_path), str(md_path), str(meta_path)],
        }

    # ── private: data preparation ─────────────────────────────

    @staticmethod
    def _prepare_correction_data(raw_data: list[dict]) -> list[dict]:
        """Prepare original + best DBLP match pairs for LLM correction."""
        correction_data = []

        for item in raw_data:
            if not isinstance(item, dict):
                continue
            entry = item.get("entry", {})
            if not isinstance(entry, dict):
                continue
            dblp_hits = item.get("dblp_hits", [])
            if not isinstance(dblp_hits, list) or not dblp_hits:
                continue
            best_hit = dblp_hits[0]
            if not isinstance(best_hit, dict):
                continue

            correction_data.append({
                "entry_id":        entry.get("id"),
                "original_entry":  entry,
                "dblp_corrected":  best_hit,
                "similarity_score": best_hit.get("similarity_score", 0),
            })

        return correction_data

    # ── private: LLM call ─────────────────────────────────────

    async def _generate_corrections_with_llm(
        self,
        correction_data: list[dict],
        validation_context: str,
    ) -> str:
        """Ask the LLM to generate corrected entries using the active strategy."""

        # Zero-shot: strip DBLP evidence so LLM relies on pre-trained knowledge
        if self.strategy == PromptStrategy.ZERO_SHOT:
            payload = [
                {
                    "entry_id":       d["entry_id"],
                    "original_entry": d["original_entry"],
                }
                for d in correction_data
            ]
        else:
            payload = correction_data  # RAG and CoT receive full DBLP evidence

        response = await self.llm.ainvoke([
            SystemMessage(content=STRATEGY_PROMPTS[self.strategy]),
            HumanMessage(content=(
                "Correct these BibTeX entries:\n\n"
                f"```json\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n```\n\n"
                f"Validation context:\n{validation_context[:800]}"
            )),
        ])
        # update to use _extract_text for consistent response parsing across backends
        raw_text = _extract_text(response)
        return raw_text

    # ── private: response parsing ─────────────────────────────

    @staticmethod
    def _extract_json(llm_response: str) -> dict:
        """
        Robustly extract JSON from LLM response.
        Handles: ```json fences, plain JSON, JSON embedded in prose (CoT).
        """
        # 1. Try fenced ```json block first (most reliable for CoT)
        if "```json" in llm_response:
            fenced = llm_response.split("```json", 1)[1].split("```")[0].strip()
            try:
                return json.loads(fenced)
            except json.JSONDecodeError:
                pass

        # 2. Fall back to outermost { ... }
        json_start = llm_response.find("{")
        json_end   = llm_response.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            try:
                return json.loads(llm_response[json_start:json_end])
            except json.JSONDecodeError:
                pass

        return {"corrected_entries": [], "markdown_summary": ""}

    @staticmethod
    def _parse_bibtex_fields(bibtex_str: str) -> dict:
        """
        Extract a field dict from a BibTeX string.
        Used to populate corrections_list["corrected"] for EvaluationAgent.
        """
        fields = {}
        for line in bibtex_str.splitlines():
            if "=" in line and not line.strip().startswith("@"):
                key, _, value = line.partition("=")
                cleaned = value.strip().strip(",").strip()
                # Remove outer braces or quotes
                if (cleaned.startswith("{") and cleaned.endswith("}")) or \
                   (cleaned.startswith('"') and cleaned.endswith('"')):
                    cleaned = cleaned[1:-1]
                fields[key.strip().lower()] = cleaned
        return fields

    def _parse_llm_response(
        self,
        llm_response: str,
        raw_data: list[dict],
    ) -> tuple[list[dict], list[dict], str]:
        """Parse LLM JSON response into corrected entries and corrections list."""
        llm_data = self._extract_json(llm_response)

        corrected_entries = []
        corrections_list  = []
        markdown_report   = llm_data.get("markdown_summary", "")

        # Build lookup for original entries
        raw_map = {}
        for item in raw_data:
            if not isinstance(item, dict):
                continue
            entry = item.get("entry", {})
            if not isinstance(entry, dict):
                continue
            entry_id = entry.get("id")
            if entry_id:
                raw_map[entry_id] = entry

        llm_corrected = llm_data.get("corrected_entries", [])
        if not isinstance(llm_corrected, list):
            llm_corrected = []

        for llm_entry in llm_corrected:
            if not isinstance(llm_entry, dict):
                continue

            entry_id        = llm_entry.get("entry_id")
            original        = raw_map.get(entry_id, {})
            corrections     = llm_entry.get("corrections_applied", [])
            corrected_bibtex = llm_entry.get("corrected_bibtex", "")

            if not isinstance(corrections, list):
                corrections = []

            # Parse corrected field values from BibTeX string for EvaluationAgent
            corrected_fields = self._parse_bibtex_fields(corrected_bibtex) or original

            corrected_entries.append({
                "entry_id":        entry_id,
                "original":        original,
                "corrected_bibtex": corrected_bibtex,
            })

            corrections_list.append({
                "entry_id":        entry_id,
                "corrected":       corrected_fields,   # ← actual field values
                "changes":         corrections,
                "corrections":     corrections,
                "num_corrections": len(corrections),
                "corrected_bibtex": corrected_bibtex,
            })

        # Fallback: if LLM output is empty/malformed, emit no-op records
        if not corrections_list:
            for item in raw_data:
                if not isinstance(item, dict):
                    continue
                entry    = item.get("entry", {})
                entry_id = entry.get("id") if isinstance(entry, dict) else None
                if not entry_id:
                    continue

                corrected_entries.append({
                    "entry_id":        entry_id,
                    "original":        entry,
                    "corrected_bibtex": "",
                })
                corrections_list.append({
                    "entry_id":        entry_id,
                    "corrected":       entry,
                    "changes":         [],
                    "corrections":     [],
                    "num_corrections": 0,
                    "corrected_bibtex": "",
                })

            if not markdown_report:
                markdown_report = (
                    "# Corrections Report\n\n"
                    "No structured corrections were parsed from the LLM response.\n"
                    "No-op correction records were generated so evaluation can proceed."
                )

        return corrected_entries, corrections_list, markdown_report

    # ── private: stats & BibTeX generation ───────────────────

    @staticmethod
    def _compute_stats(corrections_list: list[dict]) -> dict:
        """Compute aggregate correction statistics."""
        safe_items        = [c for c in corrections_list if isinstance(c, dict)]
        total_entries     = len(safe_items)
        total_corrections = sum(c.get("num_corrections", 0) for c in safe_items)

        field_counts = {}
        for corr in safe_items:
            fixes = corr.get("corrections", [])
            if not isinstance(fixes, list):
                fixes = []
            for fix in fixes:
                if not isinstance(fix, dict):
                    continue
                field = fix.get("field", "unknown")
                field_counts[field] = field_counts.get(field, 0) + 1

        return {
            "total_corrected":      total_entries,
            "total_corrections":    total_corrections,
            "corrections_by_field": field_counts,
        }

    @staticmethod
    def _generate_bib(corrected_entries: list[dict]) -> str:
        """Generate BibTeX text from corrected entries."""
        lines = []

        for item in corrected_entries:
            if not isinstance(item, dict):
                continue

            if item.get("corrected_bibtex"):
                lines.append(item["corrected_bibtex"])
                continue

            entry      = item.get("corrected", item.get("original", {}))
            if not isinstance(entry, dict):
                continue
            entry_type = entry.get("type", "article").lower()
            entry_id   = entry.get("id", "unknown")

            lines.append(f"@{entry_type}{{{entry_id},")
            for field in [
                "author", "title", "year", "journal", "booktitle",
                "publisher", "doi", "url", "pages", "volume", "number", "note",
            ]:
                if field in entry and entry[field]:
                    value = str(entry[field]).replace("{", "{{").replace("}", "}}")
                    lines.append(f"\t{field} = {{{value}}},")

            if lines and lines[-1].endswith(","):
                lines[-1] = lines[-1][:-1]
            lines.append("}\n")

        return "\n".join(lines)