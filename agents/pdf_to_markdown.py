import json
import re
from pathlib import Path
from typing import Optional

# ── Pydantic ──────────────────────────────────────────────────────────────────
from pydantic import BaseModel

# ── LangChain tools ───────────────────────────────────────────────────────────
from langchain_core.tools import tool

# ── LLM ───────────────────────────────────────────────────────────────────────
from langchain_ollama import ChatOllama

# ── Docling ───────────────────────────────────────────────────────────────────
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class Reference(BaseModel):
    ref_id: str
    title: str
    authors: list[str] = []
    year: Optional[str] = None
    journal: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    publisher: Optional[str] = None
    edition: Optional[str] = None
    ref_type: str = "article"
    raw_text: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BIBTEX FORMATTER
# ═══════════════════════════════════════════════════════════════════════════════

def _bibtex_entry(ref: Reference) -> str:
    lines = [f"@{ref.ref_type}{{{ref.ref_id},"]

    def add(field, value):
        if value:
            lines.append(f"  {field} = {{{value}}},")

    add("title", ref.title)
    if ref.authors:
        add("author", " and ".join(ref.authors))
    add("year", ref.year)
    add("journal", ref.journal)
    add("volume", ref.volume)
    add("number", ref.issue)
    add("pages", ref.pages)
    add("doi", ref.doi)
    add("url", ref.url)
    add("publisher", ref.publisher)
    add("edition", ref.edition)

    lines.append("}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LLM
# ═══════════════════════════════════════════════════════════════════════════════

def _get_llm():
    return ChatOllama(
        model="qwen3-coder:480b-cloud",
        base_url="https://ollama.com",
        temperature=0.1,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def pdf_to_markdown_converter(file_path: str) -> str:
    """
    Convert PDF to markdown.
    """
    print("📥 Converting PDF → Markdown")

    loader = DoclingLoader(
        file_path=file_path,
        export_type=ExportType.MARKDOWN,
    )
    docs = loader.load()

    markdown = "\n\n".join(doc.page_content for doc in docs)
    print("✅ Markdown length:", len(markdown))

    return markdown


@tool
def extract_references_from_markdown(markdown_content: str) -> str:
    """
    Extract references using LLM.
    """
    print("📥 Extracting references with LLM")

    llm = _get_llm()

    prompt = f"""
Extract ALL references from this section.

Return ONLY JSON array of objects with:
ref_id, title, authors, year, journal

TEXT:
{markdown_content}
"""

    response = llm.invoke(prompt)
    raw = response.content.strip()

    raw = re.sub(r"```.*?\n", "", raw)
    raw = re.sub(r"```", "", raw)

    print("📄 Raw LLM output preview:", raw[:300])

    return raw


@tool
def save_references_to_files(json_content: str, output_dir: str = "output") -> str:
    """
    Save JSON + BibTeX.
    """
    print("📥 Saving files")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        data = json.loads(json_content)
    except Exception:
        return f"❌ Invalid JSON: {json_content[:300]}"

    if isinstance(data, str):
        data = json.loads(data)

    if isinstance(data, list):

        def normalize_authors(authors):
            if isinstance(authors, list):
                return authors
            if isinstance(authors, str):
                authors = re.split(r",| and ", authors)
                return [a.strip() for a in authors if a.strip()]
            return []

        refs = []
        for r in data:
            r["authors"] = normalize_authors(r.get("authors"))
            refs.append(Reference(**r))
    else:
        refs = [Reference(**r) for r in data.get("references", [])]

    json_path = out / "references.json"
    json_path.write_text(json.dumps(data, indent=2))

    bib_text = "\n\n".join(_bibtex_entry(r) for r in refs)
    bib_path = out / "references.bib"
    bib_path.write_text(bib_text)

    print("✅ Files saved")

    return f"Saved {len(refs)} references"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. HELPER: EXTRACT ONLY REFERENCES SECTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_references_section(markdown: str) -> str:
    print("🔍 Extracting References section")

    match = re.search(r"(## References.*)", markdown, re.DOTALL | re.IGNORECASE)

    if match:
        refs = match.group(1)
        print("✅ References section found")
        return refs
    else:
        print("⚠️ References section NOT found — using full text")
        return markdown


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_extraction(pdf_path: str, output_dir: str = "output"):
    print("🚀 STARTING PIPELINE\n")

    # Step 1
    markdown = pdf_to_markdown_converter.invoke(pdf_path)

    # Step 2
    refs_section = extract_references_section(markdown)

    # Step 3
    json_output = extract_references_from_markdown.invoke(refs_section)

    # Step 4
    result = save_references_to_files.invoke({
        "json_content": json_output,
        "output_dir": output_dir
    })

    print("\n🎉 DONE")
    print(result)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 7. RUN IN NOTEBOOK
# ═══════════════════════════════════════════════════════════════════════════════

pdf_path = "/content/NIPS-2017-attention-is-all-you-need-Paper.pdf"

run_extraction(pdf_path)