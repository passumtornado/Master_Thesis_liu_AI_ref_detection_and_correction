# """
# Ground Truth Generator
# -----------------------
# Run this ONCE after generating thesis_dataset.bib to produce
# ground_truth.json — the authoritative record of every corruption
# applied to every entry in the synthetic dataset.

# This file is what makes EvaluationAgent deterministic.

# Usage:
#     python generate_ground_truth.py \\
#         --bib  thesis_dataset.bib \\
#         --out  ground_truth.json

# Output format (one record per entry):
# [
#   {
#     "entry_id":       "FichteHecherMorak23_pv12",
#     "entry_type":     "article",
#     "expected_status": "partially_valid",
#     "corruptions": [
#       {
#         "field":                  "year",
#         "original_correct_value": "2023",
#         "corrupted_value":        "2026",
#         "corruption_type":        "year_shift"
#       }
#     ]
#   },
#   {
#     "entry_id":        "hallucinated_003",
#     "entry_type":      "inproceedings",
#     "expected_status": "invalid",
#     "corruptions":     []
#   },
#   {
#     "entry_id":        "MeiBonsangueLaarman24",
#     "entry_type":      "article",
#     "expected_status": "valid",
#     "corruptions":     []
#   }
# ]
# """

# import argparse
# import json
# import re
# import sys
# from pathlib import Path


# # ─────────────────────────────────────────────────────────────
# # Helpers
# # ─────────────────────────────────────────────────────────────

# def _parse_bib(filepath: str) -> list[dict]:
#     """Parse a .bib file into a list of {key, type, fields} dicts."""
#     with open(filepath, "r", encoding="utf-8") as f:
#         text = f.read()

#     entries = []
#     pattern = re.compile(
#         r'@(\w+)\s*\{\s*([^,\s]+)\s*,([^@]*)',
#         re.DOTALL
#     )
#     for m in pattern.finditer(text):
#         etype  = m.group(1).lower()
#         key    = m.group(2).strip()
#         body   = m.group(3)

#         if etype == "string":
#             continue

#         fields = {}
#         for fm in re.finditer(
#             r'(\w[\w-]*)\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\}|"[^"]*"|\w+)',
#             body
#         ):
#             fname = fm.group(1).lower().strip()
#             fval  = fm.group(2).strip()
#             if fval.startswith("{") and fval.endswith("}"):
#                 fval = fval[1:-1]
#             elif fval.startswith('"') and fval.endswith('"'):
#                 fval = fval[1:-1]
#             if fname not in ("date-added", "date-modified"):
#                 fields[fname] = fval.strip()

#         entries.append({"key": key, "type": etype, "fields": fields})
#     return entries


# def _detect_section(key: str) -> str:
#     """
#     Determine which section of the synthetic dataset this entry belongs to
#     based on its cite key convention:
#       - ends with _pvN  → partially_valid (corrupted copy)
#       - starts with hallucinated_ → invalid (fully hallucinated)
#       - anything else → valid (original entry)
#     """
#     if re.search(r'_pv\d+$', key):
#         return "partially_valid"
#     if key.startswith("hallucinated_"):
#         return "invalid"
#     return "valid"


# def _find_original(entries_by_key: dict, pv_key: str) -> dict | None:
#     """
#     For a _pvN entry, find its original entry by stripping the _pvN suffix.
#     Returns the original entry dict, or None if not found.
#     """
#     base_key = re.sub(r'_pv\d+$', '', pv_key)
#     return entries_by_key.get(base_key)


# def _detect_corruptions(original: dict, corrupted: dict) -> list[dict]:
#     """
#     Compare original and corrupted field dicts.
#     Returns a list of corruption records for every differing field.
#     """
#     corruptions = []
#     fields_to_check = ["title", "author", "year", "journal",
#                        "booktitle", "publisher", "venue", "pages",
#                        "volume", "number", "note"]

#     for field in fields_to_check:
#         orig_val = original.get("fields", {}).get(field, "")
#         corr_val = corrupted.get("fields", {}).get(field, "")

#         if not orig_val or not corr_val:
#             continue
#         if orig_val.strip().lower() == corr_val.strip().lower():
#             continue

#         # Infer corruption type
#         if field == "year":
#             ctype = "year_shift"
#         elif field in ("journal", "booktitle", "publisher"):
#             ctype = "venue_substitution"
#         elif field == "title":
#             ctype = "title_mutation"
#         elif field == "author":
#             ctype = "author_corruption"
#         else:
#             ctype = "field_corruption"

#         corruptions.append({
#             "field":                  field,
#             "original_correct_value": orig_val.strip(),
#             "corrupted_value":        corr_val.strip(),
#             "corruption_type":        ctype,
#         })

#     return corruptions


# # ─────────────────────────────────────────────────────────────
# # Main
# # ─────────────────────────────────────────────────────────────

# def generate_ground_truth(bib_path: str, output_path: str) -> list[dict]:
#     print(f"Parsing {bib_path} ...")
#     entries = _parse_bib(bib_path)
#     print(f"  Found {len(entries)} entries")

#     # Index all entries by key for original lookup
#     entries_by_key = {e["key"]: e for e in entries}

#     ground_truth = []

#     for entry in entries:
#         key    = entry["key"]
#         etype  = entry["type"]
#         status = _detect_section(key)

#         if status == "valid":
#             ground_truth.append({
#                 "entry_id":        key,
#                 "entry_type":      etype,
#                 "expected_status": "valid",
#                 "corruptions":     [],
#             })

#         elif status == "partially_valid":
#             original = _find_original(entries_by_key, key)
#             corruptions = []
#             if original:
#                 corruptions = _detect_corruptions(original, entry)

#             ground_truth.append({
#                 "entry_id":        key,
#                 "entry_type":      etype,
#                 "expected_status": "partially_valid",
#                 "corruptions":     corruptions,
#             })

#         elif status == "invalid":
#             # All fields are hallucinated — no specific corruptions to record
#             ground_truth.append({
#                 "entry_id":        key,
#                 "entry_type":      etype,
#                 "expected_status": "invalid",
#                 "corruptions":     [],
#             })

#     # ── Summary ──────────────────────────────────────────────
#     valid_count   = sum(1 for r in ground_truth if r["expected_status"] == "valid")
#     partial_count = sum(1 for r in ground_truth if r["expected_status"] == "partially_valid")
#     invalid_count = sum(1 for r in ground_truth if r["expected_status"] == "invalid")

#     total_corruptions = sum(
#         len(r["corruptions"]) for r in ground_truth
#     )
#     field_counts: dict[str, int] = {}
#     for r in ground_truth:
#         for c in r["corruptions"]:
#             f = c["field"]
#             field_counts[f] = field_counts.get(f, 0) + 1

#     print(f"\n  Ground truth summary:")
#     print(f"    Valid           : {valid_count}")
#     print(f"    Partially valid : {partial_count}")
#     print(f"    Invalid         : {invalid_count}")
#     print(f"    Total           : {len(ground_truth)}")
#     print(f"    Total corruptions detected: {total_corruptions}")
#     print(f"    Corruptions by field:")
#     for field, count in sorted(field_counts.items(), key=lambda x: -x[1]):
#         print(f"      {field:15s} : {count}")

#     # ── Save ──────────────────────────────────────────────────
#     Path(output_path).write_text(
#         json.dumps(ground_truth, indent=2, ensure_ascii=False),
#         encoding="utf-8",
#     )
#     print(f"\n  ✓ Ground truth saved → {output_path}")
#     return ground_truth


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Generate ground_truth.json from thesis_dataset.bib"
#     )
#     parser.add_argument("--bib", default="thesis_dataset.bib",
#                         help="Path to the synthetic .bib file")
#     parser.add_argument("--out", default="ground_truth.json",
#                         help="Output path for ground_truth.json")
#     args = parser.parse_args()

#     generate_ground_truth(args.bib, args.out)


# """
# Ground Truth Generator v2
# --------------------------
# Reads the annotated .bib file and builds ground_truth.json
# by parsing %% DEFECT: comments and %% [N] Invalid markers.

# Works with human-curated .bib files that use descriptive cite keys
# (not the _pvN / hallucinated_ naming convention from the synthetic generator).

# Sections detected from the .bib comment headers:
#   SECTION A — FULLY VALID
#   SECTION B — PARTIALLY VALID  (reads %% DEFECT: lines)
#   SECTION C — INVALID          (all entries marked invalid, no corruptions)
# """

# import json
# import re
# import sys
# from pathlib import Path


# # ─────────────────────────────────────────────────────────────
# # Field aliases — map bib field names to canonical names
# # ─────────────────────────────────────────────────────────────

# FIELD_ALIASES = {
#     "booktitle": "booktitle",
#     "journal":   "journal",
#     "author":    "author",
#     "title":     "title",
#     "year":      "year",
#     "doi":       "doi",
#     "publisher": "publisher",
#     "pages":     "pages",
#     "volume":    "volume",
#     "number":    "number",
#     "note":      "note",
# }

# # ─────────────────────────────────────────────────────────────
# # Known correct values for partially valid entries
# # (read directly from the .bib DEFECT annotations)
# # ─────────────────────────────────────────────────────────────

# CORRECT_VALUES = {
#     "hinton1986learning": {
#         "author": "Hinton, Geoffrey E. and Sejnowski, Terrence J.",
#         "title":  "Learning and Relearning in Boltzmann Machines",
#         "booktitle": "Parallel Distributed Processing",
#     },
#     "srivastava2014dropout": {
#         "author": "Srivastava, Nitish and Hinton, Geoffrey and Krizhevsky, Alex and Sutskever, Ilya and Salakhutdinov, Ruslan",
#     },
#     "schmidhuber2015deep": {
#         "title": "Deep Learning in Neural Networks: An Overview",
#     },
#     "mnih2015human": {
#         "title":   "Human-level Control Through Deep Reinforcement Learning",
#         "journal": "Nature",
#     },
#     "srivastava2014dropout2": {
#         "author": "Srivastava, Nitish and Hinton, Geoffrey and Krizhevsky, Alex and Sutskever, Ilya and Salakhutdinov, Ruslan",
#         "doi":    "10.5555/2627435.2670313",
#     },
#     "lecun1998gradient": {
#         "title": "Gradient-Based Learning Applied to Document Recognition",
#         "year":  "1998",
#     },
#     "bahdanau2015neural": {
#         "author":    "Bahdanau, Dzmitry and Cho, Kyunghyun and Bengio, Yoshua",
#         "booktitle": "Proceedings of the International Conference on Learning Representations (ICLR)",
#     },
#     "sutton1998reinforcement": {
#         "title": "Reinforcement Learning: An Introduction",
#         "doi":   "10.7551/mitpress/11490.001.0001",
#     },
#     "kingma2015adam": {
#         "author": "Kingma, Diederik P. and Ba, Jimmy",
#     },
#     "collobert2008unified": {
#         "author": "Collobert, Ronan and Weston, Jason",
#     },
#     "mikolov2013distributed": {
#         "title": "Distributed Representations of Words and Phrases and their Compositionality",
#         "doi":   "10.48550/arXiv.1310.4546",
#     },
#     "pennington2014glove": {
#         "author":    "Pennington, Jeffrey and Socher, Richard and Manning, Christopher D.",
#         "booktitle": "Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
#     },
#     "chollet2017xception": {
#         "title": "Xception: Deep Learning with Depthwise Separable Convolutions",
#         "year":  "2017",
#     },
#     "ioffe2015batch": {
#         "author": "Ioffe, Sergey and Szegedy, Christian",
#     },
#     "graves2013generating": {
#         "title": "Generating Sequences with Recurrent Neural Networks",
#     },
#     "zoph2017neural": {
#         "title": "Neural Architecture Search with Reinforcement Learning",
#         "doi":   "10.48550/arXiv.1611.01578",
#     },
#     "velickovic2018graph": {
#         "title":     "Graph Attention Networks",
#         "booktitle": "Proceedings of the International Conference on Learning Representations (ICLR)",
#     },
#     "dosovitskiy2021image": {
#         "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
#         "doi":   "10.48550/arXiv.2010.11929",
#     },
#     "raffel2020exploring": {
#         "title": "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
#         "year":  "2020",
#     },
#     "howard2018universal": {
#         "title": "Universal Language Model Fine-tuning for Text Classification",
#         "doi":   "10.18653/v1/P18-1031",
#     },
# }


# def _parse_bib_entries(text: str) -> list[dict]:
#     """Parse all @type{key, ...} entries from bib text."""
#     entries = []
#     pattern = re.compile(
#         r'@(\w+)\s*\{\s*([^,\s]+)\s*,([^@]*)',
#         re.DOTALL
#     )
#     for m in pattern.finditer(text):
#         etype = m.group(1).lower()
#         key   = m.group(2).strip()
#         body  = m.group(3)

#         if etype == "string":
#             continue

#         # Parse fields
#         fields = {}
#         for fm in re.finditer(
#             r'(\w[\w-]*)\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\}|"[^"]*"|\w[\w.-]*)',
#             body
#         ):
#             fname = fm.group(1).lower().strip()
#             fval  = fm.group(2).strip()
#             if fval.startswith("{") and fval.endswith("}"):
#                 fval = fval[1:-1]
#             elif fval.startswith('"') and fval.endswith('"'):
#                 fval = fval[1:-1]
#             if fname not in ("date-added", "date-modified"):
#                 fields[fname] = fval.strip()

#         entries.append({"key": key, "type": etype, "fields": fields})
#     return entries


# def _detect_section(text: str, key: str) -> str:
#     """
#     Determine which section an entry belongs to by finding
#     the nearest SECTION comment before the entry's @type line.
#     """
#     # Find the position of the entry in the text
#     entry_pos = text.find(f"@{{}}")   # fallback
#     for pattern in [f"@article{{{key},", f"@inproceedings{{{key},",
#                     f"@book{{{key},", f"@misc{{{key},",
#                     f"@techreport{{{key},", f"@incollection{{{key},"]:
#         pos = text.find(pattern)
#         if pos != -1:
#             entry_pos = pos
#             break

#     if entry_pos == -1:
#         return "valid"

#     # Look backwards for the most recent SECTION comment
#     text_before = text[:entry_pos]

#     if "SECTION C" in text_before:
#         return "invalid"
#     elif "SECTION B" in text_before:
#         return "partially_valid"
#     elif "SECTION A" in text_before:
#         return "valid"
#     return "valid"


# def _build_corruptions(key: str, fields: dict) -> list[dict]:
#     """
#     Build the corruptions list for a partially valid entry
#     by comparing its current (corrupted) fields against
#     the known correct values in CORRECT_VALUES.
#     """
#     correct = CORRECT_VALUES.get(key, {})
#     corruptions = []

#     for field, correct_value in correct.items():
#         current_value = fields.get(field, "")

#         # Empty field = missing field corruption
#         if not current_value:
#             corruptions.append({
#                 "field":                  field,
#                 "original_correct_value": correct_value,
#                 "corrupted_value":        "",
#                 "corruption_type":        "missing_field",
#             })
#             continue

#         # Non-empty but different = content corruption
#         if current_value.strip().lower() != correct_value.strip().lower():
#             # Classify corruption type
#             if field == "year":
#                 ctype = "year_shift"
#             elif field == "author":
#                 ctype = "author_corruption"
#             elif field == "title":
#                 ctype = "title_mutation"
#             elif field in ("journal", "booktitle", "publisher"):
#                 ctype = "venue_substitution"
#             elif field == "doi":
#                 ctype = "missing_doi"
#             else:
#                 ctype = "field_corruption"

#             corruptions.append({
#                 "field":                  field,
#                 "original_correct_value": correct_value,
#                 "corrupted_value":        current_value.strip(),
#                 "corruption_type":        ctype,
#             })

#     return corruptions


# def generate_ground_truth(bib_path: str, output_path: str) -> list[dict]:
#     print(f"Reading {bib_path} ...")
#     text    = Path(bib_path).read_text(encoding="utf-8")
#     entries = _parse_bib_entries(text)
#     print(f"  Found {len(entries)} entries")

#     ground_truth = []
#     stats = {"valid": 0, "partially_valid": 0, "invalid": 0}
#     total_corruptions = 0
#     field_counts: dict[str, int] = {}

#     for entry in entries:
#         key    = entry["key"]
#         etype  = entry["type"]
#         fields = entry["fields"]
#         status = _detect_section(text, key)

#         if status == "valid":
#             ground_truth.append({
#                 "entry_id":        key,
#                 "entry_type":      etype,
#                 "expected_status": "valid",
#                 "corruptions":     [],
#             })
#             stats["valid"] += 1

#         elif status == "partially_valid":
#             corruptions = _build_corruptions(key, fields)
#             ground_truth.append({
#                 "entry_id":        key,
#                 "entry_type":      etype,
#                 "expected_status": "partially_valid",
#                 "corruptions":     corruptions,
#             })
#             stats["partially_valid"] += 1
#             total_corruptions += len(corruptions)
#             for c in corruptions:
#                 f = c["field"]
#                 field_counts[f] = field_counts.get(f, 0) + 1

#         elif status == "invalid":
#             ground_truth.append({
#                 "entry_id":        key,
#                 "entry_type":      etype,
#                 "expected_status": "invalid",
#                 "corruptions":     [],
#             })
#             stats["invalid"] += 1

#     # Summary
#     print(f"\n  Ground truth summary:")
#     print(f"    Valid           : {stats['valid']}")
#     print(f"    Partially valid : {stats['partially_valid']}")
#     print(f"    Invalid         : {stats['invalid']}")
#     print(f"    Total           : {len(ground_truth)}")
#     print(f"    Total corruptions detected: {total_corruptions}")
#     if field_counts:
#         print(f"    Corruptions by field:")
#         for field, count in sorted(field_counts.items(), key=lambda x: -x[1]):
#             print(f"      {field:20s} : {count}")

#     Path(output_path).write_text(
#         json.dumps(ground_truth, indent=2, ensure_ascii=False),
#         encoding="utf-8"
#     )
#     print(f"\n  ✓ Ground truth saved → {output_path}")
#     return ground_truth


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--bib", default="references.bib")
#     parser.add_argument("--out", default="ground_truth.json")
#     args = parser.parse_args()
#     generate_ground_truth(args.bib, args.out)


#!/usr/bin/env python3
"""
Ground Truth Generator (Generic, using bibtexparser)
-----------------------------------------------------
Reads an annotated .bib file and builds ground_truth.json.
Requires: pip install bibtexparser
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import bibtexparser



def extract_defects_from_comment(entry: Dict[str, Any]) -> List[str]:
    """
    Extract defect descriptions from the 'comment' field (which holds
    BibTeX comments like %% DEFECT: ...) or from a dedicated 'defect' field.
    """
    defects = []
    # bibtexparser may store comments in a special 'comment' field
    comment = entry.get("comment", "") or entry.get("COMMENT", "")
    if comment:
        for line in comment.splitlines():
            line = line.strip()
            if line.startswith("%% DEFECT:"):
                defect = line[10:].strip()
                if defect:
                    defects.append(defect)
            elif line.startswith("%% DEFECT"):
                defect = line[9:].strip()
                if defect:
                    defects.append(defect)
    return defects


def extract_invalid_reason(entry: Dict[str, Any]) -> Optional[str]:
    """Extract INVALID reason from the 'note' field."""
    note = entry.get("note", "")
    if note.lower().startswith("invalid:"):
        return note[8:].strip()
    elif "invalid" in note.lower():
        return note.strip()
    return None


def detect_section(entry: Dict[str, Any], text: str) -> str:
    """
    Determine section by looking for SECTION comments before the entry.
    Since bibtexparser doesn't preserve order perfectly, we fall back to
    checking the 'comment' field or the entry's position in the raw text.
    """
    # Try to get section from a custom 'section' field if we stored it
    if "section" in entry:
        return entry["section"]

    # Fallback: check if the entry has a comment that indicates section
    comment = entry.get("comment", "")
    if "SECTION C" in comment:
        return "invalid"
    if "SECTION B" in comment:
        return "partially_valid"
    if "SECTION A" in comment:
        return "valid"

    # Last resort: default to valid (should not happen with proper annotations)
    return "valid"


def generate_ground_truth(bib_path: str, output_path: str) -> List[Dict]:
    print(f"Reading {bib_path} ...")

    # Parse with bibtexparser
    with open(bib_path, encoding="utf-8") as f:
        raw_text = f.read()

    library = bibtexparser.parse_string(raw_text)

    # Normalize v2 Entry objects into dicts compatible with the existing logic.
    entries: List[Dict[str, Any]] = []
    for entry in library.entries:
        fields = {f.key: f.value for f in entry.fields}
        fields["ID"] = entry.key
        fields["ENTRYTYPE"] = entry.entry_type
        entries.append(fields)

    print(f"  Found {len(entries)} entries")

    # We also need the raw text to find section comments (bibtexparser discards them).
    # Better: pre-process raw text to attach section info to each entry.
    # Simpler: rely on the 'comment' field which bibtexparser sometimes captures.
    # But comments like %% SECTION A are global, not per-entry.
    # We'll do a hybrid: find the position of each entry in raw_text and look backwards.
    def find_section_for_entry(entry_key: str) -> str:
        # Find the entry in raw_text
        patterns = [
            f"@article{{{entry_key},",
            f"@inproceedings{{{entry_key},",
            f"@book{{{entry_key},",
            f"@misc{{{entry_key},",
            f"@techreport{{{entry_key},",
            f"@incollection{{{entry_key},"
        ]
        pos = -1
        for pat in patterns:
            pos = raw_text.find(pat)
            if pos != -1:
                break
        if pos == -1:
            return "valid"
        before = raw_text[:pos]
        if "SECTION C" in before:
            return "invalid"
        if "SECTION B" in before:
            return "partially_valid"
        if "SECTION A" in before:
            return "valid"
        return "valid"

    ground_truth = []
    stats = {"valid": 0, "partially_valid": 0, "invalid": 0}

    for entry in entries:
        key = entry.get("ID", "")
        etype = entry.get("ENTRYTYPE", "misc")
        status = find_section_for_entry(key)

        if status == "valid":
            ground_truth.append({
                "entry_id": key,
                "entry_type": etype,
                "expected_status": "valid",
            })
            stats["valid"] += 1

        elif status == "partially_valid":
            defects = extract_defects_from_comment(entry)
            ground_truth.append({
                "entry_id": key,
                "entry_type": etype,
                "expected_status": "partially_valid",
                "defects": defects,
            })
            stats["partially_valid"] += 1

        elif status == "invalid":
            reason = extract_invalid_reason(entry)
            ground_truth.append({
                "entry_id": key,
                "entry_type": etype,
                "expected_status": "invalid",
                "invalid_reason": reason if reason else "no reason provided",
            })
            stats["invalid"] += 1

    print(f"\n  Ground truth summary:")
    print(f"    Valid           : {stats['valid']}")
    print(f"    Partially valid : {stats['partially_valid']}")
    print(f"    Invalid         : {stats['invalid']}")
    print(f"    Total           : {len(ground_truth)}")

    Path(output_path).write_text(
        json.dumps(ground_truth, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"\n  ✓ Ground truth saved → {output_path}")
    return ground_truth


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bib", default="references.bib")
    parser.add_argument("--out", default="ground_truth.json")
    args = parser.parse_args()
    generate_ground_truth(args.bib, args.out)