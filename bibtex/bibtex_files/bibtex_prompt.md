# Synthetic Academic Reference Dataset Generation Prompt

---

## Overview

Generate a dataset of **250 synthetic academic references** spanning a wide range of publication years (historical, mid-20th century, and recent). The references should draw from diverse source types including research papers, novels, arXiv preprints, and outdated or retracted works. Organize the output into three clearly labeled sections.

---

## Section A — 100 Fully Valid References

Each reference must include **all** of the following fields, correctly formatted:

| Field | Requirement |
|---|---|
| Author(s) | Full names, correctly spelled |
| Title | Complete and accurate |
| Venue | Journal, conference, or publisher |
| DOI | Valid and resolvable |
| Year | Correct publication year |
| Type | See entry types below |

**Entry types must vary and include:**
- Journal articles
- Conference proceedings (e.g. NeurIPS, CVPR, ACL, ICML, ICLR, AAAI)
- Books and book chapters

> All metadata must be accurate and verifiable.

---

## Section B — 80 Partially Valid References

Each reference must contain **at least one deliberate defect**, clearly annotated using `%% DEFECT:` comments. Defect types should vary across entries and include:

- Misspelled author name(s) or title
- Missing author, venue, title, DOI, or year
- Incomplete or truncated fields
- Incorrect but plausible metadata (e.g. wrong volume, swapped initials)

> Each defect must be explicitly annotated so the error is identifiable without external lookup.

---

## Section C — 70 Invalid References

Each reference must be **entirely or substantially fabricated**. Invalid types should vary and include:

- Hallucinated papers (plausible-sounding but non-existent)
- Impossible author pairings (e.g. figures from different historical eras)
- Fictional or non-existent journals and conferences
- Non-existent arXiv IDs (e.g. `arXiv:2199.99999`)
- Retracted papers that have since been removed from databases
- Anachronistic entries (e.g. a 1940s paper citing a 2020 framework)
- Future-dated entries (e.g. proceedings from the year 2099)

> Each entry must include a `note = {INVALID: ...}` field explaining specifically why it is invalid.

---

## Year Range Requirements

References must be distributed across three broad eras:

| Era | Approximate Range |
|---|---|
| Historical / Old | Pre-1970 |
| Mid-period | 1970–2010 |
| Recent | 2011–present |

---

## Output Format

- **Format:** Fully formatted BibTeX (`.bib` file)
- **Entry types:** Use appropriate BibTeX types — `@article`, `@inproceedings`, `@book`, `@incollection`, `@misc`, `@techreport` etc.
- **Special characters:** Escape non-ASCII characters correctly (e.g. `{\"u}` for ü, `{\'{e}}` for é)
- **Protected casing:** Wrap proper nouns and acronyms in braces (e.g. `{BERT}`, `{GPU}`, `{Go}`)
- **Section headers:** Do not Separate the three sections with clearly labeled `%%` block comments

---

## Example Annotations

### Section B — Partial Defect Comment
```bibtex
%% DEFECT: Misspelled author ("Benjio" should be "Bengio"); venue field missing
@inproceedings{bahdanau2015neural,
  author    = {Bahdanau, Dzmitry and Cho, Kyunghyun and Benjio, Yoshua},
  title     = {Neural Machine Translation by Jointly Learning to Align and Translate},
  booktitle = {},
  year      = {2015},
  doi       = {10.48550/arXiv.1409.0473}
}
```

### Section C — Invalid Entry with Note
```bibtex
@inproceedings{feigenbaum1985unified,
  author    = {Feigenbaum, Edward A. and Turing, Alan M. and McCarthy, John},
  title     = {Unified Theory of Artificial Minds: Bridging Symbolic and Subsymbolic {AI}},
  booktitle = {AAAI 1985 Workshop on Artificial General Intelligence},
  year      = {1985},
  doi       = {10.1234/aaai.agi.1985.7741},
}
```

---

## Checklist

- [ ] Exactly 250 entries total (100 + 80 + 70)
- [ ] No duplicate citation keys
- [ ] All three eras represented in each section where possible
- [ ] Entry types vary within each section
<!-- - [ ] Section B defects are annotated with `%% DEFECT:`
- [ ] Section C entries include `note = {INVALID: ...}` -->
- [ ] Special characters and proper nouns correctly formatted
- [ ] Output is a single valid `.bib` file with section header comments