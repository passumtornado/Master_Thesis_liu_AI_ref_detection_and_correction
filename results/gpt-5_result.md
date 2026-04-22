# Evaluation Report: Bibliographic Correction Performance

## 1) Overall Metrics Summary

| Metric | Value |
|---|---:|
| True Positives (TP) | 22 |
| False Positives (FP) | 1 |
| False Negatives (FN) | 13 |
| Precision | 0.957 |
| Recall | 0.629 |
| F1-score | 0.759 |

**Interpretation.** The correction agent exhibits **very high precision** (few incorrect changes) but **moderate recall** (a substantial proportion of existing errors remained uncorrected). The resulting F1-score indicates overall good performance, driven primarily by conservative behavior.

---

## 2) Field-level Accuracy

| Field | Errors in Original | Errors Corrected | False Corrections | Accuracy |
|---|---:|---:|---:|---:|
| author | 8 | 5 | 0 | 0.625 |
| title | 13 | 13 | 1 | 1.000 |
| booktitle | 4 | 0 | 0 | 0.000 |
| journal | 1 | 1 | 0 | 1.000 |
| doi | 6 | 0 | 0 | 0.000 |
| year | 3 | 3 | 0 | 1.000 |

**Summary.** Performance is **excellent for `title`, `year`, and `journal`**, moderate for `author`, and **fails entirely for `booktitle` and `doi`** in this evaluation set.

---

## 3) Entry Outcomes (Grouped)

### A. Correctly fixed (entries with ≥1 TP and no FP; may still include FNs)
These entries contained errors and were at least partially corrected accurately.

| Entry ID | TP Fields | Missed (FN) Fields |
|---|---|---|
| hinton1986learning | author, title | booktitle |
| schmidhuber2015deep | title | — |
| mnih2015human | title, journal | — |
| lecun1998gradient | title, year | — |
| bahdanau2015neural | author | booktitle |
| collobert2008unified | author | — |
| mikolov2013distributed | title | doi |
| pennington2014glove | author | booktitle |
| chollet2017xception | title, year | — |
| graves2013generating | title | — |
| sutton1998reinforcement | title | doi |

### B. Missed errors (entries where all corruptions were missed; only FNs)
These entries had corrupted/missing fields, but the agent did not produce the correct field value(s) according to the evaluation criteria.

| Entry ID | Missed (FN) Fields | Notes |
|---|---|---|
| srivastava2014dropout | author | Output appears semantically correct but not in the expected canonical/BibTeX author-string form. |
| kingma2015adam | author | Same pattern: name list produced, but not in expected canonical format. |
| ioffe2015batch | author | Same pattern: name list produced, but not in expected canonical format. |

### C. False corrections (entries with ≥1 FP)
Entries where the agent changed a field that was already correct, producing an incorrect value.

| Entry ID | FP Field(s) | Correct Value | Incorrect Corrected Value |
|---|---|---|---|
| srivastava2014dropout2 | title | *Dropout: Preventing Overfitting* | *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* |

### D. No-change / already valid entries
Entries expected to be valid and showing no recorded field outcomes (i.e., no detected issues and no applied corrections).

- turing1950computing
- shannon1948mathematical
- vaswani2017attention
- lecun2015deep
- goodfellow2014generative
- devlin2019bert
- silver2016mastering
- hochreiter1997long
- minsky1969perceptrons
- rumelhart1986learning
- brown2020language
- he2016deep
- bengio2013representation
- krizhevsky2012imagenet
- radford2021learning

---

## 4) Key Insights

1. **Strength in lexical/typographic title repair.**  
   All title corruptions were corrected (13/13), indicating robust handling of common OCR-like mutations (e.g., *Recogniton → Recognition*, *Recurent → Recurrent*). The single FP suggests that, despite strong performance, the system can over-normalise titles toward more famous variants.

2. **Systematic failure to recover `doi` fields.**  
   All DOI errors were missed (0/6). This pattern strongly suggests the agent lacks a DOI retrieval strategy (e.g., Crossref/arXiv lookup) or is configured to avoid adding identifiers when missing.

3. **Systematic failure to recover `booktitle` fields.**  
   All booktitle errors were missed (0/4), including both typographic corruption and missing values. This indicates insufficient venue inference or an overly conservative policy for conference/book venues.

4. **Author correction is partially successful but sensitive to canonical formatting requirements.**  
   Where the author field contained minor spelling errors, corrections were successful (e.g., *Benjio → Bengio*). However, when the author field was missing, the agent often produced a plausible author list but in a **non-canonical string format**, which was scored as FN. This indicates a *representation/normalisation* weakness rather than a pure entity-resolution failure.

5. **Conservative behaviour yields high precision.**  
   Precision (0.957) indicates the agent rarely introduces new errors. The trade-off is reduced recall (0.629), concentrated in metadata fields that typically require external lookup or stronger inference (DOI, booktitle).

---

## 5) Actionable Recommendations

1. **Implement identifier-aware enrichment for DOIs.**  
   Add a deterministic lookup stage (e.g., Crossref by title+author+year; arXiv API for preprints) and populate `doi` when confidence is high. Consider storing provenance (source and timestamp) to support auditability.

2. **Add venue inference and controlled vocabulary for `booktitle`.**  
   Introduce a venue normalisation module (string matching + known conference acronym expansions such as EMNLP, ICLR). For missing `booktitle`, use title/author/year queries to retrieve proceedings metadata.

3. **Enforce BibTeX-compliant author canonicalisation.**  
   Ensure output uses the expected “Last, First and Last, First” format, with correct delimitering (“and”) and consistent initials. This is likely to convert several current FNs into TPs without changing the underlying extraction.

4. **Reduce title over-correction via equivalence checking.**  
   Before replacing a title, verify that the proposed correction matches the known record for the *same work* (e.g., via DOI, stable venue-year-author matching). This would mitigate the observed FP where a correct short title was replaced by a longer canonical variant.

5. **Adopt a field-prioritised strategy to raise recall without sacrificing precision.**  
   Maintain conservative policies for already-correct fields, but apply more assertive correction/enrichment specifically to **high-value missing fields** (`doi`, `booktitle`, missing `author`) using confidence thresholds and external validation.

---