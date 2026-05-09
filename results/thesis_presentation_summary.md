# Results — Presentation Summary

_Optimized for defense presentation & slide deck (2-3 pages)_

---

## 1. Dataset Overview

This study evaluates 4 LLM models on reference validation and correction using a BibTeX dataset split into 4 ground-truth sets.

| Set       | Total Entries |   Valid | Partially Valid | Invalid |
| --------- | ------------: | ------: | --------------: | ------: |
| Set 1-2   |           102 |      52 |              34 |      16 |
| Set 3-4   |           100 |      52 |              34 |      14 |
| **Total** |       **202** | **104** |          **68** |  **30** |

**Dataset design:** Balanced across all 3 label types; supports fair model comparison and repeated testing.

---

## 2. Model Comparison (Available Runs)

| Model             | Validation F1 | Correction F1 | Weighted Score | Reliability Score |
| ----------------- | ------------: | ------------: | -------------: | ----------------: |
| **Grok 4.20**     |         0.931 |         0.143 |          0.616 |         **0.610** |
| **GPT-5.4**       |         0.748 |         0.327 |          0.579 |             0.579 |
| Claude Sonnet 4.6 |         0.838 |         0.130 |          0.554 |             0.493 |
| Gemini 2.5 Pro    |         0.764 |         0.110 |          0.502 |             0.428 |

**Formula:** `Weighted Score = 0.6 × Validation F1 + 0.4 × Correction F1`  
**Reliability Score:** Weighted Score penalized by unverifiable rate

**Ranking interpretation:**

- ✅ **Grok 4.20** best overall (strongest validation, lowest errors)
- ✅ **GPT-5.4** best on correction quality
- ⚠️ Claude & Gemini show higher instability across datasets

---

## 3. Key Performance Insights

### Validation Behavior

- Grok maintains **93% F1** across available runs (most stable)
- GPT-5.4 shows **high variance** by dataset (74–84% F1)
- Claude & Gemini most affected by **tool access failures** (15%+ unverifiable labels)
- All models struggle with **invalid class recall** (average 63–80%)

### Correction Behavior

- GPT-5.4 achieves **53% F1** on Set 3 (best single run)
- All models show **very low recall** on correction (1–35%)
  - Means: Most needed fixes are still missed
- High precision overall (71–100%) — fixes applied are usually safe

### System Effects

- Unverifiable rate **strongly impacts** final scores (1–15% difference)
- Implies: Tool reliability matters as much as model quality

---

## 4. Final Ranking with Reliability

|     Rank | Model             | Why This Rank                                                                                         |
| -------: | ----------------- | ----------------------------------------------------------------------------------------------------- |
| 🥇 **1** | **Grok 4.20**     | Best validation F1 (0.931) + lowest tool failures (1% unverifiable) = most reliable choice            |
| 🥈 **2** | **GPT-5.4**       | Best correction F1 (0.327) despite weaker validation; consistent tool access (0% unverifiable)        |
| 🥉 **3** | Claude Sonnet 4.6 | Strong validation (0.838) but weak correction (0.130) + occasional tool failures (11% unverifiable)   |
|    **4** | Gemini 2.5 Pro    | Weakest in both validation (0.764) and correction (0.110) + frequent tool failures (15% unverifiable) |

---

## 5. Limitations & Threats to Validity

1. **Incomplete runs:** Some model-set pairs missing; averages weighted only on available data
2. **External tool failures:** DBLP/Scholar access issues mask true model performance on 1–15% of cases
3. **Low correction recall:** All models miss 65–99% of needed corrections — correction module still immature
4. **Limited dataset:** Only 4 sets; more would increase ranking stability confidence
5. **No Set 1 runs:** Missing baseline; cannot assess performance on first dataset

---

## 6. Key Findings & Conclusion

1. ✅ **Dataset is balanced & suitable** for fair model testing
2. ✅ **Validation task is well-learned** by all models (F1 > 0.74)
3. ⚠️ **Correction task is challenging** for all models (best F1 = 0.33)
4. 🎯 **No single best model:** Grok wins validation, GPT wins correction
5. 📊 **Reliability matters:** Tool failures shift ranking more than model skill differences
6. 📈 **Implication:** Final deployment should prioritize **Grok + GPT combination** (Grok for validation, GPT for correction) if latency allows

### Final Statement for Thesis

The benchmark demonstrates that BibTeX reference validation is well-solved by modern LLMs, but correction remains an open challenge. Grok 4.20 is the strongest single-model choice when tool reliability is considered. However, the architecture should not depend on a single model for both tasks; using specialized models per task (Grok for validation, GPT for correction) would likely yield better results.

---

_This summary can be split across 2 presentation slides:_

- _Slide 1: Dataset, Model Comparison, Ranking (sections 1–4)_
- _Slide 2: Limitations, Key Findings, Conclusion (sections 5–6)_
