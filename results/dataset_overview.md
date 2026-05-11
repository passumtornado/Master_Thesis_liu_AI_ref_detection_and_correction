# Dataset Overview & Statistics

This document provides a comprehensive overview of the BibTeX reference datasets used in this research,
including detailed statistics on validation quality, entry type distributions, and field usage patterns.

## Overall Dataset Summary

**Total Entries Across All Datasets:** 202

| Validation Status | Count | Percentage |
|---|---:|---:|
| Valid | 104 | 51.5% |
| Partially Valid | 68 | 33.7% |
| Invalid | 30 | 14.9% |

The dataset includes 202 BibTeX entries distributed across 4 training sets, with approximately
51.5% clean entries, 33.7% entries with field-level errors, and 14.9% completely invalid entries.

## Individual Dataset Details

### Stefan Training Set 1 (Set 1)

**Total Entries:** 51

#### Validation Status Distribution

| Status | Count | Percentage |
|---|---:|---:|
| Valid | 26 | 51.0% |
| Partially Valid | 17 | 33.3% |
| Invalid | 8 | 15.7% |

#### Entry Type Distribution

| Entry Type | Count | Percentage |
|---|---:|---:|
| @article | 32 | 62.7% |
| @inproceedings | 12 | 23.5% |
| @book | 4 | 7.8% |
| @misc | 2 | 3.9% |
| @techreport | 1 | 2.0% |

#### Dataset Characteristics

- **Quality Assessment:** 26/51 (51.0%) of entries are fully valid
- **Error Prevalence:** 17/51 (33.3%) contain field-level errors
- **Corruption Rate:** 8/51 (15.7%) are fundamentally broken

### Stefan Training Set 2 (Set 2)

**Total Entries:** 51

#### Validation Status Distribution

| Status | Count | Percentage |
|---|---:|---:|
| Valid | 26 | 51.0% |
| Partially Valid | 17 | 33.3% |
| Invalid | 8 | 15.7% |

#### Entry Type Distribution

| Entry Type | Count | Percentage |
|---|---:|---:|
| @article | 30 | 58.8% |
| @inproceedings | 17 | 33.3% |
| @misc | 2 | 3.9% |
| @book | 1 | 2.0% |
| @techreport | 1 | 2.0% |

#### Dataset Characteristics

- **Quality Assessment:** 26/51 (51.0%) of entries are fully valid
- **Error Prevalence:** 17/51 (33.3%) contain field-level errors
- **Corruption Rate:** 8/51 (15.7%) are fundamentally broken

### Stefan Training Set 3 (Set 3)

**Total Entries:** 50

#### Validation Status Distribution

| Status | Count | Percentage |
|---|---:|---:|
| Valid | 26 | 52.0% |
| Partially Valid | 17 | 34.0% |
| Invalid | 7 | 14.0% |

#### Entry Type Distribution

| Entry Type | Count | Percentage |
|---|---:|---:|
| @article | 31 | 62.0% |
| @inproceedings | 12 | 24.0% |
| @misc | 4 | 8.0% |
| @book | 2 | 4.0% |
| @techreport | 1 | 2.0% |

#### Dataset Characteristics

- **Quality Assessment:** 26/50 (52.0%) of entries are fully valid
- **Error Prevalence:** 17/50 (34.0%) contain field-level errors
- **Corruption Rate:** 7/50 (14.0%) are fundamentally broken

### Stefan Training Set 4 (Set 4)

**Total Entries:** 50

#### Validation Status Distribution

| Status | Count | Percentage |
|---|---:|---:|
| Valid | 26 | 52.0% |
| Partially Valid | 17 | 34.0% |
| Invalid | 7 | 14.0% |

#### Entry Type Distribution

| Entry Type | Count | Percentage |
|---|---:|---:|
| @article | 31 | 62.0% |
| @inproceedings | 14 | 28.0% |
| @misc | 4 | 8.0% |
| @book | 1 | 2.0% |

#### Dataset Characteristics

- **Quality Assessment:** 26/50 (52.0%) of entries are fully valid
- **Error Prevalence:** 17/50 (34.0%) contain field-level errors
- **Corruption Rate:** 7/50 (14.0%) are fundamentally broken

## Comparative Analysis Across Datasets

### Size Comparison

| Dataset | Total Entries | Valid | Partially Valid | Invalid |
|---|---:|---:|---:|---:|
| Stefan Training Set 1 | 51 | 26 | 17 | 8 |
| Stefan Training Set 2 | 51 | 26 | 17 | 8 |
| Stefan Training Set 3 | 50 | 26 | 17 | 7 |
| Stefan Training Set 4 | 50 | 26 | 17 | 7 |

### Quality Ranking (by Valid Entry Ratio)

1. **Stefan Training Set 3** - 52.0% valid entries (26/50)
2. **Stefan Training Set 4** - 52.0% valid entries (26/50)
3. **Stefan Training Set 1** - 51.0% valid entries (26/51)
4. **Stefan Training Set 2** - 51.0% valid entries (26/51)

## Entry Type Analysis

| Entry Type | Total Across All Sets | Most Common In |
|---|---:|---|
| @article | 124 | Stefan Training Set 1 (32) |
| @inproceedings | 55 | Stefan Training Set 2 (17) |
| @misc | 12 | Stefan Training Set 3 (4) |
| @book | 8 | Stefan Training Set 1 (4) |
| @techreport | 3 | Stefan Training Set 1 (1) |

## Dataset Usage Recommendations

### For Model Training
- All four datasets are sufficiently large (50-51 entries each) for evaluation purposes
- The datasets cover diverse BibTeX entry types, enabling comprehensive model testing
- Mixed quality levels provide realistic data conditions

### For Model Evaluation
- Use datasets in sequence (Stefan1→4) to assess model consistency across distributions
- Focus on Sets 2 & 3 for strategy comparison (uniform coverage across all strategies)
- Use Set 4 for RAG-specific evaluation (only RAG results available)

### Data Characteristics
- **Balanced Distribution:** Dataset quality varies reasonably across sets (valid ratios: 51.0%–52.0%)
- **Type Diversity:** @article, @inproceedings, and @book types well-represented
- **Error Distribution:** Realistic mix of completely invalid vs. partially-valid entries

