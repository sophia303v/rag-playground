# Experiment Comparison: Baseline (TOP_K=3) vs TOP_K=5

Date: 2026-02-23

## Summary Metrics

| Metric | Baseline (K=3) | TOP_K=5 | Delta |
|---|---|---|---|
| **TOP_K** | 3 | 5 | +2 |
| **Context Precision (mean)** | 0.321 | 0.216 | -0.105 |
| **Context Precision (max)** | 1.000 | 0.667 | -0.333 |
| **Context Recall (mean)** | 0.642 | 0.694 | **+0.052** |
| **Context Recall (max)** | 1.000 | 1.000 | -- |

## Category-Level Comparison

| Category | Precision (K=3) | Precision (K=5) | Recall (K=3) | Recall (K=5) |
|---|---|---|---|---|
| comparative | 0.217 | 0.222 | 0.233 | 0.342 |
| diagnostic | 0.300 | 0.257 | 0.800 | **1.000** |
| factual | 0.350 | 0.188 | 0.800 | 0.800 |
| procedural | 0.433 | 0.273 | 0.667 | 0.667 |

## Recall Improvements (recall=0 -> recall>0)

| Question ID | Category | Recall (K=3) | Recall (K=5) | Notes |
|---|---|---|---|---|
| q22 | comparative | 0.0 | 0.5 | Now retrieves CXR_0017 (1/2 relevant UIDs) |
| q29 | comparative | 0.0 | 0.25 | Now retrieves CXR_0010 (1/4 relevant UIDs) |
| q34 | diagnostic | 0.0 | **1.0** | Now retrieves CXR_0011 -- fully fixed |

Regressions: **none** (no question had recall decrease)

## Remaining Recall=0 Questions (8 total)

| Question ID | Category | Question |
|---|---|---|
| q01 | factual | What are the findings in report CXR_0001? |
| q03 | factual | Clinical indication for CXR_0002? |
| q08 | factual | Findings suggest malignancy in CXR_0006? |
| q09 | factual | What does CXR_0009 show about lungs? |
| q26 | comparative | Compare interstitial findings CXR_0007 and CXR_0014 |
| q28 | comparative | Right upper lobe findings CXR_0005, CXR_0006, CXR_0011 |
| q30 | comparative | Which reports recommend CT and why? |
| q37 | procedural | ETT tip position in CXR_0016? |

These questions are not helped by increasing TOP_K -- the embedding model fails to match the query to the correct chunks.

## Precision Drop Examples

| Question ID | Precision (K=3) | Precision (K=5) |
|---|---|---|
| q07 | 1.000 | 0.333 |
| q02 | 0.333 | 0.200 |
| q04 | 0.333 | 0.200 |
| q05 | 0.333 | 0.200 |
| q10 | 0.500 | 0.250 |

## Conclusion

Classic precision-recall tradeoff. TOP_K=5 modestly improves recall (+0.052) at the cost of substantial precision drop (-0.105). The 8 stubborn recall=0 questions need better embeddings, chunking strategy, or a re-ranking step rather than further TOP_K increases.

## Source Experiments

- Baseline: `experiments/2026-02-20_baseline/`
- TOP_K=5: `experiments/2026-02-23_topk5/`
