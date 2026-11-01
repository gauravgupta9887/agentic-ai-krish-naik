# Naive RAG vs RAG + reranking

15 questions · generator `minimax-local` · judge `gpt-oss:20b` · top_k=3 · reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` over top-20 candidates

## Headline

| metric | naive | reranked | delta | q better / worse / tied | sign-test p |
|---|---:|---:|---:|---:|---:|
| context_precision | 0.833 | 0.878 | +0.044 | 4 / 1 / 10 | 0.375 |
| context_recall | 0.833 | 0.967 | +0.133 | 3 / 0 / 12 | 0.250 |
| faithfulness | 0.720 | 0.983 | +0.263 | 7 / 0 / 8 | 0.016 |
| answer_relevancy | 0.585 | 0.757 | +0.172 | 5 / 8 / 2 | 0.581 |
| hit_at_k | 0.733 | 0.933 | +0.200 | 3 / 0 / 12 | 0.250 |
| reciprocal_rank | 0.700 | 0.856 | +0.156 | 4 / 1 / 10 | 0.375 |

## Per question (reranked minus naive)

| id | type | context_precision | context_recall | faithfulness | answer_relevancy | hit_at_k | reciprocal_rank |
|---|---|---:|---:|---:|---:|---:|---:|
| q01 | definition | 1.00 | 0.50 | 0.75→1.00 | 0.87→0.71 | 0.00→1.00 | 0.00→1.00 |
| q02 | numeric-lookup | 1.00 | 1.00 | 0.50→1.00 | 0.84→0.84 | 1.00 | 1.00→0.50 |
| q03 | table-lookup | 1.00 | 1.00 | 1.00 | 0.56→0.53 | 1.00 | 1.00 |
| q04 | table-aggregate | 1.00 | 1.00 | 1.00 | 0.98→0.98 | 1.00 | 1.00 |
| q05 | list-multi-chunk | 0.00 | 0.50→1.00 | 0.50→1.00 | 0.43→0.39 | 1.00 | 1.00 |
| q06 | reasoning | 0.83→1.00 | 1.00 | 0.00→1.00 | 0.00→0.95 | 0.00→1.00 | 0.00→1.00 |
| q07 | table-lookup | 0.00→0.33 | 0.00→1.00 | 0.00→1.00 | 0.00→0.80 | 0.00→1.00 | 0.00→0.33 |
| q08 | definition | 0.83→1.00 | 0.50→1.00 | 0.25→0.75 | 0.00 | 0.00 | 0.00 |
| q09 | factoid | 0.83→1.00 | 1.00 | 1.00 | 0.91→0.94 | 1.00 | 1.00 |
| q10 | definition | 1.00 | 1.00 | 1.00 | 1.00→0.88 | 1.00 | 1.00 |
| q11 | reasoning | 1.00→0.83 | 1.00 | 1.00 | 0.67→0.65 | 1.00 | 0.50→1.00 |
| q12 | factoid | 1.00 | 1.00 | 0.80→1.00 | 0.96→0.95 | 1.00 | 1.00 |
| q13 | factoid | 1.00 | 1.00 | 1.00 | 0.33→0.84 | 1.00 | 1.00 |
| q14 | factoid | 1.00 | 1.00 | 1.00 | 0.31→0.99 | 1.00 | 1.00 |
| q15 | list-lookup | 1.00 | 1.00 | 1.00 | 0.91 | 1.00 | 1.00 |

First-stage ceiling: gold evidence is inside the top-20 candidate pool for 15/15 questions (a reranker cannot recover what the first stage never retrieved).
