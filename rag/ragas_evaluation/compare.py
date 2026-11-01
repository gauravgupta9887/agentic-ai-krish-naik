"""
STAGE 3 - put the two variants side by side.

Writes results/COMPARISON.md with:
  1. headline table (RAGAS means + deterministic retrieval metrics + deltas)
  2. per-question deltas, so you can see WHICH questions moved and why
  3. a paired sign test on the per-question differences, because with n=15 a mean
     difference alone says very little about whether the change is real
"""
import json
from math import comb

import pandas as pd

import config

METRICS = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
RETRIEVAL = ["hit_at_k", "reciprocal_rank"]


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact binomial sign test, ties discarded. H0: P(better) = 0.5."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def main() -> None:
    a = pd.read_csv(config.RESULTS_DIR / "naive_ragas.csv").set_index("id")
    b = pd.read_csv(config.RESULTS_DIR / "reranked_ragas.csv").set_index("id")
    cols = METRICS + RETRIEVAL
    a[RETRIEVAL] = a[RETRIEVAL].astype(float)
    b[RETRIEVAL] = b[RETRIEVAL].astype(float)

    lines = ["# Naive RAG vs RAG + reranking", "",
             f"{len(a)} questions · generator `{config.GENERATOR_MODEL}` · judge `{config.JUDGE_MODEL}` · "
             f"top_k={config.TOP_K} · reranker `{config.RERANK_MODEL}` over top-{config.RERANK_CANDIDATES} candidates", "",
             "## Headline", "",
             "| metric | naive | reranked | delta | q better / worse / tied | sign-test p |",
             "|---|---:|---:|---:|---:|---:|"]
    for c in cols:
        d = (b[c] - a[c]).dropna()
        w, l, t = int((d > 1e-9).sum()), int((d < -1e-9).sum()), int((d.abs() <= 1e-9).sum())
        lines.append(f"| {c} | {a[c].mean():.3f} | {b[c].mean():.3f} | {b[c].mean() - a[c].mean():+.3f} "
                     f"| {w} / {l} / {t} | {sign_test_p(w, l):.3f} |")

    lines += ["", "## Per question (reranked minus naive)", "",
              "| id | type | " + " | ".join(cols) + " |", "|---|---|" + "---:|" * len(cols)]
    for qid in a.index:
        cells = []
        for c in cols:
            x, y = a.loc[qid, c], b.loc[qid, c]
            cells.append("n/a" if pd.isna(x) or pd.isna(y) else f"{x:.2f}→{y:.2f}" if abs(y - x) > 1e-9 else f"{x:.2f}")
        lines.append(f"| {qid} | {a.loc[qid, 'type']} | " + " | ".join(cells) + " |")

    ans = {r["id"]: r for r in json.loads((config.RESULTS_DIR / "reranked_answers.json").read_text())}
    pool = [r.get("hit_at_pool") for r in ans.values()]
    lines += ["", f"First-stage ceiling: gold evidence is inside the top-{config.RERANK_CANDIDATES} candidate pool "
                  f"for {sum(bool(p) for p in pool)}/{len(pool)} questions "
                  f"(a reranker cannot recover what the first stage never retrieved).", ""]
    out = config.RESULTS_DIR / "COMPARISON.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
