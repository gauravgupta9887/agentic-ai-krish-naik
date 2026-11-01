"""
STAGE 1 - run a RAG variant over the QA set and cache everything the evaluator needs.

    python generate.py --variant naive
    python generate.py --variant reranked

Output: results/<variant>_answers.json  (question, answer, contexts, retrieval log, hit flags)

Why a separate, cached stage: generation and judging are the two expensive, noisy,
LLM-bound steps. Splitting them means (a) you can re-judge the SAME answers with a
different judge/metric config without regenerating, and (b) the two variants are compared
on stored artefacts you can open and read, not on numbers from a black box.

Also computed here, with NO LLM involved (so it is noise-free and a useful cross-check on
the LLM-judged RAGAS numbers):
    hit@k  : does any of the k retrieved chunks contain the gold evidence span?
    mrr    : 1 / (rank of the first chunk containing the gold span), 0 if none
    hit@N  : (reranked only) same check over the whole first-stage candidate pool -
             the CEILING: a reranker can only reorder what the first stage found.
"""
import argparse
import json
import time

import config
from rag import NaiveRAG, RerankedRAG, VectorIndex, evidence_hit

VARIANTS = {"naive": NaiveRAG, "reranked": RerankedRAG}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    args = ap.parse_args()

    qa = json.loads(config.GROUND_TRUTH.read_text())[: args.limit]
    index = VectorIndex()
    rag = VARIANTS[args.variant](index)
    print(f"[{args.variant}] {len(index.chunks)} chunks, top_k={config.TOP_K}, {len(qa)} questions")

    rows = []
    for item in qa:
        t0 = time.time()
        out = rag.run(item["question"])
        ev = item["evidence"]
        hits = [evidence_hit(ev, [c]) for c in out["contexts"]]
        out.update(
            id=item["id"],
            type=item["type"],
            ground_truth=item["ground_truth"],
            evidence=ev,
            hit_at_k=any(hits),
            reciprocal_rank=(1 / (hits.index(True) + 1)) if any(hits) else 0.0,
            gen_seconds=round(time.time() - t0, 2),
        )
        if args.variant == "reranked":                    # the ceiling: whole candidate pool
            pool = index.search(item["question"], config.RERANK_CANDIDATES)
            out["hit_at_pool"] = evidence_hit(ev, [r.chunk.text for r in pool])
        rows.append(out)
        print(f"  {item['id']} hit@k={out['hit_at_k']!s:<5} rr={out['reciprocal_rank']:.2f}  "
              f"-> {out['answer'][:90]!r}")

    config.RESULTS_DIR.mkdir(exist_ok=True)
    path = config.RESULTS_DIR / f"{args.variant}_answers.json"
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    n = len(rows)
    print(f"\nhit@{config.TOP_K} = {sum(r['hit_at_k'] for r in rows)}/{n}   "
          f"MRR = {sum(r['reciprocal_rank'] for r in rows) / n:.3f}   saved -> {path}")


if __name__ == "__main__":
    main()
