"""
STAGE 2 - score a variant's cached answers with the four RAGAS metrics.

    python evaluate_ragas.py --variant naive
    python evaluate_ragas.py --variant reranked

Metric  <->  what it needs  <->  what it diagnoses
  context_precision (LLMContextPrecisionWithReference)  question, contexts, reference
        RETRIEVER RANKING: are the relevant chunks at the TOP of the list?
  context_recall (LLMContextRecall)                      question, contexts, reference
        RETRIEVER COVERAGE: is everything the reference answer says present in the contexts?
  faithfulness (Faithfulness)                            question, answer, contexts
        GENERATOR GROUNDING: is every claim in the answer supported by the contexts?
  answer_relevancy (ResponseRelevancy)                   question, answer (+ embeddings)
        GENERATOR FOCUS: does the answer address the question (not waffle / refuse)?

The judge is a different model from the generator and runs at temperature 0.
"""
import argparse
import json
import warnings

import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

import config

warnings.filterwarnings("ignore", category=DeprecationWarning)   # legacy-path notice, see README
from ragas.metrics import (  # noqa: E402
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

METRIC_COLUMNS = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]


def build_judge():
    llm = ChatOpenAI(
        model=config.JUDGE_MODEL,
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
        temperature=0,
        timeout=config.JUDGE_TIMEOUT_S,
        max_retries=config.JUDGE_MAX_RETRIES,
    )
    emb = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    # bypass_n=True: our LLM gateway (LiteLLM -> Ollama) rejects the OpenAI `n` parameter.
    # RAGAS metrics that want several samples (answer_relevancy asks for 3 generated
    # questions) then get n separate single-sample requests instead of one request with n=3.
    return LangchainLLMWrapper(llm, bypass_n=True), LangchainEmbeddingsWrapper(emb)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    args = ap.parse_args()

    rows = json.loads((config.RESULTS_DIR / f"{args.variant}_answers.json").read_text())[: args.limit]
    dataset = EvaluationDataset.from_list([
        {
            "user_input": r["question"],
            "retrieved_contexts": r["contexts"],
            "response": r["answer"],
            "reference": r["ground_truth"],
        }
        for r in rows
    ])
    judge_llm, judge_emb = build_judge()
    result = evaluate(
        dataset,
        metrics=[
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
            Faithfulness(),
            ResponseRelevancy(),
        ],
        llm=judge_llm,
        embeddings=judge_emb,
        run_config=RunConfig(timeout=config.JUDGE_TIMEOUT_S, max_workers=config.JUDGE_MAX_WORKERS,
                             max_retries=config.JUDGE_MAX_RETRIES),
        raise_exceptions=False,          # a failed judge call becomes NaN, not a crash
    )
    df = result.to_pandas()
    # RAGAS names this column after the class; shorten it to the conventional metric name.
    df = df.rename(columns={"llm_context_precision_with_reference": "context_precision"})
    df.insert(0, "id", [r["id"] for r in rows])
    df.insert(1, "type", [r["type"] for r in rows])
    missing = set(METRIC_COLUMNS) - set(df.columns)
    assert not missing, f"RAGAS did not return expected metric columns: {missing} (see ENGINEERING_LOG #3)"
    score_cols = METRIC_COLUMNS
    df["hit_at_k"] = [r["hit_at_k"] for r in rows]
    df["reciprocal_rank"] = [r["reciprocal_rank"] for r in rows]

    out_csv = config.RESULTS_DIR / f"{args.variant}_ragas.csv"
    df.to_csv(out_csv, index=False)
    summary = {
        "variant": args.variant,
        "n_questions": len(df),
        "judge": config.JUDGE_MODEL,
        "generator": config.GENERATOR_MODEL,
        "top_k": config.TOP_K,
        "metrics_mean": {c: round(float(df[c].mean()), 4) for c in score_cols},   # NaN skipped
        "metrics_nan_count": {c: int(df[c].isna().sum()) for c in score_cols},
        "hit_at_k": round(float(df["hit_at_k"].mean()), 4),
        "mrr": round(float(df["reciprocal_rank"].mean()), 4),
    }
    (config.RESULTS_DIR / f"{args.variant}_summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 200)
    print(df[["id", "type", *score_cols, "hit_at_k"]].round(3).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
