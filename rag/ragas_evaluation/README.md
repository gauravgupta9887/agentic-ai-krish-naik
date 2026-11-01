# Evaluating RAG with RAGAS: naive retrieval vs. retrieval + reranking

A hands-on experiment, written to be **learned from**, not just run.

We take a small document corpus, hand-write **15 ground-truth questions**, build a **naive RAG**
pipeline, score it with the **four core RAGAS metrics**, then add a **cross-encoder reranker**
and score it again. Everything else is held constant, so the metric deltas are attributable to
the reranker (with caveats, covered honestly below).

| Document | What it is | Read this when |
|---|---|---|
| **README.md** (this file) | Overview, results, analysis, how to run | Start here |
| [CONCEPTS.md](CONCEPTS.md) | Theory: the 4 metrics (formulas + how they work), bi- vs cross-encoders, LLM-as-judge, statistics for small eval sets | You want to understand *why* |
| [ENGINEERING_LOG.md](ENGINEERING_LOG.md) | Every problem hit while building this, and the fix | You want to avoid the same traps |
| [results/COMPARISON.md](results/COMPARISON.md) | Auto-generated side-by-side tables | You want raw numbers |

---

## 1. Results at a glance

15 questions · generator `minimax-local` · judge `gpt-oss:20b` (a *different* model) · top-k = 3 ·
reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` re-scoring the top-20 dense candidates.

| Metric | Naive | + Rerank | Δ | questions better / worse / tied | sign-test p |
|---|---:|---:|---:|:---:|---:|
| **Context Precision** | 0.833 | 0.878 | +0.044 | 4 / 1 / 10 | 0.375 |
| **Context Recall** | 0.833 | 0.967 | +0.133 | 3 / 0 / 12 | 0.250 |
| **Faithfulness** | 0.720 | 0.983 | **+0.263** | 7 / 0 / 8 | **0.016** |
| **Answer Relevancy** | 0.585 | 0.757 | +0.172 | 5 / 8 / 2 | 0.581 |
| *hit@3 (no LLM)* | 0.733 | 0.933 | +0.200 | 3 / 0 / 12 | 0.250 |
| *MRR (no LLM)* | 0.700 | 0.856 | +0.156 | 4 / 1 / 10 | 0.375 |

The last two rows are **deterministic** retrieval metrics (is the gold evidence span inside the
retrieved chunks? at what rank?). They need no LLM and are the cross-check on the four judged metrics.

### The one-paragraph takeaway

Reranking helped, mostly by **getting the right chunk into the top 3**: hit@3 went from 11/15 to
14/15. Because the generator can only answer from what it is shown, that flowed through to
**recall (+0.13) and faithfulness (+0.26)**. Context *precision* barely moved (+0.04) because it was
already high. Answer relevancy rose, but for a mostly-wrong reason (§4.3). **With n = 15 and a single
run, only the faithfulness gain is statistically distinguishable from noise, and even that does not
survive a multiple-comparisons correction.** Read these numbers as "consistent, plausible evidence",
not proof. §5 explains how to firm them up.

---

## 2. What is in this folder

```
rag/ragas_evaluation/
├── README.md / CONCEPTS.md / ENGINEERING_LOG.md   documentation
├── config.py                  ALL knobs (models, chunking, k, judge settings) in one file
├── corpus.py                  PDF/TXT loading + NFKC normalisation + empty-PDF handling
├── rag.py                     VectorIndex, NaiveRAG, RerankedRAG, evidence_hit()
├── ground_truth.json          the 15 hand-written QA items (+ gold evidence span each)
├── validate_ground_truth.py   checks every gold span really exists in the corpus
├── generate.py                STAGE 1: run a variant over the QA set, cache answers + retrieval log
├── evaluate_ragas.py          STAGE 2: score cached answers with the 4 RAGAS metrics
├── compare.py                 STAGE 3: side-by-side tables + paired sign test
├── run_all.sh                 validate → generate ×2 → evaluate ×2 → compare
├── corpus/                    5 documents (see §3.1)
├── results/                   *_answers.json, *_ragas.csv, *_summary.json, COMPARISON.md
└── requirements.txt           isolated-venv dependencies (note the pins!)
```

Pipeline data flow:

```
 ground_truth.json ──► generate.py ──► results/<variant>_answers.json ──► evaluate_ragas.py ──► results/<variant>_ragas.csv
                         │  (retrieve → [rerank] → LLM answer)                │ (judge LLM + embeddings)
                         └─ also computes hit@k / MRR / hit@pool (no LLM)     └─► compare.py ──► results/COMPARISON.md
```

**Why two cached stages?** Generation and judging are the two slow, noisy, LLM-bound steps.
Separating them means you can re-judge the *same* answers with a different judge or metric config
without regenerating, and every number can be traced to a stored artefact you can open and read.

---

## 3. Experimental design

### 3.1 Corpus (5 documents, 87 chunks)

| File | Content | Role |
|---|---|---|
| `Information_Retrieval_and_Text_Processing_Guide_6308.pdf` | 15-page guide on chunking; has prose, a table, a numbered list, a hierarchy | Main source: 12 of 15 questions |
| `BASIC_DSA_TRCIKS.pdf`, `Tricks_for_DSA_Problem_Solving.pdf` | Slide-deck-style DSA interview tips | **Distractors** + 2 questions |
| `machine_learning.txt`, `python_intro.txt` | Tiny text files | Distractors + 1 question |
| ~~`git.pdf`~~ | **Excluded**: image-only, `pypdf` extracts 0 characters | See ENGINEERING_LOG #5 |

A multi-document corpus on *unrelated* topics matters: with one homogeneous document, retrieval is
trivially easy and there is nothing for a reranker to fix.

### 3.2 The only variable: the reranker

| | Naive | Reranked |
|---|---|---|
| Corpus, chunking (500 chars / 50 overlap), embedding model | same | same |
| Generator model, prompt, temperature 0 | same | same |
| Judge model, metric config | same | same |
| Chunks given to the generator (**k**) | **3** | **3** |
| First-stage retrieval | dense top-**3** | dense top-**20** |
| Re-ordering | none | cross-encoder re-scores the 20, keep top **3** |

`k` is held at 3 **deliberately**. If the reranked variant fed the LLM 5 chunks, you could not
tell whether gains came from better *ordering* or just from *more context*.

### 3.3 Ground truth: hand-written, with evidence spans

Each of the 15 items has: `question`, `ground_truth` (reference answer), `type`
(factoid / definition / table-lookup / table-aggregate / list-multi-chunk / reasoning / numeric-lookup),
`source` file, and **`evidence`**, a short verbatim span from the corpus that must be present for the
question to be answerable.

Why hand-written rather than LLM-generated (RAGAS can synthesize test sets)? The judge LLM is going
to grade answers *against* these references. If an LLM also wrote them, you are measuring agreement
between two language models, not correctness. For a staff-level eval, humans own the ground truth.

`validate_ground_truth.py` verifies, before any LLM call, that each evidence span occurs in its
claimed source, occurs in **no other** file (ambiguity check), and that ids/fields are valid. A typo in
ground truth otherwise becomes a silent, permanent ceiling on your scores.

The evidence span also powers the **deterministic** metrics `hit@k` / `MRR` / `hit@pool`, which is
independent of any LLM judge.

---

## 4. Analysis: what actually happened

### 4.1 Retrieval: what the reranker did

`hit@3`: **11/15 → 14/15**. The three questions it fixed (q01, q06, q07) were all cases where the
right chunk sat at dense rank 6, 19 and 6 respectively. A top-3 cutoff could never reach it:

| q | Gold chunk's dense rank | Naive top-3 | Reranked top-3 |
|---|---|---|---|
| q06 (*token overhead*) | **19** | missed, answered "I don't know" | cross-encoder pulled it from rank 19 into the top 3 |
| q01 (*hard cut*) | 6 | missed | pulled into top 3 |
| q07 (*semantic chunking drawback*) | 6 | missed, answered "I don't know" | pulled into top 3 |

q06 is the clearest illustration of **why you rerank**: a bi-encoder had the answer at position 19
out of 87, and a top-3 cutoff would never have seen it. The cross-encoder, which reads question and
chunk *jointly*, recognised it.

**First-stage ceiling: 15/15.** The gold evidence was inside the top-20 pool for *every* question.
A reranker can only reorder what the first stage found, so it had the maximum possible headroom here.
In real systems, measure this number first (`hit@pool`): if it is low, fix recall (chunking, hybrid
search, query rewriting) before buying a reranker.

### 4.2 Where reranking did NOT help (read these; they teach more than the wins)

**q08 (*dilution effect*): missed by both.** The gold chunk was at dense rank 5; the cross-encoder
ranked it **4th**, one place outside the top 3. Both variants then produced an honest "not defined in
the context". Reranking is a probabilistic improvement, not a guarantee. A reranker with k=3 can also
*demote* a gold chunk it previously had in range: q02's MRR fell from 1.0 to 0.5 (gold chunk went from
rank 1 to rank 2; still in top-3, so the answer was fine, but ordering got worse).

**q05 (*first and last shutdown steps*) is the "hard cut" problem live, in our own pipeline.** The
numbered list was split by the chunker. The naive top-1 chunk begins at *step 2* (so its step 1 is
elsewhere), and the model confidently answered "the first step is to disable the primary load
balancer", **a wrong answer, grounded in the retrieved text.** Reranking pulled in the chunk containing
step 1 and the answer became correct. The guide we are indexing literally warns about this failure.

### 4.3 Reading the four RAGAS metrics honestly

**Context Precision (+0.044).** Barely moved, mainly because it was already 0.83 (little headroom).
Note a **judge-vs-deterministic disagreement**: for q05 the gold chunk *was* retrieved at rank 1
(`hit@k = True`) yet context precision is **0.00** in both variants. The LLM judge decided none of the
chunks "was useful to arrive at the reference answer" (the reference needs *two* facts, step 1 and
step 6, and no single chunk had both). This is a known behaviour of reference-based precision on
multi-fact answers. It is why we kept the deterministic hit/MRR next to it.

**Context Recall (+0.133).** Tracks hit@3 closely (+0.20): once the right chunk is present, the
judge can attribute the reference's claims to it. The 3 improving questions are the 3 retrieval fixes.

**Faithfulness (+0.263), a real gain with a misleading component.** 7 questions improved, 0 got
worse, the only statistically notable result. But look *why*. Faithfulness = (answer claims supported
by context) ÷ (all answer claims). For q06 and q07 the naive answer was the refusal *"I don't know
based on the provided context"*, which scored **0.00**. A refusal contains no supported factual claim, so
the judge scores it zero. So part of the gain is "model answered instead of refusing", **not** "model
hallucinated less". An honest refusal is arguably the *most* faithful behaviour; the metric does not see
it that way. Always read answers next to scores.

**Answer Relevancy (+0.172; but 8 questions got *worse* and 5 better).** The mean rose because four
questions jumped a lot (q06 0.00→0.95, q07 0.00→0.80, q13 0.33→0.84, q14 0.31→0.99) while the eight
"worse" ones moved by tiny amounts (−0.001 to −0.16; six of the eight within ±0.05, i.e. judge/wording noise). Two lessons:

1. **Magnitude vs sign:** the sign test says 5-vs-8 (no effect), the mean says +0.17. Both are
   "right"; they answer different questions. Look at the distribution, not one summary.
2. **The metric rewards echoing the question.** q13 and q14 *both* retrieved the right chunk in *both*
   variants. The jump came purely from answer wording: the reranked answer restated the question
   ("…the DSA tips recommend using a Heap when dealing with top/maximum/minimum/closest 'K'
   elements…") so questions regenerated *from the answer* were closer to the original. Answer
   relevancy measures topical similarity, **not correctness**. Naive q01's answer was *wrong* ("the
   context does not define it") yet it once scored 0.87.

### 4.4 Judge noise is not hypothetical, we measured it by accident

During a 2-question smoke test, naive **q02** received `faithfulness = 1.0`. In the full run the
**identical answer text** ("Based on the context, Chunk 2 covers tokens 400 to 900.") with identical
retrieved chunks received `faithfulness = 0.5`. Same input, same judge, temperature 0, scores a full
0.5 apart. Naive q01 shows a similar pattern (`answer_relevancy` 0.00 in the smoke test, 0.87 in the
full run; same retrieved chunks, same answer prefix).

This is the single most important thing to internalise about LLM-as-judge: **a per-question score is a
noisy sample, not a measurement.** At n=15 with one run each, a 0.04 delta (context precision) is
indistinguishable from noise, and even 0.17 deltas deserve suspicion.

---

## 5. Limitations, and how a staff engineer would harden this

| Threat to validity | Present here? | Mitigation |
|---|---|---|
| **Small n** (15) | yes | 100+ questions; report bootstrap CIs, not point estimates |
| **Single run, no judge-variance estimate** | yes (§4.4) | Re-judge each answer 3-5× (cheap: answers are cached) and report mean ± sd; only trust deltas larger than the judge's own sd |
| **Multiple comparisons** | yes: 6 sign tests at α=0.05; faithfulness p=0.016 → 0.096 after Bonferroni | Pre-register the primary metric; correct for the rest |
| **Generator sampling noise** | partly (temperature 0, but backend is not guaranteed deterministic) | Generate several times |
| **Judge model weaker than the task** (20B local) | likely | Spot-check 20 judgments by hand; compare against a stronger judge |
| **Self-preference bias** | mitigated: generator ≠ judge | Keep them different |
| **Ground-truth coverage**: all questions are answerable from the corpus | yes | Add *unanswerable* questions; faithfulness/"refuse correctly" behaviour is invisible otherwise |
| **Question mix skews to one document** (12/15) | yes | Stratify by source/type; report per-type results |
| **No latency / cost axis** | yes | Reranking adds a cross-encoder pass per candidate; measure p50/p95 latency before shipping. Measured here: 0.64 s/question naive vs 1.06 s reranked (LLM call included), so the cross-encoder over 20 candidates on CPU added ≈ 0.4 s |

---

## 6. How to run it

### Prerequisites
* Python 3.12, [`uv`](https://docs.astral.sh/uv/)
* An OpenAI-compatible LLM endpoint. This repo's `.env` provides `CHAT_API_BASE_URL` and
  `CHAT_API_KEY` (company gateway serving `minimax-local` and `gpt-oss:20b`). Change models in `config.py`.
* Internet once, to download `all-MiniLM-L6-v2` and the cross-encoder from Hugging Face.

### Setup (isolated venv: required, see ENGINEERING_LOG #1)
```bash
cd rag/ragas_evaluation
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

### Run everything
```bash
./run_all.sh                       # ≈ 12-15 min, dominated by the judge LLM
```
or stage by stage (this is the recommended way to *learn*; open each output between steps):
```bash
.venv/bin/python validate_ground_truth.py            # 0 LLM calls
.venv/bin/python generate.py --variant naive         # inspect results/naive_answers.json
.venv/bin/python generate.py --variant reranked
.venv/bin/python evaluate_ragas.py --variant naive   # ~6 min
.venv/bin/python evaluate_ragas.py --variant reranked
.venv/bin/python compare.py
```
Smoke-test cheaply with `--limit 2` on `generate.py` / `evaluate_ragas.py`.

`.env` not found (e.g. in a git worktree, where the git-ignored file isn't checked out)?
`export RAGAS_ENV_FILE=/path/to/.env`.

---

## 7. Exercises (in rough order of learning value)

1. **Measure judge noise.** Re-run `evaluate_ragas.py --variant naive` three times. Compute the
   per-question sd. Which metric is noisiest? Is any reported delta smaller than that sd?
2. **Read the claims.** Open the RAGAS per-claim verdicts (faithfulness statements, recall attributions)
   for q05 and q06 and decide whether you agree with the judge.
3. **Vary k.** Set `TOP_K = 5` for naive. Does naive k=5 close the gap to reranked k=3? (This is the
   "just retrieve more" baseline every reranker must beat. Reranking's real value is *quality per
   context token*.)
4. **Break the ceiling.** Set `RERANK_CANDIDATES = 5`. `hit@pool` falls; watch reranking's benefit
   shrink. Then set it to 40 and watch latency rise for little gain.
5. **Chunk size sweep** (200 / 500 / 1000) with and without reranking. Does the reranker matter
   more when chunks are small? (Hypothesis: yes: smaller chunks → more ranking errors to fix.)
6. **Add unanswerable questions** and a metric for correct refusal. Faithfulness currently *penalises*
   refusals (§4.3); design something that does not.
7. **Hybrid retrieval.** Add BM25 and fuse with dense (reciprocal-rank fusion) *before* reranking.
8. **Swap the judge** for a different model and compare rankings of the two variants. Do they agree?
9. **Stratify** the results by question `type`. Which kinds of questions benefit from reranking?

---

## 8. Further reading
* Es et al., *RAGAS: Automated Evaluation of Retrieval Augmented Generation* (2023), arXiv:2309.15217
* Nogueira & Cho, *Passage Re-ranking with BERT* (2019), arXiv:1901.04085: the origin of cross-encoder reranking
* Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (2023), arXiv:2306.05685: judge biases (position, verbosity, self-preference)
* RAGAS docs: metric definitions: <https://docs.ragas.io>
