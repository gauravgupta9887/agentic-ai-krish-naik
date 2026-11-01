# Concepts: the theory behind the experiment

Read this to understand **why** each piece exists. Every formula below was checked against the
installed RAGAS 0.4.3 source (`ragas/metrics/_context_precision.py`, `_context_recall.py`,
`_faithfulness.py`, `_answer_relevance.py`), not recalled from memory.

Contents
1. [Why RAG needs its own evaluation](#1-why-rag-needs-its-own-evaluation)
2. [Where RAG fails: a taxonomy](#2-where-rag-fails-a-taxonomy)
3. [What RAGAS is (and what "LLM-as-judge" costs you)](#3-what-ragas-is)
4. [The four metrics, in depth](#4-the-four-metrics-in-depth)
5. [Which metric diagnoses which component](#5-which-metric-diagnoses-which-component)
6. [Reranking: bi-encoders vs cross-encoders](#6-reranking)
7. [Deterministic retrieval metrics: hit@k, MRR, the ceiling](#7-deterministic-retrieval-metrics)
8. [Evaluating with small samples: statistics you actually need](#8-statistics-for-small-eval-sets)
9. [LLM-as-judge: biases and defences](#9-llm-as-judge-biases-and-defences)
10. [Glossary](#10-glossary)

---

## 1. Why RAG needs its own evaluation

A RAG system is **two systems in a trench coat**: a *retriever* (search) and a *generator* (LLM).
End-to-end "did the answer look right?" hides which half broke. A wrong answer can mean:

* the right information was never retrieved (retriever's fault), or
* it *was* retrieved but buried / outvoted by noise (ranking's fault), or
* it was in the context but the model ignored or contradicted it (generator's fault), or
* the model answered something true but off-topic (generator's focus).

Each has a different fix (chunking / reranking / prompting / model choice). Metrics that map onto
those failure modes let you spend effort in the right place. That is the entire point of RAGAS's
four-metric design.

Classical NLP metrics (BLEU, ROUGE, exact match) fail here: "Chunk 2 covers tokens 400 to 900" and
"It spans 400–900" are the same answer with almost zero n-gram overlap. So RAGAS uses an **LLM as the
judge** to compare *meaning*.

---

## 2. Where RAG fails: a taxonomy

| Failure | Symptom | Example from *this* experiment | Fix lives in |
|---|---|---|---|
| **Retrieval miss**: gold info not in top-k | answer wrong or "I don't know" | q06: gold chunk at dense rank **19**; naive top-3 never saw it | chunking, hybrid search, bigger k, reranking |
| **Ranking error**: gold in the pool but low | same as above | q01, q07: gold at dense rank 6 | **reranking** |
| **Chunking damage**: answer split across chunks | partially-right, confident answer | q05: list split; top chunk began at step 2 → model said "first step is disable the load balancer" | chunker (structure-aware, overlap) |
| **Context noise**: irrelevant chunks in prompt | distraction, lower precision | DSA-deck chunks retrieved for IR questions | ranking, smaller k, filters |
| **Unfaithful generation**: claims not in context | hallucination | naive faithfulness averaged 0.72 (q02 0.50, q05 0.50), but see the refusal quirk in §4.3: part of that is refusals scoring 0 | prompt, model, decoding |
| **Irrelevant generation**: true but off-question | low relevancy | verbose/hedging answers | prompt |
| **Over-refusal**: "I don't know" though the answer was retrievable | refusal score 0 | q06/q07 naive | retrieval (the refusal was *correct* given the context!) |

Note the last row: the generator behaved **correctly** (refused when context lacked the answer), and
the root cause was upstream. Generation-only metrics cannot see that. Retrieval metrics can.

---

## 3. What RAGAS is

RAGAS (*Retrieval Augmented Generation Assessment*; Es et al. 2023) decomposes RAG quality into
metrics computed by prompting an LLM (and sometimes an embedding model) to make small, checkable
judgments, then aggregating them arithmetically. It is not magic: each metric is a short, readable
algorithm. That is good news: **you can audit it**.

Each metric is **reference-based** (needs your hand-written `reference`/`ground_truth` answer) or
**reference-free** (needs only the question, contexts, answer):

| Metric | Needs `reference`? | Needs `answer`? | Needs `contexts`? | Needs embeddings? |
|---|:-:|:-:|:-:|:-:|
| Context Precision (`LLMContextPrecisionWithReference`) | ✔ | | ✔ | |
| Context Recall (`LLMContextRecall`) | ✔ | | ✔ | |
| Faithfulness | | ✔ | ✔ | |
| Answer Relevancy (`ResponseRelevancy`) | | ✔ | | ✔ |

Reference-free metrics can run on **production traffic** with no labels; reference-based ones need a
curated set and are how you do **regression testing**. A mature setup uses both.

### Data format RAGAS expects (0.4.x)

```python
{"user_input": question,
 "retrieved_contexts": [chunk_text, ...],   # in RANK ORDER; context precision depends on order
 "response": generated_answer,
 "reference": ground_truth_answer}
```
Note `retrieved_contexts` **order matters** (see Context Precision). Older tutorials use
`question / contexts / answer / ground_truth`; same ideas, renamed columns.

---

## 4. The four metrics, in depth

Notation: question **q**, retrieved chunks **c₁…c_K** in rank order, generated answer **a**,
reference answer **r**. All scores ∈ [0, 1], higher is better.

### 4.1 Context Precision: *"Is the useful stuff ranked at the top?"*

**Algorithm.** For each chunk cₖ the judge LLM is asked: *given the question and the reference answer,
was this chunk useful in arriving at the reference answer?* → verdict vₖ ∈ {0, 1}. Then:

```
precision@k  = (v₁ + … + vₖ) / k
ContextPrecision = Σₖ ( precision@k · vₖ ) / Σₖ vₖ          (0 if no chunk is relevant)
```

This is **average precision**: it rewards relevant chunks appearing *early*.

Worked example, K = 3:

| verdicts (v₁,v₂,v₃) | calculation | score |
|---|---|---|
| 1, 0, 0 | (1/1·1) / 1 | **1.00** |
| 0, 1, 0 | (1/2·1) / 1 | 0.50 |
| 0, 0, 1 | (1/3·1) / 1 | 0.33 |
| 1, 1, 0 | (1·1 + 1·1)/2 | 1.00 |
| 1, 0, 1 | (1·1 + (2/3)·1)/2 | 0.83 |
| 0, 0, 0 | no relevant chunk | **0.00** |

**What it diagnoses:** ranking quality and noise. Same chunks, different *order* → different score.
**Blind spot:** it does not penalise *missing* information. One relevant chunk at rank 1 with two
junk chunks scores a perfect 1.0. Pair it with recall.
**Gotcha seen here (q05):** the reference needed **two** facts (step 1 *and* step 6). No single chunk
supported both, the judge marked each chunk "not useful on its own", and the score was **0.00** even
though the key chunk was retrieved at rank 1. Reference-based precision is harsh on multi-fact answers.

### 4.2 Context Recall: *"Did we retrieve everything needed?"*

**Algorithm.** The *reference answer* is split into sentences/claims. For each, the judge decides:
*can this claim be attributed to the retrieved contexts?* → attributed ∈ {0, 1}.

```
ContextRecall = (# reference claims attributable to contexts) / (# reference claims)
```

**What it diagnoses:** retriever coverage. It is the closest RAGAS gets to "was the information even
there?". **Key property:** it measures the *contexts against the reference*, not the answer, so the
generator's behaviour cannot affect it. A low recall + high faithfulness pattern means "the model was
honest about incomplete context", a retrieval problem.
**Depends on reference quality:** a long, multi-claim reference sets a high bar; a one-word reference
("Use a Heap.") has one claim and is nearly binary. Our reference lengths vary (1-3 claims); see
README §5 on stratification.

### 4.3 Faithfulness: *"Is every claim in the answer supported by the context?"*

**Algorithm (two LLM calls).**
1. *Decompose* the answer into atomic statements.
2. For each statement, an NLI-style check: *can this be inferred from the context?* → verdict ∈ {0,1}.

```
Faithfulness = (# statements supported by context) / (# statements)
```

**What it diagnoses:** hallucination / grounding, **relative to what was retrieved, not to the truth.**
If the retriever fetched a wrong passage and the model faithfully repeats it, faithfulness is **1.0**
and the answer is wrong. (Our q05 naive answer, "first step is to disable the load balancer", was
lifted straight from the chunk it saw, whose list began at step 2, and is still wrong. The judge happened to score it 0.50, not 1.0, so the metric is a noisy proxy for "grounded" too.)
**Known quirk:** an honest refusal ("I don't know based on the provided context") contains no
supported factual statement and scores **0.0** (observed: q06, q07 naive). So faithfulness gains can
partly reflect "answered instead of refusing". Always inspect answers beside scores.

### 4.4 Answer Relevancy: *"Does the answer address the question?"*

**Algorithm (reverse-engineer the question).**
1. From the *answer alone*, the judge generates **N = 3** questions this answer would respond to, and
   flags each as *noncommittal* (evasive / "I don't know").
2. Embed the original question and each generated question; take cosine similarities.

```
AnswerRelevancy = mean( cos(embed(q), embed(q̃ᵢ)) ) × [ 0 if ALL N are noncommittal else 1 ]
```

Intuition: if the answer is on-topic, the questions reconstructed from it resemble the original.

**What it diagnoses:** generator focus (waffle, off-topic, refusals).
**What it does NOT measure: correctness.** A fluent, on-topic, *wrong* answer scores high (naive q01's
wrong "context doesn't define it" answer scored 0.87 in the full run). It also **rewards answers that
echo the question's vocabulary**: q13/q14 jumped +0.5 / +0.7 with identical retrieval, purely because
the longer answer restated the question.
**Why it is the noisiest metric:** it multiplies a continuous cosine by a **discrete flag**. One judge
sample flipping to "noncommittal" for *all three* generated questions zeroes the score; otherwise it is
~0.85. That is how the same question went 0.00 → 0.87 between two runs (README §4.4).
**Uses an embedding model**, here the same local `all-MiniLM-L6-v2` as the retriever.

---

## 5. Which metric diagnoses which component

```
 question ─► [RETRIEVER] ─► chunks c₁..c_K ─► [GENERATOR] ─► answer
                 │                │                 │            │
         Context Recall    Context Precision    Faithfulness   Answer Relevancy
        (coverage vs ref)  (ranking/noise)     (answer vs ctx)  (answer vs q)
```

Diagnostic patterns (a staff-level cheat sheet):

| Recall | Precision | Faithfulness | Relevancy | Likely story |
|:-:|:-:|:-:|:-:|---|
| low | any | high | high-ish | Retriever missed it; model honestly worked with scraps → **fix retrieval** |
| high | low | any | any | Right chunk present but buried in noise → **rerank / smaller k** |
| high | high | **low** | any | Context is good, model strayed → **fix prompt / model** |
| high | high | high | **low** | Grounded but off-question or evasive → **fix prompt** |
| high | high | high | high | Healthy (verify with spot-checks, metrics can be fooled) |

---

## 6. Reranking

### 6.1 Two ways to score "how relevant is chunk d to query q?"

| | **Bi-encoder** (retriever) | **Cross-encoder** (reranker) |
|---|---|---|
| Architecture | Encode q and d **separately** → vectors; score = dot/cosine | Concatenate `[CLS] q [SEP] d` → **one** transformer pass → relevance score |
| Interaction between q and d | none until the final dot product | **full token-level cross-attention** |
| Accuracy | good, coarse | much higher |
| Cost | doc vectors computed **once offline**; query = 1 forward pass; search = ANN lookup | **one forward pass per (q, d) pair at query time**; nothing precomputable |
| Scales to 1M docs? | yes | no (1M passes per query) |
| Model here | `all-MiniLM-L6-v2` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

### 6.2 Why we combine them: the funnel

```
 87 chunks ─[bi-encoder, cheap]─► top-20 candidates ─[cross-encoder, accurate]─► top-3 ─► LLM
```

The bi-encoder is a **recall-oriented filter** ("don't lose the answer"); the cross-encoder is a
**precision-oriented sorter** ("put it first"). Size the pool so the answer is almost always inside
(`hit@pool`, §7) but small enough to rerank cheaply. In production: pool 50-200, final k 3-10.

### 6.3 Behaviours to expect (all observed here)
* **Big wins when the gold chunk is mid-pack** (q06: rank 19 → top 3).
* **No help when the pool lacks the answer** (ceiling), **or when the cross-encoder itself errs** (q08: ranked 4th).
* **Can demote a gold chunk** it previously ranked first (q02: rank 1 → 2).
* **Domain shift:** this cross-encoder was trained on MS MARCO web-search passages. It generalised to
  this technical text, but a domain-tuned reranker (or a bigger one, or an LLM reranker) may do better.
* **Latency cost:** +≈0.4 s/question measured (20 pairs, CPU). GPU or a smaller pool cuts it.

### 6.4 The fair baseline question
"Rerank 20 → 3" vs "just retrieve 3" is the comparison we ran. The tougher baseline is "just
retrieve **10** and let the LLM sort it out". Reranking's honest value proposition is *quality per
context token*: it keeps the prompt short (cheaper, less distraction, less "lost in the middle") while
approaching big-k recall. Exercise 3 in the README tests exactly this.

---

## 7. Deterministic retrieval metrics

LLM-judged metrics are noisy and cost money. Pair them with metrics needing **no LLM**, using the gold
**evidence span** stored per question.

* **hit@k**: does *any* of the top-k chunks contain the evidence span? (binary per question; mean = hit rate)
* **MRR** (mean reciprocal rank): `1/rank` of the first chunk containing the evidence; 0 if none.
  Rank 1 → 1.0, rank 2 → 0.5, rank 3 → 0.33.
* **hit@pool (the ceiling)**: is the evidence anywhere in the first-stage candidate pool? No reranker can
  exceed this. If hit@pool is low, **don't buy a reranker, fix first-stage recall.**

Caveat: span matching is a *proxy*. If the answer straddles two chunks and the span lands in one, hit
is true yet context may still be insufficient (q05). And the span must be chosen carefully: a span cut by a
chunk boundary produces false misses. `validate_ground_truth.py` checks spans exist in the source;
it cannot tell you a span is *chunk-boundary-fragile*. Understanding that is part of evaluator craft.

---

## 8. Statistics for small eval sets

**Per-question scores are not independent of questions**, so compare variants with a **paired** test
(same 15 questions, two conditions), not two independent means.

**Sign test** (what `compare.py` does): for each question, did variant B beat A (win), lose, or tie?
Discard ties. Under H₀ ("no difference") wins ~ Binomial(wins+losses, 0.5). Exact two-sided p-value:

```
p = 2 · Σ_{i=0}^{min(w,l)} C(w+l, i) / 2^(w+l)         (capped at 1)
```

Example, faithfulness: 7 wins, 0 losses → p = 2·C(7,0)/2⁷ = 2/128 = **0.016**.
It is deliberately crude: it ignores *magnitude* (a −0.001 loss counts as a full loss; see answer
relevancy: 5 wins / 8 losses, p = 0.58, despite a +0.17 mean). Use it as a sanity gate, not the headline.

**Better tools when you have more data:** paired bootstrap CIs on the mean difference; Wilcoxon
signed-rank (uses magnitude); mixed-effects models if you have repeated runs.

**Multiple comparisons.** We ran 6 tests. At α = 0.05, expect ~0.3 false positives by chance alone.
Bonferroni: multiply p by 6 → faithfulness 0.096 (no longer < 0.05). Declare your **primary metric
before** looking at results, or correct.

**Run-to-run noise (judge variance)** is a separate, additive source of uncertainty from sampling
questions. We saw a 0.5 swing on one question with identical inputs (README §4.4). The cure is
repeated judging: score each answer k times, average, and require effect sizes bigger than the observed
sd. Since answers are cached, re-judging costs only judge tokens.

**Effect size vs significance.** With n=15, only huge effects reach p<0.05. "Not significant" ≠ "no
effect": it means "this experiment couldn't tell". Plan sample size from the effect you care about.

---

## 9. LLM-as-judge: biases and defences

| Bias | What it is | Defence used / advised |
|---|---|---|
| **Self-preference** | Judges favour text resembling their own output | Generator ≠ judge (`minimax-local` vs `gpt-oss:20b`) |
| **Verbosity** | Longer, more detailed answers score higher | Watch answer relevancy (q13/q14); constrain answer length in the prompt (we ask for 1-3 sentences) |
| **Position** | In pairwise judging, the first option wins more | N/A to RAGAS pointwise scoring, but relevant if you add pairwise judges |
| **Instruction/format fragility** | Small prompt/format changes shift scores | Pin the RAGAS version (`ragas==0.4.3`) and judge model; record both with results |
| **Non-determinism** | Same input, different scores even at temperature 0 | Repeated judging; report variance |
| **Capability ceiling** | A weak judge mis-grades subtle claims | Spot-check judgments by hand; compare to a stronger judge on a sample |

**Treat the judge as a measuring instrument you must calibrate**: periodically hand-label ~20-30
items, compute agreement (Cohen's κ or simple accuracy) with the judge, and document it.

---

## 10. Glossary

* **Chunk**: a slice of a document embedded and retrieved as a unit.
* **Dense retrieval**: nearest-neighbour search over learned embedding vectors.
* **Sparse retrieval**: keyword-style (TF-IDF, BM25) matching.
* **Bi-encoder / cross-encoder**: §6.1.
* **Top-k**: the k highest-ranked chunks given to the generator.
* **Candidate pool**: the larger first-stage result list a reranker re-orders.
* **Ground truth / reference**: the human-written correct answer.
* **Evidence span**: verbatim corpus text that must be present to answer a question.
* **Hit rate / MRR / NDCG**: ranking metrics. NDCG additionally handles graded relevance.
* **LLM-as-judge**: using an LLM to grade outputs.
* **Noncommittal answer**: evasive/"I don't know"; RAGAS zeroes answer relevancy for these.
* **Paired test**: comparing two conditions on the *same* items.
* **Ceiling (hit@pool)**: the best any reranker could do given its input pool.
