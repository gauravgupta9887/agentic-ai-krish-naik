# Engineering log: what went wrong while building this, and what it teaches

Real problems hit during this build, in the order they occurred. Evaluation pipelines fail in
quiet ways, and most of these did **not** raise an error: they produced plausible-looking, wrong
output. That is why they are worth reading.

---

## Decisions made up front

| Decision | Reasoning |
|---|---|
| **Isolated venv** inside this folder | RAGAS pulls a different dependency set than the repo's main `.venv` (see #1). Isolation protects the main environment. |
| **Generator ≠ judge** (`minimax-local` vs `gpt-oss:20b`) | Avoid self-preference bias. Also: `minimax-local` answered in ~0.3 s (cheap generation), `gpt-oss:20b` is a reasoning model, slower but better for judging. |
| **Hand-written ground truth + evidence spans** | The judge grades against it; LLM-written references would be circular. Spans enable LLM-free metrics. |
| **Cache stage-1 outputs to JSON** | Re-judge without regenerating; every number traceable to an inspectable artefact. |
| **`k` fixed at 3 for both variants** | Isolates *ordering* quality from *amount of context*. |
| **Local embeddings** (`all-MiniLM-L6-v2`) | Gateway embedding endpoint unusable (#7); local = free + deterministic; same model the rest of this repo uses. |
| **Temperature 0** everywhere | Reduce noise (it does not eliminate it, #9). |

---

## #1: RAGAS 0.4.3 crashes on import: `No module named 'langchain_community.chat_models.vertexai'`

**Symptom.** `import ragas` → `ModuleNotFoundError` from `ragas/llms/base.py`.
**Cause.** RAGAS 0.4.3 unconditionally imports `ChatVertexAI` from `langchain_community`; that module was
removed in `langchain-community` 0.4. A fresh `pip install ragas` pulled the *new* langchain family
(`langchain 1.x`, `langchain-community 0.4.2`), which RAGAS does not yet support.
**Fix.** Pin the langchain family to the 0.3 line in `requirements.txt` (`langchain-community<0.4`,
`langchain-core<1.0`, `langchain-openai<1.0`, …).
**Why a separate venv rather than installing into the main one.** `uv pip install --dry-run ragas` on
the main environment showed it would *uninstall/downgrade* `huggingface-hub` (1.27→2.4),
`aiohttp`, `jiter`, `rich`, and with constraints to avoid that, would downgrade `openai`,
`langchain-openai` and `instructor`. Breaking the repo's other work to evaluate RAG is a bad trade.
**Lesson.** Eval frameworks are fast-moving and pin-sensitive. Isolate them, pin exact versions, and
record the version in your results (`ragas==0.4.3`).

## #2: The gateway rejects `n`; `answer_relevancy` came back `NaN`

**Symptom (first smoke test, 2 questions).** Several RAGAS jobs failed with
`litellm.UnsupportedParamsError: ollama does not support parameters: ['n']`. Result: `answer_relevancy`
was NaN for both questions, and some other cells NaN too.
**Cause.** `answer_relevancy` asks the judge for 3 samples (`strictness=3`) using OpenAI's `n` parameter.
Our endpoint is LiteLLM in front of Ollama, which does not support `n`.
**Fix.** `LangchainLLMWrapper(llm, bypass_n=True)`, a RAGAS option that sends `n` separate requests
instead of one request with `n=n`. Found by reading `ragas/llms/base.py`.
**Why we smoke-tested on 2 questions first.** The full run is ~14 minutes. A 1-minute 2-question run
exposed #2 and #3 before they could waste a full run. **Always smoke-test an eval pipeline on a tiny slice.**

## #3: `context_precision` silently missing from the output

**Symptom.** After the fix for #2, the summary listed three metrics, not four. No error.
**Cause.** RAGAS names the dataframe column after the metric *class*:
`llm_context_precision_with_reference`, not `context_precision`. My code filtered on the short name,
so the metric was computed, then dropped from the report.
**Fix.** Rename the column right after `to_pandas()`.
**Lesson.** "Missing" is the most dangerous failure mode in a report. Assert that every expected
metric is present. `evaluate_ragas.py` now does exactly that (`assert not missing`), which would have
caught this immediately. (Added after the full run; it only adds a check, it does not change any score.)

## #4: Deprecation warnings: legacy API is deprecated but currently required

RAGAS 0.4.x marks `ragas.metrics.*` imports and `LangchainLLMWrapper` as deprecated in favour of
`ragas.metrics.collections` and `llm_factory`. This project uses the **legacy path** because that is the
path `evaluate()` + `EvaluationDataset` + `LangchainLLMWrapper` documents for the four classic metrics.
**What was not verified:** I did not test the newer `collections` API end-to-end against this gateway.
Before upgrading to RAGAS 1.0, expect to migrate (and pin `ragas<1.0` until then). The warnings are
suppressed in `evaluate_ragas.py` with a comment pointing here, so they don't drown out real output.

## #5: Silent ingestion failures in the corpus

Two separate traps in PDF loading, both silent:

* **`git.pdf` has zero extractable text.** `pypdf` returned `0` characters for all 13 pages (it is
  scanned/image-only). A pipeline that does not count characters would "ingest successfully" and
  index nothing. We check `if text:` per page, skip it, and document it. *Fix, if you need that file:* OCR (Tesseract / a vision model).
* **Typographic ligatures.** PDF extraction yields single code points like `ﬁ` and `ﬂ`: `definition`
  arrived as `deﬁnition`. A query for "definition" would not lexically match, an embedding model sees
  different tokens, and ground-truth string matching fails. Fix: `unicodedata.normalize("NFKC", …)` in
  `corpus.normalise()`, applied to the corpus **and** to the matching in `evidence_hit()`.

**Lesson.** Ingestion bugs masquerade as "retrieval quality problems" later. Print character/page counts
at ingest time and eyeball a sample of extracted text.

## #6: Configured chunk size ≠ actual chunk size

`CHUNK_SIZE = 500` produced 87 chunks with an **average of 330 characters**. `RecursiveCharacterTextSplitter`
splits on `"\n\n"`, `"\n"`, `". "` before falling back to characters, and PDF text is full of short lines
(slide decks and table rows especially), so many chunks end well below the limit. `chunk_size` is a
*maximum*, not a target. Report **measured** chunk-size distributions, not configured ones.
This also fed the q05 failure: the shutdown list was split across chunks even though it is short.

## #7: Gateway limitations

* `gemma-local` returned a connection error, so it was not usable as a judge or generator.
* The embeddings endpoint (`nomic-embed-text`) rejects `encoding_format=base64`, which the OpenAI client
  sends by default. Rather than fight the gateway, we embed locally with sentence-transformers.
  Cost: none. Benefit: deterministic, offline-reproducible retrieval.

## #8: Git worktree has no `.env`

The session ran in a git **worktree**, a separate checkout containing only tracked files. The repo's
`.env` is git-ignored, so it does not exist there. `config.py` searches upward for `.env` and honours
`RAGAS_ENV_FILE` to point at one explicitly. (Likewise the main checkout's new folder needs its own `.venv`.)

## #9: Judge noise discovered by accident (the most important finding)

The 2-question smoke test and the full run both scored naive **q02**, with the **identical answer text**
and identical retrieved chunks. `faithfulness` was **1.0** once and **0.5** once. Naive **q01**'s
`answer_relevancy` was 0.00 once and 0.87 once. Temperature 0 does **not** make an LLM judge
deterministic (server batching, floating-point non-associativity, reasoning-model sampling). For
`answer_relevancy` the mechanism is visible in the source: the final score is
`mean(cosine) × int(not all_noncommittal)`, so one judge sample flipping a discrete "noncommittal"
flag swings the score between ~0 and ~0.85.

**Consequences for how to read this experiment:**
* A single-run delta smaller than ~0.2 on any LLM-judged metric is **not** evidence of anything.
* The deterministic metrics (hit@k, MRR) are the reliable anchor. Their direction (+0.20, +0.16) is
  consistent with the judged recall/faithfulness gains, which is what makes the overall story credible
  even though no individual judged number is.
* The proper fix (not done here, because it multiplies judge cost): judge each answer 3-5 times.

## #10: Reading answers beside scores caught three metric quirks

Spot-reading `results/*_answers.json` next to the CSVs revealed behaviours no summary table shows:
refusals scoring `faithfulness = 0` (q06, q07), `answer_relevancy` rewarding answers that restate the
question (q13, q14), and `context_precision = 0.00` despite the gold chunk being retrieved (q05).
**Never ship an eval you have not read at the row level.**

---

## What was *not* verified (honest list)

* Agreement between the judge and a human (no hand-labelled spot-check done).
* Judge repeatability beyond the two incidental comparisons in #9.
* Behaviour with the non-legacy RAGAS API (#4).
* Whether the cross-encoder gain generalises beyond this 87-chunk corpus.
* The q08 miss (gold chunk ranked 4th, outside top-3) was observed once; it is not known whether it is
  stable across reranker model versions.

## Wall-clock (this machine, CPU, remote LLM gateway)

| Step | Time |
|---|---|
| Install isolated venv (torch + CUDA wheels dominate) | several minutes, one-off |
| `generate.py` per variant | ≈ 10 s (naive) / ≈ 16 s (reranked) of per-question time, plus ~10 s model loading |
| `evaluate_ragas.py` per variant | ~5.5-6.5 min (60 judge jobs; reasoning model) |
| Mean generation latency per question | naive 0.64 s · reranked 1.06 s |
| Whole `run_all.sh` | ≈ 14 min |
