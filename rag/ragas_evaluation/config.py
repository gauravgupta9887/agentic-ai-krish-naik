"""
Single source of truth for every knob in the experiment.

Why one file: when you compare two RAG variants, the ONLY thing allowed to differ is the
variable under test (here: the reranker). Everything else - corpus, chunking, embedding
model, generator, prompt, judge, k - must be identical, and keeping them here makes that
auditable at a glance.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
CORPUS_DIR = ROOT / "corpus"
RESULTS_DIR = ROOT / "results"
GROUND_TRUTH = ROOT / "ground_truth.json"

# --- credentials ---------------------------------------------------------------------
# The repo's .env (CHAT_API_BASE_URL / CHAT_API_KEY) lives at the repo root. We search
# upward from here; RAGAS_ENV_FILE overrides (needed in git worktrees, where the
# git-ignored .env is not checked out).
_env = os.getenv("RAGAS_ENV_FILE")
if _env:
    load_dotenv(_env)
else:
    for parent in [ROOT, *ROOT.parents]:
        if (parent / ".env").exists():
            load_dotenv(parent / ".env")
            break

LLM_BASE_URL = os.getenv("CHAT_API_BASE_URL")
LLM_API_KEY = os.getenv("CHAT_API_KEY")

# --- models --------------------------------------------------------------------------
# Generator and judge are DIFFERENT models on purpose: LLM judges systematically prefer
# text that looks like their own output ("self-preference bias"). See README section 7.
GENERATOR_MODEL = "minimax-local"
JUDGE_MODEL = "gpt-oss:20b"

# Embeddings: used (a) by the retriever and (b) by RAGAS answer_relevancy. Local, free,
# deterministic. Same model the rest of this repo uses.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Cross-encoder reranker (trained on MS MARCO passage ranking). ~90 MB, CPU-friendly.
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- chunking / retrieval --------------------------------------------------------------
CHUNK_SIZE = 500          # characters. Small on purpose: makes retrieval non-trivial.
CHUNK_OVERLAP = 50
TOP_K = 3                 # chunks handed to the generator. IDENTICAL for both variants.
RERANK_CANDIDATES = 20    # first-stage pool the reranker re-orders (reranked variant only)

# --- RAGAS runtime -----------------------------------------------------------------------
JUDGE_TIMEOUT_S = 300
JUDGE_MAX_WORKERS = 4
JUDGE_MAX_RETRIES = 3
