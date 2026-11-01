"""
Two RAG pipelines that share EVERYTHING except the retrieval ordering step.

    NaiveRAG     : embed query -> FAISS top-k  -> LLM
    RerankedRAG  : embed query -> FAISS top-N  -> cross-encoder rescoring -> top-k -> LLM

Two-stage retrieval in one sentence: the bi-encoder (embedding model) is FAST because
documents are embedded once offline, but it compresses query and chunk into separate
vectors and compares them with a dot product, so it is a coarse relevance signal. The
cross-encoder reads (query, chunk) TOGETHER with full cross-attention, which is far more
accurate but too slow to run over the whole corpus - so we only run it on the N candidates
the bi-encoder already shortlisted.
"""
from dataclasses import dataclass

import faiss
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder, SentenceTransformer

import config
from corpus import Page, load_corpus, normalise


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_id: int


@dataclass
class Retrieved:
    chunk: Chunk
    score: float       # cosine similarity (naive) or cross-encoder logit (reranked)
    dense_rank: int    # 1-based position in the first-stage dense ranking


ANSWER_PROMPT = """You are a careful assistant answering questions using ONLY the context below.
Rules:
- Use only facts stated in the context. Do not use outside knowledge.
- If the context does not contain the answer, reply exactly: "I don't know based on the provided context."
- Be concise: 1-3 sentences.

Context:
{context}

Question: {question}

Answer:"""


class VectorIndex:
    """Chunk -> embed -> FAISS. Cosine similarity via L2-normalised vectors + inner product."""

    def __init__(self, pages: list[Page] | None = None):
        pages = pages or load_corpus()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.chunks: list[Chunk] = []
        for p in pages:
            for piece in splitter.split_text(p.text):
                self.chunks.append(Chunk(piece, p.source, p.page, len(self.chunks)))
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        vecs = self.embedder.encode(
            [c.text for c in self.chunks], normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        self.index = faiss.IndexFlatIP(vecs.shape[1])   # exact search; corpus is tiny
        self.index.add(vecs)

    def search(self, query: str, k: int) -> list[Retrieved]:
        q = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q, k)
        return [
            Retrieved(self.chunks[i], float(s), rank)
            for rank, (s, i) in enumerate(zip(scores[0], ids[0]), start=1)
        ]


class NaiveRAG:
    name = "naive"

    def __init__(self, index: VectorIndex):
        self.index = index
        self.llm = ChatOpenAI(
            model=config.GENERATOR_MODEL,
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
            temperature=0,
        )

    def retrieve(self, question: str) -> list[Retrieved]:
        return self.index.search(question, config.TOP_K)

    def generate(self, question: str, retrieved: list[Retrieved]) -> str:
        context = "\n\n".join(f"[{i}] {r.chunk.text}" for i, r in enumerate(retrieved, 1))
        prompt = ANSWER_PROMPT.format(context=context, question=question)
        return self.llm.invoke(prompt).content.strip()

    def run(self, question: str) -> dict:
        retrieved = self.retrieve(question)
        return {
            "question": question,
            "answer": self.generate(question, retrieved),
            "contexts": [r.chunk.text for r in retrieved],
            "retrieved": [
                {"source": r.chunk.source, "page": r.chunk.page, "chunk_id": r.chunk.chunk_id,
                 "score": round(r.score, 4), "dense_rank": r.dense_rank}
                for r in retrieved
            ],
        }


class RerankedRAG(NaiveRAG):
    name = "reranked"

    def __init__(self, index: VectorIndex):
        super().__init__(index)
        self.reranker = CrossEncoder(config.RERANK_MODEL)

    def retrieve(self, question: str) -> list[Retrieved]:
        candidates = self.index.search(question, config.RERANK_CANDIDATES)
        scores = self.reranker.predict([(question, c.chunk.text) for c in candidates])
        for c, s in zip(candidates, scores):
            c.score = float(s)                       # overwrite cosine with cross-encoder logit
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[: config.TOP_K]            # SAME k as naive: only ORDER may differ


def evidence_hit(evidence: str, texts: list[str]) -> bool:
    """Deterministic retrieval check: does any retrieved text contain the gold span?
    Whitespace-insensitive because chunking/PDF extraction moves newlines around."""
    squash = lambda s: " ".join(normalise(s).split()).lower()
    needle = squash(evidence)
    return any(needle in squash(t) for t in texts)
