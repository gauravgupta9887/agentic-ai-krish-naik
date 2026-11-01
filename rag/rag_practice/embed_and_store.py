import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from pathlib import Path
from chunking import (
    fixed_size_chunking,
    recursive_character_split_chunking,
    semantic_search_chunking
)
from document_ingestion import get_pdf_document

# Local embedding model (offline, no API key needed)
# all-MiniLM-L6-v2 -> 384-dim vectors, max 256 tokens per chunk
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Embedding dim: {embed_model.get_sentence_embedding_dimension()}")


def embed_chunks(chunks):
    """Turn a list of Document chunks into a 2D float32 numpy array of embeddings.
    We embed the *text* of each chunk; the vector is the numerical meaning of that chunk."""
    texts = [c.page_content for c in chunks]
    vecs = embed_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return np.asarray(vecs, dtype="float32")


def store_chunks(strategy_name, chunks):
    """Embed a set of chunks and save them into a dedicated FAISS index + metadata file.
    Each strategy gets its own folder so we can compare retrieval independently."""
    out_dir = Path(strategy_name)
    out_dir.mkdir(exist_ok=True)

    embeddings = embed_chunks(chunks)
    dim = embeddings.shape[1]
    print(f"[{strategy_name}] embedded {embeddings.shape[0]} chunks -> shape {embeddings.shape}")

    # FAISS index: IndexFlatL2 measures Euclidean distance between query and chunk vectors.
    # Smaller distance = more similar.
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)  # all chunk vectors go into the index

    faiss.write_index(index, str(out_dir / "faiss.index"))

    # We must store the original text alongside each vector, because FAISS only keeps
    # numbers -- it can't tell us *what* a matched chunk says. Metadata bridges that gap.
    metadata = [{"text": c.page_content, "page": c.metadata.get("page")} for c in chunks]
    with open(out_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"[{strategy_name}] saved FAISS index + metadata to ./{out_dir}/")
    return out_dir


docs = get_pdf_document()
# ---- 1) Fixed-size chunking ----
fixed_splitter = fixed_size_chunking(chunk_size=1000, chunk_overlap=200)
fixed_chunks = fixed_splitter.split_documents(docs)
print(f"Fixed-size chunks: {len(fixed_chunks)}")

# ---- 2) Recursive chunking ----
recursive_splitter = recursive_character_split_chunking(chunk_size=1000, chunk_overlap=200)
recursive_chunks = recursive_splitter.split_documents(docs)
print(f"Recursive chunks: {len(recursive_chunks)}")

# ---- 3) Semantic chunking ----
# NOTE: SemanticChunker here is configured with OpenAIEmbeddings in your chunking.py.
# Since we are going fully local, we instead build a SemanticChunker with the local model.
semantic_splitter = semantic_search_chunking()
semantic_chunks = semantic_splitter.split_documents(docs)
print(f"Semantic chunks: {len(semantic_chunks)}")

# Store each strategy's chunks in its own FAISS index folder.
store_chunks("faiss_fixed", fixed_chunks)
store_chunks("faiss_recursive", recursive_chunks)
store_chunks("faiss_semantic", semantic_chunks)