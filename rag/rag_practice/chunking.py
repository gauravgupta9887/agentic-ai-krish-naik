from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)


def fixed_size_chunking(chunk_size, chunk_overlap):
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
        )
    return splitter


def recursive_character_split_chunking(chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
    return splitter


def semantic_search_chunking():
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_community.embeddings import HuggingFaceEmbeddings
    local_emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    semantic_splitter = SemanticChunker(
        local_emb,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
    )
    return semantic_splitter
