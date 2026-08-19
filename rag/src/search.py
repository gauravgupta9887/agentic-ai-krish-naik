import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from vector_store import FaissVectorStore

load_dotenv()


class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 llm_model: str = "gemma2-9b-it"):
        self.vector_store = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vector_store
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vector_store.build_from_documents(docs)
        else:
            self.vector_store.load()
        self.llm = ChatOpenAI(
            model="minimax-local",
            api_key=os.getenv("CHAT_API_KEY"),
            base_url=os.getenv("CHAT_API_BASE_URL"),
            temperature=0,
            verbose=True
        )

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vector_store.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "")
                 for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = f"""Summarize the following context for the query: \
            '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self.llm.invoke([prompt])
        return response.content


# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "DSA Interview Preparation"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
