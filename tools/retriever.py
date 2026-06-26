from langchain_core.tools import tool
from vector_store.store import get_retriever

_retriever = None


def _get_retriever_instance():
    global _retriever
    if _retriever is None:
        _retriever = get_retriever()
    return _retriever


@tool
def search_docs(query: str) -> str:
    """Search through local documents for relevant information.
    Use this to find information from uploaded PDFs and text files."""
    try:
        retriever = _get_retriever_instance()
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant documents found."
        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            results.append(f"[Doc {i} - {source}]:\n{doc.page_content}")
        return "\n\n".join(results)
    except Exception as e:
        return f"Document search failed: {str(e)}"
