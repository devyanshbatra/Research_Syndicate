import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from config import CHROMA_PATH, EMBED_MODEL, TOP_K_RESULTS


def load_vector_store() -> Chroma:
    """Load existing Chroma DB from disk."""
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(
            f"Vector store not found at '{CHROMA_PATH}'. "
            "Run 'python ingest.py' first to create it."
        )

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    return vectorstore


def get_retriever():
    """Return retriever ready to query."""
    vectorstore = load_vector_store()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS}
    )

    return retriever
