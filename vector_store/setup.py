import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHROMA_PATH, DOCS_PATH, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


def load_documents(docs_path: str) -> list:
    """Load all PDF and TXT files from docs/ folder."""
    documents = []

    for filename in os.listdir(docs_path):
        filepath = os.path.join(docs_path, filename)

        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
            print(f"Loaded PDF: {filename}")

        elif filename.endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
            documents.extend(loader.load())
            print(f"Loaded TXT: {filename}")

    return documents


def split_documents(documents: list) -> list:
    """Split documents into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks: list) -> Chroma:
    """Embed chunks and save to Chroma DB on disk."""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print(f"Vector store created at: {CHROMA_PATH}")
    return vectorstore


def setup_vector_store() -> Chroma:
    """Full pipeline: load → split → embed → save."""
    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError(f"Docs folder not found: {DOCS_PATH}")

    if not os.listdir(DOCS_PATH):
        raise ValueError(f"No files in {DOCS_PATH}. Add PDFs or TXT files first.")

    print("Loading documents...")
    documents = load_documents(DOCS_PATH)

    print("Splitting documents...")
    chunks = split_documents(documents)

    print("Creating vector store...")
    vectorstore = create_vector_store(chunks)

    print("Done. Vector store ready.")
    return vectorstore
