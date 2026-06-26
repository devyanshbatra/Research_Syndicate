import os
from dotenv import load_dotenv

load_dotenv()

# LangSmith
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "multi-agent-research")

# Ollama models
LLM_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"
TEMPERATURE = 0.7

# Vector DB
CHROMA_PATH = "./chroma_db"
DOCS_PATH = "./docs"

# Retrieval
TOP_K_RESULTS = 3
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Agent behavior
CRITIQUE_THRESHOLD = 7      # score below this → writer revises
MAX_REVISIONS = 3           # max writer revisions before force finish
