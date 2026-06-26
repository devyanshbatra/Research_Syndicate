# Research Syndicate

A multi-agent research system built with LangGraph + LangChain + LangSmith, running fully locally via Ollama.

4 specialized agents collaborate to research any topic, write a structured report, and iteratively improve it through critique cycles.

---

## Architecture

```
User Query
    ↓
Supervisor Agent (routes dynamically based on state)
    ↓
Research Agent (3 parallel researchers: web + academic + docs)
    ↓
Supervisor Agent
    ↓
Writer Agent (structured markdown report)
    ↓
Supervisor Agent
    ↓
Critic Agent (3x scoring, averaged for stability)
    ↓
Supervisor Agent
    ↓
Writer Agent (revision based on critique) ← loops until score ≥ 7 or max revisions hit
    ↓
Final Report
```

---

## What Makes This Different

| Feature | Basic Tutorial Agent | Research Syndicate |
|---|---|---|
| Agent count | 1 | 4 specialized |
| Routing | Hardcoded edges | Supervisor LLM decides dynamically |
| Research | Sequential tools | 3 parallel researchers (asyncio) |
| Quality check | Self-evaluation | External critic agent |
| Scoring | Single LLM run | 3 runs averaged (statistical stability) |
| Revision | None | Writer-critic loop up to 3 cycles |
| Observability | None | Full LangSmith tracing |
| Cost | OpenAI API | 100% local via Ollama (free) |

---

## Agents

### Supervisor
- Reads full state after every agent run
- Decides who runs next: `research → writer → critic → END`
- Uses low temperature (0.1) for deterministic routing
- Has fallback logic if LLM hallucinates a destination

### Research Agent
- Runs 3 parallel researchers using `asyncio.gather()`
- **Researcher A**: Web search — current news and trends
- **Researcher B**: Wikipedia + Arxiv — background and academic papers
- **Researcher C**: Local documents — domain-specific content
- Combines all findings into structured research block

### Writer Agent
- Converts research findings into structured markdown report
- On revision runs: receives `revision_notes` from critic and addresses each point
- Report structure: Executive Summary → Key Findings → Detailed Analysis → Sources → Conclusion

### Critic Agent
- Runs 3 critic evaluations at different temperatures (0.2, 0.3, 0.4)
- Averages scores for stability — single LLM scores are noisy
- Picks response closest to average for critique text
- Scores on: Accuracy, Completeness, Clarity, Depth, Sources
- Generates `REVISION_NOTES` with specific actionable fixes for writer
- Runs `final_report_prompt` when approved (score ≥ 7 or max revisions hit)

---

## Project Structure

```
research_syndicate/
├── .env                        # LangSmith credentials
├── requirements.txt
├── config.py                   # All settings in one place
├── ingest.py                   # CLI to load documents into vector DB
├── main.py                     # Entry point
│
├── state/
│   └── schema.py               # MultiAgentState — shared memory (14 fields)
│
├── vector_store/
│   ├── store.py                # Load Chroma DB, get retriever
│   └── setup.py                # Load docs → split → embed → save
│
├── tools/
│   ├── web_search.py           # DuckDuckGo, Wikipedia, Arxiv, date tools
│   ├── retriever.py            # Local document search tool
│   └── __init__.py             # Exports all_tools list
│
├── prompts/
│   └── templates.py            # 5 prompts — one per agent role
│
├── agents/
│   ├── supervisor.py           # Routing brain
│   ├── research_agent.py       # Parallel research execution
│   ├── writer_agent.py         # Report generation
│   └── critic_agent.py         # Averaged quality scoring
│
└── graph/
    ├── nodes.py                # Node wrappers for LangGraph
    ├── edges.py                # Conditional routing function
    └── graph.py                # Graph assembly and compilation
```

---

## Setup

### 1. Install Ollama
Download from [ollama.ai](https://ollama.ai) and pull required models:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure LangSmith (optional but recommended)
Create account at [smith.langchain.com](https://smith.langchain.com) and add to `.env`:
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key_here
LANGCHAIN_PROJECT=research-syndicate
```

### 4. Add Documents (optional)
Drop PDF or TXT files into a `docs/` folder, then run:
```bash
python ingest.py
```

### 5. Run
```bash
python main.py
```

---

## Configuration

All settings in `config.py`:

```python
CRITIQUE_THRESHOLD = 7    # Score needed to approve report (0-10)
MAX_REVISIONS = 3         # Max writer revision cycles before force finish
TOP_K_RESULTS = 3         # Chunks returned from vector DB per query
LLM_MODEL = "llama3.2"   # Ollama model for all agents
EMBED_MODEL = "nomic-embed-text"  # Embedding model for vector store
```

---

## Tech Stack

- **LangGraph** — multi-agent graph orchestration
- **LangChain** — tools, prompts, chains, document loaders
- **LangSmith** — tracing and observability
- **Ollama** — local LLM hosting (llama3.2 + nomic-embed-text)
- **Chroma** — local vector database
- **asyncio** — parallel research execution

---

## Requirements

- Python 3.10+
- Ollama running locally
- 8GB+ RAM recommended (16GB for smooth performance)
- GPU optional but speeds up inference significantly
