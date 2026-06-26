import os
from dotenv import load_dotenv

# load env vars first — before any langchain imports
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "multi-agent-research")

from graph.graph import app


def run_multi_agent(query: str) -> str:
    """Run full multi-agent pipeline. Returns final polished report."""

    print(f"\nQuery: {query}")
    print("=" * 60)

    initial_state = {
        "query": query,
        "messages": [],
        "doc_results": "",
        "search_results": "",
        "research_findings": "",
        "draft_report": "",
        "critique": "",
        "critique_score": 0,
        "revision_notes": "",
        "next_agent": "",
        "current_agent": "",
        "agent_history": [],
        "revision_count": 0,
        "final_report": "",
    }

    result = app.invoke(initial_state)

    final = result.get("final_report", "")
    if not final:
        # fallback — return draft if final not generated
        final = result.get("draft_report", "No report generated.")

    return final


def main():
    print("=" * 60)
    print("Multi-Agent Research System")
    print("Supervisor + Research + Writer + Critic")
    print("Powered by LangGraph + Ollama + LangSmith")
    print("=" * 60)
    print("Type 'quit' to exit\n")

    while True:
        query = input("Ask anything: ").strip()

        if not query:
            continue

        if query.lower() in ["quit", "exit", "q"]:
            print("Goodbye.")
            break

        try:
            report = run_multi_agent(query)
            print("\n" + "=" * 60)
            print("FINAL REPORT")
            print("=" * 60)
            print(report)
            print("=" * 60 + "\n")
        except Exception as e:
            print(f"\nError: {e}")
            print("Make sure ollama is running and ingest.py was executed.")


if __name__ == "__main__":
    main()
