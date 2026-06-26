from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from config import LLM_MODEL, TEMPERATURE, MAX_REVISIONS, CRITIQUE_THRESHOLD
from prompts.templates import supervisor_prompt

llm = ChatOllama(model=LLM_MODEL, temperature=0.1)  # low temp — routing must be deterministic
parser = StrOutputParser()

chain = supervisor_prompt | llm | parser


def run_supervisor(state: dict) -> dict:
    """Decide which agent runs next. Writes next_agent to state."""

    response = chain.invoke({
        "query": state["query"],
        "agent_history": state.get("agent_history", []),
        "has_research": "yes" if state.get("research_findings") else "no",
        "has_draft": "yes" if state.get("draft_report") else "no",
        "critique_score": state.get("critique_score", 0),
        "revision_count": state.get("revision_count", 0),
        "max_revisions": MAX_REVISIONS,
    })

    # extract just the routing word — LLM sometimes adds extra text
    decision = response.strip().lower().split()[0]

    # guard: only allow valid destinations
    valid = {"research", "writer", "critic", "end"}
    if decision not in valid:
        # fallback logic if LLM hallucinates
        if not state.get("research_findings"):
            decision = "research"
        elif not state.get("draft_report"):
            decision = "writer"
        elif not state.get("critique"):
            decision = "critic"
        else:
            decision = "end"

    return {
        "next_agent": decision,
        "current_agent": "supervisor",
        "agent_history": ["supervisor"],
    }
