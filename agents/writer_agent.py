from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from config import LLM_MODEL, TEMPERATURE
from prompts.templates import writer_prompt

llm = ChatOllama(model=LLM_MODEL, temperature=TEMPERATURE)
parser = StrOutputParser()

chain = writer_prompt | llm | parser


def run_writer_agent(state: dict) -> dict:
    """Write markdown report from research findings. Writes draft_report to state."""

    draft = chain.invoke({
        "research_findings": state.get("research_findings", ""),
        "revision_notes": state.get("revision_notes", "None — this is the first draft."),
    })

    return {
        "draft_report": draft,
        "current_agent": "writer",
        "agent_history": ["writer"],
        "revision_count": state.get("revision_count", 0) + 1,
    }
