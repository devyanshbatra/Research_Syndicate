import re
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from config import LLM_MODEL, CRITIQUE_THRESHOLD, MAX_REVISIONS
from prompts.templates import critic_prompt, final_report_prompt

# 3 critic instances with slight temp variation for score diversity
critic_llm_1 = ChatOllama(model=LLM_MODEL, temperature=0.2)
critic_llm_2 = ChatOllama(model=LLM_MODEL, temperature=0.3)
critic_llm_3 = ChatOllama(model=LLM_MODEL, temperature=0.4)

parser = StrOutputParser()

critic_chain_1 = critic_prompt | critic_llm_1 | parser
critic_chain_2 = critic_prompt | critic_llm_2 | parser
critic_chain_3 = critic_prompt | critic_llm_3 | parser

final_chain = final_report_prompt | ChatOllama(model=LLM_MODEL, temperature=0.3) | parser


def _parse_score(text: str) -> int:
    """Extract integer score from critic response."""
    match = re.search(r"SCORE:\s*(\d+)", text)
    if match:
        return min(10, max(0, int(match.group(1))))
    return 0


def _parse_section(text: str, section: str) -> str:
    """Extract a named section from critic response."""
    pattern = rf"{section}:\s*\n(.*?)(?=\n[A-Z_]+:|$)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def run_critic_agent(state: dict) -> dict:
    """Run critic 3x, average scores for stability. Parse best critique."""

    inputs = {
        "query": state["query"],
        "draft_report": state.get("draft_report", ""),
    }

    # run all 3 critics
    response_1 = critic_chain_1.invoke(inputs)
    response_2 = critic_chain_2.invoke(inputs)
    response_3 = critic_chain_3.invoke(inputs)

    score_1 = _parse_score(response_1)
    score_2 = _parse_score(response_2)
    score_3 = _parse_score(response_3)

    # averaged score — more stable than single LLM evaluation
    avg_score = round((score_1 + score_2 + score_3) / 3)

    # use response with score closest to average for critique + revision notes
    scores = [(score_1, response_1), (score_2, response_2), (score_3, response_3)]
    best_response = min(scores, key=lambda x: abs(x[0] - avg_score))[1]

    critique = _parse_section(best_response, "CRITIQUE")
    revision_notes = _parse_section(best_response, "REVISION_NOTES")

    revision_count = state.get("revision_count", 0)
    approved = avg_score >= CRITIQUE_THRESHOLD or revision_count >= MAX_REVISIONS

    final_report = ""
    if approved:
        final_report = final_chain.invoke({
            "query": state["query"],
            "draft_report": state.get("draft_report", ""),
            "critique_score": avg_score,
        })

    return {
        "critique": critique,
        "critique_score": avg_score,
        "revision_notes": revision_notes,
        "final_report": final_report,
        "current_agent": "critic",
        "agent_history": ["critic"],
    }
