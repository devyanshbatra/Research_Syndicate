from typing import TypedDict, Annotated
import operator


class MultiAgentState(TypedDict):
    # shared across all agents
    query: str                                    # original user question — never changes
    messages: Annotated[list, operator.add]       # full conversation history — accumulates

    # research agent outputs
    doc_results: str                              # chunks from Chroma vector DB
    search_results: str                           # web + wiki + arxiv results
    research_findings: str                        # combined summary of all research

    # writer agent output
    draft_report: str                             # formatted markdown report

    # critic agent output
    critique: str                                 # full critique feedback
    critique_score: int                           # 0-10 quality rating
    revision_notes: str                           # specific points writer must fix

    # supervisor control fields
    next_agent: str                               # supervisor writes next destination
    current_agent: str                            # which agent is running now
    agent_history: Annotated[list, operator.add]  # log of which agents ran — accumulates

    # revision tracking
    revision_count: int                           # how many times writer revised

    # final output
    final_report: str                             # approved polished report shown to user
