from langgraph.graph import StateGraph, END
from state.schema import MultiAgentState
from graph.nodes import supervisor_node, research_node, writer_node, critic_node
from graph.edges import route_from_supervisor


def build_graph():
    """Assemble multi-agent graph. Supervisor routes between all agents."""

    graph = StateGraph(MultiAgentState)

    # ── register nodes ────────────────────────────────────────────────────────
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("research",   research_node)
    graph.add_node("writer",     writer_node)
    graph.add_node("critic",     critic_node)

    # ── entry point ───────────────────────────────────────────────────────────
    graph.set_entry_point("supervisor")

    # ── supervisor routes dynamically to any agent or END ────────────────────
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "research": "research",
            "writer":   "writer",
            "critic":   "critic",
            "__end__":  END,
        }
    )

    # ── every agent returns to supervisor after finishing ─────────────────────
    graph.add_edge("research", "supervisor")
    graph.add_edge("writer",   "supervisor")
    graph.add_edge("critic",   "supervisor")

    return graph.compile()


app = build_graph()
