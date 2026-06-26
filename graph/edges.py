from state.schema import MultiAgentState


def route_from_supervisor(state: MultiAgentState) -> str:
    """Read next_agent from state. Return destination node name."""
    decision = state.get("next_agent", "research").lower()

    routing_map = {
        "research": "research",
        "writer": "writer",
        "critic": "critic",
        "end": "__end__",
    }

    return routing_map.get(decision, "research")
