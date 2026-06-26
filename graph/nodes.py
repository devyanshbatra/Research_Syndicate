from state.schema import MultiAgentState
from agents.supervisor import run_supervisor
from agents.research_agent import run_research_agent
from agents.writer_agent import run_writer_agent
from agents.critic_agent import run_critic_agent


def supervisor_node(state: MultiAgentState) -> dict:
    """Supervisor decides who runs next."""
    return run_supervisor(state)


def research_node(state: MultiAgentState) -> dict:
    """Research agent gathers information using tools."""
    return run_research_agent(state)


def writer_node(state: MultiAgentState) -> dict:
    """Writer agent produces markdown report."""
    return run_writer_agent(state)


def critic_node(state: MultiAgentState) -> dict:
    """Critic agent scores report and gives revision notes."""
    return run_critic_agent(state)
