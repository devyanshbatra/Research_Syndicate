import asyncio
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from config import LLM_MODEL, TEMPERATURE
from tools.web_search import search_web, search_wikipedia, search_arxiv, get_current_date
from tools.retriever import search_docs

llm = ChatOllama(model=LLM_MODEL, temperature=TEMPERATURE)
react_prompt = hub.pull("hwchase17/react")

# separate tool lists per researcher angle
_web_tools = [search_web, get_current_date]
_wiki_tools = [search_wikipedia, search_arxiv]
_doc_tools  = [search_docs]


def _make_executor(tools):
    agent = create_react_agent(llm, tools, react_prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=5,
        handle_parsing_errors=True,
    )


def _run_web_research(query: str) -> str:
    """Researcher A — current/news angle via web."""
    executor = _make_executor(_web_tools)
    result = executor.invoke({
        "input": f"Find the most recent and current information about: {query}. Focus on news, trends, and latest developments."
    })
    return result.get("output", "")


def _run_academic_research(query: str) -> str:
    """Researcher B — academic/background angle via wikipedia + arxiv."""
    executor = _make_executor(_wiki_tools)
    result = executor.invoke({
        "input": f"Find background context, history, definitions, and academic research about: {query}. Focus on established facts and papers."
    })
    return result.get("output", "")


def _run_doc_research(query: str) -> str:
    """Researcher C — domain-specific angle via local documents."""
    executor = _make_executor(_doc_tools)
    result = executor.invoke({
        "input": f"Search local documents for relevant information about: {query}."
    })
    return result.get("output", "")


async def _run_all_parallel(query: str):
    """Run all 3 researchers in parallel using asyncio."""
    loop = asyncio.get_event_loop()

    # run each blocking executor in thread pool so they run simultaneously
    web_task      = loop.run_in_executor(None, _run_web_research,      query)
    academic_task = loop.run_in_executor(None, _run_academic_research, query)
    doc_task      = loop.run_in_executor(None, _run_doc_research,      query)

    web_result, academic_result, doc_result = await asyncio.gather(
        web_task, academic_task, doc_task
    )

    return web_result, academic_result, doc_result


def run_research_agent(state: dict) -> dict:
    """Run 3 parallel researchers. Combine findings. Writes research_findings to state."""

    query = state["query"]

    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    web_result, academic_result, doc_result = loop.run_until_complete(
        _run_all_parallel(query)
    )

    # combine all 3 angles into one findings block
    findings = f"""## Research Findings for: {query}

### Current Information (Web)
{web_result or 'No web results found.'}

### Academic & Background (Wikipedia + Arxiv)
{academic_result or 'No academic results found.'}

### Domain Documents (Local)
{doc_result or 'No local documents found.'}
"""

    return {
        "research_findings": findings,
        "current_agent": "research",
        "agent_history": ["research"],
    }
