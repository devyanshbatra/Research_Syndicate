from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from datetime import datetime

_search = DuckDuckGoSearchRun()


@tool
def search_web(query: str) -> str:
    """Search the web for recent information about a topic.
    Use this when you need current information not found in documents."""
    try:
        results = _search.run(query)
        return results if results else "No results found."
    except Exception as e:
        return f"Web search failed: {str(e)}"


@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for factual background information about a topic.
    Use this for definitions, history, and established facts."""
    try:
        from langchain_community.tools import WikipediaQueryRun
        from langchain_community.utilities import WikipediaAPIWrapper
        wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))
        result = wiki.run(query)
        return result if result else "No Wikipedia results found."
    except Exception as e:
        return f"Wikipedia search failed: {str(e)}"


@tool
def search_arxiv(query: str) -> str:
    """Search academic research papers on arxiv.org.
    Use this for scientific papers, AI research, and technical topics."""
    try:
        from langchain_community.tools import ArxivQueryRun
        from langchain_community.utilities import ArxivAPIWrapper
        arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=2))
        result = arxiv.run(query)
        return result if result else "No arxiv papers found."
    except Exception as e:
        return f"Arxiv search failed: {str(e)}"


@tool
def get_current_date(dummy: str = "") -> str:
    """Get today's current date and time.
    Use this when the query involves current time, recent events, or date calculations."""
    return datetime.now().strftime("Today is %A, %B %d, %Y. Current time: %H:%M")
