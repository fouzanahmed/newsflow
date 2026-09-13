"""LangChain tool wrapping the local retriever, for the live-agent path."""
from rag.retriever import NewsRetriever

_retriever = NewsRetriever()


def search_recent_news(query: str) -> str:
    """Search for recent AI/news snippets relevant to a topic.

    Falls back to a local sample corpus (rag/data/sample_news.json) --
    there's no live web-search API wired in. Swap this function's body for
    a real search API call (Tavily, Bing, etc.) to make it live.
    """
    results = _retriever.retrieve(query, k=3)
    if not results:
        return "No relevant articles found in the local sample corpus."
    return "\n\n".join(f"- {r['title']}: {r['text']}" for r in results)


def get_langchain_tool():
    """Lazily built so importing this module never requires langchain to
    be installed -- only the live-agent path (generate_bulletin_live) does."""
    from langchain.tools import Tool

    return Tool(
        name="search_recent_news",
        func=search_recent_news,
        description=(
            "Search for recent AI/news snippets relevant to a topic. "
            "Input should be a short topic string."
        ),
    )
