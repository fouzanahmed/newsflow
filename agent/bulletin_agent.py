"""Generates a news bulletin for a topic.

Two paths:
  - Live agent path (generate_bulletin_live): a real LangChain tool-calling
    agent backed by Claude. Requires ANTHROPIC_API_KEY and langchain +
    langchain-anthropic installed. Not exercised in this environment (no
    API key available here) -- exists as genuine, real agent code for a
    reader who supplies their own key.
  - Fallback path (generate_bulletin): no LLM call at all. Retrieves
    relevant snippets from the local corpus and formats them directly.
    This is the path tests and CI actually exercise.

generate_bulletin() tries the live path first and falls back on any
failure (missing key, missing package, API error), so the public
interface always returns a real, structured result either way.
"""
import os

from rag.retriever import NewsRetriever

_retriever = NewsRetriever()


def _fallback_bulletin(topic: str) -> dict:
    snippets = _retriever.retrieve(topic, k=3)
    return {
        "topic": topic,
        "source": "fallback_retrieval",
        "headline": f"AI & Agentic AI Developer Bulletin: {topic}",
        "items": [{"title": s["title"], "summary": s["text"]} for s in snippets],
    }


def generate_bulletin_live(topic: str) -> dict:
    """Real LangChain tool-calling agent. Raises on any failure -- callers
    should catch and fall back, this function does not swallow errors."""
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate

    from agent.tools import get_langchain_tool

    llm = ChatAnthropic(model="claude-sonnet-5", temperature=0)
    tools = [get_langchain_tool()]
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a news bulletin writer. Use the search_recent_news tool "
            "to find relevant items, then write a short structured bulletin "
            "as 2-4 bullet points. Only use information the tool returns -- "
            "do not invent details."
        )),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    result = executor.invoke({"input": f"Write a bulletin about: {topic}"})
    return {
        "topic": topic,
        "source": "langchain_agent_claude",
        "headline": f"AI & Agentic AI Developer Bulletin: {topic}",
        "bulletin_text": result["output"],
    }


def generate_bulletin(topic: str) -> dict:
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            return generate_bulletin_live(topic)
        except Exception:
            pass  # any missing package / API failure -> fall back below
    return _fallback_bulletin(topic)
