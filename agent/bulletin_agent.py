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

generate_bulletin() tries the live path first -- retrying transient
failures with exponential backoff -- and falls back on any remaining
failure (missing key, missing package, API error), so the public
interface always returns a real, structured result either way. Which
path served each request is logged, and is also visible in the
result's "source" field.
"""
import logging
import os
import time

from rag.retriever import NewsRetriever

logger = logging.getLogger(__name__)

# Live-path retry policy: up to LIVE_MAX_ATTEMPTS calls, sleeping
# LIVE_BACKOFF_BASE_SECONDS * 2**(attempt-1) between them (0.5s, 1s, ...).
LIVE_MAX_ATTEMPTS = 3
LIVE_BACKOFF_BASE_SECONDS = 0.5

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


def _generate_bulletin_live_with_retry(topic: str) -> dict:
    """Calls the live path, retrying transient failures with exponential
    backoff. ImportError (langchain / langchain-anthropic not installed) is
    not retried -- it cannot succeed on a later attempt. Raises the last
    exception once attempts are exhausted."""
    for attempt in range(1, LIVE_MAX_ATTEMPTS + 1):
        try:
            return generate_bulletin_live(topic)
        except ImportError:
            raise
        except Exception as exc:
            if attempt == LIVE_MAX_ATTEMPTS:
                raise
            delay = LIVE_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "Live agent attempt %d/%d failed for topic %r (%s: %s); "
                "retrying in %.1fs",
                attempt, LIVE_MAX_ATTEMPTS, topic,
                type(exc).__name__, exc, delay,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")  # loop always returns or raises


def generate_bulletin(topic: str) -> dict:
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            result = _generate_bulletin_live_with_retry(topic)
            logger.info("Bulletin for topic %r served by live agent path", topic)
            return result
        except Exception as exc:
            logger.warning(
                "Live agent path failed for topic %r (%s: %s); "
                "falling back to retrieval-only path",
                topic, type(exc).__name__, exc,
            )
    else:
        logger.info("ANTHROPIC_API_KEY not set; skipping live agent path")
    logger.info("Bulletin for topic %r served by fallback retrieval path", topic)
    return _fallback_bulletin(topic)
