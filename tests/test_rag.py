from rag.retriever import NewsRetriever, get_retriever


def test_retrieve_returns_relevant_doc_for_matching_query():
    retriever = NewsRetriever()
    results = retriever.retrieve("Model Context Protocol MCP tools", k=1)
    assert len(results) == 1
    assert "MCP" in results[0]["title"] or "Model Context Protocol" in results[0]["title"]


def test_retrieve_respects_k():
    retriever = NewsRetriever()
    results = retriever.retrieve("agent", k=2)
    assert len(results) <= 2


def test_retrieve_returns_empty_for_unrelated_query():
    retriever = NewsRetriever()
    results = retriever.retrieve("zzz_no_such_topic_qqq", k=3)
    assert results == []


def test_get_retriever_returns_cached_instance():
    assert isinstance(get_retriever(), NewsRetriever)
    assert get_retriever() is get_retriever()


def test_bulletin_agent_and_tools_share_one_retriever(monkeypatch):
    from agent import bulletin_agent, tools

    get_retriever.cache_clear()
    seen = []
    real_retrieve = NewsRetriever.retrieve

    def recording_retrieve(self, query, k=3):
        seen.append(self)
        return real_retrieve(self, query, k)

    monkeypatch.setattr(NewsRetriever, "retrieve", recording_retrieve)

    bulletin_agent._fallback_bulletin("agent")
    tools.search_recent_news("agent")

    assert len(seen) == 2
    assert seen[0] is seen[1] is get_retriever()
