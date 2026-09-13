from rag.retriever import NewsRetriever


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
