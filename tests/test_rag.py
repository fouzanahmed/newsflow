import json

import pytest

from rag.retriever import DATA_PATH, NewsRetriever


def _write_corpus(tmp_path, documents):
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(documents), encoding="utf-8")
    return path


def test_sample_corpus_integrity():
    documents = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    assert isinstance(documents, list) and documents
    ids = [doc["id"] for doc in documents]
    assert len(ids) == len(set(ids)), "duplicate ids in sample_news.json"
    for doc in documents:
        for key in ("id", "title", "text"):
            assert isinstance(doc.get(key), str) and doc[key].strip(), (
                f"document {doc.get('id')!r} has empty or missing '{key}'"
            )


def test_empty_corpus_raises_clear_error(tmp_path):
    with pytest.raises(ValueError, match="corpus is empty"):
        NewsRetriever(_write_corpus(tmp_path, []))


def test_non_list_corpus_raises_clear_error(tmp_path):
    with pytest.raises(ValueError, match="expected a JSON list"):
        NewsRetriever(_write_corpus(tmp_path, {"id": "n1"}))


def test_missing_key_raises_clear_error(tmp_path):
    docs = [{"id": "n1", "title": "Agents ship"}]
    with pytest.raises(ValueError, match=r"index 0 is missing required key\(s\): text"):
        NewsRetriever(_write_corpus(tmp_path, docs))


def test_blank_field_raises_clear_error(tmp_path):
    docs = [{"id": "n1", "title": "Agents ship", "text": "   "}]
    with pytest.raises(ValueError, match="empty or non-string 'text'"):
        NewsRetriever(_write_corpus(tmp_path, docs))


def test_duplicate_ids_raise_clear_error(tmp_path):
    docs = [
        {"id": "n1", "title": "Agents ship", "text": "Agent frameworks release."},
        {"id": "n1", "title": "Retrieval news", "text": "Vector stores improve."},
    ]
    with pytest.raises(ValueError, match="duplicate document id 'n1' at indexes 0 and 1"):
        NewsRetriever(_write_corpus(tmp_path, docs))


def test_stop_word_only_corpus_raises_clear_error(tmp_path):
    docs = [{"id": "n1", "title": "the", "text": "and of the"}]
    with pytest.raises(ValueError, match="no indexable terms"):
        NewsRetriever(_write_corpus(tmp_path, docs))


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
