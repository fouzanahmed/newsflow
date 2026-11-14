"""Retrieval-quality eval for rag/retriever.py against rag/data/sample_news.json.

This is the executable version of docs/RETRIEVAL_EVAL.md: every claim in that
doc's results tables is a case here, so CI re-checks them on every run.

Metric: top-1 hit -- retrieve(query, k=1) must return the expected doc id
(or nothing, for queries with no relevant document).
"""
import pytest

from rag.retriever import NewsRetriever

# One query per corpus document, reusing words from that document's title.
DIRECT_CASES = [
    ("Claude Agent SDK autonomous coding", "n1"),
    ("durable resumable agent state checkpoint", "n2"),
    ("Model Context Protocol MCP tools", "n3"),
    ("RAG retrieval evaluation benchmarks", "n4"),
    ("n8n workflow automation nodes", "n5"),
    ("open source pull request agents", "n6"),
    ("draft transparency rules for AI regulators", "n7"),
    ("standardized red-teaming reports AI safety", "n8"),
    ("data center power for AI training clusters", "n9"),
    ("inference accelerators cost per token", "n10"),
    ("open-weight models reasoning benchmarks", "n11"),
    ("small distilled models on-device assistants", "n12"),
    ("vector database hybrid keyword embedding search", "n13"),
    ("generative AI pilots stall before production", "n14"),
    ("startups versus big tech small AI companies", "n15"),
    ("hospitals AI scribes clinician documentation", "n16"),
    ("protein structure drug candidate screening", "n17"),
    ("humanoid robots foundation models manipulation", "n18"),
    ("multimodal models understand long videos", "n19"),
    ("prompt injection security risk tool-using agents", "n20"),
]

# Paraphrases that describe a document without reusing its title wording.
PARAPHRASE_CASES = [
    (
        "software that keeps working state saved so a long computation is not "
        "lost if it stops",
        "n2",
    ),
    ("universal plugin standard for connecting assistants to external systems", "n3"),
    ("drag and drop visual automation platform gains new AI capabilities", "n5"),
    ("big companies vs small startups in AI", "n15"),
    ("electricity grid strain from machine learning", "n9"),
    ("doctors spending less time on paperwork", "n16"),
    ("malicious instructions hidden in web pages hijack assistants", "n20"),
]

# Paraphrases TF-IDF currently gets wrong (no shared vocabulary with the
# target). Strict xfail: if one starts passing, the test run fails so the
# doc's "where it's weaker" section gets updated.
KNOWN_PARAPHRASE_MISSES = [
    ("government oversight of frontier models", "n7"),
    ("chips that make running models cheaper", "n10"),
]

NO_MATCH_QUERIES = ["weather forecast tomorrow"]


@pytest.fixture(scope="module")
def retriever():
    return NewsRetriever()


def _top1_id(retriever, query):
    results = retriever.retrieve(query, k=1)
    return results[0]["id"] if results else None


def test_direct_cases_cover_every_corpus_document(retriever):
    corpus_ids = {doc["id"] for doc in retriever.documents}
    assert {doc_id for _, doc_id in DIRECT_CASES} == corpus_ids


@pytest.mark.parametrize("query,expected_id", DIRECT_CASES)
def test_direct_vocabulary_query_hits_top1(retriever, query, expected_id):
    assert _top1_id(retriever, query) == expected_id


@pytest.mark.parametrize("query,expected_id", PARAPHRASE_CASES)
def test_paraphrased_query_hits_top1(retriever, query, expected_id):
    assert _top1_id(retriever, query) == expected_id


@pytest.mark.parametrize(
    "query,expected_id",
    [
        pytest.param(q, e, marks=pytest.mark.xfail(strict=True, reason="no shared vocabulary"))
        for q, e in KNOWN_PARAPHRASE_MISSES
    ],
)
def test_known_paraphrase_miss(retriever, query, expected_id):
    assert _top1_id(retriever, query) == expected_id


@pytest.mark.parametrize("query", NO_MATCH_QUERIES)
def test_unrelated_query_returns_nothing(retriever, query):
    assert retriever.retrieve(query, k=3) == []
