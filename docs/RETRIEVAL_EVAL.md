# Retrieval evaluation: `rag/retriever.py` against `rag/data/sample_news.json`

The README's retrieval section shows one worked example (`search_recent_news` /
`retrieve(query, k)`). This doc goes further. It runs a small, repeatable set
of query → expected-document pairs against the real `NewsRetriever` and
records what came back, so retrieval quality is shown, not just claimed.

Corpus: `rag/data/sample_news.json`, 20 synthetic articles (`n1`–`n20`)
covering agents, MCP, RAG, regulation, AI safety, compute, models, enterprise
adoption, healthcare, science, robotics and security. Retrieval is TF-IDF
plus cosine similarity, with no embeddings (the `rag/retriever.py` docstring
explains why).

## Method

For each query, `retrieve(query, k=1)` is called and the top-ranked document
id is compared with the expected id. A query passes if the top-1 result
matches. This is a top-1 hit-rate check, not a full precision/recall study.
That fits a corpus where each query has exactly one relevant document.

The eval is a parametrized pytest, so CI runs it on every push and PR:

```bash
pytest tests/test_retrieval_eval.py -v
```

`tests/test_retrieval_eval.py` holds four groups of cases:

- **Direct-vocabulary queries**: one per corpus document. A guard test checks
  that the set of expected ids equals the set of ids in the corpus, so adding
  an article without adding an eval case fails CI.
- **Paraphrased queries**: they describe a document without reusing its title
  wording.
- **Known paraphrase misses**: marked `xfail(strict=True)`. If retrieval
  improves and one starts passing, the run fails so this doc gets updated.
- **No-match queries**: they must return an empty result list.

## Results: direct-vocabulary queries

Scores are the top-1 cosine similarity from the test run on the current
corpus.

| Query | Expected | Top-1 | Score | Result |
|---|---|---|---|---|
| "Claude Agent SDK autonomous coding" | n1 | n1 | 0.573 | PASS |
| "durable resumable agent state checkpoint" | n2 | n2 | 0.528 | PASS |
| "Model Context Protocol MCP tools" | n3 | n3 | 0.483 | PASS |
| "RAG retrieval evaluation benchmarks" | n4 | n4 | 0.435 | PASS |
| "n8n workflow automation nodes" | n5 | n5 | 0.470 | PASS |
| "open source pull request agents" | n6 | n6 | 0.564 | PASS |
| "draft transparency rules for AI regulators" | n7 | n7 | 0.392 | PASS |
| "standardized red-teaming reports AI safety" | n8 | n8 | 0.541 | PASS |
| "data center power for AI training clusters" | n9 | n9 | 0.493 | PASS |
| "inference accelerators cost per token" | n10 | n10 | 0.576 | PASS |
| "open-weight models reasoning benchmarks" | n11 | n11 | 0.598 | PASS |
| "small distilled models on-device assistants" | n12 | n12 | 0.598 | PASS |
| "vector database hybrid keyword embedding search" | n13 | n13 | 0.707 | PASS |
| "generative AI pilots stall before production" | n14 | n14 | 0.536 | PASS |
| "startups versus big tech small AI companies" | n15 | n15 | 0.600 | PASS |
| "hospitals AI scribes clinician documentation" | n16 | n16 | 0.444 | PASS |
| "protein structure drug candidate screening" | n17 | n17 | 0.553 | PASS |
| "humanoid robots foundation models manipulation" | n18 | n18 | 0.325 | PASS |
| "multimodal models understand long videos" | n19 | n19 | 0.613 | PASS |
| "prompt injection security risk tool-using agents" | n20 | n20 | 0.494 | PASS |
| "weather forecast tomorrow" | (none) | *no results* | — | PASS |

21/21 pass. That is expected: each query reuses several content words from
its target's title, which is the case TF-IDF handles best. In every case the
runner-up scores well below the top hit. The closest gap is n1, where the
runner-up is n2 at 0.186.

## Where it's weaker: paraphrases

TF-IDF matches surface tokens, not meaning. Queries that describe a document
without reusing its wording are a harder and more realistic test:

| Query (paraphrased) | Expected | Top-1 | Score | Result |
|---|---|---|---|---|
| "software that keeps working state saved so a long computation is not lost if it stops" | n2 | n2 | 0.245 | PASS (runner-up n19 at 0.182 is close) |
| "universal plugin standard for connecting assistants to external systems" | n3 | n3 | 0.178 | PASS (weak) |
| "drag and drop visual automation platform gains new AI capabilities" | n5 | n5 | 0.275 | PASS (mostly "automation"/"AI" overlap) |
| "big companies vs small startups in AI" | n15 | n15 | 0.505 | PASS (n15's body shares "companies"/"startups") |
| "electricity grid strain from machine learning" | n9 | n9 | 0.150 | PASS (the only nonzero match) |
| "doctors spending less time on paperwork" | n16 | n16 | 0.117 | PASS (barely: runner-up n7 at 0.110) |
| "malicious instructions hidden in web pages hijack assistants" | n20 | n20 | 0.236 | PASS |
| "government oversight of frontier models" | n7 | n11 | 0.214 | **FAIL** (xfail): "models" pulls toward n11, and n7 never says "government" or "oversight" |
| "chips that make running models cheaper" | n10 | n12 | 0.184 | **FAIL** (xfail): n10 ranks second at 0.104 |

7/9 paraphrases pass. The passing scores are mostly 2–4x lower than the
direct-vocabulary scores, and some pass by a thin margin (n16 beats n7 by
0.007). The two failures are the real failure mode. When the query and the
relevant document share no vocabulary, TF-IDF ranks whichever document shares
a generic word like "models" highest.

The earlier 6-article version of this doc listed "big companies vs small
startups in AI" as a questionable no-good-match case. With n15 now in the
corpus, the query has a real target and hits it with a high score.

## Takeaways

- **Direct-vocabulary queries** are the common case for a small news corpus,
  where users often search with words from headlines. Here TF-IDF retrieval
  was correct for every corpus document (20/20, plus the no-match case).
- **Paraphrased queries** do noticeably worse: 7/9 correct, with lower
  confidence scores. Success depends on incidental word overlap, and 2 of 9
  failed outright.
- **"No relevant document" detection** is limited to `retrieve()`'s raw
  `score > 0` cutoff. Fully out-of-domain queries return nothing, but a
  vaguely related query always returns something. A caller that needs
  stricter precision should apply a minimum-score threshold. A cutoff of
  about 0.15 would keep every direct hit, but it would also drop the two
  weakest correct paraphrase hits (0.117, and 0.1496, which rounds to
  0.150). Treat any
  threshold as a precision/recall trade-off, not a free fix.
- This matches the retriever's design intent (see the `rag/retriever.py`
  docstring): a real, local, deterministic retrieval baseline, not a
  replacement for semantic or embeddings-based search. The strict-xfail cases
  are where an embeddings retriever should show its value first.
