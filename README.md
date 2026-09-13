# NewsFlow

An AI news bulletin system combining a scheduled n8n workflow with a Python agent/RAG layer.

## What's real here

- **`n8n/ai_news_workflow.json`** — a working n8n workflow ("Daily AI News Bulletin") that runs on a schedule and generates a daily AI/dev news bulletin via Gemini. This is the original, actually-deployed automation.
- **`agent/` + `rag/`** — a separate Python demonstration of agentic tool-calling and retrieval-augmented generation, built to show the underlying techniques (LangChain tool-calling, retrieval) as real, testable code rather than only a workflow-tool configuration.

## Architecture

```
                    ┌────────────────────┐
   n8n (schedule) → │  Gemini prompt      │ → daily bulletin (existing, deployed)
                    └────────────────────┘

                    ┌────────────────────┐
   POST /bulletin → │ agent/app.py         │
                    │  → agent/bulletin_agent.py
                    │     ├─ has ANTHROPIC_API_KEY? → real LangChain tool-calling
                    │     │   agent (agent/tools.py + langchain-anthropic)
                    │     └─ otherwise → retrieval-only fallback
                    │           (rag/retriever.py, TF-IDF over rag/data/)
                    └────────────────────┘
```

The retrieval layer (`rag/retriever.py`) is a local TF-IDF retriever over a small sample news corpus (`rag/data/sample_news.json` — synthetic sample data for local development/testing, not scraped or live content). It's deliberately not an embeddings-based vector store (FAISS/Chroma): that would require either an embeddings API key or a heavy local model just to run tests. The retrieval interface (`retrieve(query, k)`) is the same shape either way, so swapping in a real vector store later is a contained change.

The live agent path (`generate_bulletin_live`) builds a genuine LangChain tool-calling agent (`create_tool_calling_agent` + `AgentExecutor`) with a `search_recent_news` tool backed by the retriever, and a Claude model via `langchain-anthropic`. **This path requires your own `ANTHROPIC_API_KEY`** and has not been exercised against a live API in this repo's own development — the tests and CI only exercise the fallback path, which needs no API key and is fully deterministic.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements-dev.txt
cp .env.example .env  # add your own ANTHROPIC_API_KEY to try the live agent path
uvicorn agent.app:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/bulletin -H "Content-Type: application/json" -d '{"topic": "LangGraph"}'
```

## Tests

```bash
pytest
```

Tests exercise the fallback (no-API-key) path only, by design — they're deterministic and don't depend on external services.

## The n8n workflow

Import `n8n/ai_news_workflow.json` into an n8n instance to run the scheduled bulletin generation. It uses a Gemini node and a schedule trigger; see the node parameters in the JSON for the exact prompt/schedule.
