# NewsFlow

An AI news bulletin system combining a scheduled n8n workflow with a Python agent/RAG layer.

## What's real here

- **`n8n/ai_news_workflow.json`** — a working n8n workflow ("Daily AI News Bulletin") that runs on a schedule, generates a daily AI/dev news bulletin via Gemini, and delivers it to Telegram, Slack and WhatsApp Business. This is the original, actually-deployed automation.
- **`agent/` + `rag/`** — a separate Python demonstration of agentic tool-calling and retrieval-augmented generation, built to show the underlying techniques (LangChain tool-calling, retrieval) as real, testable code rather than only a workflow-tool configuration.

## Architecture

```
                    ┌────────────────────┐                     ┌→ Telegram
   n8n (schedule) → │  Gemini prompt      │ → Format Message → ├→ Slack
                    └────────────────────┘                     └→ WhatsApp Business

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

## API

The FastAPI app (`agent/app.py`) has two endpoints.

### `GET /health`

Liveness check. Always returns:

```json
{"ok": true}
```

### `POST /bulletin`

Request body:

```json
{"topic": "LangGraph"}
```

Validation (`BulletinRequest` in `agent/app.py`):

- `topic` is required and must be a string.
- The length is capped at `TOPIC_MAX_LENGTH = 200` characters. The cap is checked on the raw value, before stripping.
- Leading and trailing whitespace is stripped, and the stripped value is what the agent receives. An empty or whitespace-only topic is rejected.

A request that fails validation gets FastAPI's standard **422 Unprocessable Entity** response, for example for a 201-character topic:

```json
{
  "detail": [
    {
      "type": "string_too_long",
      "loc": ["body", "topic"],
      "msg": "String should have at most 200 characters",
      "input": "xxxx…",
      "ctx": {"max_length": 200}
    }
  ]
}
```

(A blank topic produces `"type": "value_error"` with `"msg": "Value error, topic must not be empty"`.)

The response's `source` field shows which path produced the bulletin.

**`"source": "fallback_retrieval"`**: no `ANTHROPIC_API_KEY` is set, or the live agent failed. The top 3 TF-IDF matches from the local corpus are returned as `items` (shown here with the summaries shortened):

```json
{
  "topic": "LangGraph",
  "source": "fallback_retrieval",
  "headline": "AI & Agentic AI Developer Bulletin: LangGraph",
  "items": [
    {
      "title": "LangGraph adds native support for durable, resumable agent state",
      "summary": "LangGraph's latest release lets long-running agent graphs checkpoint their state to disk or a database, ..."
    },
    {"title": "...", "summary": "..."},
    {"title": "...", "summary": "..."}
  ]
}
```

`items` can hold fewer than 3 entries, or none at all, if few corpus documents match the topic.

**`"source": "langchain_agent_claude"`**: `ANTHROPIC_API_KEY` is set and the LangChain tool-calling agent succeeded. Instead of `items`, the response has the model's free-text bulletin (the text below is only an illustration; the real output comes from the LLM):

```json
{
  "topic": "LangGraph",
  "source": "langchain_agent_claude",
  "headline": "AI & Agentic AI Developer Bulletin: LangGraph",
  "bulletin_text": "- LangGraph now checkpoints long-running agent state so runs can be paused and resumed...\n- ..."
}
```

## Tests

```bash
ruff check .
pytest
```

Tests exercise the fallback (no-API-key) path only, by design — they're deterministic and don't depend on external services.

## The n8n workflow

Import `n8n/ai_news_workflow.json` into an n8n instance to run the scheduled bulletin generation. The nodes run in this order:

1. **Daily 8AM Trigger** (`scheduleTrigger`): cron `0 8 * * *`.
2. **Generate AI News** (Google Gemini chat node, `gemini-2.0-flash-exp`): writes the bulletin from the prompt stored in the node parameters.
3. **Format Message** (`set`): copies the Gemini response into a `content` field.
4. Three delivery nodes that run in parallel from Format Message:
   - **Send to Telegram** (`telegram` node): sends `content` as Markdown to a chat ID.
   - **Send to Slack** (`slack` node): posts `content` to a channel.
   - **Send to WhatsApp Business** (`httpRequest` node): POSTs a `messaging_product: whatsapp` text message using HTTP header auth.

The chat ID, Slack channel and WhatsApp phone number in the JSON are placeholders (`your-telegram-chat-id`, `your-slack-channel`, `your-phone-number`). Replace them, and attach your own Gemini, Telegram, Slack and WhatsApp credentials in n8n, before you activate the workflow.
