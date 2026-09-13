from fastapi.testclient import TestClient

from agent.app import app
from agent.bulletin_agent import generate_bulletin

client = TestClient(app)


def test_generate_bulletin_fallback_path_no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = generate_bulletin("LangGraph durable agent state")
    assert result["source"] == "fallback_retrieval"
    assert result["topic"] == "LangGraph durable agent state"
    assert len(result["items"]) > 0


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_bulletin_endpoint_fallback_path(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    resp = client.post("/bulletin", json={"topic": "MCP adoption"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "fallback_retrieval"
    assert "items" in body
