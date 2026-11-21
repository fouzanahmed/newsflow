import pytest
from fastapi.testclient import TestClient

from agent.app import TOPIC_MAX_LENGTH, app
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


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"topic": ""}, id="empty-string"),
        pytest.param({"topic": "   \t\n "}, id="whitespace-only"),
        pytest.param({}, id="missing-field"),
        pytest.param({"topic": "x" * (TOPIC_MAX_LENGTH + 1)}, id="too-long"),
    ],
)
def test_bulletin_rejects_invalid_topic(payload):
    resp = client.post("/bulletin", json=payload)
    assert resp.status_code == 422
    errors = resp.json()["detail"]
    assert any(err["loc"][-1] == "topic" for err in errors)


def test_bulletin_accepts_topic_at_max_length(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    resp = client.post("/bulletin", json={"topic": "x" * TOPIC_MAX_LENGTH})
    assert resp.status_code == 200


def test_openapi_documents_bulletin_response_schema():
    spec = client.get("/openapi.json").json()
    ok = spec["paths"]["/bulletin"]["post"]["responses"]["200"]
    ref = ok["content"]["application/json"]["schema"]["$ref"]
    assert ref.endswith("/BulletinResponse")

    schemas = spec["components"]["schemas"]
    resp_schema = schemas["BulletinResponse"]
    assert set(resp_schema["required"]) == {"topic", "source", "headline"}
    assert {"items", "bulletin_text"} <= set(resp_schema["properties"])
    assert set(resp_schema["properties"]["source"]["enum"]) == {
        "fallback_retrieval",
        "langchain_agent_claude",
    }
    assert set(schemas["BulletinItem"]["required"]) == {"title", "summary"}


def test_bulletin_fallback_response_omits_bulletin_text(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    body = client.post("/bulletin", json={"topic": "MCP adoption"}).json()
    assert "bulletin_text" not in body
    assert body["items"]
    assert all(set(item) == {"title", "summary"} for item in body["items"])


def test_bulletin_live_agent_shape_passes_response_model(monkeypatch):
    def fake_generate(topic):
        return {
            "topic": topic,
            "source": "langchain_agent_claude",
            "headline": f"AI & Agentic AI Developer Bulletin: {topic}",
            "bulletin_text": "- point one\n- point two",
        }

    monkeypatch.setattr("agent.app.generate_bulletin", fake_generate)
    resp = client.post("/bulletin", json={"topic": "MCP"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "langchain_agent_claude"
    assert body["bulletin_text"] == "- point one\n- point two"
    assert "items" not in body


def test_bulletin_strips_padded_topic(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    resp = client.post("/bulletin", json={"topic": "  MCP  "})
    assert resp.status_code == 200
    assert resp.json()["topic"] == "MCP"
