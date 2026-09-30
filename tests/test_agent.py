import sys
import types

import pytest
from fastapi.testclient import TestClient

import agent.tools
from agent.app import app
from agent.bulletin_agent import generate_bulletin, generate_bulletin_live

client = TestClient(app)


@pytest.fixture
def fake_langchain(monkeypatch):
    """Replace the modules generate_bulletin_live imports lazily with fakes,
    so the live path runs without an API key (or even langchain installed).
    Returns a dict that records what the live path passed to each piece."""
    calls = {}
    fake_tool = object()

    class FakeChatAnthropic:
        def __init__(self, **kwargs):
            calls["llm_kwargs"] = kwargs

    class FakeChatPromptTemplate:
        @classmethod
        def from_messages(cls, messages):
            calls["prompt_messages"] = messages
            return cls()

    def fake_create_tool_calling_agent(llm, tools, prompt):
        calls["agent_args"] = (llm, tools, prompt)
        return "fake-agent"

    class FakeAgentExecutor:
        output = "- Item one\n- Item two"

        def __init__(self, agent, tools):
            calls["executor_args"] = (agent, tools)

        def invoke(self, inputs):
            calls["invoke_inputs"] = inputs
            return {"output": self.output}

    agents_mod = types.ModuleType("langchain.agents")
    agents_mod.AgentExecutor = FakeAgentExecutor
    agents_mod.create_tool_calling_agent = fake_create_tool_calling_agent
    anthropic_mod = types.ModuleType("langchain_anthropic")
    anthropic_mod.ChatAnthropic = FakeChatAnthropic
    prompts_mod = types.ModuleType("langchain_core.prompts")
    prompts_mod.ChatPromptTemplate = FakeChatPromptTemplate

    monkeypatch.setitem(sys.modules, "langchain.agents", agents_mod)
    monkeypatch.setitem(sys.modules, "langchain_anthropic", anthropic_mod)
    monkeypatch.setitem(sys.modules, "langchain_core.prompts", prompts_mod)
    monkeypatch.setattr(agent.tools, "get_langchain_tool", lambda: fake_tool)

    calls["fake_tool"] = fake_tool
    calls["executor_cls"] = FakeAgentExecutor
    return calls


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


def test_generate_bulletin_live_returns_agent_output(fake_langchain):
    result = generate_bulletin_live("MCP adoption")
    assert result == {
        "topic": "MCP adoption",
        "source": "langchain_agent_claude",
        "headline": "AI & Agentic AI Developer Bulletin: MCP adoption",
        "bulletin_text": "- Item one\n- Item two",
    }


def test_generate_bulletin_live_wires_llm_tools_and_prompt(fake_langchain):
    generate_bulletin_live("MCP adoption")

    assert fake_langchain["llm_kwargs"]["model"].startswith("claude-")
    assert fake_langchain["llm_kwargs"]["temperature"] == 0

    llm, tools, _prompt = fake_langchain["agent_args"]
    assert tools == [fake_langchain["fake_tool"]]
    assert fake_langchain["executor_args"] == ("fake-agent", tools)
    assert fake_langchain["invoke_inputs"] == {
        "input": "Write a bulletin about: MCP adoption"
    }

    roles = [role for role, _ in fake_langchain["prompt_messages"]]
    assert roles == ["system", "human", "placeholder"]
    assert "search_recent_news" in fake_langchain["prompt_messages"][0][1]


def test_live_prompt_messages_build_a_real_chat_prompt_template(fake_langchain):
    # The fakes capture the messages; if langchain_core is installed, check
    # they form a valid template with the variables the executor supplies.
    generate_bulletin_live("MCP adoption")
    messages = fake_langchain["prompt_messages"]
    sys.modules.pop("langchain_core.prompts")
    prompts = pytest.importorskip("langchain_core.prompts")

    template = prompts.ChatPromptTemplate.from_messages(messages)
    assert "input" in template.input_variables
    rendered = template.format_messages(input="hi", agent_scratchpad=[])
    assert rendered[-1].content == "hi"


def test_generate_bulletin_uses_live_path_when_key_set(monkeypatch, fake_langchain):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    result = generate_bulletin("MCP adoption")
    assert result["source"] == "langchain_agent_claude"
    assert result["bulletin_text"] == "- Item one\n- Item two"


def test_generate_bulletin_falls_back_when_live_agent_raises(
    monkeypatch, fake_langchain
):
    def boom(self, inputs):
        raise RuntimeError("simulated Anthropic API error")

    monkeypatch.setattr(fake_langchain["executor_cls"], "invoke", boom)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    result = generate_bulletin("MCP adoption")
    assert result["source"] == "fallback_retrieval"
    assert len(result["items"]) > 0
