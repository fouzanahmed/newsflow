"""Structural checks for the deployed n8n workflow export.

These tests use only the stdlib ``json`` module, so they catch a broken export
or a dangling connection without needing an n8n instance.
"""

import json
from pathlib import Path

import pytest

WORKFLOW_PATH = Path(__file__).resolve().parent.parent / "n8n" / "ai_news_workflow.json"


@pytest.fixture(scope="module")
def workflow():
    with WORKFLOW_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def nodes_by_name(workflow):
    return {node["name"]: node for node in workflow["nodes"]}


def test_workflow_has_nodes_and_connections(workflow):
    assert isinstance(workflow.get("nodes"), list) and workflow["nodes"]
    assert isinstance(workflow.get("connections"), dict)


def test_node_names_are_unique(workflow):
    # n8n wires connections by node name, so duplicates would be ambiguous.
    names = [node["name"] for node in workflow["nodes"]]
    assert len(names) == len(set(names))


def test_connections_reference_existing_nodes(workflow, nodes_by_name):
    for source, outputs in workflow["connections"].items():
        assert source in nodes_by_name, f"connection source {source!r} is not a node"
        for output_type, branches in outputs.items():
            for branch in branches:
                for link in branch:
                    target = link["node"]
                    assert target in nodes_by_name, (
                        f"{source!r} ({output_type}) connects to missing node {target!r}"
                    )


@pytest.mark.parametrize(
    ("name", "node_type"),
    [
        ("Daily 8AM Trigger", "n8n-nodes-base.scheduleTrigger"),
        ("Generate AI News", "@n8n/n8n-nodes-langchain.lmChatGoogleGemini"),
    ],
)
def test_readme_described_nodes_present(nodes_by_name, name, node_type):
    # README: "It uses a Gemini node and a schedule trigger".
    assert name in nodes_by_name, f"missing node {name!r}"
    assert nodes_by_name[name]["type"] == node_type


def test_trigger_feeds_gemini_node(workflow):
    targets = [
        link["node"]
        for branch in workflow["connections"].get("Daily 8AM Trigger", {}).get("main", [])
        for link in branch
    ]
    assert "Generate AI News" in targets
