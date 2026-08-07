"""Integration tests: the graph engine wired into Engine.execute() behind a flag.

Uses the FakeRouter (no model, no network), mirroring test_audit_hardening.py.
Proves: (1) with MARTINUS_GRAPH_ENGINE=1 the feature graph runs and produces the
same file + success as the linear path; (2) the graph trace lands in the manifest;
(3) default-off is byte-for-byte the linear behavior.
"""
from __future__ import annotations

import json
from pathlib import Path

from agent import orchestrator as orch
from agent.orchestrator import Engine
from agent.providers import ChatResponse


class FakeRouter:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self._usage = {"model_requests": 0, "prompt_tokens": 0,
                       "completion_tokens": 0, "total_tokens": 0}

    def resolve(self, provider):
        return "fake"

    def chat(self, provider, req):
        self.calls += 1
        self._usage["model_requests"] += 1
        item = self.script.pop(0) if self.script else "ok"
        return ChatResponse(provider="fake", model="fake-model", content=item)

    def usage_totals(self):
        return dict(self._usage)


def make_engine(script):
    return Engine(router=FakeRouter(script), memory=None, routing=None)


def test_graph_engine_runs_feature_graph_and_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    monkeypatch.setenv("MARTINUS_GRAPH_ENGINE", "1")
    # desktop profile: Discovery + Implementation phases → 2 phase calls.
    impl = '<file path="main.py">print("hi")\n</file>'
    engine = make_engine(["discovery", impl])

    events = []
    result = engine.execute(
        session_id="g1", task="create main.py", workspace_path=str(tmp_path),
        goals=[], provider_name="ollama", hardware_profile="desktop",
        on_event=events.append,
    )

    assert result.status == "success"
    assert (tmp_path / "main.py").exists()

    types = [e.get("type") for e in events]
    assert "graph_compiled" in types
    assert "node_started" in types
    assert "node_completed" in types

    # Graph trace recorded in the manifest.
    manifest = json.loads((tmp_path / ".martinus-run.json").read_text())
    graph = manifest["verification"]["graph"]
    assert graph["graphId"] == "feature"
    assert "repository.inspect" in graph["completed"]
    assert "Implementation" in graph["completed"]


def test_graph_compiled_event_carries_node_list(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    monkeypatch.setenv("MARTINUS_GRAPH_ENGINE", "1")
    engine = make_engine(["discovery", '<file path="a.py">x=1\n</file>'])

    captured = {}
    def on_event(evt):
        if evt.get("type") == "graph_compiled":
            captured["graph"] = evt["graph"]

    engine.execute(
        session_id="g2", task="create a.py", workspace_path=str(tmp_path),
        goals=[], provider_name="ollama", hardware_profile="desktop",
        on_event=on_event,
    )
    assert "graph" in captured
    node_ids = [n["id"] for n in captured["graph"]["nodes"]]
    assert node_ids[0] == "repository.inspect"
    assert "Implementation" in node_ids


def test_graph_engine_off_by_default_no_graph_events(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    monkeypatch.delenv("MARTINUS_GRAPH_ENGINE", raising=False)
    engine = make_engine(["discovery", '<file path="main.py">print("x")\n</file>'])

    events = []
    result = engine.execute(
        session_id="g3", task="create main.py", workspace_path=str(tmp_path),
        goals=[], provider_name="ollama", hardware_profile="desktop",
        on_event=events.append,
    )
    assert result.status == "success"
    assert (tmp_path / "main.py").exists()
    types = [e.get("type") for e in events]
    assert "graph_compiled" not in types
    # Linear path emits phase_started, not node_started.
    assert "phase_started" in types
    assert "node_started" not in types
    manifest = json.loads((tmp_path / ".martinus-run.json").read_text())
    assert "graph" not in manifest["verification"]


def test_graph_and_linear_produce_same_file(tmp_path, monkeypatch):
    """Golden-ish: same script + task yields the same generated file either way."""
    monkeypatch.setattr(orch, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    impl = '<file path="calc.py">def add(a, b):\n    return a + b\n</file>'

    # Linear
    monkeypatch.delenv("MARTINUS_GRAPH_ENGINE", raising=False)
    lin_dir = tmp_path / "lin"
    lin_dir.mkdir()
    make_engine(["discovery", impl]).execute(
        session_id="lin", task="create calc.py", workspace_path=str(lin_dir),
        goals=[], provider_name="ollama", hardware_profile="desktop",
    )

    # Graph
    monkeypatch.setenv("MARTINUS_GRAPH_ENGINE", "1")
    g_dir = tmp_path / "graph"
    g_dir.mkdir()
    make_engine(["discovery", impl]).execute(
        session_id="grf", task="create calc.py", workspace_path=str(g_dir),
        goals=[], provider_name="ollama", hardware_profile="desktop",
    )

    assert (lin_dir / "calc.py").read_text() == (g_dir / "calc.py").read_text()
