"""Integration tests for sandbox isolation wired into Engine.execute().

Uses the FakeRouter (no model, no network), mirroring test_audit_hardening.py.
Forces sandbox isolation on via the MARTINUS_SANDBOX_ISOLATION env opt-in and
asserts that verified files still land in the ORIGINAL workspace after
apply-back, and that a failed run leaves the workspace untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent import orchestrator as orch
from agent.orchestrator import Engine, RunBudget
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


def test_sandbox_run_applies_files_back_to_original_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    monkeypatch.setenv("MARTINUS_SANDBOX_ISOLATION", "1")
    impl = '<file path="main.py">print("hi")\n</file>'
    engine = make_engine(["discovery", impl])

    events = []
    result = engine.execute(
        session_id="sbx1", task="create main.py", workspace_path=str(tmp_path),
        goals=[], provider_name="ollama", hardware_profile="desktop",
        on_event=events.append,
    )

    # Verified success → files applied back to the REAL workspace.
    assert result.status == "success"
    assert (tmp_path / "main.py").exists()
    assert (tmp_path / "main.py").read_text(encoding="utf-8") == 'print("hi")\n'

    # A sandbox was created and applied.
    types = [e.get("type") for e in events]
    assert "sandbox_created" in types
    assert "sandbox_applied" in types

    # Manifest lands in the ORIGINAL workspace (path was restored), and records
    # the apply-back.
    manifest = json.loads((tmp_path / ".martinus-run.json").read_text())
    assert manifest["status"] == "success"
    assert "main.py" in manifest["verification"]["sandbox"]["applied"]


def test_sandbox_created_in_temp_not_workspace_during_run(tmp_path, monkeypatch):
    """The sandbox_created event must point outside the user's workspace."""
    monkeypatch.setattr(orch, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    monkeypatch.setenv("MARTINUS_SANDBOX_ISOLATION", "1")
    impl = '<file path="main.py">x = 1\n</file>'
    engine = make_engine(["discovery", impl])

    captured = {}
    def on_event(evt):
        if evt.get("type") == "sandbox_created":
            captured["path"] = evt["sandboxPath"]

    engine.execute(
        session_id="sbx2", task="create main.py", workspace_path=str(tmp_path),
        goals=[], provider_name="ollama", hardware_profile="desktop",
        on_event=on_event,
    )
    assert "path" in captured
    # Sandbox path is not inside the user's workspace.
    assert str(tmp_path) not in captured["path"]


def test_failed_run_leaves_workspace_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    monkeypatch.setenv("MARTINUS_SANDBOX_ISOLATION", "1")
    # Only prose, no files → deterministic failure.
    engine = make_engine(["discovery", "All done, everything passed."])

    result = engine.execute(
        session_id="sbx3", task="create main.py", workspace_path=str(tmp_path),
        goals=[], provider_name="ollama", hardware_profile="desktop",
    )
    assert result.success is False
    # Nothing applied back to the real workspace.
    assert not (tmp_path / "main.py").exists()
    # Only the manifest is written to the original workspace.
    entries = {p.name for p in tmp_path.iterdir()}
    assert entries <= {".martinus-run.json", "MARTINUS_SESSION.md"}


def test_sandbox_off_by_default_writes_in_place(tmp_path, monkeypatch):
    """Without the env opt-in and on a non-sandbox profile, behavior is unchanged."""
    monkeypatch.setattr(orch, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    monkeypatch.delenv("MARTINUS_SANDBOX_ISOLATION", raising=False)
    impl = '<file path="main.py">print("x")\n</file>'
    engine = make_engine(["discovery", impl])

    events = []
    result = engine.execute(
        session_id="sbx4", task="create main.py", workspace_path=str(tmp_path),
        goals=[], provider_name="ollama", hardware_profile="desktop",
        on_event=events.append,
    )
    assert result.status == "success"
    assert (tmp_path / "main.py").exists()
    assert "sandbox_created" not in [e.get("type") for e in events]
