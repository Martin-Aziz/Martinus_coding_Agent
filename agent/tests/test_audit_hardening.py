"""Independent audit tests for the hardened orchestrator.

These tests use a fake router (no model, no network) and are deliberately
independent of any thesis comparison tasks. They cover: retry policy,
run budgets, cancellation, truthful final status, truncation detection,
workspace confinement, and behavioral-test evidence propagation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent import orchestrator as orch
from agent.gates import run_workspace_tests
from agent.orchestrator import (
    Engine,
    RunAborted,
    RunBudget,
    _detect_truncated_file_blocks,
    _safe_abs_write_path,
    _extract_and_write_files,
)
from agent.providers import ChatResponse, DeterministicProviderError


class FakeRouter:
    """Scripted router: pops responses in order; entries may be strings
    (returned as content) or Exceptions (raised)."""

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
        if isinstance(item, Exception):
            raise item
        return ChatResponse(provider="fake", model="fake-model", content=item)

    def usage_totals(self):
        return dict(self._usage)


def make_engine(script):
    router = FakeRouter(script)
    return Engine(router=router, memory=None, routing=None), router


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------

def test_deterministic_error_is_not_retried(monkeypatch):
    monkeypatch.setattr(orch, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    engine, router = make_engine([DeterministicProviderError("no model selected")])
    engine._budget = RunBudget(max_seconds=60, max_model_calls=10)
    with pytest.raises(DeterministicProviderError):
        engine._call_with_retry("fake", "sys", "prompt", 0.1, "", 3, "Test", None)
    assert router.calls == 1  # no second attempt


def test_transient_error_retried_bounded(monkeypatch):
    monkeypatch.setattr(orch, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    engine, router = make_engine([RuntimeError("boom"), RuntimeError("boom"), RuntimeError("boom")])
    engine._budget = RunBudget(max_seconds=60, max_model_calls=10)
    with pytest.raises(RuntimeError, match="all 3 attempts failed"):
        engine._call_with_retry("fake", "sys", "prompt", 0.1, "", 3, "Test", None)
    assert router.calls == 3


def test_empty_response_counts_as_failure_then_recovers(monkeypatch):
    monkeypatch.setattr(orch, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    engine, router = make_engine(["", "real content"])
    engine._budget = RunBudget(max_seconds=60, max_model_calls=10)
    content, _, _, attempts = engine._call_with_retry(
        "fake", "sys", "prompt", 0.1, "", 3, "Test", None)
    assert content == "real content"
    assert attempts == 2


# ---------------------------------------------------------------------------
# Run budget and cancellation
# ---------------------------------------------------------------------------

def test_model_call_budget_aborts_run_truthfully(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "RUN_MAX_MODEL_CALLS", 1)
    monkeypatch.setattr(orch, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    engine, _ = make_engine(["discovery prose", "should never be reached"])
    result = engine.execute(
        session_id="s1", task="write something", workspace_path=str(tmp_path),
        goals=[], provider_name="ollama", hardware_profile="desktop",
    )
    assert result.status == "aborted"
    assert result.success is False
    assert "model-call budget" in result.failure_reason
    manifest = json.loads((tmp_path / ".martinus-run.json").read_text())
    assert manifest["status"] == "aborted"
    assert manifest["success"] is False


def test_cancellation_stops_run(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    engine, router = make_engine(["a", "b", "c", "d"])
    cancelled = {"flag": False}

    def on_event(evt):
        if evt.get("type") == "phase_completed":
            cancelled["flag"] = True  # cancel after first phase completes

    result = engine.execute(
        session_id="s2", task="anything", workspace_path=str(tmp_path),
        goals=[], provider_name="ollama", hardware_profile="desktop",
        on_event=on_event, cancel_check=lambda: cancelled["flag"],
    )
    assert result.status == "aborted"
    assert "cancelled" in result.failure_reason
    assert router.calls <= 2


def test_budget_check_raises_on_deadline():
    budget = RunBudget(max_seconds=1, max_model_calls=100)
    budget.deadline = 0  # force expiry
    with pytest.raises(RunAborted, match="wall-clock"):
        budget.check("test")


# ---------------------------------------------------------------------------
# Truthful final status
# ---------------------------------------------------------------------------

def test_run_with_valid_files_reports_success(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    impl = '<file path="main.py">print("hi")\n</file>'
    engine, _ = make_engine(["discovery", impl])
    result = engine.execute(
        session_id="s3", task="create main.py", workspace_path=str(tmp_path),
        goals=[], provider_name="ollama", hardware_profile="desktop",
    )
    assert result.status == "success"
    assert result.success is True
    assert (tmp_path / "main.py").exists()
    manifest = json.loads((tmp_path / ".martinus-run.json").read_text())
    assert manifest["verification"]["passed"] is True
    assert manifest["usage"]["model_requests"] >= 2


def test_run_with_no_files_reports_failure_not_success(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    # Model emits only prose claiming success — status must still be failure.
    engine, _ = make_engine(["discovery", "All tests passed. Verification successful."])
    result = engine.execute(
        session_id="s4", task="do a thing with no named artifacts",
        workspace_path=str(tmp_path), goals=[],
        provider_name="ollama", hardware_profile="desktop",
    )
    assert result.status == "failure"
    assert result.success is False


def test_static_gate_failure_repaired_then_success(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    broken = '<file path="main.py">def f(:\n</file>'
    fixed = '<file path="main.py">def f():\n    return 1\n</file>'
    engine, router = make_engine(["discovery", broken, fixed])
    result = engine.execute(
        session_id="s5", task="create main.py", workspace_path=str(tmp_path),
        goals=[], provider_name="ollama", hardware_profile="desktop",
    )
    assert result.status == "success"
    assert (tmp_path / "main.py").read_text().startswith("def f():")
    assert router.calls == 3  # discovery + implementation + one repair


def test_gate_failure_with_exhausted_repairs_is_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    broken = '<file path="main.py">def f(:\n</file>'
    engine, _ = make_engine(["discovery", broken, broken, broken, broken])
    result = engine.execute(
        session_id="s6", task="create main.py", workspace_path=str(tmp_path),
        goals=[], provider_name="ollama", hardware_profile="desktop",
    )
    assert result.status == "failure"
    assert result.success is False


# ---------------------------------------------------------------------------
# Malformed / truncated model output
# ---------------------------------------------------------------------------

def test_truncated_file_block_detected():
    assert _detect_truncated_file_blocks('<file path="a.py">x</file>') == 0
    assert _detect_truncated_file_blocks('<file path="a.py">x</file>\n<file path="b.py">cut of') == 1


def test_truncation_warning_appended_to_phase_output(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    truncated = '<file path="a.py">print(1)</file>\n<file path="b.py">print(2'
    engine, _ = make_engine(["discovery", truncated])
    events = []
    engine.execute(
        session_id="s7", task="create a.py and b.py", workspace_path=str(tmp_path),
        goals=[], provider_name="ollama", hardware_profile="desktop",
        on_event=events.append,
    )
    impl_outputs = [e["output"] for e in events
                    if e.get("type") == "phase_completed" and e.get("phase") == "Implementation"]
    assert impl_outputs and "[TRUNCATION WARNING]" in impl_outputs[0]


# ---------------------------------------------------------------------------
# Workspace confinement
# ---------------------------------------------------------------------------

def test_traversal_and_absolute_paths_refused(tmp_path):
    assert _safe_abs_write_path(str(tmp_path), "../evil.py") is None
    assert _safe_abs_write_path(str(tmp_path), "a/../../evil.py") is None
    assert _safe_abs_write_path(str(tmp_path), "C:evil.py") is None
    written = _extract_and_write_files(
        str(tmp_path), '<file path="../escape.py">x</file>')
    assert written == []
    assert not (tmp_path.parent / "escape.py").exists()


def test_symlink_escape_refused(tmp_path):
    outside = tmp_path.parent / "outside-target"
    outside.mkdir(exist_ok=True)
    (tmp_path / "link").symlink_to(outside)
    assert _safe_abs_write_path(str(tmp_path), "link/evil.py") is None


def test_workspace_root_confinement(tmp_path, monkeypatch):
    root = tmp_path / "allowed"
    root.mkdir()
    monkeypatch.setenv("ORCHESTRATOR_WORKSPACE_ROOT", str(root))
    engine, _ = make_engine(["x"])
    with pytest.raises(ValueError, match="outside ORCHESTRATOR_WORKSPACE_ROOT"):
        engine.execute(
            session_id="s8", task="t", workspace_path=str(tmp_path / "elsewhere"),
            goals=[], provider_name="ollama", hardware_profile="desktop",
        )


# ---------------------------------------------------------------------------
# Cross-run isolation
# ---------------------------------------------------------------------------

def test_temperature_scheduler_reset_between_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    engine, _ = make_engine(["a", '<file path="m.py">x=1\n</file>'])
    engine._temp.record_failure("Implementation")
    engine._temp.record_failure("Implementation")
    engine.execute(
        session_id="s9", task="create m.py", workspace_path=str(tmp_path),
        goals=[], provider_name="ollama", hardware_profile="desktop",
    )
    # execute() must have replaced the scheduler; old failure counts are gone.
    assert engine._temp._failures.get("Implementation", 0) == 0


# ---------------------------------------------------------------------------
# Behavioral test evidence
# ---------------------------------------------------------------------------

def test_run_workspace_tests_none_when_no_tests(tmp_path):
    assert run_workspace_tests(str(tmp_path)) is None


def test_run_workspace_tests_pass_and_fail(tmp_path):
    (tmp_path / "test_ok.py").write_text("def test_a():\n    assert True\n")
    res = run_workspace_tests(str(tmp_path), timeout=60)
    assert res is not None and res["passed"] is True

    (tmp_path / "test_ok.py").write_text("def test_a():\n    assert False\n")
    res = run_workspace_tests(str(tmp_path), timeout=60)
    assert res is not None and res["passed"] is False


def test_behavioral_failure_overrides_static_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    monkeypatch.setenv("MARTINUS_RUN_WORKSPACE_TESTS", "1")
    impl = ('<file path="main.py">x = 1\n</file>'
            '<file path="test_main.py">def test_bad():\n    assert False\n</file>')
    engine, _ = make_engine(["discovery", impl])
    result = engine.execute(
        session_id="s10", task="create main.py and test_main.py",
        workspace_path=str(tmp_path), goals=[],
        provider_name="ollama", hardware_profile="desktop",
    )
    assert result.status == "failure"
    assert result.success is False
