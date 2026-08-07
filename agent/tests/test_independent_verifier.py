"""Tests for the independent verifier (LOCALFORGE Rule 2.5 / §9).

Uses a FakeRouter that records the messages it receives so we can assert the
verifier prompt is built with FRESH context (no implementer phase history).
"""
from __future__ import annotations

from pathlib import Path

from agent import orchestrator as orch
from agent.orchestrator import Engine, _parse_verifier_verdict
from agent.providers import ChatResponse


class RecordingRouter:
    """Scripted router that records (system, user) message pairs per call."""

    def __init__(self, script):
        self.script = list(script)
        self.messages = []  # list of (system, user)
        self._usage = {"model_requests": 0, "prompt_tokens": 0,
                       "completion_tokens": 0, "total_tokens": 0}

    def resolve(self, provider):
        return "fake"

    def chat(self, provider, req):
        self._usage["model_requests"] += 1
        sys_msg = next((m.content for m in req.messages if m.role == "system"), "")
        user_msg = next((m.content for m in req.messages if m.role == "user"), "")
        self.messages.append((sys_msg, user_msg))
        item = self.script.pop(0) if self.script else "ok"
        return ChatResponse(provider="fake", model="fake-model", content=item)

    def usage_totals(self):
        return dict(self._usage)


def make_engine(script):
    router = RecordingRouter(script)
    return Engine(router=router, memory=None, routing=None), router


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------

def test_parse_verdict_variants() -> None:
    assert _parse_verifier_verdict("...\nVERDICT: APPROVED (CONFIDENCE: 88)") == ("APPROVED", 88)
    assert _parse_verifier_verdict("VERDICT: REJECTED (CONFIDENCE: 10)") == ("REJECTED", 10)
    assert _parse_verifier_verdict("garbage with no verdict") == ("HUMAN_REVIEW", 0)
    # Out-of-range confidence clamped.
    assert _parse_verifier_verdict("VERDICT: APPROVED (CONFIDENCE: 999)") == ("APPROVED", 100)


# ---------------------------------------------------------------------------
# Integration through execute()
# ---------------------------------------------------------------------------

def test_rejected_verdict_downgrades_a_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    monkeypatch.setenv("MARTINUS_INDEPENDENT_VERIFIER", "1")
    impl = '<file path="main.py">print("hi")\n</file>'
    # desktop profile: Discovery, Implementation → 2 calls; verifier → 3rd call.
    engine, router = make_engine(["discovery", impl, "VERDICT: REJECTED (CONFIDENCE: 20)"])

    result = engine.execute(
        session_id="v1", task="create main.py", workspace_path=str(tmp_path),
        goals=[], provider_name="ollama", hardware_profile="desktop",
    )
    # File was produced (deterministic pass) but the verifier rejected it.
    assert (tmp_path / "main.py").exists()
    assert result.success is False
    assert result.status == "failure"
    assert "REJECTED" in result.failure_reason


def test_approved_verdict_keeps_a_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    monkeypatch.setenv("MARTINUS_INDEPENDENT_VERIFIER", "1")
    impl = '<file path="main.py">print("hi")\n</file>'
    engine, _ = make_engine(["discovery", impl, "VERDICT: APPROVED (CONFIDENCE: 95)"])

    result = engine.execute(
        session_id="v2", task="create main.py", workspace_path=str(tmp_path),
        goals=[], provider_name="ollama", hardware_profile="desktop",
    )
    assert result.success is True
    assert result.status == "success"


def test_verifier_cannot_upgrade_a_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    monkeypatch.setenv("MARTINUS_INDEPENDENT_VERIFIER", "1")
    # Prose only → deterministic failure. Verifier "APPROVED" must NOT flip it.
    engine, _ = make_engine(["discovery", "all good, done!", "VERDICT: APPROVED (CONFIDENCE: 99)"])

    result = engine.execute(
        session_id="v3", task="create main.py", workspace_path=str(tmp_path),
        goals=[], provider_name="ollama", hardware_profile="desktop",
    )
    assert result.success is False
    assert result.status == "failure"


def test_verifier_prompt_has_no_phase_history(tmp_path, monkeypatch):
    """Independence check: the verifier prompt must NOT contain the implementer's
    phase transcript (spec §2.5)."""
    monkeypatch.setattr(orch, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    monkeypatch.setenv("MARTINUS_INDEPENDENT_VERIFIER", "1")
    secret_reasoning = "IMPLEMENTER_SECRET_REASONING_TOKEN"
    impl = f'<file path="main.py">print("hi")\n</file>\n{secret_reasoning}'
    engine, router = make_engine(["discovery reasoning here", impl, "VERDICT: APPROVED (CONFIDENCE: 90)"])

    engine.execute(
        session_id="v4", task="create main.py", workspace_path=str(tmp_path),
        goals=["main.py prints hi"], provider_name="ollama", hardware_profile="desktop",
    )
    # The last model call is the verifier.
    verifier_system, verifier_user = router.messages[-1]
    assert "INDEPENDENT verifier" in verifier_system
    # It must include the original request + goals...
    assert "create main.py" in verifier_user
    assert "main.py prints hi" in verifier_user
    # ...but NOT the discovery-phase reasoning transcript.
    assert "discovery reasoning here" not in verifier_user


def test_verdict_recorded_in_manifest(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(orch, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    monkeypatch.setenv("MARTINUS_INDEPENDENT_VERIFIER", "1")
    impl = '<file path="main.py">print("hi")\n</file>'
    engine, _ = make_engine(["discovery", impl, "VERDICT: HUMAN_REVIEW (CONFIDENCE: 50)"])

    engine.execute(
        session_id="v5", task="create main.py", workspace_path=str(tmp_path),
        goals=[], provider_name="ollama", hardware_profile="desktop",
    )
    manifest = json.loads((tmp_path / ".martinus-run.json").read_text())
    verdict = manifest["verification"]["independentVerdict"]
    assert verdict["verdict"] == "HUMAN_REVIEW"
    assert verdict["confidence"] == 50
