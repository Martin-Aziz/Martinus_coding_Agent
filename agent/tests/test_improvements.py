"""Tests for the four research-driven improvements (deep-research follow-up):

1. `simple` graph pattern — fast-path routing for small greenfield tasks.
2. Behavioral-test default-on for medium/high profiles.
3. Context compression between phases (deterministic budget trimmer).
4. Budget-aware verifier refinement loop (REJECTED → retry → re-verify).

Uses a scripted FakeRouter (no model, no network), mirroring the existing
verifier/graph integration tests.
"""
from __future__ import annotations

from agent import orchestrator as orch
from agent.orchestrator import (
    Engine,
    SessionEntry,
    _compress_history,
    _COMPRESS_BUDGET,
    _HISTORY_TOTAL_CHAR_BUDGET,
)
from agent.providers import ChatResponse
from agent.profiles import resolve_profile
from agent.graph_patterns import classify_task, compile_graph, PATTERNS
from agent.graph import validate_graph


class FakeRouter:
    def __init__(self, script):
        self.script = list(script)
        self._usage = {"model_requests": 0, "prompt_tokens": 0,
                       "completion_tokens": 0, "total_tokens": 0}

    def resolve(self, provider):
        return "fake"

    def chat(self, provider, req):
        self._usage["model_requests"] += 1
        item = self.script.pop(0) if self.script else "ok"
        return ChatResponse(provider="fake", model="fake-model", content=item)

    def usage_totals(self):
        return dict(self._usage)


def make_engine(script):
    return Engine(router=FakeRouter(script), memory=None, routing=None)


# ---------------------------------------------------------------------------
# Priority 1: simple pattern classification + graph shape
# ---------------------------------------------------------------------------

def test_simple_pattern_selected_for_small_greenfield_task() -> None:
    assert classify_task("create a calculator script", []) == "simple"
    assert classify_task("write a function to reverse a string", []) == "simple"
    assert classify_task("make a hello world program", []) == "simple"


def test_simple_pattern_not_selected_for_complex_tasks() -> None:
    # Feature-scale tasks must NOT route to simple.
    assert classify_task("create a login feature", []) == "feature"
    assert classify_task("build a simple authentication system", []) != "simple"
    assert classify_task("create a new API endpoint", []) == "feature"
    # Bug / refactor keywords still win.
    assert classify_task("fix a crash in the parser", []) == "bug_repair"


def test_simple_graph_skips_architecture_and_clarification() -> None:
    profile = resolve_profile("medium")
    graph = compile_graph("create a calculator script", [], profile)
    assert graph.id == "simple"
    node_ids = [n.id for n in graph.nodes]
    # Fast path: Discovery + Implementation only, no Architecture/Clarification/Review.
    assert "Architecture" not in node_ids
    assert "Clarification" not in node_ids
    assert "Review" not in node_ids
    assert "Discovery" in node_ids
    assert "Implementation" in node_ids


def test_simple_graph_is_valid() -> None:
    profile = resolve_profile("high")
    graph = PATTERNS["simple"](
        __import__("agent.graph_patterns", fromlist=["CompileInput"]).CompileInput(
            task="write a script", goals=[], profile=profile,
        )
    )
    assert validate_graph(graph) == []


def test_simple_graph_fewer_model_phases_than_feature() -> None:
    profile = resolve_profile("medium")
    simple = compile_graph("create a small utility script", [], profile)
    feature = compile_graph("add CSV export to the reports page", [], profile)
    simple_phases = [n for n in simple.nodes if n.executor == "phase.run"]
    feature_phases = [n for n in feature.nodes if n.executor == "phase.run"]
    assert len(simple_phases) < len(feature_phases)


# ---------------------------------------------------------------------------
# Priority 2: behavioral tests default-on for medium/high
# ---------------------------------------------------------------------------

def test_behavioral_tests_flag_on_for_medium_and_high() -> None:
    assert resolve_profile("medium").behavioral_tests is True
    assert resolve_profile("high").behavioral_tests is True


def test_behavioral_tests_flag_off_for_desktop_and_low() -> None:
    assert resolve_profile("desktop").behavioral_tests is False
    assert resolve_profile("low").behavioral_tests is False


# ---------------------------------------------------------------------------
# Priority 3: context compression
# ---------------------------------------------------------------------------

def test_compress_history_empty_is_noop() -> None:
    assert _compress_history([], "Implementation") == []


def test_compress_history_truncates_long_entries() -> None:
    long_content = "x" * 5000
    entries = [
        SessionEntry(phase="Discovery", content=long_content),
        SessionEntry(phase="Architecture", content=long_content),
    ]
    compressed = _compress_history(entries, "Implementation")
    # Each entry trimmed to its per-phase budget (+ marker), original untouched.
    for e in compressed:
        assert len(e.content) <= _COMPRESS_BUDGET.get(e.phase, 400) + len(" …[compressed]")
    assert entries[0].content == long_content  # originals unchanged


def test_compress_history_keeps_recent_verbatim_for_review() -> None:
    recent = "y" * 5000
    entries = [
        SessionEntry(phase="Discovery", content="short"),
        SessionEntry(phase="Implementation", content=recent),
    ]
    compressed = _compress_history(entries, "Verification")
    # Verification is a reading phase — last entry kept verbatim.
    assert compressed[-1].content == recent


def test_compress_history_drops_oldest_when_over_total_budget() -> None:
    # Many entries each at their budget; total must be trimmed to fit.
    entries = [
        SessionEntry(phase="Implementation", content="z" * 2000)
        for _ in range(10)
    ]
    compressed = _compress_history(entries, "Discovery")
    total = sum(len(e.content) for e in compressed)
    assert total <= _HISTORY_TOTAL_CHAR_BUDGET or len(compressed) == 1


def test_structure_aware_compression_keeps_signal_drops_prose() -> None:
    """Multi-line entry: file paths / decisions survive, filler prose is dropped."""
    from agent.orchestrator import _compress_entry_structurally
    lines = ["Summary of work done."]
    # 40 lines of pure prose filler.
    lines += [f"This is some narrative sentence number {i} with no facts." for i in range(40)]
    # High-signal lines interleaved.
    lines += [
        '<file path="src/app.py">',
        "def compute_total(items):",
        "- Decision: use a dict for O(1) lookups",
        "import json",
    ]
    content = "\n".join(lines)
    budget = 400
    out = _compress_entry_structurally(content, budget)
    # Signal lines survive.
    assert "src/app.py" in out
    assert "def compute_total" in out
    assert "Decision: use a dict" in out
    assert "import json" in out
    # A lot of the prose filler is gone (structure beats blind truncation, which
    # would have kept only the first ~400 chars = the summary + first few prose lines).
    assert out.count("narrative sentence") < 40
    # Compression marker present.
    assert "compressed" in out


def test_structure_aware_compression_single_line_falls_back() -> None:
    """No line structure → honest plain truncation, bounded by budget + marker."""
    from agent.orchestrator import _compress_entry_structurally
    out = _compress_entry_structurally("x" * 5000, 400)
    assert len(out) <= 400 + len(" …[compressed]")
    assert out.endswith("…[compressed]")


# ---------------------------------------------------------------------------
# Priority 4: budget-aware refinement loop
# ---------------------------------------------------------------------------

def test_refinement_retries_on_rejected_then_approves(tmp_path, monkeypatch):
    """REJECTED verdict → refine Implementation → re-verify APPROVED → success."""
    monkeypatch.setattr(orch, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    monkeypatch.setenv("MARTINUS_INDEPENDENT_VERIFIER", "1")
    impl = '<file path="main.py">print("v1")\n</file>'
    impl2 = '<file path="main.py">print("v2 fixed")\n</file>'
    # desktop: Discovery, Implementation → 2 phase calls; verifier REJECTED;
    # refinement re-runs Implementation; re-verify APPROVED.
    engine = make_engine([
        "discovery", impl,
        "VERDICT: REJECTED (CONFIDENCE: 30)",   # first verdict
        impl2,                                   # refinement Implementation
        "VERDICT: APPROVED (CONFIDENCE: 95)",    # re-verify
    ])
    events = []
    result = engine.execute(
        session_id="r1", task="create main.py", workspace_path=str(tmp_path),
        goals=[], provider_name="ollama", hardware_profile="desktop",
        on_event=events.append,
    )
    # A refinement round fired.
    assert any(e.get("type") == "refinement_round" for e in events)
    # Final approved result is a success.
    assert result.success is True


def test_refinement_gives_up_after_max_rounds(tmp_path, monkeypatch):
    """Persistent REJECTED across rounds → failure stands, not silently upgraded."""
    monkeypatch.setattr(orch, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    monkeypatch.setenv("MARTINUS_INDEPENDENT_VERIFIER", "1")
    impl = '<file path="main.py">print("hi")\n</file>'
    # Every verdict is REJECTED — after max rounds the result must be a failure.
    engine = make_engine([
        "discovery", impl,
        "VERDICT: REJECTED (CONFIDENCE: 10)",
        impl, "VERDICT: REJECTED (CONFIDENCE: 10)",
        impl, "VERDICT: REJECTED (CONFIDENCE: 10)",
    ])
    result = engine.execute(
        session_id="r2", task="create main.py", workspace_path=str(tmp_path),
        goals=[], provider_name="ollama", hardware_profile="desktop",
    )
    assert result.success is False
    assert result.status == "failure"


def test_no_refinement_on_human_review(tmp_path, monkeypatch):
    """HUMAN_REVIEW must NOT trigger automated refinement rounds."""
    monkeypatch.setattr(orch, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    monkeypatch.setenv("MARTINUS_INDEPENDENT_VERIFIER", "1")
    impl = '<file path="main.py">print("hi")\n</file>'
    engine = make_engine(["discovery", impl, "VERDICT: HUMAN_REVIEW (CONFIDENCE: 50)"])
    events = []
    engine.execute(
        session_id="r3", task="create main.py", workspace_path=str(tmp_path),
        goals=[], provider_name="ollama", hardware_profile="desktop",
        on_event=events.append,
    )
    # No refinement round fired.
    assert not any(e.get("type") == "refinement_round" for e in events)


# ---------------------------------------------------------------------------
# Improvement 8: adaptive profile downgrade for simple tasks
# ---------------------------------------------------------------------------

def test_adaptive_downgrade_trims_phases_for_simple_task(tmp_path, monkeypatch):
    """A 'simple' task on a medium profile drops to a lean phase set for the run."""
    monkeypatch.setattr(orch, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    monkeypatch.setenv("MARTINUS_ADAPTIVE_PROFILE", "1")
    # medium normally runs 6 phases. A simple task should trim to <= 3.
    engine = make_engine(["discovery", "clarify", "arch",
                          '<file path="calc.py">print(1)\n</file>',
                          "review", "verify"])
    events = []
    engine.execute(
        session_id="a1", task="create a calculator script",
        workspace_path=str(tmp_path), goals=[],
        provider_name="ollama", hardware_profile="medium",
        on_event=events.append,
    )
    downgrade = [e for e in events if e.get("type") == "profile_downgraded"]
    assert downgrade, "expected a profile_downgraded event for a simple task"
    assert set(downgrade[0]["phases"]) <= {"Discovery", "Implementation", "Verification"}
    # The heavy phases must not have run.
    ran_phases = {e.get("phase") for e in events if e.get("type") == "phase_started"}
    assert "Architecture" not in ran_phases
    assert "Clarification" not in ran_phases


def test_adaptive_downgrade_off_keeps_full_pipeline(tmp_path, monkeypatch):
    """With the flag off, a simple task still runs the full medium pipeline."""
    monkeypatch.setattr(orch, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    monkeypatch.setenv("MARTINUS_ADAPTIVE_PROFILE", "0")
    engine = make_engine(["discovery", "clarify", "arch",
                          '<file path="calc.py">print(1)\n</file>',
                          "review", "verify"])
    events = []
    engine.execute(
        session_id="a2", task="create a calculator script",
        workspace_path=str(tmp_path), goals=[],
        provider_name="ollama", hardware_profile="medium",
        on_event=events.append,
    )
    assert not any(e.get("type") == "profile_downgraded" for e in events)


def test_adaptive_downgrade_skips_complex_task(tmp_path, monkeypatch):
    """A feature-scale task must NOT be downgraded even with the flag on."""
    monkeypatch.setattr(orch, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    monkeypatch.setenv("MARTINUS_ADAPTIVE_PROFILE", "1")
    engine = make_engine(["discovery", "clarify", "arch",
                          '<file path="auth.py">print(1)\n</file>',
                          "review", "verify"])
    events = []
    engine.execute(
        session_id="a3", task="build a user authentication system",
        workspace_path=str(tmp_path), goals=[],
        provider_name="ollama", hardware_profile="medium",
        on_event=events.append,
    )
    assert not any(e.get("type") == "profile_downgraded" for e in events)


# ---------------------------------------------------------------------------
# Improvement 9: early-exit verifier on conclusive deterministic evidence
# ---------------------------------------------------------------------------

def test_verifier_early_exit_when_gates_and_tests_pass(tmp_path, monkeypatch):
    """Simple task + gates pass + behavioral tests pass → skip the verifier call."""
    monkeypatch.setattr(orch, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    monkeypatch.setenv("MARTINUS_INDEPENDENT_VERIFIER", "1")
    monkeypatch.setenv("MARTINUS_VERIFY_EARLY_EXIT", "1")
    monkeypatch.setenv("MARTINUS_ADAPTIVE_PROFILE", "0")  # isolate the early-exit
    monkeypatch.setenv("MARTINUS_RUN_WORKSPACE_TESTS", "1")
    # Force behavioral tests to run and report a pass, deterministically.
    monkeypatch.setattr(
        orch._gates, "run_workspace_tests",
        lambda ws: {"passed": True, "runner": "pytest", "collected": 1, "summary": "1 passed"},
    )
    impl = '<file path="calc.py">def add(a, b):\n    return a + b\n</file>'
    # Only Discovery + Implementation are scripted. If the verifier were called,
    # it would consume a (missing) script entry and get the default "ok" → a
    # HUMAN_REVIEW verdict. Early-exit means no verifier call at all.
    engine = make_engine(["discovery", impl])
    events = []
    result = engine.execute(
        session_id="e1", task="create a calculator script",
        workspace_path=str(tmp_path), goals=[],
        provider_name="ollama", hardware_profile="desktop",
        on_event=events.append,
    )
    assert any(e.get("type") == "verifier_early_exit" for e in events)
    # No independent-verifier phase event fired.
    assert not any(e.get("type") == "independent_verdict" for e in events)


def test_no_early_exit_when_tests_fail(tmp_path, monkeypatch):
    """If behavioral tests fail, the verifier must still run (no early-exit)."""
    monkeypatch.setattr(orch, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    monkeypatch.setenv("MARTINUS_INDEPENDENT_VERIFIER", "1")
    monkeypatch.setenv("MARTINUS_VERIFY_EARLY_EXIT", "1")
    monkeypatch.setenv("MARTINUS_ADAPTIVE_PROFILE", "0")
    monkeypatch.setenv("MARTINUS_RUN_WORKSPACE_TESTS", "1")
    monkeypatch.setattr(
        orch._gates, "run_workspace_tests",
        lambda ws: {"passed": False, "runner": "pytest", "collected": 1, "summary": "1 failed"},
    )
    impl = '<file path="calc.py">def add(a, b):\n    return a + b\n</file>'
    engine = make_engine(["discovery", impl, "VERDICT: HUMAN_REVIEW (CONFIDENCE: 50)"])
    events = []
    engine.execute(
        session_id="e2", task="create a calculator script",
        workspace_path=str(tmp_path), goals=[],
        provider_name="ollama", hardware_profile="desktop",
        on_event=events.append,
    )
    # Early-exit must NOT fire when tests failed.
    assert not any(e.get("type") == "verifier_early_exit" for e in events)


def test_no_early_exit_for_complex_task(tmp_path, monkeypatch):
    """A non-simple task never early-exits even with gates+tests passing."""
    monkeypatch.setattr(orch, "RETRY_BACKOFF_BASE_SECONDS", 0.0)
    monkeypatch.setenv("MARTINUS_INDEPENDENT_VERIFIER", "1")
    monkeypatch.setenv("MARTINUS_VERIFY_EARLY_EXIT", "1")
    monkeypatch.setenv("MARTINUS_ADAPTIVE_PROFILE", "0")
    monkeypatch.setenv("MARTINUS_RUN_WORKSPACE_TESTS", "1")
    monkeypatch.setattr(
        orch._gates, "run_workspace_tests",
        lambda ws: {"passed": True, "runner": "pytest", "collected": 1, "summary": "1 passed"},
    )
    impl = '<file path="auth.py">def login():\n    return True\n</file>'
    engine = make_engine(["discovery", impl, "VERDICT: APPROVED (CONFIDENCE: 90)"])
    events = []
    engine.execute(
        session_id="e3", task="build a user authentication system",
        workspace_path=str(tmp_path), goals=[],
        provider_name="ollama", hardware_profile="desktop",
        on_event=events.append,
    )
    assert not any(e.get("type") == "verifier_early_exit" for e in events)


