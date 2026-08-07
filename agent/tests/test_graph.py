"""Unit tests for the graph IR, registries, validator, and runtime.

No model, no network — deterministic nodes and a tiny fake engine exercise the
runtime; validator tests are pure.
"""
from __future__ import annotations

import pytest

from agent.graph import (
    CONDITION_REGISTRY,
    NODE_REGISTRY,
    Graph,
    GraphEdge,
    GraphNode,
    GraphResult,
    GraphRuntime,
    GraphState,
    LoopContract,
    NodeKind,
    validate_graph,
)
from agent.permissions import RiskLevel
from agent.profiles import resolve_profile


# ---------------------------------------------------------------------------
# IR round-trip
# ---------------------------------------------------------------------------

def test_graph_to_dict_round_trip() -> None:
    g = Graph(
        id="t", objective="obj",
        nodes=[
            GraphNode("a", NodeKind.DETERMINISTIC, "repository.inspect"),
            GraphNode("b", NodeKind.MODEL, "phase.run", depends_on=["a"]),
        ],
        edges=[GraphEdge("a", "b")],
    )
    d = g.to_dict()
    assert d["id"] == "t"
    assert [n["id"] for n in d["nodes"]] == ["a", "b"]
    assert d["edges"][0] == {"from": "a", "to": "b", "condition": "always"}


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def test_validator_accepts_valid_graph() -> None:
    g = Graph("t", "o", nodes=[
        GraphNode("a", NodeKind.DETERMINISTIC, "repository.inspect"),
        GraphNode("b", NodeKind.MODEL, "phase.run", depends_on=["a"]),
    ], edges=[GraphEdge("a", "b")])
    assert validate_graph(g) == []


def test_validator_flags_unknown_executor() -> None:
    g = Graph("t", "o", nodes=[GraphNode("a", NodeKind.MODEL, "does.not.exist")])
    defects = validate_graph(g)
    assert any("unknown executor" in d for d in defects)


def test_validator_flags_unknown_dependency() -> None:
    g = Graph("t", "o", nodes=[
        GraphNode("a", NodeKind.MODEL, "phase.run", depends_on=["ghost"]),
    ])
    defects = validate_graph(g)
    assert any("unknown node 'ghost'" in d for d in defects)


def test_validator_flags_unknown_edge_condition() -> None:
    g = Graph("t", "o", nodes=[
        GraphNode("a", NodeKind.DETERMINISTIC, "repository.inspect"),
        GraphNode("b", NodeKind.MODEL, "phase.run", depends_on=["a"]),
    ], edges=[GraphEdge("a", "b", condition="no_such_predicate")])
    defects = validate_graph(g)
    assert any("unknown condition" in d for d in defects)


def test_validator_flags_ungoverned_cycle() -> None:
    g = Graph("t", "o", nodes=[
        GraphNode("a", NodeKind.MODEL, "phase.run", depends_on=["b"]),
        GraphNode("b", NodeKind.MODEL, "phase.run", depends_on=["a"]),
    ])
    defects = validate_graph(g)
    assert any("cycle" in d for d in defects)


def test_validator_allows_cycle_with_loop_contract() -> None:
    g = Graph("t", "o", nodes=[
        GraphNode("a", NodeKind.MODEL, "phase.run", depends_on=["b"]),
        GraphNode("b", NodeKind.MODEL, "phase.run", depends_on=["a"]),
    ], loops=[
        LoopContract("a", max_iterations=2),
        LoopContract("b", max_iterations=2),
    ])
    # Both cycle members are governed by loop contracts → no ungoverned-cycle defect.
    assert not any("ungoverned cycle" in d for d in validate_graph(g))


def test_validator_flags_duplicate_ids() -> None:
    g = Graph("t", "o", nodes=[
        GraphNode("a", NodeKind.DETERMINISTIC, "repository.inspect"),
        GraphNode("a", NodeKind.MODEL, "phase.run"),
    ])
    assert any("duplicate node ids" in d for d in validate_graph(g))


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

def test_registry_risk_levels() -> None:
    assert NODE_REGISTRY["repository.inspect"].risk == RiskLevel.READ_ONLY
    assert NODE_REGISTRY["phase.run"].risk == RiskLevel.SANDBOXED_MUTATION
    assert NODE_REGISTRY["browser.inspect"].stub is True
    assert NODE_REGISTRY["repository.inspect"].stub is False


def test_condition_registry_has_core_predicates() -> None:
    for key in ("always", "passed", "failed", "verdict_rejected"):
        assert key in CONDITION_REGISTRY


# ---------------------------------------------------------------------------
# Runtime — deterministic nodes + a fake engine
# ---------------------------------------------------------------------------

class _FakeSession:
    session_id = "gsess"
    workspace_path = ""
    task = "t"
    goals: list = []


class _FakeEngine:
    """Minimal engine: only needs `_budget` (None) and `_execute_phase_step`."""
    _budget = None

    def __init__(self):
        self.phase_calls = []

    def _execute_phase_step(self, phase, profile, state, provider, model_override,
                            requested_files, generated_files, result, on_event):
        self.phase_calls.append(phase)
        # Simulate a phase writing a file.
        generated_files.append(f"{phase.lower()}.py")

        class _PR:
            output = "ok"
            attempts = 1
        return _PR()

    def _run_independent_verifier(self, state, verification, provider, on_event):
        return {"verdict": "APPROVED", "confidence": 90}


def _mk_state():
    profile = resolve_profile("desktop")
    return GraphState(
        session=_FakeSession(), profile=profile, provider="fake",
        model_override="", requested_files=[], generated_files=[],
        result=None,  # not used by deterministic/phase executors here
    )


def test_runtime_runs_deterministic_then_phase_in_order() -> None:
    g = Graph("t", "o", nodes=[
        GraphNode("repository.inspect", NodeKind.DETERMINISTIC, "repository.inspect"),
        GraphNode("Implementation", NodeKind.MODEL, "phase.run", depends_on=["repository.inspect"]),
    ], edges=[GraphEdge("repository.inspect", "Implementation")])
    assert validate_graph(g) == []

    engine = _FakeEngine()
    gstate = _mk_state()
    events = []
    res = GraphRuntime().run(g, gstate, engine, events.append)

    assert isinstance(res, GraphResult)
    assert res.completed == ["repository.inspect", "Implementation"]
    assert engine.phase_calls == ["Implementation"]  # deterministic node made no phase call
    types = [e["type"] for e in events]
    assert "node_started" in types and "node_completed" in types


def test_runtime_skips_node_when_edge_condition_false() -> None:
    # 'passed' is false (no verification set) → the gated target is skipped.
    g = Graph("t", "o", nodes=[
        GraphNode("repository.inspect", NodeKind.DETERMINISTIC, "repository.inspect"),
        GraphNode("only_if_passed", NodeKind.DETERMINISTIC, "integration",
                  depends_on=["repository.inspect"]),
    ], edges=[GraphEdge("repository.inspect", "only_if_passed", condition="passed")])

    engine = _FakeEngine()
    gstate = _mk_state()
    res = GraphRuntime().run(g, gstate, engine, None)
    assert "repository.inspect" in res.completed
    assert "only_if_passed" in res.skipped


def test_runtime_edge_condition_true_runs_node() -> None:
    g = Graph("t", "o", nodes=[
        GraphNode("repository.inspect", NodeKind.DETERMINISTIC, "repository.inspect"),
        GraphNode("gated", NodeKind.DETERMINISTIC, "integration", depends_on=["repository.inspect"]),
    ], edges=[GraphEdge("repository.inspect", "gated", condition="passed")])

    engine = _FakeEngine()
    gstate = _mk_state()
    gstate.verification["passed"] = True   # make the 'passed' predicate true
    res = GraphRuntime().run(g, gstate, engine, None)
    assert "gated" in res.completed


def test_stub_node_emits_not_implemented_without_crashing() -> None:
    g = Graph("t", "o", nodes=[
        GraphNode("b", NodeKind.HYBRID, "browser.inspect"),
    ])
    engine = _FakeEngine()
    gstate = _mk_state()
    res = GraphRuntime().run(g, gstate, engine, None)
    assert "b" in res.completed
    assert gstate.evidence[-1]["status"] == "not_implemented"
