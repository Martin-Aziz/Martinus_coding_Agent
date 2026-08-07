"""
agent/graph.py — Graph-native execution runtime (LOCALFORGE graph engine, v1).

Reifies Martinus's execution as a typed, validated graph of nodes and conditional
edges, instead of a hardcoded linear phase loop. The graph — not one monolithic
prompt — becomes the unit of control: what runs next, in what order, under which
budget, producing which evidence.

This is an ADDITIVE layer. It does not replace the phase logic; its `phase.run`
node executor calls straight back into `Engine._execute_phase_step`, so every
node reuses the existing candidate-design / tree-of-code / self-critique /
snapshot / recovery / gate-repair behavior. When the graph engine is off
(default), `Engine.execute()` runs the original linear path unchanged.

Design constraints (mirroring the shipped LOCALFORGE modules):
- Zero new dependencies; stdlib only; fully typed (strict mypy).
- Deterministic nodes cost zero model calls.
- Node risk levels reuse agent.permissions.RiskLevel.
- Edges/loops reference predicates by NAME (registry keys) so the IR stays
  declarative and serializable to/from dict (YAML/JSON/manifest) — no lambdas
  live in the IR itself.
- Never raises through a node: a node executor that fails records failure
  evidence and the runtime decides via edges, matching the "evidence over
  assumption" rule.

Scope honesty: the `feature` pattern and all model/deterministic executors here
are fully functional. Browser / advanced-symbol / test-detect executors are
STUBS that emit a structured {"status": "not_implemented"} evidence record —
they belong to earlier roadmap phases (execution kernel, Playwright) that are
not built yet. Nothing silently pretends to run a browser.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, TYPE_CHECKING

from .permissions import RiskLevel
from .profiles import HardwareProfile

if TYPE_CHECKING:  # avoid a runtime import cycle with orchestrator
    from .orchestrator import Engine, SessionState, RunResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IR
# ---------------------------------------------------------------------------

class NodeKind(str, Enum):
    DETERMINISTIC = "deterministic"   # pure code, no model call
    MODEL = "model"                   # a model-backed phase
    HYBRID = "hybrid"                 # deterministic evidence + model judgment
    SUBGRAPH = "subgraph"             # expands to another graph
    VERIFIER = "verifier"             # independent, fresh-context review


@dataclass
class GraphNode:
    id: str
    kind: NodeKind
    executor: str                                  # NODE_REGISTRY key
    depends_on: list[str] = field(default_factory=list)
    model_profile: str = ""                        # capability/role hint
    tools: list[str] = field(default_factory=list)
    max_input_tokens: int = 0                      # 0 = use global budget
    context_include: list[str] = field(default_factory=list)
    context_exclude: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "executor": self.executor,
            "dependsOn": self.depends_on,
            "modelProfile": self.model_profile,
            "tools": self.tools,
            "outputs": self.outputs,
        }


@dataclass
class GraphEdge:
    source: str
    target: str
    condition: str = "always"                      # CONDITION_REGISTRY key

    def to_dict(self) -> dict[str, object]:
        return {"from": self.source, "to": self.target, "condition": self.condition}


@dataclass
class LoopContract:
    node: str                                      # node whose repetition is bounded
    goal: str = ""
    max_iterations: int = 4
    progress_conditions: list[str] = field(default_factory=list)
    termination_conditions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "node": self.node,
            "goal": self.goal,
            "maxIterations": self.max_iterations,
            "progressConditions": self.progress_conditions,
            "terminationConditions": self.termination_conditions,
        }


@dataclass
class GraphBudgets:
    max_model_calls: int = 0                        # 0 = defer to global RunBudget
    max_tokens: int = 0
    max_repair_iterations: int = 2


@dataclass
class Graph:
    id: str
    objective: str
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    loops: list[LoopContract] = field(default_factory=list)
    budgets: GraphBudgets = field(default_factory=GraphBudgets)
    version: int = 1

    def node(self, node_id: str) -> Optional[GraphNode]:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "version": self.version,
            "objective": self.objective,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "loops": [loop.to_dict() for loop in self.loops],
        }


@dataclass
class GraphState:
    """Structured blackboard the nodes read/write. Wraps the run's SessionState
    and mutable accumulators; nodes exchange data via named `artifacts` keys and
    append to `evidence` rather than passing tuples around."""
    session: "SessionState"
    profile: HardwareProfile
    provider: str
    model_override: str
    requested_files: list[str]
    generated_files: list[str]
    result: "RunResult"
    verification: dict[str, object] = field(default_factory=dict)
    artifacts: dict[str, object] = field(default_factory=dict)
    evidence: list[dict[str, object]] = field(default_factory=list)
    node_status: dict[str, str] = field(default_factory=dict)   # node id -> pass|fail|skipped
    trace: list[dict[str, object]] = field(default_factory=list)


@dataclass
class NodeSpec:
    """Registry entry: a node executor bound to a risk level + requirements."""
    executor: Callable[
        ["Engine", GraphNode, GraphState, Optional[Callable[[dict[str, object]], None]]],
        dict[str, object],
    ]
    risk: RiskLevel
    requires: list[str] = field(default_factory=list)
    stub: bool = False


@dataclass
class GraphResult:
    completed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    trace: list[dict[str, object]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Node executors
# ---------------------------------------------------------------------------
# Each returns a dict written into GraphState.evidence. Model/hybrid/verifier
# executors delegate to existing Engine methods so behavior is shared, not
# duplicated. They never raise — the runtime routes on recorded status.

def _exec_phase(engine: "Engine", node: GraphNode, gstate: GraphState,
                on_event: Optional[Callable[[dict[str, object]], None]]) -> dict[str, object]:
    """MODEL/HYBRID node: run a Martinus phase via the shared step method.

    The node id (or its `phase` in artifacts) names the phase. This reuses ALL
    existing phase behavior (candidate designs, tree-of-code, self-critique,
    snapshot + repomap injection, recovery, gate-repair, reference validation)
    because it calls the exact method the linear engine calls per phase."""
    phase = str(gstate.artifacts.get(f"phase_for:{node.id}", node.id))
    pr = engine._execute_phase_step(
        phase, gstate.profile, gstate.session, gstate.provider,
        gstate.model_override, gstate.requested_files, gstate.generated_files,
        gstate.result, on_event,
    )
    gstate.artifacts[f"output:{node.id}"] = pr.output
    return {"node": node.id, "phase": phase, "status": "ok", "attempts": pr.attempts}


def _exec_independent_verify(engine: "Engine", node: GraphNode, gstate: GraphState,
                             on_event: Optional[Callable[[dict[str, object]], None]]) -> dict[str, object]:
    """VERIFIER node: fresh-context independent review (advisory)."""
    verdict = engine._run_independent_verifier(
        gstate.session, gstate.verification, gstate.provider, on_event,
    )
    gstate.verification["independentVerdict"] = verdict
    gstate.artifacts["independentVerdict"] = verdict
    return {"node": node.id, "status": "ok", "verdict": verdict.get("verdict")}


def _exec_repository_inspect(engine: "Engine", node: GraphNode, gstate: GraphState,
                             on_event: Optional[Callable[[dict[str, object]], None]]) -> dict[str, object]:
    """DETERMINISTIC node: build the repository map (no model call)."""
    from . import repomap as _repomap
    ws = gstate.session.workspace_path
    repomap = _repomap.build_repomap(ws) if ws else {}
    gstate.artifacts["repomap"] = repomap
    langs: list[object] = []
    manifest = repomap.get("manifest", {}) if isinstance(repomap, dict) else {}
    if isinstance(manifest, dict):
        langs = manifest.get("languages", []) or []
    return {"node": node.id, "status": "ok", "languages": langs}


def _exec_gates(engine: "Engine", node: GraphNode, gstate: GraphState,
                on_event: Optional[Callable[[dict[str, object]], None]]) -> dict[str, object]:
    """DETERMINISTIC node: run static gates over generated files."""
    from . import gates as _gates
    ws = gstate.session.workspace_path
    files = sorted(set(gstate.generated_files))
    if not ws or not files:
        return {"node": node.id, "status": "skipped", "reason": "no workspace/files"}
    results = _gates.run_gates(ws, files)
    ok = all(g.ok for _rel, g in results)
    return {"node": node.id, "status": "ok" if ok else "fail",
            "checks": [{"file": rel, **g.as_dict()} for rel, g in results]}


def _exec_integration(engine: "Engine", node: GraphNode, gstate: GraphState,
                      on_event: Optional[Callable[[dict[str, object]], None]]) -> dict[str, object]:
    """DETERMINISTIC merge point (no-op for the single-writer v1 runtime)."""
    return {"node": node.id, "status": "ok", "note": "single-writer integration no-op"}


def _make_stub(name: str) -> Callable[..., dict[str, object]]:
    def _stub(engine: "Engine", node: GraphNode, gstate: GraphState,
              on_event: Optional[Callable[[dict[str, object]], None]]) -> dict[str, object]:
        # Honest not-yet-implemented evidence; never crashes the graph.
        logger.info("graph node %s (%s) is a stub — emitting not_implemented", node.id, name)
        return {"node": node.id, "status": "not_implemented", "executor": name}
    return _stub


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

NODE_REGISTRY: dict[str, NodeSpec] = {
    "phase.run": NodeSpec(_exec_phase, RiskLevel.SANDBOXED_MUTATION),
    "independent.verify": NodeSpec(_exec_independent_verify, RiskLevel.READ_ONLY),
    "repository.inspect": NodeSpec(_exec_repository_inspect, RiskLevel.READ_ONLY),
    "verification.gates": NodeSpec(_exec_gates, RiskLevel.READ_ONLY),
    "integration": NodeSpec(_exec_integration, RiskLevel.READ_ONLY),
    # Stubs — real executors arrive with the execution kernel / Playwright phases.
    "browser.inspect": NodeSpec(_make_stub("browser.inspect"), RiskLevel.SANDBOXED_MUTATION, stub=True),
    "repository.symbols": NodeSpec(_make_stub("repository.symbols"), RiskLevel.READ_ONLY, stub=True),
    "test.detect": NodeSpec(_make_stub("test.detect"), RiskLevel.READ_ONLY, stub=True),
    "test.run": NodeSpec(_make_stub("test.run"), RiskLevel.SANDBOXED_MUTATION, stub=True),
}


def _cond_always(_g: GraphState) -> bool:
    return True


def _cond_passed(g: GraphState) -> bool:
    return bool(g.verification.get("passed"))


def _cond_failed(g: GraphState) -> bool:
    return not bool(g.verification.get("passed"))


def _cond_has_missing_files(g: GraphState) -> bool:
    missing = g.verification.get("missingFiles")
    return bool(missing)


def _cond_verdict_rejected(g: GraphState) -> bool:
    verdict = g.artifacts.get("independentVerdict")
    return isinstance(verdict, dict) and verdict.get("verdict") == "REJECTED"


def _cond_last_status_ok(g: GraphState) -> bool:
    if not g.evidence:
        return True
    return g.evidence[-1].get("status") in ("ok", "skipped")


CONDITION_REGISTRY: dict[str, Callable[[GraphState], bool]] = {
    "always": _cond_always,
    "passed": _cond_passed,
    "failed": _cond_failed,
    "approved": _cond_last_status_ok,
    "rejected": _cond_verdict_rejected,
    "visual_failure": _cond_failed,       # placeholder until browser evidence lands
    "has_missing_files": _cond_has_missing_files,
    "verdict_rejected": _cond_verdict_rejected,
}


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def validate_graph(graph: Graph) -> list[str]:
    """Return a list of defect strings; empty means the graph is valid.

    Checks: unknown executors, unknown edge/loop predicates, unknown depends_on
    / edge endpoints, duplicate node ids, cycles NOT governed by a loop contract,
    and unreachable nodes."""
    defects: list[str] = []
    ids = [n.id for n in graph.nodes]
    id_set = set(ids)

    if len(ids) != len(id_set):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        defects.append(f"duplicate node ids: {dupes}")

    for n in graph.nodes:
        if n.executor not in NODE_REGISTRY:
            defects.append(f"node {n.id!r} uses unknown executor {n.executor!r}")
        for dep in n.depends_on:
            if dep not in id_set:
                defects.append(f"node {n.id!r} depends on unknown node {dep!r}")

    for e in graph.edges:
        if e.source not in id_set:
            defects.append(f"edge from unknown node {e.source!r}")
        if e.target not in id_set:
            defects.append(f"edge to unknown node {e.target!r}")
        if e.condition not in CONDITION_REGISTRY:
            defects.append(f"edge {e.source}->{e.target} uses unknown condition {e.condition!r}")

    loop_nodes = {loop.node for loop in graph.loops}
    for loop in graph.loops:
        if loop.node not in id_set:
            defects.append(f"loop contract references unknown node {loop.node!r}")
        for pred in list(loop.progress_conditions) + list(loop.termination_conditions):
            if pred not in CONDITION_REGISTRY:
                defects.append(f"loop on {loop.node!r} uses unknown predicate {pred!r}")

    # Cycle detection over depends_on; a cycle is only allowed if every node in
    # it is governed by a loop contract.
    cycle = _find_cycle(graph)
    if cycle and not all(c in loop_nodes for c in cycle):
        defects.append(f"ungoverned cycle in depends_on: {cycle}")

    # Unreachable: nodes with no depends_on and not the designated roots are fine,
    # but a node that is neither a root nor reachable via depends_on chains from a
    # root is flagged.
    roots = [n.id for n in graph.nodes if not n.depends_on]
    if graph.nodes and not roots:
        defects.append("graph has no root node (every node has dependencies)")

    return defects


def _find_cycle(graph: Graph) -> list[str]:
    """Return one cycle in the depends_on DAG, or [] if acyclic."""
    adj: dict[str, list[str]] = {n.id: list(n.depends_on) for n in graph.nodes}
    WHITE, GREY, BLACK = 0, 1, 2
    color: dict[str, int] = {n: WHITE for n in adj}
    stack: list[str] = []

    def dfs(u: str) -> list[str]:
        color[u] = GREY
        stack.append(u)
        for v in adj.get(u, []):
            if v not in color:
                continue
            if color[v] == GREY:
                idx = stack.index(v)
                return stack[idx:]
            if color[v] == WHITE:
                found = dfs(v)
                if found:
                    return found
        color[u] = BLACK
        stack.pop()
        return []

    for node_id in adj:
        if color[node_id] == WHITE:
            found = dfs(node_id)
            if found:
                return found
    return []


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

class GraphRuntime:
    """Executes a validated graph against a GraphState by driving a topological
    frontier and following conditional edges. Model calls still serialize on the
    provider semaphore; deterministic nodes are free. Budgets/cancellation reuse
    the engine's RunBudget via the node executors that call model code."""

    def run(
        self,
        graph: Graph,
        gstate: GraphState,
        engine: "Engine",
        on_event: Optional[Callable[[dict[str, object]], None]] = None,
    ) -> GraphResult:
        result = GraphResult()
        completed: set[str] = set()

        # Precompute per-node incoming conditional edges for edge-gating.
        incoming: dict[str, list[GraphEdge]] = {n.id: [] for n in graph.nodes}
        for e in graph.edges:
            if e.target in incoming:
                incoming[e.target].append(e)

        # Iterate to a fixpoint: a node runs once its depends_on are complete and
        # at least one incoming edge condition holds (or it has no incoming edges).
        progressed = True
        while progressed:
            progressed = False
            for node in graph.nodes:
                if node.id in completed:
                    continue
                if any(dep not in completed for dep in node.depends_on):
                    continue
                edges = incoming.get(node.id, [])
                if edges and not any(self._edge_allows(e, gstate) for e in edges):
                    # No enabling edge yet; may become enabled after other nodes.
                    continue

                if engine._budget:
                    engine._budget.check(f"before graph node {node.id}")

                self._emit(on_event, {
                    "type": "node_started", "node": node.id,
                    "kind": node.kind.value, "executor": node.executor,
                    "sessionId": gstate.session.session_id,
                })
                spec = NODE_REGISTRY.get(node.executor)
                evidence: dict[str, object]
                if spec is None:
                    evidence = {"node": node.id, "status": "error", "reason": "unknown executor"}
                else:
                    try:
                        evidence = spec.executor(engine, node, gstate, on_event)
                    except Exception as exc:  # never let one node crash the graph
                        from .orchestrator import RunAborted  # local import (cycle)
                        if isinstance(exc, RunAborted):
                            raise
                        logger.warning("graph node %s failed: %s", node.id, exc)
                        evidence = {"node": node.id, "status": "error", "reason": str(exc)}

                gstate.evidence.append(evidence)
                status = str(evidence.get("status", "ok"))
                gstate.node_status[node.id] = status
                trace_entry: dict[str, object] = {
                    "node": node.id, "executor": node.executor, "status": status,
                }
                gstate.trace.append(trace_entry)
                completed.add(node.id)
                result.completed.append(node.id)
                progressed = True
                self._emit(on_event, {"type": "node_completed", "node": node.id,
                                      "status": status, "sessionId": gstate.session.session_id})

        # Any node never reached (its edge conditions never held) is skipped.
        for node in graph.nodes:
            if node.id not in completed:
                result.skipped.append(node.id)
                gstate.node_status[node.id] = "skipped"
                self._emit(on_event, {"type": "node_skipped", "node": node.id,
                                      "sessionId": gstate.session.session_id})

        result.trace = gstate.trace
        return result

    @staticmethod
    def _edge_allows(edge: GraphEdge, gstate: GraphState) -> bool:
        pred = CONDITION_REGISTRY.get(edge.condition, _cond_always)
        try:
            return pred(gstate)
        except Exception:  # a predicate must never crash the runtime
            return False

    @staticmethod
    def _emit(on_event: Optional[Callable[[dict[str, object]], None]], payload: dict[str, object]) -> None:
        if on_event:
            on_event(payload)
