"""
agent/graph_patterns.py — Graph compiler + reusable pattern library.

Turns a request into a validated execution Graph (agent/graph.py). A deterministic
classifier picks one of five patterns; each pattern is a fixed, validated graph
built from approved registry primitives. This is the "constrained graph compiler"
of the roadmap: the shapes are curated, not invented per request — safe and
predictable, while still task-specific.

Pattern coverage / honesty:
- `feature` — FULLY functional: maps 1:1 onto Martinus's canonical phase order,
  so running the graph engine on an ordinary task reproduces the linear pipeline.
- `bug_repair`, `ui_redesign`, `refactor`, `unknown_repo` — real, validated IR
  graphs, but several of their specialized nodes (browser.inspect, test.run,
  repository.symbols, test.detect) are STUB executors until the execution-kernel
  and Playwright roadmap phases land. They emit structured not_implemented
  evidence rather than pretending to run.

Model-proposed composition (a model selecting/wiring approved primitives, then
validating) is a documented extension point; v1 classification is deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass

from .graph import (
    Graph,
    GraphBudgets,
    GraphEdge,
    GraphNode,
    NodeKind,
    validate_graph,
)
from .profiles import HardwareProfile

# Reuse the orchestrator's existing hint sets so classification stays consistent
# with how the pipeline already reasons about task type.
from .orchestrator import _BROWSER_APP_HINTS, _CLI_HINTS

# Task-class keyword hints (deterministic, cheap — no model call).
_BUG_HINTS = ("bug", "fix", "broken", "error", "crash", "failing", "regression", "not working")
_REFACTOR_HINTS = ("refactor", "clean up", "cleanup", "restructure", "rename", "extract", "simplify")
_UI_HINTS = ("redesign", "restyle", "layout", "css", "styling", "dark mode", "responsive", "ui ")

# Signals that a task is small and greenfield — skip Architecture + Clarification.
# Kapoor et al. (2407.01502): complex scaffolding costs 55x more for equal/worse
# quality on simple tasks. A 2-phase path (Discovery → Implementation → gates)
# closes the latency and cost gap with single-turn tools on these tasks.
_SIMPLE_HINTS = (
    "create a", "write a", "write the", "make a", "add a", "generate a",
    "implement a", "build a simple", "hello world", "scaffold a",
)
_SIMPLE_EXCLUSIONS = (
    "system", "service", "architecture", "refactor", "migrate", "integrate",
    "authentication", "authorization", "database", "distributed",
    "feature", "module", "login", "signup", "auth", "api", "endpoint",
    "page", "component", "class", "framework",
)


@dataclass
class CompileInput:
    task: str
    goals: list[str]
    profile: HardwareProfile


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify_task(task: str, goals: list[str]) -> str:
    """Return one of: simple | feature | bug_repair | ui_redesign | refactor | unknown_repo.

    Deterministic keyword heuristic. Order matters: bug/refactor/ui are more
    specific than the generic feature default. `simple` short-circuits first for
    unambiguous small greenfield tasks (Kapoor et al. 2407.01502 cost finding)."""
    text = (task + " " + " ".join(goals)).lower()

    def has(hints: tuple[str, ...]) -> bool:
        return any(h in text for h in hints)

    if has(_BUG_HINTS):
        return "bug_repair"
    if has(_REFACTOR_HINTS):
        return "refactor"
    # A UI/redesign task that is browser-flavored.
    if has(_UI_HINTS) and has(_BROWSER_APP_HINTS):
        return "ui_redesign"
    # "understand/explore this repo" style requests.
    if any(h in text for h in ("understand", "explore", "map the", "what does this")) and not has(_CLI_HINTS):
        return "unknown_repo"
    # Small greenfield task: skip the full pipeline overhead.
    if has(_SIMPLE_HINTS) and not has(_SIMPLE_EXCLUSIONS):
        return "simple"
    return "feature"


# ---------------------------------------------------------------------------
# Pattern builders
# ---------------------------------------------------------------------------

def _phase_node(node_id: str, kind: NodeKind = NodeKind.MODEL,
                depends_on: list[str] | None = None, role: str = "") -> GraphNode:
    return GraphNode(
        id=node_id, kind=kind, executor="phase.run",
        depends_on=depends_on or [], model_profile=role,
    )


def _budgets(profile: HardwareProfile) -> GraphBudgets:
    return GraphBudgets(max_repair_iterations=profile.repair_iterations)


def _linear_edges(node_ids: list[str]) -> list[GraphEdge]:
    return [GraphEdge(node_ids[i], node_ids[i + 1]) for i in range(len(node_ids) - 1)]


def _simple_graph(ci: CompileInput) -> Graph:
    """Fast-path for small, greenfield tasks: skip Clarification and Architecture.

    Graph: repository.inspect → Discovery → Implementation → verification.gates
    → independent.verify (if profile enables it).

    Rationale: Kapoor et al. (arXiv:2407.01502) show a simple retry strategy
    beats complex agent scaffolding at 1/55th the cost on HumanEval. For tasks
    that match _SIMPLE_HINTS, the full 6-phase pipeline is pure overhead."""
    nodes: list[GraphNode] = [
        GraphNode("repository.inspect", NodeKind.DETERMINISTIC, "repository.inspect"),
        _phase_node("Discovery", depends_on=["repository.inspect"], role="planner"),
        _phase_node("Implementation", depends_on=["Discovery"], role="implementer"),
        GraphNode("verification.gates", NodeKind.DETERMINISTIC, "verification.gates",
                  depends_on=["Implementation"]),
        GraphNode("independent.verify", NodeKind.VERIFIER, "independent.verify",
                  depends_on=["verification.gates"], model_profile="verifier"),
    ]
    ordered = [n.id for n in nodes]
    return Graph(
        id="simple", objective=ci.task, nodes=nodes,
        edges=_linear_edges(ordered), budgets=_budgets(ci.profile),
    )


def _feature_graph(ci: CompileInput) -> Graph:
    """The canonical Martinus pipeline as a graph. Node ids ARE the phase names,
    so `phase.run` dispatches each through the existing engine, preserving order
    and behavior. A deterministic repo-inspect node precedes Discovery, and the
    independent verifier closes the graph."""
    phases = list(ci.profile.phases)
    nodes: list[GraphNode] = [
        GraphNode("repository.inspect", NodeKind.DETERMINISTIC, "repository.inspect"),
    ]
    prev = "repository.inspect"
    ordered = ["repository.inspect"]
    for ph in phases:
        role = _role_for(ph)
        nodes.append(_phase_node(ph, NodeKind.MODEL, depends_on=[prev], role=role))
        ordered.append(ph)
        prev = ph
    # Independent verifier as a fresh-context closing node.
    nodes.append(GraphNode("independent.verify", NodeKind.VERIFIER, "independent.verify",
                           depends_on=[prev], model_profile="verifier"))
    ordered.append("independent.verify")
    edges = _linear_edges(ordered)
    return Graph(
        id="feature", objective=ci.task, nodes=nodes, edges=edges,
        budgets=_budgets(ci.profile),
    )


def _role_for(phase: str) -> str:
    if phase in ("Discovery", "Clarification"):
        return "planner"
    if phase == "Review":
        return "reviewer"
    return "implementer"


def _bug_repair_graph(ci: CompileInput) -> Graph:
    """Reproduce → collect evidence → diagnose → repair → re-verify. Diagnosis and
    repair reuse phase nodes; reproduction/evidence use gates + stub test nodes."""
    nodes = [
        GraphNode("repository.inspect", NodeKind.DETERMINISTIC, "repository.inspect"),
        GraphNode("test.detect", NodeKind.DETERMINISTIC, "test.detect", depends_on=["repository.inspect"]),
        _phase_node("Discovery", depends_on=["test.detect"], role="planner"),
        _phase_node("Implementation", depends_on=["Discovery"], role="implementer"),
        GraphNode("verification.gates", NodeKind.DETERMINISTIC, "verification.gates",
                  depends_on=["Implementation"]),
        _phase_node("Verification", depends_on=["verification.gates"], role="implementer"),
        GraphNode("independent.verify", NodeKind.VERIFIER, "independent.verify",
                  depends_on=["Verification"], model_profile="verifier"),
    ]
    ordered = [n.id for n in nodes]
    return Graph(id="bug_repair", objective=ci.task, nodes=nodes,
                 edges=_linear_edges(ordered), budgets=_budgets(ci.profile))


def _ui_redesign_graph(ci: CompileInput) -> Graph:
    """Inspect → design → implement → browser-inspect → visual-repair. The browser
    node is a stub until Playwright lands (roadmap Phase 3)."""
    nodes = [
        GraphNode("repository.inspect", NodeKind.DETERMINISTIC, "repository.inspect"),
        _phase_node("Architecture", depends_on=["repository.inspect"], role="planner"),
        _phase_node("Implementation", depends_on=["Architecture"], role="implementer"),
        GraphNode("browser.inspect", NodeKind.HYBRID, "browser.inspect",
                  depends_on=["Implementation"]),
        _phase_node("Review", depends_on=["browser.inspect"], role="reviewer"),
        GraphNode("independent.verify", NodeKind.VERIFIER, "independent.verify",
                  depends_on=["Review"], model_profile="verifier"),
    ]
    ordered = [n.id for n in nodes]
    return Graph(id="ui_redesign", objective=ci.task, nodes=nodes,
                 edges=_linear_edges(ordered), budgets=_budgets(ci.profile))


def _refactor_graph(ci: CompileInput) -> Graph:
    """Map deps → baseline (stub tests) → incremental change → gates → regression
    review."""
    nodes = [
        GraphNode("repository.inspect", NodeKind.DETERMINISTIC, "repository.inspect"),
        GraphNode("test.run", NodeKind.HYBRID, "test.run", depends_on=["repository.inspect"]),
        _phase_node("Architecture", depends_on=["test.run"], role="planner"),
        _phase_node("Implementation", depends_on=["Architecture"], role="implementer"),
        GraphNode("verification.gates", NodeKind.DETERMINISTIC, "verification.gates",
                  depends_on=["Implementation"]),
        GraphNode("independent.verify", NodeKind.VERIFIER, "independent.verify",
                  depends_on=["verification.gates"], model_profile="verifier"),
    ]
    ordered = [n.id for n in nodes]
    return Graph(id="refactor", objective=ci.task, nodes=nodes,
                 edges=_linear_edges(ordered), budgets=_budgets(ci.profile))


def _unknown_repo_graph(ci: CompileInput) -> Graph:
    """Fingerprint → symbols → summarize. Read-only exploration; ends with a
    Discovery summary phase."""
    nodes = [
        GraphNode("repository.inspect", NodeKind.DETERMINISTIC, "repository.inspect"),
        GraphNode("repository.symbols", NodeKind.DETERMINISTIC, "repository.symbols",
                  depends_on=["repository.inspect"]),
        _phase_node("Discovery", depends_on=["repository.symbols"], role="planner"),
    ]
    ordered = [n.id for n in nodes]
    return Graph(id="unknown_repo", objective=ci.task, nodes=nodes,
                 edges=_linear_edges(ordered), budgets=_budgets(ci.profile))


PATTERNS = {
    "simple": _simple_graph,
    "feature": _feature_graph,
    "bug_repair": _bug_repair_graph,
    "ui_redesign": _ui_redesign_graph,
    "refactor": _refactor_graph,
    "unknown_repo": _unknown_repo_graph,
}


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------

class GraphCompileError(ValueError):
    """Raised when a compiled graph fails validation (a programming error in a
    pattern builder, not user input)."""


def compile_graph(task: str, goals: list[str], profile: HardwareProfile) -> Graph:
    """Classify the task, build the matching pattern graph, validate it, and
    return it. Falls back to the `feature` pattern if classification yields an
    unknown key (defensive)."""
    pattern = classify_task(task, goals)
    builder = PATTERNS.get(pattern, _feature_graph)
    graph = builder(CompileInput(task=task, goals=goals, profile=profile))
    defects = validate_graph(graph)
    if defects:
        # A pattern builder produced an invalid graph — fail loudly; this is our
        # bug, never silently run a broken graph.
        raise GraphCompileError(f"pattern {graph.id!r} produced an invalid graph: {defects}")
    return graph
