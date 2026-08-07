"""Tests for the graph compiler + pattern library."""
from __future__ import annotations

import pytest

from agent.graph import validate_graph
from agent.graph_patterns import PATTERNS, CompileInput, classify_task, compile_graph
from agent.profiles import resolve_profile


@pytest.mark.parametrize("task,expected", [
    ("create a login feature", "feature"),
    ("add CSV export to the reports page", "feature"),
    ("fix the crash when logging in", "bug_repair"),
    ("the build is broken, error in parser", "bug_repair"),
    ("refactor the auth module for clarity", "refactor"),
    ("rename the session helper everywhere", "refactor"),
    ("redesign the css layout of the web app", "ui_redesign"),
    ("understand this repository structure", "unknown_repo"),
])
def test_classifier_routes(task: str, expected: str) -> None:
    assert classify_task(task, []) == expected


def test_all_patterns_compile_and_validate() -> None:
    profile = resolve_profile("high")
    for name, builder in PATTERNS.items():
        graph = builder(CompileInput(task="do a thing", goals=[], profile=profile))
        assert graph.id == name
        assert validate_graph(graph) == [], f"pattern {name} invalid"
        assert graph.nodes, f"pattern {name} has no nodes"


def test_feature_graph_contains_profile_phases_in_order() -> None:
    profile = resolve_profile("medium")
    graph = compile_graph("build a feature", [], profile)
    assert graph.id == "feature"
    node_ids = [n.id for n in graph.nodes]
    # repository.inspect first, phases in order, verifier last.
    assert node_ids[0] == "repository.inspect"
    assert node_ids[-1] == "independent.verify"
    for phase in profile.phases:
        assert phase in node_ids
    # Phase order preserved between the two book-end nodes.
    phase_positions = [node_ids.index(p) for p in profile.phases]
    assert phase_positions == sorted(phase_positions)


def test_compile_graph_defaults_to_feature() -> None:
    profile = resolve_profile("desktop")
    graph = compile_graph("some neutral request", [], profile)
    assert graph.id == "feature"


def test_bug_pattern_selected_and_valid() -> None:
    profile = resolve_profile("medium")
    graph = compile_graph("fix the failing test", [], profile)
    assert graph.id == "bug_repair"
    assert validate_graph(graph) == []
