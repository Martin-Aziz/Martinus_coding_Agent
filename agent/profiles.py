"""
agent/profiles.py — Hardware profile definitions and role-based model routing.

Three profiles gate which phases and techniques the pipeline runs,
scaling to the user's available RAM and GPU tier.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HardwareProfile:
    name: str
    phases: list[str]
    step_back: bool = False
    candidate_designs: bool = False
    tree_of_code: bool = False
    self_critique: bool = False
    reflection: bool = False
    hierarchical_summary: bool = False
    memory_retrieval: bool = False
    role_routing: bool = False
    max_retries: int = 3
    # Algorithm 4.12 — Executable Verification with Automated Repair.
    # When enabled, every file the pipeline writes is passed through a
    # language-appropriate static gate (py_compile / node --check / html.parser /
    # css brace-balance / json.loads). Failures trigger a Repair phase that
    # gets the exact compiler error text and the current source, up to
    # `repair_iterations` times. Zero cost when files pass on the first try.
    executable_verification: bool = True
    repair_iterations: int = 2
    # LOCALFORGE §6.2 — inject a deterministic Layer 1-3 repository map into the
    # Discovery / Architecture prompts so the agent reasons about the existing
    # codebase instead of only files it generated this run. Costs one cheap
    # deterministic walk; no model call. On for profiles that run Architecture.
    repo_map: bool = False
    # LOCALFORGE Rule 2.2 / Node H — run the whole pipeline in an isolated
    # sandbox (git worktree or temp-dir copy) and apply results back only on a
    # verified pass. Default OFF: preserves the in-place-write contract.
    sandbox_isolation: bool = False
    # LOCALFORGE Rule 2.5 / §9 — a final independent verifier pass with fresh
    # context (no implementer reasoning). Advisory: can downgrade a pass to
    # human_review but never upgrades a deterministic failure.
    independent_verifier: bool = False
    # Graph-native execution — compile the request into a typed, validated graph
    # and run it via GraphRuntime instead of the fixed linear phase loop. The
    # compiled "feature" graph reproduces the linear order, so behavior is
    # preserved. Default OFF; opt in via MARTINUS_GRAPH_ENGINE=1.
    graph_engine: bool = False
    # Behavioral test gate — when True, run pytest (or detected test runner) in
    # the workspace after generation. A failing test suite overrides a static-
    # gate pass: no "success" claim without runtime evidence.
    # Research basis: Anthropic "Building Effective Agents" §B — automated tests
    # are the bias-free quality signal; Kapoor et al. show static-only gates
    # accept broken outputs that runtime execution catches.
    # Default ON for medium/high (they run a Verification phase anyway);
    # OFF for desktop/low to keep wall-clock short on resource-constrained hosts.
    behavioral_tests: bool = False


PROFILES: dict[str, HardwareProfile] = {
    "desktop": HardwareProfile(
        name="desktop",
        phases=["Discovery", "Implementation"],
        step_back=False,
        candidate_designs=False,
        tree_of_code=False,
        self_critique=False,
        reflection=False,
        hierarchical_summary=False,
        memory_retrieval=False,
        role_routing=False,
        # Bumped from 1 → 2: desktop skips Verification so we get exactly one
        # chance to produce the requested files. A single retry is enough to
        # recover from a formatting slip without doubling wall-clock on happy paths.
        max_retries=2,
        # Desktop keeps the Repair loop short — one repair pass is enough for
        # a formatting slip and keeps wall-clock predictable on 7B local models.
        executable_verification=True,
        repair_iterations=1,
    ),
    "low": HardwareProfile(
        name="low",
        phases=["Discovery", "Implementation", "Verification"],
        step_back=False,
        candidate_designs=False,
        tree_of_code=False,
        self_critique=False,
        reflection=False,
        hierarchical_summary=False,
        memory_retrieval=False,
        role_routing=False,
        max_retries=3,
        executable_verification=True,
        repair_iterations=1,
    ),
    "medium": HardwareProfile(
        name="medium",
        phases=["Discovery", "Clarification", "Architecture", "Implementation", "Review", "Verification"],
        step_back=True,
        candidate_designs=False,
        tree_of_code=False,
        self_critique=False,
        reflection=True,
        hierarchical_summary=True,
        memory_retrieval=True,
        role_routing=True,
        max_retries=3,
        executable_verification=True,
        repair_iterations=2,
        repo_map=True,
        behavioral_tests=True,
    ),
    "high": HardwareProfile(
        name="high",
        phases=["Discovery", "Clarification", "Architecture", "Implementation", "Review", "Verification"],
        step_back=True,
        candidate_designs=True,
        tree_of_code=True,
        self_critique=True,
        reflection=True,
        hierarchical_summary=True,
        memory_retrieval=True,
        role_routing=True,
        max_retries=3,
        executable_verification=True,
        repair_iterations=2,
        repo_map=True,
        independent_verifier=True,
        behavioral_tests=True,
    ),
}


def resolve_profile(name: str, fallback: str = "medium") -> HardwareProfile:
    p = PROFILES.get(name.strip().lower())
    if p:
        return p
    p = PROFILES.get(fallback.strip().lower())
    return p or PROFILES["medium"]


# ---------------------------------------------------------------------------
# Role-based model routing
# ---------------------------------------------------------------------------

def _phase_role(phase: str) -> str:
    if phase in ("Discovery", "Clarification"):
        return "planner"
    if phase == "Review":
        return "reviewer"
    # The independent verifier (LOCALFORGE §9) is not a pipeline phase, but it
    # routes like one: a dedicated `verifier_model` in the routing config lets
    # it run on a stronger / different model than the implementer.
    if phase == "IndependentVerify":
        return "verifier"
    return "implementer"


@dataclass
class RoleRouting:
    """Maps (provider, role) to a model name override."""
    providers: dict[str, dict[str, str]] = field(default_factory=dict)

    def model_for(self, provider: str, phase: str) -> str:
        role = _phase_role(phase)
        provider_map = self.providers.get(provider.strip().lower(), {})
        return provider_map.get(role, "").strip()


def load_role_routing(path: str) -> Optional[RoleRouting]:
    """Parse the model-routing YAML (hand-rolled, no yaml dependency)."""
    path = path.strip()
    if not path or not os.path.isfile(path):
        return None

    routing = RoleRouting()
    in_providers = False
    current_provider = ""

    try:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.rstrip()
                if not line or line.lstrip().startswith("#"):
                    continue
                indent = len(raw) - len(raw.lstrip(" \t"))
                trimmed = raw.lstrip(" \t").rstrip()

                if indent == 0:
                    in_providers = trimmed.startswith("providers:")
                    current_provider = ""
                    continue
                if not in_providers:
                    continue
                if indent <= 2:
                    if ":" in trimmed:
                        current_provider = trimmed.split(":")[0].strip().lower()
                        if current_provider:
                            routing.providers.setdefault(current_provider, {})
                else:
                    if not current_provider or ":" not in trimmed:
                        continue
                    key, _, value = trimmed.partition(":")
                    key = key.strip().lower()
                    value = value.strip().strip("\"'")
                    if key.endswith("_model") and value:
                        role = key[: -len("_model")]
                        routing.providers[current_provider][role] = value
    except OSError:
        return None

    return routing
