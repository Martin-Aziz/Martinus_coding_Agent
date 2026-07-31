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
