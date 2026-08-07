"""
agent/permissions.py — Tiered action risk model (LOCALFORGE §7).

Classifies every consequential action the orchestrator can take into one of
four risk levels and returns an allow / deny / (record-and-deny) decision.

Design for a HEADLESS server (this matters):
`Engine.execute()` runs in a thread-pool executor with only a fire-and-forget
`on_event` callback and a `cancel_check` — there is NO request/response channel
back into a running phase. So "requires explicit human approval" (L3) cannot be
an interactive prompt mid-run. The only truthful behavior is:

- L0 (read-only)            → always allow.
- L1 (sandboxed mutation)   → allow (this is the agent's normal operation).
- L2 (sensitive local)      → policy-configured via env; the one real L2 action
                              is applying sandbox results back to the user's
                              workspace. Default allow, but the caller still
                              enforces hard guards (workspace-root confinement,
                              newer-file conflict) that OVERRIDE an allow.
- L3 (external/destructive) → deny by default, always. The decision is surfaced
                              as an event/record; we never fake approval.

Env configuration:
- MARTINUS_PERMISSION_L2 = allow | deny   (default: allow)
- MARTINUS_PERMISSION_L3 = allow | deny   (default: deny; `allow` only honored
                            if MARTINUS_PERMISSION_ALLOW_L3_OVERRIDE=1, so it
                            can't be flipped on by accident)

Zero dependencies; standard library only. Pure and unit-testable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class RiskLevel(IntEnum):
    """LOCALFORGE §7 risk tiers."""
    READ_ONLY = 0            # read files, search, inspect git/config
    SANDBOXED_MUTATION = 1   # edit sandbox files, install sandbox deps, run tests
    SENSITIVE_LOCAL = 2      # modify files outside sandbox, read secrets, bind public net
    EXTERNAL_DESTRUCTIVE = 3  # push, open PRs, deploy, delete remote, send messages


# Canonical action → risk level. Actions the orchestrator actually performs are
# listed explicitly; unknown actions default to the SENSITIVE tier (fail-safe).
_ACTION_RISK: dict[str, RiskLevel] = {
    # L0
    "read_file": RiskLevel.READ_ONLY,
    "search_code": RiskLevel.READ_ONLY,
    "inspect_git": RiskLevel.READ_ONLY,
    "build_repomap": RiskLevel.READ_ONLY,
    # L1
    "write_sandbox_file": RiskLevel.SANDBOXED_MUTATION,
    "run_workspace_tests": RiskLevel.SANDBOXED_MUTATION,
    "create_sandbox": RiskLevel.SANDBOXED_MUTATION,
    "cleanup_sandbox": RiskLevel.SANDBOXED_MUTATION,
    # L2
    "apply_back_to_workspace": RiskLevel.SENSITIVE_LOCAL,
    "write_outside_sandbox": RiskLevel.SENSITIVE_LOCAL,
    "read_secret": RiskLevel.SENSITIVE_LOCAL,
    "bind_public_interface": RiskLevel.SENSITIVE_LOCAL,
    # L3
    "git_push": RiskLevel.EXTERNAL_DESTRUCTIVE,
    "open_pull_request": RiskLevel.EXTERNAL_DESTRUCTIVE,
    "deploy": RiskLevel.EXTERNAL_DESTRUCTIVE,
    "delete_remote": RiskLevel.EXTERNAL_DESTRUCTIVE,
    "send_message": RiskLevel.EXTERNAL_DESTRUCTIVE,
}


@dataclass
class Decision:
    """Outcome of a permission check."""
    allowed: bool
    level: RiskLevel
    action: str
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "allowed": self.allowed,
            "level": int(self.level),
            "levelName": self.level.name,
            "action": self.action,
            "reason": self.reason,
        }


def _env_flag(name: str, default: bool) -> bool:
    val = os.getenv(name, "").strip().lower()
    if val in ("1", "true", "yes", "allow", "on"):
        return True
    if val in ("0", "false", "no", "deny", "off"):
        return False
    return default


@dataclass
class PermissionPolicy:
    """Configurable policy. Construct from env with `from_env()`."""
    allow_l2: bool = True
    allow_l3: bool = False

    @classmethod
    def from_env(cls) -> "PermissionPolicy":
        allow_l2 = _env_flag("MARTINUS_PERMISSION_L2", True)
        # L3 is deny unless BOTH the tier is set to allow AND the explicit
        # override guard is set — defence against an accidental blanket allow.
        allow_l3 = (
            _env_flag("MARTINUS_PERMISSION_L3", False)
            and _env_flag("MARTINUS_PERMISSION_ALLOW_L3_OVERRIDE", False)
        )
        return cls(allow_l2=allow_l2, allow_l3=allow_l3)

    @staticmethod
    def classify(action: str) -> RiskLevel:
        """Map an action name to its risk level. Unknown → SENSITIVE (fail-safe)."""
        return _ACTION_RISK.get(action, RiskLevel.SENSITIVE_LOCAL)

    def decide(self, action: str, detail: str = "") -> Decision:
        """Return the allow/deny decision for an action under this policy.

        Note: an `allowed=True` here is necessary but not sufficient for L2 —
        the caller (e.g. sandbox apply-back) still enforces hard guards
        (workspace-root confinement, newer-file conflict) that can override it.
        """
        level = self.classify(action)
        suffix = f" ({detail})" if detail else ""
        if level <= RiskLevel.SANDBOXED_MUTATION:
            return Decision(True, level, action, f"L{int(level)} auto-allowed{suffix}")
        if level == RiskLevel.SENSITIVE_LOCAL:
            if self.allow_l2:
                return Decision(True, level, action,
                                f"L2 allowed by policy; caller must enforce hard guards{suffix}")
            return Decision(False, level, action,
                            f"L2 denied by policy (MARTINUS_PERMISSION_L2=deny){suffix}")
        # L3
        if self.allow_l3:
            return Decision(True, level, action, f"L3 explicitly enabled via override{suffix}")
        return Decision(False, level, action,
                        f"L3 denied: no interactive approval channel in a headless run{suffix}")


# Module-level default so callers can share one policy without re-reading env.
_default_policy: Optional[PermissionPolicy] = None


def default_policy() -> PermissionPolicy:
    """Process-wide default policy, built once from the environment."""
    global _default_policy
    if _default_policy is None:
        _default_policy = PermissionPolicy.from_env()
    return _default_policy
