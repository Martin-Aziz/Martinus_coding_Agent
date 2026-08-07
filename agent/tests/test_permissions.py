from __future__ import annotations

import pytest

from agent.permissions import Decision, PermissionPolicy, RiskLevel, default_policy


def test_classify_known_actions() -> None:
    assert PermissionPolicy.classify("read_file") == RiskLevel.READ_ONLY
    assert PermissionPolicy.classify("write_sandbox_file") == RiskLevel.SANDBOXED_MUTATION
    assert PermissionPolicy.classify("apply_back_to_workspace") == RiskLevel.SENSITIVE_LOCAL
    assert PermissionPolicy.classify("git_push") == RiskLevel.EXTERNAL_DESTRUCTIVE


def test_unknown_action_is_fail_safe_sensitive() -> None:
    assert PermissionPolicy.classify("something_new") == RiskLevel.SENSITIVE_LOCAL


def test_l0_l1_always_allowed() -> None:
    policy = PermissionPolicy(allow_l2=False, allow_l3=False)
    assert policy.decide("read_file").allowed is True
    assert policy.decide("run_workspace_tests").allowed is True


def test_l2_configurable() -> None:
    allow = PermissionPolicy(allow_l2=True)
    deny = PermissionPolicy(allow_l2=False)
    assert allow.decide("apply_back_to_workspace").allowed is True
    assert deny.decide("apply_back_to_workspace").allowed is False


def test_l3_denied_by_default() -> None:
    policy = PermissionPolicy(allow_l2=True, allow_l3=False)
    d = policy.decide("git_push")
    assert d.allowed is False
    assert d.level == RiskLevel.EXTERNAL_DESTRUCTIVE
    assert "headless" in d.reason


def test_l3_allowed_only_with_explicit_override() -> None:
    policy = PermissionPolicy(allow_l2=True, allow_l3=True)
    assert policy.decide("deploy").allowed is True


def test_from_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MARTINUS_PERMISSION_L2", raising=False)
    monkeypatch.delenv("MARTINUS_PERMISSION_L3", raising=False)
    monkeypatch.delenv("MARTINUS_PERMISSION_ALLOW_L3_OVERRIDE", raising=False)
    policy = PermissionPolicy.from_env()
    assert policy.allow_l2 is True
    assert policy.allow_l3 is False


def test_from_env_l2_deny(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARTINUS_PERMISSION_L2", "deny")
    policy = PermissionPolicy.from_env()
    assert policy.allow_l2 is False
    assert policy.decide("apply_back_to_workspace").allowed is False


def test_from_env_l3_needs_both_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    # L3=allow alone is NOT enough — the override guard is required.
    monkeypatch.setenv("MARTINUS_PERMISSION_L3", "allow")
    monkeypatch.delenv("MARTINUS_PERMISSION_ALLOW_L3_OVERRIDE", raising=False)
    assert PermissionPolicy.from_env().allow_l3 is False

    monkeypatch.setenv("MARTINUS_PERMISSION_ALLOW_L3_OVERRIDE", "1")
    assert PermissionPolicy.from_env().allow_l3 is True


def test_decision_as_dict() -> None:
    d = Decision(True, RiskLevel.SANDBOXED_MUTATION, "write_sandbox_file", "ok")
    payload = d.as_dict()
    assert payload["allowed"] is True
    assert payload["level"] == 1
    assert payload["levelName"] == "SANDBOXED_MUTATION"


def test_default_policy_is_singleton() -> None:
    assert default_policy() is default_policy()
