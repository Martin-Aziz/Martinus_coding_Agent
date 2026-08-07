"""Tests for the disk-backed semantic response cache (Improvement 7)."""
from __future__ import annotations

import importlib

from agent import response_cache as rc
from agent.providers import ChatResponse


def _reload_with_env(monkeypatch, tmp_path, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("MARTINUS_CACHE_DIR", str(tmp_path))
    importlib.reload(rc)
    return rc


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MARTINUS_RESPONSE_CACHE", raising=False)
    importlib.reload(rc)
    assert rc.is_enabled() is False


def test_cacheable_only_low_temperature(monkeypatch, tmp_path):
    _reload_with_env(monkeypatch, tmp_path, MARTINUS_RESPONSE_CACHE="1")
    assert rc.cacheable(0.1) is True
    assert rc.cacheable(0.9) is False


def test_key_is_stable_and_model_sensitive(monkeypatch, tmp_path):
    _reload_with_env(monkeypatch, tmp_path, MARTINUS_RESPONSE_CACHE="1")
    msgs = [{"role": "user", "content": "hello"}]
    k1 = rc.make_key("ollama", "m1", 0.1, msgs)
    k2 = rc.make_key("ollama", "m1", 0.1, msgs)
    k3 = rc.make_key("ollama", "m2", 0.1, msgs)  # different model
    assert k1 == k2
    assert k1 != k3


def test_key_normalizes_temp_workspace_paths(monkeypatch, tmp_path):
    """Same task in two different temp workspaces must hash to the SAME key.

    Regression guard: workspace paths leak into prompts (repo-map / snapshot
    headers). Without normalization the cache never hits across runs."""
    _reload_with_env(monkeypatch, tmp_path, MARTINUS_RESPONSE_CACHE="1")
    prompt_a = "Task: square a number\nWorkspace: /var/folders/92/abc/T/e_A_r0uf9c3q\nProvider: x"
    prompt_b = "Task: square a number\nWorkspace: /var/folders/92/abc/T/e_B_ycut_ui4\nProvider: x"
    ka = rc.make_key("claude", "m", 0.1, [{"role": "user", "content": prompt_a}])
    kb = rc.make_key("claude", "m", 0.1, [{"role": "user", "content": prompt_b}])
    assert ka == kb


def test_key_still_distinguishes_real_task_differences(monkeypatch, tmp_path):
    """Normalization must not collapse genuinely different tasks."""
    _reload_with_env(monkeypatch, tmp_path, MARTINUS_RESPONSE_CACHE="1")
    ka = rc.make_key("claude", "m", 0.1, [{"role": "user", "content": "square a number"}])
    kb = rc.make_key("claude", "m", 0.1, [{"role": "user", "content": "cube a number"}])
    assert ka != kb


class _CountingClient:
    """Fake provider client that counts real chat calls."""
    def __init__(self):
        self.calls = 0

    def chat(self, req):
        self.calls += 1
        return ChatResponse(provider="ollama", model="m", content="live answer",
                            prompt_tokens=10, completion_tokens=5, total_tokens=15)


def _router_with_fake(monkeypatch, tmp_path):
    # Reload response_cache bound to tmp dir + enabled, then reload providers so
    # it picks up the reloaded module reference.
    _reload_with_env(monkeypatch, tmp_path, MARTINUS_RESPONSE_CACHE="1")
    import agent.providers as providers
    importlib.reload(providers)
    router = providers.Router(default_provider="ollama")
    fake = _CountingClient()
    router._ollama = fake  # type: ignore[assignment]
    return providers, router, fake


def test_router_second_identical_call_hits_cache(monkeypatch, tmp_path):
    providers, router, fake = _router_with_fake(monkeypatch, tmp_path)
    req = providers.ChatRequest(
        messages=[providers.Message(role="user", content="same prompt")],
        temperature=0.1,
    )
    r1 = router.chat("ollama", req)
    assert r1.content == "live answer"
    assert fake.calls == 1
    # Second identical call must not hit the provider.
    r2 = router.chat("ollama", providers.ChatRequest(
        messages=[providers.Message(role="user", content="same prompt")],
        temperature=0.1,
    ))
    assert r2.content == "live answer"
    assert fake.calls == 1  # no new provider call
    assert router.usage_totals().get("cache_hits", 0) == 1


def test_router_high_temperature_not_cached(monkeypatch, tmp_path):
    providers, router, fake = _router_with_fake(monkeypatch, tmp_path)
    for _ in range(2):
        router.chat("ollama", providers.ChatRequest(
            messages=[providers.Message(role="user", content="creative")],
            temperature=0.9,
        ))
    # High temperature must never be served from cache.
    assert fake.calls == 2


def test_router_cache_hit_replays_to_on_token(monkeypatch, tmp_path):
    providers, router, fake = _router_with_fake(monkeypatch, tmp_path)
    router.chat("ollama", providers.ChatRequest(
        messages=[providers.Message(role="user", content="stream me")],
        temperature=0.1,
    ))
    tokens = []
    router.chat("ollama", providers.ChatRequest(
        messages=[providers.Message(role="user", content="stream me")],
        temperature=0.1,
        on_token=tokens.append,
    ))
    assert fake.calls == 1
    assert "".join(tokens) == "live answer"
