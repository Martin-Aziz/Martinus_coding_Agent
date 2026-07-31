"""
agent/providers.py — Ollama and LM Studio LLM clients with streaming support.

Algorithm 3.1: Bounded Retry with Self-Correction.
Every LLM call retries at most MAX_RETRIES times. On failure the error is
appended as a self-correction instruction so the model can identify the root
cause and recover without changing phase logic or temperature.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

import httpx

FALLBACK_RETRY_INSTRUCTION = (
    "\n\nPrevious attempt failed. Before retrying: "
    "(1) identify the root cause explicitly, "
    "(2) correct the specific error, "
    "(3) produce a minimal but complete response that fully satisfies the output contract."
)

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


# Split connect/read timeouts, both env-configurable. The old flat 1800s
# timeout meant a dead runtime could stall a phase for 30 minutes; connect
# failures now surface in seconds while long generations still get a generous
# (but bounded) read window.
_CONNECT_TIMEOUT = _env_float("MARTINUS_CONNECT_TIMEOUT_SECONDS", 10.0)
_READ_TIMEOUT = _env_float("MARTINUS_READ_TIMEOUT_SECONDS", 600.0)
_DEFAULT_TIMEOUT = httpx.Timeout(
    connect=_CONNECT_TIMEOUT, read=_READ_TIMEOUT, write=60.0, pool=30.0
)
_HEALTH_TIMEOUT = 5.0
_LIST_TIMEOUT = 5.0

# Hardware safety: at most N model requests in flight machine-wide (default 1
# — a 24GB unified-memory machine cannot serve two 7B generations at once
# without swapping). Router.chat serializes on this.
_MAX_CONCURRENT = max(1, int(_env_float("MARTINUS_MAX_CONCURRENT_MODEL_REQUESTS", 1)))
_MODEL_CALL_SEMAPHORE = threading.BoundedSemaphore(_MAX_CONCURRENT)


class DeterministicProviderError(RuntimeError):
    """A provider error that will not change on retry (missing model,
    malformed request, 4xx). Callers must NOT retry these."""

# Short TTL cache: health and model-list calls are hit on every UI refresh and
# every phase start; caching for a few seconds eliminates the 8-12s stall path
# when a runtime is briefly unreachable without hiding real state changes.
_CACHE_TTL_SECONDS = 6.0


@dataclass
class Message:
    role: str
    content: str


@dataclass
class ChatRequest:
    messages: list[Message]
    temperature: float = 0.1
    disable_reasoning: bool = True
    model: str = ""
    on_token: Optional[Callable[[str], None]] = None


@dataclass
class ChatResponse:
    provider: str
    model: str
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ProviderHealth:
    provider: str
    reachable: bool
    model: str
    model_available: bool
    message: str = ""


class _TTLCache:
    """Tiny per-key TTL cache used to smooth over repeated health/list checks."""

    def __init__(self, ttl_seconds: float = _CACHE_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._data: dict[str, tuple[float, object]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[object]:
        with self._lock:
            entry = self._data.get(key)
            if not entry:
                return None
            ts, value = entry
            if time.monotonic() - ts > self._ttl:
                self._data.pop(key, None)
                return None
            return value

    def set(self, key: str, value: object) -> None:
        with self._lock:
            self._data[key] = (time.monotonic(), value)

    def invalidate(self, key: Optional[str] = None) -> None:
        with self._lock:
            if key is None:
                self._data.clear()
            else:
                self._data.pop(key, None)


def _clamp(t: float) -> float:
    return max(0.0, min(1.0, t))


def _actionable_error(provider_label: str, base_url: str, exc: BaseException) -> str:
    """Turn raw httpx errors into copy that tells the user what to do next."""
    kind = exc.__class__.__name__
    raw = str(exc).strip() or kind
    if isinstance(exc, httpx.ConnectError):
        return (
            f"{provider_label} is not reachable at {base_url}. "
            f"Start the local server — e.g. `ollama serve` for Ollama or click "
            f"'Start Server' in LM Studio — then try again."
        )
    if isinstance(exc, httpx.ReadTimeout):
        return f"{provider_label} at {base_url} timed out. The model may be loading; wait a moment and retry."
    if isinstance(exc, httpx.HTTPStatusError):
        return f"{provider_label} returned HTTP {exc.response.status_code}. Check the server logs."
    # Fallback: never surface a bare traceback line to the UI.
    return f"{provider_label} error: {kind}. Details: {raw[:160]}"


class OllamaClient:
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model  # may be empty ("auto-detect")
        self._cache = _TTLCache()
        # Long-lived client — reuses TCP connections across health/list/chat calls.
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=_DEFAULT_TIMEOUT,
            limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
        )

    def health_check(self) -> ProviderHealth:
        cached = self._cache.get("health")
        if isinstance(cached, ProviderHealth):
            return cached
        health = ProviderHealth(
            provider="ollama",
            reachable=False,
            model=self.model,
            model_available=False,
        )
        try:
            resp = self._client.get("/api/tags", timeout=_HEALTH_TIMEOUT)
            resp.raise_for_status()
            health.reachable = True
            data = resp.json()
            names = [m.get("name", "") for m in data.get("models", [])]
            if not self.model:
                # Empty model default: the runtime picks whatever is loaded.
                health.model_available = bool(names)
                health.message = (
                    f"Ollama is ready with {len(names)} model(s) — pick one in Settings."
                    if names
                    else "Ollama is reachable but has no models installed. Run e.g. `ollama pull qwen2.5-coder:7b`."
                )
            else:
                health.model_available = any(
                    n.strip().lower() == self.model.strip().lower() for n in names
                )
                health.message = (
                    "ready"
                    if health.model_available
                    else f"model {self.model!r} not found. Available: {', '.join(names[:5]) or '(none)'}"
                )
        except Exception as exc:
            health.message = _actionable_error("Ollama", self.base_url, exc)
        self._cache.set("health", health)
        return health

    def list_models(self) -> list[str]:
        cached = self._cache.get("models")
        if isinstance(cached, list):
            return cached
        try:
            resp = self._client.get("/api/tags", timeout=_LIST_TIMEOUT)
            resp.raise_for_status()
            models = sorted(m.get("name", "") for m in resp.json().get("models", []))
        except Exception:
            models = []
        self._cache.set("models", models)
        return models

    def chat(self, req: ChatRequest) -> ChatResponse:
        model = req.model.strip() or self.model
        if not model:
            raise DeterministicProviderError(
                "Ollama has no model selected. Load a model in Ollama or pick one in Settings."
            )
        streaming = req.on_token is not None
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in req.messages],
            "stream": streaming,
            "think": not req.disable_reasoning,
            "options": {"temperature": _clamp(req.temperature)},
        }
        full_content: list[str] = []
        usage: dict = {}
        if streaming:
            with self._client.stream("POST", "/api/chat", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        req.on_token(token)
                        full_content.append(token)
                    if chunk.get("done"):
                        usage = chunk
                        break
        else:
            resp = self._client.post("/api/chat", json=payload)
            resp.raise_for_status()
            usage = resp.json()
            full_content.append(usage.get("message", {}).get("content", ""))
        # Chat succeeded — invalidate the health cache so the UI notices state changes quickly.
        self._cache.invalidate("health")
        prompt_tokens = int(usage.get("prompt_eval_count", 0) or 0)
        completion_tokens = int(usage.get("eval_count", 0) or 0)
        return ChatResponse(
            provider="ollama", model=model, content="".join(full_content),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )


class LMStudioClient:
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model  # may be empty ("use whichever is loaded")
        self._cache = _TTLCache()
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=_DEFAULT_TIMEOUT,
            limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
        )

    def health_check(self) -> ProviderHealth:
        cached = self._cache.get("health")
        if isinstance(cached, ProviderHealth):
            return cached
        health = ProviderHealth(
            provider="lmstudio",
            reachable=False,
            model=self.model,
            model_available=False,
        )
        try:
            resp = self._client.get("/v1/models", timeout=_HEALTH_TIMEOUT)
            resp.raise_for_status()
            health.reachable = True
            data = resp.json()
            ids = [m.get("id", "") for m in data.get("data", [])]
            if not self.model:
                health.model_available = bool(ids)
                health.message = (
                    f"LM Studio is ready with {len(ids)} model(s) — pick one in Settings."
                    if ids
                    else "LM Studio is reachable but has no models loaded. Load a model in LM Studio."
                )
            else:
                health.model_available = any(
                    i.strip().lower() == self.model.strip().lower() for i in ids
                )
                health.message = (
                    "ready"
                    if health.model_available
                    else f"model {self.model!r} not found. Loaded: {', '.join(ids[:5]) or '(none)'}"
                )
        except Exception as exc:
            health.message = _actionable_error("LM Studio", self.base_url, exc)
        self._cache.set("health", health)
        return health

    def list_models(self) -> list[str]:
        cached = self._cache.get("models")
        if isinstance(cached, list):
            return cached
        try:
            resp = self._client.get("/v1/models", timeout=_LIST_TIMEOUT)
            resp.raise_for_status()
            models = sorted(m.get("id", "") for m in resp.json().get("data", []))
        except Exception:
            models = []
        self._cache.set("models", models)
        return models

    def chat(self, req: ChatRequest) -> ChatResponse:
        model = req.model.strip() or self.model
        if not model:
            raise DeterministicProviderError(
                "LM Studio has no model selected. Load a model in LM Studio or pick one in Settings."
            )
        streaming = req.on_token is not None
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in req.messages],
            "temperature": _clamp(req.temperature),
            "max_tokens": 4096,
            "stream": streaming,
            "reasoning": {"enabled": not req.disable_reasoning},
        }
        full_content: list[str] = []
        usage: dict = {}
        if streaming:
            with self._client.stream(
                "POST", "/v1/chat/completions", json=payload
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    choices = chunk.get("choices", [])
                    if choices:
                        token = choices[0].get("delta", {}).get("content", "")
                        if token:
                            req.on_token(token)
                            full_content.append(token)
        else:
            resp = self._client.post("/v1/chat/completions", json=payload)
            resp.raise_for_status()
            body = resp.json()
            usage = body.get("usage", {})
            choices = body.get("choices", [])
            if not choices:
                raise ValueError("lmstudio returned no choices")
            full_content.append(choices[0].get("message", {}).get("content", ""))
        self._cache.invalidate("health")
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        return ChatResponse(
            provider="lmstudio", model=model, content="".join(full_content),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0),
        )


class Router:
    """Routes LLM calls to Ollama, LM Studio, or opt-in Claude Code."""

    def __init__(
        self,
        default_provider: str = "ollama",
        ollama_base_url: str = "http://127.0.0.1:11434",
        ollama_model: str = "",
        lmstudio_base_url: str = "http://127.0.0.1:1234",
        lmstudio_model: str = "",
    ) -> None:
        self._default = self._normalize(default_provider) or "ollama"
        self._ollama = OllamaClient(ollama_base_url, ollama_model)
        self._lmstudio = LMStudioClient(lmstudio_base_url, lmstudio_model)
        self._usage = {
            "model_requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        # Optional Claude Code provider. It remains disabled unless explicitly
        # enabled with MARTINUS_ENABLE_CLAUDE_ADAPTER=1.
        self._claude: Optional["ClaudeCodeClient"] = None
        if os.getenv("MARTINUS_ENABLE_CLAUDE_ADAPTER", "").strip() in ("1", "true", "yes"):
            self._claude = ClaudeCodeClient()

    @staticmethod
    def _normalize(name: str) -> str:
        n = name.strip().lower()
        if n in ("ollama", "lmstudio", "claude"):
            return n
        return ""

    def resolve(self, provider: str) -> str:
        return self._normalize(provider) or self._default

    def chat(self, provider: str, req: ChatRequest) -> ChatResponse:
        r = self.resolve(provider)
        # Serialize model requests machine-wide (hardware safety on 24GB
        # unified memory). Bounded wait so a wedged holder cannot block forever.
        acquired = _MODEL_CALL_SEMAPHORE.acquire(timeout=_READ_TIMEOUT + 120.0)
        if not acquired:
            raise RuntimeError(
                "timed out waiting for the model-request slot; another run may be wedged"
            )
        try:
            if r == "claude" and self._claude is None:
                raise DeterministicProviderError(
                    "Claude Code is disabled. Set MARTINUS_ENABLE_CLAUDE_ADAPTER=1 "
                    "before starting the orchestrator."
                )
            if r == "claude":
                assert self._claude is not None
                response = self._claude.chat(req)
            elif r == "lmstudio":
                response = self._lmstudio.chat(req)
            else:
                response = self._ollama.chat(req)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if 400 <= status < 500:
                # Client-side errors are deterministic — retrying the same
                # payload will fail the same way.
                raise DeterministicProviderError(
                    f"{r} returned HTTP {status}: {exc.response.text[:200]}"
                ) from exc
            raise
        finally:
            _MODEL_CALL_SEMAPHORE.release()
        self._usage["model_requests"] += 1
        self._usage["prompt_tokens"] += response.prompt_tokens
        self._usage["completion_tokens"] += response.completion_tokens
        self._usage["total_tokens"] += response.total_tokens
        return response

    def usage_totals(self) -> dict[str, int]:
        """Return cumulative measurement-only usage for this Router instance."""
        return dict(self._usage)

    def health(self, provider: str) -> ProviderHealth:
        r = self.resolve(provider)
        if r == "claude" and self._claude is None:
            return ProviderHealth(
                provider="claude",
                reachable=False,
                model="claude-code",
                model_available=False,
                message="Claude Code is disabled. Enable it before starting the orchestrator.",
            )
        if r == "claude":
            assert self._claude is not None
            return self._claude.health_check()
        if r == "lmstudio":
            return self._lmstudio.health_check()
        return self._ollama.health_check()

    def list_models(self, provider: str) -> list[str]:
        resolved = self.resolve(provider)
        if resolved == "claude" and self._claude is None:
            return []
        if resolved == "claude":
            assert self._claude is not None
            return self._claude.list_models()
        if resolved == "lmstudio":
            return self._lmstudio.list_models()
        return self._ollama.list_models()


# Optional Claude Code provider.
# Enable with MARTINUS_ENABLE_CLAUDE_ADAPTER=1.

class ClaudeCodeClient:
    """Adapter that shells out to the local `claude` CLI for each chat call.

    Each request becomes one `claude -p --output-format json` subprocess.
    Streaming is not implemented — the CLI does support --output-format
    stream-json but a subprocess-per-call keeps this adapter reversible and
    matches the semantics of the Ollama non-streaming path.
    """

    def __init__(self) -> None:
        import shutil
        self._binary = os.getenv("MARTINUS_CLAUDE_BINARY", "").strip() or shutil.which("claude") or "claude"
        # Extra flags — appended AFTER `-p`. Useful for pinning a model
        # (--model claude-sonnet-5) if you want to.
        extra = os.getenv("MARTINUS_CLAUDE_ARGS", "").strip()
        self._extra_args: list[str] = extra.split() if extra else []
        self._per_call_timeout = float(os.getenv("MARTINUS_CLAUDE_TIMEOUT", "300"))
        # Cache for health probes.
        self._cache = _TTLCache()

    def _probe_binary(self) -> tuple[bool, str]:
        import subprocess
        try:
            proc = subprocess.run(
                [self._binary, "--version"],
                capture_output=True,
                text=True,
                timeout=8.0,
            )
            if proc.returncode == 0:
                return True, proc.stdout.strip() or "claude CLI available"
            return False, proc.stderr.strip() or f"claude --version exited with {proc.returncode}"
        except FileNotFoundError:
            return False, f"claude binary not found at {self._binary!r}"
        except Exception as exc:
            return False, f"{exc.__class__.__name__}: {exc}"

    def health_check(self) -> ProviderHealth:
        cached = self._cache.get("health")
        if isinstance(cached, ProviderHealth):
            return cached
        ok, msg = self._probe_binary()
        health = ProviderHealth(
            provider="claude",
            reachable=ok,
            model="claude-code",
            model_available=ok,
            message=msg if ok else f"Claude Code CLI not available: {msg}",
        )
        self._cache.set("health", health)
        return health

    def list_models(self) -> list[str]:
        # The CLI decides its own model; expose a single logical model id.
        # If you want to switch, set MARTINUS_CLAUDE_ARGS='--model <id>'.
        return ["claude-code"]

    def chat(self, req: ChatRequest) -> ChatResponse:
        import subprocess

        # Merge system + user into one prompt. The CLI's -p mode takes a single
        # prompt string; we prepend the system role as a labeled preamble.
        system_content = ""
        user_content = ""
        for msg in req.messages:
            if msg.role == "system":
                system_content = msg.content
            elif msg.role == "user":
                user_content = msg.content
        merged = (
            f"[SYSTEM INSTRUCTIONS]\n{system_content}\n\n[USER REQUEST]\n{user_content}"
            if system_content
            else user_content
        )

        streaming = req.on_token is not None

        if streaming:
            # Live path: --output-format stream-json emits one JSON object per
            # event. Parse assistant text.deltas and forward them as tokens so
            # the UI sees a real progress bar instead of a 60s spinner.
            content, model_name, err = self._chat_streaming(merged, req.on_token)
            if err:
                raise RuntimeError(err)
            return ChatResponse(provider="claude", model=model_name, content=content)

        # Non-streaming path: single subprocess, single JSON envelope.
        cmd = [self._binary, "-p", "--output-format", "json"]
        cmd.extend(self._extra_args)
        try:
            proc = subprocess.run(
                cmd,
                input=merged,
                capture_output=True,
                text=True,
                timeout=self._per_call_timeout,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"claude CLI timed out after {self._per_call_timeout}s"
            ) from None
        except FileNotFoundError:
            raise RuntimeError(
                f"claude CLI binary not found at {self._binary!r}. "
                "Install Claude Code or set MARTINUS_CLAUDE_BINARY."
            ) from None
        if proc.returncode != 0:
            raise RuntimeError(
                f"claude CLI exited {proc.returncode}: {proc.stderr.strip()[:400]}"
            )
        content = ""
        model_name = "claude-code"
        raw_out = proc.stdout.strip()
        if raw_out:
            try:
                envelope = json.loads(raw_out)
                if isinstance(envelope, dict):
                    content = str(
                        envelope.get("result")
                        or envelope.get("text")
                        or envelope.get("content")
                        or ""
                    ).strip()
                    model_name = str(envelope.get("model") or model_name)
                else:
                    content = raw_out
            except json.JSONDecodeError:
                content = raw_out
        if not content:
            raise RuntimeError(
                "claude CLI returned an empty response. "
                f"stderr: {proc.stderr.strip()[:200]}"
            )
        self._cache.invalidate("health")
        return ChatResponse(provider="claude", model=model_name, content=content)

    def _chat_streaming(
        self, merged_prompt: str, on_token: Callable[[str], None]
    ) -> tuple[str, str, str]:
        """Run `claude -p --output-format stream-json --verbose` and forward
        assistant text chunks to on_token as they arrive. Returns
        (full_content, model_name, error_message). If error_message is
        non-empty the caller should raise it."""
        import subprocess
        import threading

        cmd = [
            self._binary, "-p",
            "--output-format", "stream-json",
            "--verbose",   # required by the CLI when using stream-json
        ]
        cmd.extend(self._extra_args)

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # line-buffered
            )
        except FileNotFoundError:
            return "", "claude-code", (
                f"claude CLI binary not found at {self._binary!r}. "
                "Install Claude Code or set MARTINUS_CLAUDE_BINARY."
            )

        # Write the prompt on stdin and close so the CLI actually starts work.
        try:
            proc.stdin.write(merged_prompt)
            proc.stdin.close()
        except BrokenPipeError:
            pass

        # Drain stderr in a background thread so it can't fill the pipe and
        # deadlock the parent.
        stderr_chunks: list[str] = []

        def _drain_stderr() -> None:
            try:
                for line in proc.stderr:
                    stderr_chunks.append(line)
            except Exception:
                pass

        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

        assembled: list[str] = []
        seen_texts: set[str] = set()
        model_name = "claude-code"
        deadline = time.monotonic() + self._per_call_timeout
        try:
            for raw in proc.stdout:
                if time.monotonic() > deadline:
                    proc.terminate()
                    return "".join(assembled), model_name, (
                        f"claude CLI stream exceeded {self._per_call_timeout}s"
                    )
                line = raw.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                etype = event.get("type", "")
                if etype == "assistant":
                    message = event.get("message", {})
                    if isinstance(message, dict):
                        # Record model name if the CLI reports one.
                        m = message.get("model")
                        if m:
                            model_name = str(m)
                        # `content` is a list of content-blocks; forward each
                        # new text segment as a token. Claude emits
                        # incrementally-growing text blocks, so we dedupe on
                        # the full text and forward only the delta.
                        content_blocks = message.get("content", [])
                        if isinstance(content_blocks, list):
                            for block in content_blocks:
                                if not isinstance(block, dict):
                                    continue
                                if block.get("type") != "text":
                                    continue
                                text = str(block.get("text", ""))
                                if not text or text in seen_texts:
                                    continue
                                # Find the longest previous prefix so we only
                                # emit the delta rather than the whole growing
                                # message every time.
                                delta = text
                                for prev in list(seen_texts):
                                    if text.startswith(prev):
                                        delta = text[len(prev):]
                                        break
                                seen_texts.add(text)
                                if delta:
                                    try:
                                        on_token(delta)
                                    except Exception:
                                        pass
                                    assembled.append(delta)
                elif etype == "result":
                    # Terminal envelope. `result` field is authoritative when
                    # present; fall back to what we accumulated from deltas.
                    final = event.get("result")
                    if isinstance(final, str) and final:
                        # If the assembled deltas somehow disagree with the
                        # authoritative result, prefer result but keep any
                        # extra we already streamed.
                        assembled_full = "".join(assembled)
                        if not assembled_full or assembled_full != final:
                            missing = final[len(assembled_full):] if final.startswith(assembled_full) else ""
                            if missing:
                                try:
                                    on_token(missing)
                                except Exception:
                                    pass
                                assembled.append(missing)
                            elif not assembled_full:
                                # Nothing came through deltas — emit the whole
                                # result as one token so downstream code sees
                                # at least one on_token call.
                                try:
                                    on_token(final)
                                except Exception:
                                    pass
                                assembled.append(final)
                    is_error = bool(event.get("is_error"))
                    if is_error:
                        err_msg = str(event.get("terminal_reason") or event.get("stop_reason") or "unknown")
                        return "".join(assembled), model_name, f"claude CLI reported error: {err_msg}"
        except Exception as exc:
            return "".join(assembled), model_name, f"claude CLI stream read failed: {exc}"

        rc = proc.wait(timeout=5)
        stderr_thread.join(timeout=1)
        if rc != 0:
            err = "".join(stderr_chunks).strip()[:400] or f"exit code {rc}"
            return "".join(assembled), model_name, f"claude CLI exited nonzero: {err}"

        self._cache.invalidate("health")
        return "".join(assembled), model_name, ""


def from_env() -> Router:
    return Router(
        default_provider=os.getenv("DEFAULT_PROVIDER", "ollama"),
        ollama_base_url=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
        # Empty by default: detect whatever the user has loaded. UI passes an
        # explicit model per-request; env override still supported.
        ollama_model=os.getenv("OLLAMA_MODEL", ""),
        lmstudio_base_url=os.getenv("LMSTUDIO_URL", "http://127.0.0.1:1234"),
        lmstudio_model=os.getenv("LMSTUDIO_MODEL", ""),
    )
