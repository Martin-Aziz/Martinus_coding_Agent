#!/usr/bin/env python3
"""
system_verification.py — End-to-end health + sample-task smoke test for Martinus.

Cross-platform. Uses the real SSE task endpoint (POST /v1/tasks/stream). Reads
the model name from the environment (or auto-detects the first available Ollama
tag). No hard-coded Windows paths, no phantom /v1/tasks/execute route.

Usage:
    python system_verification.py            # full check + sample task
    python system_verification.py --no-task  # health checks only
    MARTINUS_MODEL=qwen2.5-coder:7b python system_verification.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import httpx


ORCH_URL = os.getenv("ORCHESTRATOR_URL", "http://127.0.0.1:8080").rstrip("/")
AI_URL = os.getenv("AI_SERVICE_URL", "http://127.0.0.1:8090").rstrip("/")
UI_URL = os.getenv("UI_URL", "http://127.0.0.1:8091").rstrip("/")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")


def _print_header(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{title}\n{bar}")


def _check_endpoint(label: str, url: str, timeout: float = 3.0) -> bool:
    try:
        r = httpx.get(url, timeout=timeout)
        ok = 200 <= r.status_code < 300
        marker = "OK  " if ok else "FAIL"
        print(f"  [{marker}] {label:<28} {url}  (HTTP {r.status_code})")
        return ok
    except Exception as exc:
        print(f"  [FAIL] {label:<28} {url}  ({exc.__class__.__name__}: {exc})")
        return False


def _first_available_ollama_model() -> str:
    override = os.getenv("MARTINUS_MODEL", "").strip()
    if override:
        return override
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3.0)
        r.raise_for_status()
        models = r.json().get("models", [])
        for m in models:
            name = m.get("name", "")
            if name:
                return name
    except Exception:
        pass
    return ""


def run_health_checks() -> bool:
    _print_header("Health checks")
    ok_orch = _check_endpoint("Orchestrator", f"{ORCH_URL}/healthz")
    ok_ai = _check_endpoint("AI service", f"{AI_URL}/healthz")
    _check_endpoint("NiceGUI dashboard (dev UI)", UI_URL, timeout=2.0)  # advisory
    ok_ollama = _check_endpoint("Ollama", f"{OLLAMA_URL}/api/tags")

    if not (ok_orch and ok_ai):
        return False

    _print_header("Provider health via orchestrator")
    try:
        r = httpx.get(
            f"{ORCH_URL}/v1/providers/health",
            params={"provider": "ollama"},
            timeout=5.0,
        )
        r.raise_for_status()
        payload = r.json()
        for k in ("provider", "reachable", "model", "modelAvailable", "message"):
            print(f"  {k:<16}: {payload.get(k)}")
    except Exception as exc:
        print(f"  Provider health check failed: {exc}")
        return False

    return ok_orch and ok_ai and ok_ollama


def run_sample_task() -> bool:
    _print_header("Sample task (streamed via /v1/tasks/stream)")

    model = _first_available_ollama_model()
    if not model:
        print("  [SKIP] No Ollama model available. Run `ollama pull qwen2.5-coder:7b` "
              "or set MARTINUS_MODEL, then re-run.")
        return False

    workspace = Path(tempfile.mkdtemp(prefix="martinus-smoke-"))
    task = (
        "Write a single Python file named hello.py that defines a function "
        'hello() returning the string "Hello, World!". Use <file path="hello.py">'
        "...</file> tags — no markdown fences."
    )
    payload = {
        "sessionId": f"smoke-{int(time.time())}",
        "task": task,
        "workspacePath": str(workspace),
        "goals": ["one file", "no fences"],
        "provider": "ollama",
        "hardwareProfile": "desktop",
        "maxLines": 200,
        "model": model,
    }
    print(f"  Model     : {model}")
    print(f"  Workspace : {workspace}")
    print("  Streaming phases (this may take 1–3 minutes on 7B, longer on 20B):\n")

    started = time.time()
    phases_seen: list[str] = []
    try:
        with httpx.stream(
            "POST",
            f"{ORCH_URL}/v1/tasks/stream",
            json=payload,
            timeout=600.0,
            headers={"Accept": "text/event-stream"},
        ) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines():
                if not raw or not raw.startswith("data: "):
                    continue
                try:
                    event = json.loads(raw[6:])
                except json.JSONDecodeError:
                    continue
                etype = event.get("type", "")
                if etype == "phase_started":
                    phase = event.get("phase", "?")
                    phases_seen.append(phase)
                    print(f"    → {phase} started")
                elif etype == "phase_completed":
                    phase = event.get("phase", "?")
                    print(f"    ✓ {phase} completed ({event.get('model', '?')})")
                elif etype == "phase_failed":
                    print(f"    ✗ {event.get('phase', '?')} failed: {event.get('error', '?')}")
                elif etype == "error":
                    print(f"    !! error: {event.get('message', '?')}")
                    return False
                elif etype == "run_completed":
                    print(f"    ✓ run_completed")
    except Exception as exc:
        print(f"  Task stream failed: {exc}")
        return False

    elapsed = time.time() - started
    hello_file = workspace / "hello.py"
    if hello_file.exists():
        print(f"\n  [OK] hello.py written ({hello_file.stat().st_size} bytes) in {elapsed:.1f}s")
        print("  --- content preview ---")
        print("  " + hello_file.read_text()[:400].replace("\n", "\n  "))
        return True
    print(f"\n  [FAIL] hello.py NOT written after {elapsed:.1f}s. Phases seen: {phases_seen}")
    print(f"  Workspace files: {[p.name for p in workspace.iterdir()]}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-task", action="store_true", help="Skip the sample-task test")
    args = parser.parse_args()

    print("Martinus system verification")
    print(f"  Orchestrator : {ORCH_URL}")
    print(f"  AI service   : {AI_URL}")
    print(f"  UI (NiceGUI) : {UI_URL}  (optional dev client)")
    print(f"  Ollama       : {OLLAMA_URL}")

    healthy = run_health_checks()
    if not healthy:
        print("\n[!] Health checks failed. Start local services first:")
        print("    python start.py")
        return 1

    if args.no_task:
        print("\n[OK] All health checks passed. Skipping sample task.")
        return 0

    ok = run_sample_task()
    print("\n" + ("[OK] Sample task succeeded — Martinus is ready to demo."
                  if ok else "[FAIL] Sample task did not produce hello.py."))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
