#!/usr/bin/env python3
"""
Martinus — single-command launcher.

Starts all three local services in the correct boot order, then opens the
browser automatically:

  1. AI service    (Python / FastAPI  on :8090)
  2. Orchestrator  (Python / FastAPI  on :8080)
  3. UI            (NiceGUI           on :8091)

Usage
-----
    python run.py

Prerequisites
-------------
  • Ollama running at http://localhost:11434  →  start with: ollama serve
  • Python dependencies installed:
        pip install -r ai-service/requirements.txt
        pip install -r ui/requirements.txt
  • Orchestrator dependencies installed:
        pip install -r requirements.txt

Environment overrides (all optional)
--------------------------------------
  OLLAMA_BASE_URL      Ollama API base URL         (default: http://127.0.0.1:11434)
  AI_SERVICE_PORT      AI service port             (default: 8090)
  ORCHESTRATOR_PORT    Orchestrator port           (default: 8080)
  UI_PORT              NiceGUI UI port             (default: 8091)
"""

import os
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

# ── Repository paths ───────────────────────────────────────────────────────────
ROOT             = Path(__file__).parent.resolve()
AI_SERVICE_DIR   = ROOT / "ai-service"
UI_SCRIPT        = ROOT / "ui" / "main.py"

# ── Configurable ports (env overrides accepted) ────────────────────────────────
OLLAMA_URL        = os.getenv("OLLAMA_BASE_URL",    "http://127.0.0.1:11434")
AI_SERVICE_PORT   = int(os.getenv("AI_SERVICE_PORT",   "8090"))
ORCHESTRATOR_PORT = int(os.getenv("ORCHESTRATOR_PORT", "8080"))
UI_PORT           = int(os.getenv("UI_PORT",           "8091"))

# ── Tuning constants ───────────────────────────────────────────────────────────
_POLL_S   = 0.5    # seconds between health-check polls
_TIMEOUT  = 30.0   # seconds to wait for each service to become ready

# ── Internal process registry (reverse order = most-recently-started first) ───
_procs = []  # list[subprocess.Popen]


# ── Signal handler ─────────────────────────────────────────────────────────────
def _cleanup(*_):
    """Gracefully terminate all managed child processes on SIGINT / SIGTERM."""
    print("\n\nStopping all services…", flush=True)
    for proc in reversed(_procs):
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    print("Goodbye.", flush=True)
    sys.exit(0)


signal.signal(signal.SIGINT,  _cleanup)
signal.signal(signal.SIGTERM, _cleanup)


# ── Output helpers ─────────────────────────────────────────────────────────────
def _info(msg):  print(f"  ➜  {msg}", flush=True)
def _ok(msg):    print(f"  ✓  {msg}", flush=True)
def _err(msg):   print(f"  ✗  {msg}", file=sys.stderr, flush=True)


# ── Health-check helpers ───────────────────────────────────────────────────────
def _reachable(url, timeout=3.0):
    """
    Return True if *url* responds to an HTTP GET within *timeout* seconds.
    Any response (including HTTP error codes) counts as reachable; only a
    connection-refused / network error counts as not reachable.
    """
    try:
        urlopen(url, timeout=timeout)
        return True
    except URLError as exc:
        # HTTPError is a subclass of URLError and means the server IS up.
        return hasattr(exc, "code")
    except Exception:
        return False


def _wait_ready(name, url):
    """
    Poll *url* every _POLL_S seconds until it responds or _TIMEOUT expires.
    Calls _cleanup() and exits if the service never becomes ready.
    """
    _info(f"Waiting for {name} at {url} …")
    deadline = time.monotonic() + _TIMEOUT
    while time.monotonic() < deadline:
        if _reachable(url):
            _ok(f"{name} is ready.")
            return
        time.sleep(_POLL_S)
    _err(f"{name} did not start within {_TIMEOUT:.0f}s.")
    _cleanup()


# ── Subprocess helpers ─────────────────────────────────────────────────────────
def _start(args, cwd=None, extra_env=None):
    """
    Launch a subprocess, register it for cleanup, and return the Popen handle.
    stdout is suppressed; stderr is piped for crash diagnostics.
    """
    env = {**os.environ, **(extra_env or {})}
    proc = subprocess.Popen(
        args,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    _procs.append(proc)
    return proc


# ── Pre-flight checks ──────────────────────────────────────────────────────────
def check_python_deps():
    """Best-effort check that key Python dependencies are importable."""
    try:
        import fastapi, uvicorn  # noqa: F401
        _ok("Python deps for AI service found.")
    except ImportError as e:
        _info(f"Warning: {e} — run: pip install -r ai-service/requirements.txt")

    try:
        import nicegui  # noqa: F401
        _ok("Python deps for UI found.")
    except ImportError as e:
        _info(f"Warning: {e} — run: pip install -r ui/requirements.txt")


def check_ollama():
    """Verify Ollama is reachable. Print guidance and exit if not."""
    _info(f"Checking Ollama at {OLLAMA_URL} …")
    if _reachable(f"{OLLAMA_URL}/api/tags"):
        _ok("Ollama is running.")
        return
    _err(
        f"Ollama is not reachable at {OLLAMA_URL}.\n"
        "  Start it first:  ollama serve\n"
        "  Then re-run:     python run.py"
    )
    sys.exit(1)




# ── Service launchers ──────────────────────────────────────────────────────────
def start_ai_service():
    """Start the FastAPI AI service via uvicorn."""
    _info(f"Starting AI service on port {AI_SERVICE_PORT} …")
    _start(
        [
            sys.executable, "-m", "uvicorn",
            "app.main:app",
            "--host", "127.0.0.1",
            "--port", str(AI_SERVICE_PORT),
            "--log-level", "warning",
        ],
        cwd=AI_SERVICE_DIR,
    )
    _wait_ready("AI service", f"http://127.0.0.1:{AI_SERVICE_PORT}/healthz")


def start_orchestrator():
    """
    Start the Python orchestrator (agent/server.py) via uvicorn.

    CWD is the repository root so the routing config at
    config/model-routing.yaml is found by the auto-detection logic.
    """
    _info(f"Starting orchestrator on port {ORCHESTRATOR_PORT} …")
    _start(
        [
            sys.executable, "-m", "uvicorn",
            "agent.server:app",
            "--host", "127.0.0.1",
            "--port", str(ORCHESTRATOR_PORT),
            "--log-level", "warning",
        ],
        cwd=ROOT,
        extra_env={
            "AI_SERVICE_URL": f"http://127.0.0.1:{AI_SERVICE_PORT}",
            "MARTINUS_ENABLE_CLAUDE_ADAPTER": "1",
        },
    )
    _wait_ready("Orchestrator", f"http://127.0.0.1:{ORCHESTRATOR_PORT}/healthz")


def start_ui():
    """Start the NiceGUI UI."""
    _info(f"Starting UI on port {UI_PORT} …")
    _start(
        [sys.executable, str(UI_SCRIPT)],
        extra_env={
            "ORCHESTRATOR_URL": f"http://127.0.0.1:{ORCHESTRATOR_PORT}",
            "UI_HOST":          "0.0.0.0",
            "UI_PORT":          str(UI_PORT),
        },
    )
    _wait_ready("UI", f"http://127.0.0.1:{UI_PORT}")


# ── Process monitoring ─────────────────────────────────────────────────────────
def monitor_processes():
    """
    Inspect all managed child processes once.  If any has exited unexpectedly,
    print a diagnostic (reading at most 4096 bytes from stderr to avoid
    blocking) and initiate a clean shutdown.
    """
    for proc in _procs:
        if proc.poll() is not None:
            try:
                stderr_text = (proc.stderr.read(4096) or b"").decode(errors="replace").strip()
            except Exception:
                stderr_text = ""
            _err(
                f"A service exited unexpectedly (exit {proc.returncode})."
                + (f"\n  Last output: {stderr_text[:400]}" if stderr_text else "")
            )
            _cleanup()


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    print()
    print("  Martinus")
    print("  ════════════════════")
    print()

    # Pre-flight: verify deps, Ollama, ensure binary exists.
    check_python_deps()
    check_ollama()
    print()

    # Boot services in dependency order.
    start_ai_service()
    start_orchestrator()
    start_ui()

    print()
    _ok(f"All services ready.  →  http://127.0.0.1:{UI_PORT}")
    print()
    webbrowser.open(f"http://127.0.0.1:{UI_PORT}")
    print("  Press Ctrl+C to stop all services.\n", flush=True)

    # Keep the launcher alive; detect and report unexpected child crashes.
    while True:
        monitor_processes()
        time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        _err(f"Unexpected error: {exc}")
        _cleanup()
