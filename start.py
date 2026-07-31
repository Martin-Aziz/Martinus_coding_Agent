#!/usr/bin/env python3
"""
start.py — Launch the Martinus local services in one terminal.

Usage:
    python start.py            # starts orchestrator + ai-service + (legacy) ui
    python start.py orch       # orchestrator only
    python start.py ai         # ai-service only
    python start.py ui         # NiceGUI dev/inspection dashboard only

Design notes
------------
- Two shipping UI clients: the Rust/Tauri desktop app is the end-user UX; the
  NiceGUI dashboard on port 8091 is the developer/reviewer client (this is the
  one the professor uses when they want to inspect what the agent produced).
- Default bind is 127.0.0.1. Set MARTINUS_BIND_HOST=0.0.0.0 to expose services
  on the LAN — combine with ORCHESTRATOR_REQUIRE_API_KEY=true when you do.
- ai-service listens on 8090 (matches the thesis prose and system_verification).
"""
import subprocess
import sys
import os


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


DEFAULT_HOST = _env("MARTINUS_BIND_HOST", "127.0.0.1")

SERVICES = {
    "orch": [
        sys.executable, "-m", "uvicorn",
        "agent.server:app",
        "--host", _env("ORCHESTRATOR_HOST", DEFAULT_HOST),
        "--port", _env("ORCHESTRATOR_PORT", "8080"),
        "--log-level", "info",
    ],
    "ai": [
        sys.executable, "-m", "uvicorn",
        "ai-service.app.main:app",
        "--host", _env("AI_SERVICE_HOST", DEFAULT_HOST),
        "--port", _env("AI_SERVICE_PORT", "8090"),
        "--log-level", "info",
    ],
    "ui": [
        sys.executable, "ui/main.py",
    ],
}


def main() -> None:
    targets = sys.argv[1:] or list(SERVICES.keys())
    procs = []
    for target in targets:
        if target not in SERVICES:
            print(f"Unknown service: {target}. Choose from: {', '.join(SERVICES)}")
            sys.exit(1)
        print(f"Starting {target}...")
        procs.append(subprocess.Popen(SERVICES[target]))

    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
