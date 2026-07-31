# Martinus source map

Martinus is a local-first coding agent with a Python/FastAPI backend, a local
memory service, a NiceGUI inspection interface, and a Tauri desktop app.

## Core backend

- `agent/server.py` — HTTP and SSE API.
- `agent/orchestrator.py` — multi-phase execution engine and file writer.
- `agent/prompts.py` — phase instructions and prompt contracts.
- `agent/providers.py` — Ollama, LM Studio, and Claude Code adapters.
- `agent/profiles.py` — hardware profiles and role routing.
- `agent/gates.py` — deterministic source verification.
- `agent/tests/` — backend contract, safety, provider, and output tests.

## Memory service

- `ai-service/app/main.py` — memory-service API.
- `ai-service/app/memory.py` — SQLite episodic memory and hybrid retrieval.
- `ai-service/app/llm.py` — provider client for service-side operations.
- `ai-service/app/models.py` — request and response schemas.
- `ai-service/tests/` — memory tests.

The SQLite database is runtime data and is created locally; it is intentionally
not included in the submission.

## Interfaces

- `ui/main.py` — NiceGUI developer/reviewer dashboard on port 8091.
- `desktop/src/main.tsx` — native app React UI.
- `desktop/src-tauri/src/lib.rs` — Tauri commands, runtime detection, service
  lifecycle, workspace selection, and local Claude Code discovery.

## Entry points and configuration

- `start.py` — start all Python services.
- `run.py` — alternative launcher.
- `system_verification.py` — end-to-end smoke test.
- `evaluator/main.py` — optional benchmark runner.
- `config/model-routing.yaml` — active per-role provider routing.
- `.env.example` — environment-variable examples.
- `requirements.txt` — Python dependencies.
- `Makefile` — tests, linting, and type checking.

Generated workspaces, memory databases, build output, private development notes,
and evaluation transcripts are intentionally excluded from this frozen source
package.
