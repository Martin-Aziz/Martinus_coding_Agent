# Martinus application source

Martinus is a local-first autonomous coding agent. It turns a natural-language
task into files through Discovery, Clarification, Architecture,
Implementation, Review, and Verification.

## Source structure

| Path | Purpose |
|---|---|
| `agent/` | FastAPI orchestrator, phase engine, prompts, provider adapters, verification gates, and tests |
| `ai-service/` | Local episodic-memory and reflection service with tests |
| `ui/` | NiceGUI developer and reviewer interface |
| `desktop/` | React/TypeScript and Tauri/Rust native desktop application |
| `config/` | Per-provider model-routing configuration |
| `evaluator/` | Optional ten-task benchmark runner |
| `start.py` | Starts the orchestrator, memory service, and NiceGUI interface |
| `run.py` | Alternative command-line launcher |
| `system_verification.py` | Cross-platform service and task smoke test |

`AGENTS.md` contains a detailed source map. Evaluation records and surveys are
kept separately in the submission package’s `evidence/` directory.

## LOCALFORGE capabilities

Four capabilities from the LOCALFORGE graph (`graphLoop.md`) are implemented as
opt-in, backward-compatible additions (see `AGENTS.md` for flags and env vars):

- **Sandbox isolation** (`agent/sandbox.py`) — runs the pipeline in an isolated
  git worktree (or temp-dir copy) and applies files back to the real workspace
  only after a verified pass. Enable with `MARTINUS_SANDBOX_ISOLATION=1`.
- **Permission tiers** (`agent/permissions.py`) — classifies actions L0–L3;
  external/destructive actions are denied by default in a headless run.
- **Independent verifier** (`agent/orchestrator.py`) — a final fresh-context
  reviewer that gates apply-back; advisory (can downgrade a pass, never upgrade
  a failure). On for the `high` profile or `MARTINUS_INDEPENDENT_VERIFIER=1`.
- **Repository map** (`agent/repomap.py`) — a deterministic Layer 1–3 map/symbol
  index injected into Discovery/Architecture and served at `GET /v1/repomap`.
- **Graph-native execution** (`agent/graph.py`, `agent/graph_patterns.py`) —
  compiles a request into a typed, validated execution graph and runs it via
  `GraphRuntime` (reusing the existing phase logic as node executors) instead of
  the fixed linear loop. Enable with `MARTINUS_GRAPH_ENGINE=1`. See `ROADMAP.md`.

## Local setup

Python 3.11+ and one model provider are required.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python start.py
```

Open `http://127.0.0.1:8091`.

Provider defaults:

- Ollama: `http://127.0.0.1:11434`
- LM Studio: `http://127.0.0.1:1234`
- Claude Code: local `claude` CLI when the adapter is enabled

The desktop application starts its orchestrator on port `18080` and exposes
Ollama, LM Studio, and Claude Code in the provider selector.

## Tests

```sh
make test
make lint
make typecheck
cd desktop && npm ci && npm run build
cd src-tauri && cargo check
```

The agent and memory-service test suites use different import roots; the
`Makefile` invokes them correctly.

## Desktop builds

From `desktop/`:

```sh
npm ci
npm run tauri:build
```

See `MACOS_BUILD.md` and `WINDOWS_BUILD.md` for platform details. A prebuilt
macOS package is included in the submission’s `installers/` directory.
