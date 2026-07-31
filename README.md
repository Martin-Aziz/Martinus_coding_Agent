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
