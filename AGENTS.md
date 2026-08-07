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
- `agent/sandbox.py` — isolated execution (git worktree / temp-dir copy) with
  verified apply-back (LOCALFORGE Rule 2.2 / Node H).
- `agent/permissions.py` — tiered action risk model L0–L3 (LOCALFORGE §7).
- `agent/repomap.py` — deterministic repository map + symbol index, Layers 1–3
  (LOCALFORGE §6.2).
- `agent/graph.py` — graph-native execution: typed IR, node + predicate
  registries, validator, and `GraphRuntime`.
- `agent/graph_patterns.py` — graph compiler + 5 pattern builders
  (feature/bug_repair/ui_redesign/refactor/unknown_repo).
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
- `config/model-routing.yaml` — active per-role provider routing. Supports a
  `verifier_model` per provider for the independent verifier (§9).
- `.env.example` — environment-variable examples.
- `requirements.txt` — Python dependencies.
- `Makefile` — tests, linting, and type checking.

## LOCALFORGE capabilities (profile flags + env)

These four spec capabilities are gated so default behavior is unchanged. Profile
flags live in `agent/profiles.py`; env vars opt in per run.

| Capability | Profile flag | Env opt-in | Default |
|---|---|---|---|
| Repository map injection (§6.2) | `repo_map` | — | on for `medium`/`high` |
| Sandbox isolation (Rule 2.2 / Node H) | `sandbox_isolation` | `MARTINUS_SANDBOX_ISOLATION=1` | off |
| Independent verifier (Rule 2.5 / §9) | `independent_verifier` | `MARTINUS_INDEPENDENT_VERIFIER=1` | on for `high` |
| Graph-native execution | `graph_engine` | `MARTINUS_GRAPH_ENGINE=1` | off |
| Permission tiers (§7) | — | `MARTINUS_PERMISSION_L2`, `MARTINUS_PERMISSION_L3`(+`_ALLOW_L3_OVERRIDE`) | L0/L1 allow, L2 allow, L3 deny |
| Structure-aware history compression | — | `MARTINUS_STRUCTURE_COMPRESSION` (default on), `MARTINUS_HISTORY_BUDGET_CHARS` | on |
| Semantic response cache | — | `MARTINUS_RESPONSE_CACHE=1` (+ `MARTINUS_CACHE_DIR`, `MARTINUS_CACHE_MAX_TEMP`, `MARTINUS_CACHE_MAX_ENTRIES`) | off |
| Adaptive profile downgrade (simple tasks) | — | `MARTINUS_ADAPTIVE_PROFILE` (default on) | on |
| Verifier early-exit (simple + gates + tests pass) | — | `MARTINUS_VERIFY_EARLY_EXIT` (default on) | on |

Related env: `MARTINUS_SANDBOX_KEEP_ON_FAILURE=1` keeps a failed run's sandbox for
inspection. The read-only `GET /v1/repomap?workspacePath=&layers=1,2,3` endpoint
returns the deterministic map. When sandbox isolation is on, generated files are
written to an isolated worktree/copy and applied back to the real workspace only
after a verified pass; a failed run leaves the workspace untouched.

**Graph-native execution** (`MARTINUS_GRAPH_ENGINE=1`): `Engine.execute()` compiles
the request into a typed, validated graph (`agent/graph_patterns.compile_graph`)
and runs it via `GraphRuntime` instead of the linear phase loop. The `feature`
pattern maps 1:1 onto the canonical phase order (each `phase.run` node calls the
same `_execute_phase_step` the linear engine uses), so behavior is preserved; a
`graph_compiled` event + per-node `node_*` events fire, and the graph trace is
recorded in the run manifest under `verification.graph`. Roadmap detail lives in
`ROADMAP.md`. Note: the non-feature patterns' specialized nodes (browser/test/
symbol) are stubs until the execution-kernel and Playwright roadmap phases land.

**Token-cost & latency improvements** (research-driven follow-up; all reuse the
existing phase/verifier/gate logic, no new deps):

- *Structure-aware history compression* (`agent/orchestrator._compress_entry_structurally`):
  older phase outputs injected as context are compressed by KEEPING high-signal
  lines (file paths, decisions, signatures, bullets, headings) and DROPPING
  low-signal prose — a deterministic, stdlib analog of LongLLMLingua's
  perplexity-based token retention. Falls back to plain truncation for
  single-line content. On by default; `MARTINUS_STRUCTURE_COMPRESSION=0` reverts
  to blind truncation.
- *Semantic response cache* (`agent/response_cache.py`): disk-backed cache keyed
  on `(provider, model, temperature, path-normalized messages)`. A hit skips the
  model call entirely (zero tokens, near-zero latency) — the largest lever for
  re-runs and retries of the same task in the same workspace. Only caches
  near-deterministic requests (`temperature <= MARTINUS_CACHE_MAX_TEMP`, default
  0.2). Off by default (`MARTINUS_RESPONSE_CACHE=1`); a stale cache could mask a
  fix, so it is opt-in. Workspace paths are normalized out of the key so the same
  task hashes identically across temp dirs.
- *Adaptive profile downgrade* (`Engine.execute`): when the deterministic
  classifier flags a simple greenfield task and the profile runs the heavy
  6-phase pipeline, the run drops to `Discovery → Implementation → Verification`
  for THIS request — Kapoor et al. (arXiv:2407.01502): complex scaffolding costs
  ~55× more for equal/worse quality on simple tasks. Verification advantages
  (gates, behavioral tests, independent verifier) are kept. On by default;
  `MARTINUS_ADAPTIVE_PROFILE=0` disables. Emits a `profile_downgraded` event.
- *Verifier early-exit* (`Engine.execute`): the independent verifier (the single
  most expensive model call — fresh full-context re-read) is SKIPPED when a
  simple task's static gates AND behavioral tests both passed — the deterministic
  evidence is already conclusive. Only skips the extra call; never upgrades a
  failure. On by default; `MARTINUS_VERIFY_EARLY_EXIT=0` disables. Emits a
  `verifier_early_exit` event.

Generated workspaces, memory databases, build output, private development notes,
and evaluation transcripts are intentionally excluded from this frozen source
package.
