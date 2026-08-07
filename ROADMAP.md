# Martinus Engineering Roadmap: The Phase-Governed Adaptive Agent

## North Star

Martinus is evolving from an **"analyze → generate whole files → review → repair"** pipeline into an **"inspect → act → observe real output → adapt → repeat"** engineering system — but it keeps Martinus's explicit phases as **graph controllers/subgraphs**, never dissolving into an unconstrained ReAct loop. The thesis: a *phase-governed adaptive agent* beats a pure runtime (à la Kimi) because it holds seven properties simultaneously: (1) **structured control** — phases are typed graph nodes with declared reads/writes, not one giant prompt; (2) **deterministic-first** — tools and gates run before LLM calls; (3) **evidence-gated completion** — "no verified evidence, no completion claim"; (4) **isolation-by-default** — every writer acts in its own worktree; (5) **risk-tiered permissions** — L0–L3 gate every action; (6) **separated contexts** — planner/executor/verifier never share a context window; (7) **measurable efficiency** — every run is scored by Verified Engineering Efficiency. The graph is the engineering system; prompts are small, typed, role-specific node instructions.

---

## Where We Are Today

Martinus already *is* a fixed linear graph — it just isn't reified as data. Honest snapshot:

| Capability | Status | File refs |
|---|---|---|
| Linear phase engine | Working, but hardcoded `for phase in profile.phases` | `orchestrator.py:1891` `_run_phases`; `profiles.py:17,88` |
| Node dispatch | `if phase == ...` chain, not a registry | `orchestrator.py:2021-2048` `_dispatch_phase` |
| Budgets + cancellation | Working (global only) | `orchestrator.py:181-213` `RunBudget`; `RunAborted` at `:176` |
| Retries / temp / model routing | Working, per-phase | `_call_with_retry:765`; `TemperatureScheduler:228`; `RoleRouting.model_for` via `profiles.py:134,147` |
| File writing | **Whole-file, regex-extracted, last-writer-wins** | `<file>` tag `prompts.py:88-138`; `_FILE_TAG_RE orchestrator.py:92`; `_extract_and_write_files:600-631` |
| Static gates | Syntax/coherence only | `gates.py:722 run_gates`, `:657 check_browser_app_coherence`, `:747 run_workspace_tests` (opt-in) |
| Memory (episodic) | Multi-signal retrieval of prior phase outputs | `ai-service/app/memory.py:590 retrieve`, `_bm25_scores:296`, `_mmr_select:360` |
| Provider layer | Ollama/LMStudio/Claude adapter; **no native tool-calling** | `providers.py:63-85`, `:150,:263,:495` |

### The 4 shipped LOCALFORGE foundations — do NOT re-plan these

| Feature | Module | What it delivers | Default |
|---|---|---|---|
| **Isolated workspace** | `agent/sandbox.py` | git-worktree/copy sandbox (`create_sandbox`), baseline fingerprint (`_snapshot_baseline`), verified `apply_back` w/ conflict detection, `cleanup` | opt-in, one-per-run |
| **Permissions** | `agent/permissions.py` | `RiskLevel L0-L3`, `_ACTION_RISK`, `PermissionPolicy.decide` | wired into sandbox create/apply-back |
| **Repo map** | `agent/repomap.py` | deterministic Layer 1-3 (`build_manifest`, `build_architecture`, `build_symbol_index`, `compact`), `GET /v1/repomap` | injected via `_maybe_prepend_repomap`, medium/high only |
| **Independent verifier** | `agent/orchestrator.py` | `_run_independent_verifier` (fresh context, advisory), `verifier_model` routing, `_parse_verifier_verdict` | advisory: can downgrade, never upgrade a pass |
| **Graph engine (v1)** | `agent/graph.py`, `agent/graph_patterns.py` | typed IR + registry + validator + runtime; 5 patterns; `MARTINUS_GRAPH_ENGINE=1` wraps `_dispatch_phase` | opt-in, feature pattern full; others partial (stub nodes) |

All are backward-compatible and default-off/opt-in.

---

## Target Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  CONTROL PLANE            server.py /v1/tasks/stream, SSE, cancel     │
├─────────────────────────────────────────────────────────────────────┤
│  ACCEPTANCE-CONTRACT      [NEW] agent/contract.py  compile_contract() │
│  COMPILER                 (prose criteria → machine-checkable object) │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE-GOVERNED           agent/graph.py (IR) [v1 DONE] +             │
│  GRAPH ENGINE             agent/graph_patterns.py → GraphRuntime      │
│                           (wraps _dispatch_phase)                     │
├──────────────────────────────┬──────────────────────────────────────┤
│  REPOSITORY INTELLIGENCE     │  MEMORY ENGINE                         │
│  agent/repomap.py [DONE L1-3]│  ai-service/app/memory.py (episodic)   │
│  + [NEW] context_bundle.py   │  (shared scorer, separate stores)      │
├─────────────────────────────────────────────────────────────────────┤
│  MODEL ROUTING            profiles.py RoleRouting [role-based DONE]   │
│                           → [NEW] agent/capabilities.py (capability)  │
├─────────────────────────────────────────────────────────────────────┤
│  EXECUTION KERNEL         [NEW] agent/tools.py  ToolContext +         │
│  (typed tools)            AgentTool registry (Read/Glob/Grep/Edit/    │
│                           Write/Bash/Git/Test/Browser)                │
├─────────────────────────────────────────────────────────────────────┤
│  ISOLATED WORKSPACE       agent/sandbox.py [DONE] worktree/apply_back │
│                           → [NEW] scheduler.py per-writer worktrees   │
├─────────────────────────────────────────────────────────────────────┤
│  EVIDENCE ENGINE          gates.py [static DONE] + [NEW]              │
│                           evaluator.py + browser.py (runtime evidence)│
├─────────────────────────────────────────────────────────────────────┤
│  INDEPENDENT VERIFICATION agent/orchestrator.py _run_independent_     │
│                           verifier [DONE, advisory] → fed contract    │
└─────────────────────────────────────────────────────────────────────┘
```

Every safety guard (permissions, path safety, sandbox) lives in `agent/`. `desktop/` is a pure SSE client — no tool logic ever moves there or the trust boundary splits.

---

## Phase 1 — Execution Foundation (typed tool kernel)

**Goal:** Replace implicit regex side-effects with a registered, permission-gated, typed `AgentTool` set and a tool-result feedback loop. Introduce hash-anchored incremental edits.

**What already exists (do not rebuild):** git-worktree sandbox (`sandbox.py`); risk model (`permissions.py`); deterministic helpers (`repomap` reads, `gates.run_gates/run_workspace_tests`, `sandbox._git`) — unregistered/untyped; path-safety (`_safe_abs_write_path`, `_safe_rel_path`, `_strip_workspace_echo`); safe whole-file write (`_extract_and_write_files`).

**Work items:**
1. **NEW `agent/tools.py`**: `ToolContext` (workspace_path, sandbox_path, `policy`, on_event, `budget`, session_id); `AgentTool` ABC (name/description/args-schema/`risk`/`run`); `ToolRegistry`; concrete `ReadTool/GlobTool/GrepTool/EditTool/WriteTool/BashTool/GitTool/TestTool`.
2. Each tool calls `ctx.policy.decide(<action>)` before acting; add `glob_files/grep_code/apply_edit/run_bash` to `_ACTION_RISK`.
3. **Hash-anchored EditTool** at the `_extract_and_write_files` seam: read current bytes → verify caller hash → apply `old→new` → write. `WriteTool` = existing whole-file path (kept as **first-class fallback**).
4. Reuse `_safe_abs_write_path/_strip_workspace_echo`; consolidate the duplicate in `sandbox.py`.
5. Construct `ToolContext` once in `_run_phase`/`_dispatch_phase`.
6. Tool-invocation grammar in prompts (`<edit path hash>...</edit>` alongside `<file>`); **bump `PROMPT_VERSION`**. Tool loop parses invocations and feeds `ToolResult`s back — the observe→act loop. Providers unchanged (parse from prose).
7. Wrap `gates` as `TestTool`; `sandbox._git` as `GitTool`.

**Risks:** weak local models regress to fences — keep WriteTool fallback, never remove; two conflict schemes (per-edit hash vs sandbox `(mtime,size)`) — reconcile; `decide()` on hot write path could deny silent writes — classify sandbox writes as `write_sandbox_file` (L1); `_build_verification_summary`/success gate derive success from `generated_files` — update in lockstep; **BashTool = arbitrary-exec hole** — narrow allowlist.

**Exit criteria:** every mutation flows through a registered tool that called `decide()`; EditTool refuses on hash mismatch; a phase can invoke Read/Grep and receive typed results; golden-trace test shows WriteTool output identical to today.

---

## Phase 2 — Verification Engine (acceptance-contract compiler)

**Goal:** Turn orphaned prose criteria into a compiled, machine-evaluated `AcceptanceContract` that gates completion. Enforce **"no verified evidence, no completion claim."**

**What already exists:** criteria authored as prose (Clarification) but never parsed; deterministic success = files-present + gates; independent verifier (self-grades raw goals, advisory); confined-subprocess recipe (`run_workspace_tests`).

**Work items:**
1. **NEW `agent/contract.py`**: `AcceptanceCriterion(id, category, description, check{kind:command|http|log_scan|browser|file|gate, spec}, expected, status, evidence)` + `AcceptanceContract`. `compile_contract(...)` parses numbered binary lines defensively.
2. Compile seam after Clarification; store on `SessionState` (new `contract` field). No-Clarification profiles compile from task+goals.
3. **NEW `agent/evaluator.py`**: `evaluate_contract(...)` runs check runtimes — command (confined subprocess), http (`httpx`), log_scan (regex), gate (`run_gates`). Per-check timeout, counts against `RunBudget`, runs inside sandbox before apply-back.
4. Success gate additionally requires `contractPassed` (mandatory criteria pass with evidence). File-presence + gates remain necessary.
5. Verifier upgrade: inject compiled contract + machine evidence; LLM judges only residual/subjective criteria.
6. Extend manifest + `HardwareProfile` (`acceptance_contract` flag).

**Risks:** parser brittleness on weak models — degrade to file+gate-only, never fabricate a pass; executing model-authored checks = arbitrary exec — confinement + sandbox; gate-tightening flips passing runs — intended, gate behind flag; circularity — verifier adjudicates, functional/security need executed evidence; regression category needs a persisted baseline — defer; full browser deferred to Phase 3.

**Exit criteria:** a run cannot claim success with zero criteria evaluated; manifest shows per-criterion evidence; verifier receives evidence not raw goals; parse failure degrades safely.

---

## Phase 3 — Browser / Visual Intelligence (runtime instrument)

**Goal:** Move Playwright into `agent/` as an in-process runtime instrument feeding the repair loop; launch live apps, capture screenshots/console/network/a11y, emit `BrowserDefect`s.

**What already exists:** Playwright is a batch, out-of-process, `file://`-only harness in `evaluator/main.py` (no screenshots/console/network); repair loop (`_run_gate_repair_loop`); static visual gates (`check_browser_app_coherence`, `_find_stale_dom_references`); sandbox + executable-verification.

**Work items:**
1. **NEW `agent/browser.py`**: `inspect_app(...) -> BrowserReport`, lazy import so missing Playwright degrades to skipped.
2. **NEW `agent/runtime.py`** (Node M): read `repomap manifest.commands['run']` to start a dev server, poll readiness, yield base URL, tear down — reuse allowlisted subprocess/env/timeout. http not file://.
3. `BrowserDefect` shaped like `GateResult` so it slots into `failed()`/repair.
4. Hook `_run_gate_repair_loop`: browser pass extends `fails`; reuse `build_repair_prompt` + `_extract_and_write_files`. Gate behind `browser_inspection` flag.
5. Feed evidence into verification dict; stream as `browser_evidence` SSE; persist paths not bytes.
6. **Visual Critic** (Node O) = fresh-context model call modeled on `_run_independent_verifier`, fed screenshots/a11y.

**Risks:** Playwright + chromium heavy non-stdlib deps — stay optional/lazy; sync Playwright blocks the thread — hard timeout + budget; launching a live app runs model code w/ network — must run in sandbox under permissions; browser-defect→repair can loop — respect §2.6 (visual cycles: 3) + Loop Progress Rule.

**Exit criteria:** agent launches a generated web app over http in the sandbox, captures a screenshot + console errors, converts a runtime failure into a `BrowserDefect` that drives one repair iteration and terminates on no-new-evidence.

---

## Phase 4 — Repository Intelligence (token-budgeted context bundle)

**Goal:** Extend Layer 1-3 repomap into Layer 4 — import/call graphs, test association, git co-change, multi-signal symbol ranking, optional dense embeddings, and a merged token-budgeted context bundle.

**What already exists:** repomap Layers 1-3; multi-signal scorer for episodic memory only (`_bm25_scores`, `_cosine_similarity`, `_mmr_select`, `_composite_score`); token trimmer (`_trim_to_context_budget` for memory; `repomap.compact` byte-only) — they never merge. **Critical gap:** memory retrieves prior phase outputs, not repo code/symbols.

**Work items:**
1. Extend `_index_python` to collect `ast.Import/ImportFrom` (import graph) + `ast.Call` names (coarse call graph); add `build_import_graph()/build_call_graph()`.
2. `build_cochange(root)` via `git log --name-only`; degrade for no-history repos.
3. Test association by filename convention.
4. **NEW `agent/context_bundle.py`** (Node C): rank repomap symbols/files against the task query using the same composite+MMR math — lift the scorer into a **shared** module. Produce a token-budgeted bundle.
5. Wire the bundle into `_maybe_prepend_repomap` so phases get task-relevant code, not the whole map.
6. Optional dense embeddings via Ollama `/api/embeddings`, keeping the hash-vector fallback.

**Risks:** ast graph misses dynamic imports; JS regex lossy; git co-change needs history — degrade; keep stores separate (share only the scorer); dense embeddings add latency — hash fallback stays; **bundle must never drop the request-named file**; changing repomap output shape breaks its tests.

**Exit criteria:** a task query yields a token-budgeted bundle (file/lines/symbol/why/deps/dependents/tests) that always includes request-named files, fits budget, replaces whole-map injection for medium/high.

---

## Phase 5 — Parallel Graph Execution (safe concurrency)

**Goal:** Resource-locking scheduler + one-worktree-per-writer parallel workers, starting with read-only explorers, then conflict-aware writers with integration/merge verification.

**What already exists:** only concurrency primitive is a process-wide `BoundedSemaphore` default 1 (`providers.py`) for 24GB-unified-memory safety; per-request thread-pool + cancel Event (runs serialize on the semaphore); sandbox isolation ~80% of one-worktree-per-writer but **one worktree per run**; permissions classify explorers L0, apply-back L2.

**Work items:**
1. **NEW `agent/scheduler.py`**: `ResourceScheduler.acquire/release` wrapping (not replacing) `_MODEL_CALL_SEMAPHORE`; model-slot + file-ownership locks keyed by `(workspace, rel_path)`.
2. Fan-out seam between Architecture and Implementation: `_run_parallel_workers` under a `parallel_workers` flag; each worker gets its own sandbox on a distinct branch; integrate via `apply_back` with cross-worktree reconciliation.
3. Start with a read-only explorer calling `repomap.build_repomap` only (L0).
4. Extend `ApplyResult.conflicts` into a pre-flight file-ownership map.

**Risks:** local model thrash — keep model-slot concurrency at 1 for local providers while parallelizing independent resources; worktrees need resolvable HEAD, non-git dirs fall back to full copies (disk); cross-worktree merge unsolved — ownership/merge policy; self-modification warning more dangerous with multiple writers.

**Exit criteria:** ≥2 read-only explorers run concurrently at L0 with no model-slot contention on local providers; a writing worker acquires a file-ownership lock, another is blocked from the same path, integration merge-verifies before apply-back.

---

## Phase 6 — Provider / Cost Optimization + VEE Benchmark

**Goal:** Capability-based routing, local/cloud hybrid, prompt caching, context deltas, per-node telemetry, and the VEE comparison benchmark (capstone).

**What already exists:** role-based routing (provider run-global; `routing_policy` YAML thresholds are dead config); per-run telemetry (`Router.usage_totals`, `result.usage`) — no per-node/cost/latency/cache breakdown; benchmark harness (quality-only, single config, ignores `result.usage`).

**Work items (sequence: telemetry+router → caching/deltas → benchmark):**
1. **NEW `agent/capabilities.py`**: `NodeRequirements` + `ModelCatalog` + `CapabilityRouter.select(...)` returning cheapest compatible; thread per-phase provider through `_dispatch_phase`/`_call_with_retry`.
2. Telemetry: extend `ChatResponse` (latency, cache tokens), `PhaseResult`, `RunResult`; emit in manifest + `run_completed`.
3. Prompt caching: stable-prefix marker on `ChatRequest`; split system instruction; SDK client applies `cache_control`.
4. VEE benchmark arms: parameterize the evaluator with `--arm {claude-alone, martinus-local, martinus-hybrid}`; capture `result.usage`; arm-keyed matrix. **Define "Verified" via `verification.passed`, not the LLM judge.**

**Risks:** real Anthropic SDK = cloud dep + cost; caching differs between shell-out CLI and SDK; per-node provider switching breaks the one-provider-per-run invariant (wide blast radius); hand-rolled YAML parser can't express a capability catalog; VEE gameable unless anchored on deterministic evidence; claude-alone arm must use the same deterministic executors.

**Exit criteria:** router selects cheapest-compatible model per node with per-node telemetry in the manifest; prompt caching reduces prompt tokens on the Claude arm; benchmark produces a Claude-alone vs local vs hybrid matrix keyed on `verification.passed`.

---

## Graph IR + Runtime (wrap, don't rewrite)

The engine is already a fixed linear graph as data: `profile.phases` (list) + `_dispatch_phase` (if/elif) + `_run_phases` (for-loop). Reify it **additively**, guaranteeing behavior parity. **v1 of this is shipped** in `agent/graph.py` + `agent/graph_patterns.py` behind `MARTINUS_GRAPH_ENGINE=1`; later steps extend it.

| Step | Change | Phase |
|---|---|---|
| IR module | `agent/graph.py`: `Node/Edge/LoopContract/Graph/GraphState` — **v1 DONE** | 1 |
| Compiler | `agent/graph_patterns.py`: `compile_graph(...)` emitting the linear pipeline as default — **v1 DONE** | 1 |
| Driver | `GraphRuntime.run` wrapping `_dispatch_phase` — **v1 DONE** | 1 |
| Conditional edges | Reuse `_parse_verifier_verdict`/`_infer_outcome` as predicates — **v1 DONE (registry)** | 2 |
| Loop contracts | Lift `_run_gate_repair_loop`/`_run_recovery_if_needed` into `LoopContract` subgraph nodes; anchor progress predicate to deterministic evidence deltas | 2 |
| Typed blackboard | `GraphState` blackboard — **v1 DONE**; keep `entries[]` in parallel | 2 |
| Per-node permissions | Each Node declares RiskLevel actions and calls `decide()` — **v1 registry has risk** | 2 |
| Parallelism | Worklist driver dispatches disjoint-`writes` nodes concurrently via the scheduler | 5 |
| Checkpoint/resume | Serialize `{completed_nodes, blackboard}` beside the manifest | 5 |

**Separated contexts:** planner/executor/verifier never share a window. `_phase_role` already maps phases → planner/implementer/reviewer/verifier. The verifier node reads only artifacts (`_run_independent_verifier` uses a fresh context + snapshot). Formalize each node's `reads` set so this is enforced structurally.

**Guardrails:** golden-trace test (graph output == `_run_phases` output per profile); graphLoop.md node names canonical; deterministic nodes cost 0 model calls; per-node budgets default to the global `RunBudget`.

---

## Verified Engineering Efficiency (VEE)

```
VEE = weighted_acceptance / (normalized_cost × change_surface × retries)
```

| Term | Source (reuse) | Notes |
|---|---|---|
| `weighted_acceptance` | `verification.passed` + `contractPassed` (Phase 2), gates | **Deterministic only** — never the LLM judge |
| `normalized_cost` | `Router.usage_totals`, per-node telemetry (Phase 6), price table | tokens × price; local ≈ $0 but hardware-bound |
| `change_surface` | edits from EditTool/WriteTool (Phase 1) | hash-anchored edits give precise surface |
| `retries` | `RunBudget.model_calls`, `PhaseResult.attempts`, repair iterations | already tracked per-phase |

Comparison matrix (evaluator, arm-keyed): Claude-alone / Martinus-local / Martinus-hybrid × {Verified, Tokens, Cost, Wall-clock, Attempts, VEE}. Fairness: all arms scored by the same deterministic executors, fixed hardware, seeds, repeats.

---

## Non-Goals / Sequencing Guardrails

- **Do NOT** build MCP marketplaces, cron/scheduling, OAuth, or IDE protocols before these six phases.
- **Do NOT** dissolve phases into an unconstrained ReAct loop — the compiled-linear graph is always the default.
- **Do NOT** remove whole-file WriteTool — it is the local-model fallback.
- **Do NOT** build tool logic in `desktop/` — all safety guards live in `agent/`.
- **Do NOT** enable parallelism before the sequential graph is golden-trace-proven; keep model-slot concurrency at 1 for local providers.
- **Do NOT** add heavy deps (Playwright, PyYAML, Anthropic SDK, dense embeddings) as hard requirements — keep them lazy/optional.
- **Do NOT** let the acceptance contract or independent verifier fabricate a pass on parse failure — degrade to file+gate-only and flag HUMAN_REVIEW.

---

## Immediate Next Step

**Build the typed execution kernel (`agent/tools.py`) and route the existing whole-file write path through it — Phase 1, items 1-3.** This is the single load-bearing primitive every later phase depends on (context delta, model routing, verification graph, parallel workers). It plugs in at `_extract_and_write_files` (WriteTool core + hash-anchored EditTool), the extract+write block in `_run_phases` (parse tool invocations, feed `ToolResult`s back), `_run_phase`/`_dispatch_phase` (construct `ToolContext` once), and `permissions._ACTION_RISK` (new action names; every `tool.run()` calls `decide()`), with a `<edit path hash>` grammar added to `prompts.py` and `PROMPT_VERSION` bumped. This turns the shipped LOCALFORGE modules from scattered helpers into a registered, permission-gated, ToolContext-driven kernel — the moment Martinus stops "generating files" and starts "acting and observing."
