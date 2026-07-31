"""
agent/orchestrator.py — Multi-phase coding agent engine (Python replacement for Go orchestrator).

Implements all 11 MVCA algorithms from the thesis:
  Alg 3.1  Bounded Retry with Self-Correction (callWithRetry)
  Alg 4.1  Step-Back Architecture Prompting (run_architecture_with_step_back)
  Alg 4.2  Candidate-Design Exploration with Deterministic Judge (run_architecture_with_candidates)
  Alg 4.3  Tree-of-Code Implementation (run_tree_of_code)
  Alg 4.4  Single-Pass Self-Critique Revision (run_review_with_self_critique)
  Alg 4.5  ReAct Verification with Self-Debug Inner Loop (standard phase with ReAct prompt)
  Alg 4.6  Post-Run Reflection (run_reflection)
  Alg 4.7  ExpeL-Style Insight Consolidation (ai-service /v1/insights)
  Alg 4.8  Rolling Hierarchical Summarisation (apply_hierarchical_summary)
  Alg 4.9  Workspace Reference Validation (validate_workspace_references)
  Alg 4.10 Composite Top-k Memory Retrieval (ai-service /v1/memory/retrieve)
  Alg 4.11 MVCA Profile-Driven Execution Loop (execute)
"""
from __future__ import annotations

import logging
import os
import re
import time
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import httpx

from .profiles import HardwareProfile, RoleRouting, resolve_profile
from .prompts import (
    FALLBACK_RETRY,
    PROMPT_VERSION,
    build_prompt,
    build_repair_prompt,
    build_system_instruction,
    _PHASE_SYSTEM,
    _truncate,
)
from .providers import ChatRequest, DeterministicProviderError, Message, Router
from . import gates as _gates

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HISTORY_WINDOW = 6
HIERARCHICAL_SUMMARY_THRESHOLD = 10
RETRIEVAL_TOP_K = 6
MIN_RETRIEVAL_SCORE = 0.12
CONTEXT_BUDGET_TOKENS = int(os.getenv("CONTEXT_BUDGET_TOKENS", "3072"))
CHARS_PER_TOKEN = 4

# Run-level budget defaults (env-overridable). A run that exceeds either
# bound stops cleanly with a truthful failure status instead of running
# open-ended on shared local hardware.
RUN_MAX_SECONDS = float(os.getenv("MARTINUS_RUN_MAX_SECONDS", "3600"))
RUN_MAX_MODEL_CALLS = int(os.getenv("MARTINUS_RUN_MAX_MODEL_CALLS", "40"))
RETRY_BACKOFF_BASE_SECONDS = float(os.getenv("MARTINUS_RETRY_BACKOFF_SECONDS", "2"))

PHASE_TEMPERATURES: dict[str, float] = {
    "Discovery": 0.1,
    "Clarification": 0.1,
    "Architecture": 0.3,
    "Implementation": 0.1,
    "Review": 0.15,
    "Verification": 0.0,
    "stepback": 0.6,
    "judge": 0.0,
    "critique": 0.2,
}

STRONG_SUCCESS = ["all tests passed", "verification successful", "status: success",
                  "build succeeded", "all checks pass", "outcome: success"]
STRONG_FAILURE = ["all tests failed", "build failed", "status: failure",
                  "compilation error", "cannot proceed", "outcome: failure"]
WEAK_SUCCESS = ["passed", "green", "ok", "ready", "complete", "verified"]
WEAK_FAILURE = ["failed", "failure", "error", "unable", "regression", "broken", "missing"]

_KNOWN_EXTENSIONS = {".go", ".py", ".ts", ".tsx", ".js", ".css", ".html", ".md",
                     ".yaml", ".yml", ".json", ".toml", ".rs", ".java", ".cpp",
                     ".c", ".h", ".sh"}
_REQUESTED_FILE_RE = re.compile(
    r"(?<![\w/-])([\w./-]+\.(?:"
    + "|".join(re.escape(ext.lstrip(".")) for ext in _KNOWN_EXTENSIONS)
    + r"))(?![\w/-])",
    re.IGNORECASE,
)
_FILE_TAG_RE = re.compile(
    r"<file\s+path\s*=\s*([\"'“”‘’])(?P<path>.+?)(?:\1|[\"'“”‘’])\s*>(?P<content>.*?)</file>",
    re.DOTALL,
)

# Fallback: fenced code blocks with a preceding filename hint (heading or inline
# `filename` line). A well-behaved model uses <file> tags; a 7B local model often
# regresses to markdown fences. This fallback lets us recover instead of writing
# nothing, but only when the model gave us a filename to anchor on.
#
#   ### index.html            OR    **index.html**            OR    `index.html`
#   ```html                          ```                              ```html
#   ...                              ...                              ...
#   ```                              ```                              ```
_FENCED_FILE_RE = re.compile(
    r"(?:^|\n)"
    r"(?:[#>*`_\s-]*"                              # heading / list / emphasis noise
    r"(?:file\s*[:\-]\s*)?"                         # optional "File:" prefix
    r"`?(?P<path>[\w./-]+\.(?:"
    + "|".join(re.escape(ext.lstrip(".")) for ext in _KNOWN_EXTENSIONS)
    + r"))`?"                                       # captured filename
    r"[^\n]*)\n+"                                   # rest of hint line
    r"```[\w+.-]*\n"                                # opening fence + optional lang
    r"(?P<content>.*?)"                              # code body (non-greedy)
    r"```",
    re.DOTALL | re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SessionEntry:
    phase: str
    content: str
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"phase": self.phase, "content": self.content, "created_at": self.created_at}


@dataclass
class SessionState:
    session_id: str = ""
    task: str = ""
    workspace_path: str = ""
    goals: list[str] = field(default_factory=list)
    provider: str = ""
    max_lines: int = 2000
    entries: list[SessionEntry] = field(default_factory=list)


@dataclass
class PhaseResult:
    phase: str
    output: str
    attempts: int
    provider: str
    model: str
    prompt_version: str
    retrieved_count: int
    started_at: float
    finished_at: float


@dataclass
class RunResult:
    session_id: str
    provider: str
    model: str
    hardware_profile: str
    phase_results: list[PhaseResult] = field(default_factory=list)
    summary: str = ""
    reflection: str = ""
    # Truthful, evidence-based final status. `success` is derived from the
    # deterministic verification summary (files on disk + static gates), never
    # from model prose. `status` is one of success|failure|aborted.
    success: bool = False
    status: str = "failure"
    failure_reason: str = ""
    usage: dict = field(default_factory=dict)


class RunAborted(RuntimeError):
    """Raised internally when the run budget is exhausted or the caller
    cancelled; the run loop converts it into a truthful aborted status."""


class RunBudget:
    """Single explicit run budget: wall-clock deadline + model-call cap,
    plus an optional caller-supplied cancellation check."""

    def __init__(
        self,
        max_seconds: Optional[float] = None,
        max_model_calls: Optional[int] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        # Defaults resolve at call time so env/monkeypatch overrides apply.
        if max_seconds is None:
            max_seconds = RUN_MAX_SECONDS
        if max_model_calls is None:
            max_model_calls = RUN_MAX_MODEL_CALLS
        self.deadline = time.monotonic() + max(1.0, max_seconds)
        self.max_model_calls = max(1, max_model_calls)
        self.model_calls = 0
        self._cancel_check = cancel_check

    def check(self, where: str) -> None:
        if self._cancel_check and self._cancel_check():
            raise RunAborted(f"cancelled by caller at {where}")
        if time.monotonic() > self.deadline:
            raise RunAborted(f"run wall-clock budget exhausted at {where}")
        if self.model_calls >= self.max_model_calls:
            raise RunAborted(
                f"run model-call budget ({self.max_model_calls}) exhausted at {where}"
            )

    def count_model_call(self) -> None:
        self.model_calls += 1


def _detect_truncated_file_blocks(output: str) -> int:
    """Return how many `<file` openings lack a closing tag — the signature of
    a context-limit / max_tokens truncation that would otherwise silently drop
    files."""
    opens = len(re.findall(r"<file\s+path\s*=", output))
    closes = output.count("</file>")
    return max(0, opens - closes)


# ---------------------------------------------------------------------------
# Temperature scheduling  (Algorithm N.1 — Adaptive Temperature Scheduling)
# ---------------------------------------------------------------------------

class TemperatureScheduler:
    STEP = 0.05
    MAX_BUMP = 0.20

    def __init__(self) -> None:
        self._failures: dict[str, int] = {}

    def temperature(self, phase: str) -> float:
        base = PHASE_TEMPERATURES.get(phase, 0.1)
        failures = self._failures.get(phase, 0)
        if failures <= 0:
            return base
        bump = min(self.STEP * failures, self.MAX_BUMP)
        return min(base + bump, 1.0)

    def record_success(self, phase: str) -> None:
        if self._failures.get(phase, 0) > 0:
            self._failures[phase] -= 1

    def record_failure(self, phase: str) -> None:
        self._failures[phase] = self._failures.get(phase, 0) + 1


# ---------------------------------------------------------------------------
# AI-service memory client
# ---------------------------------------------------------------------------

class AIMemoryClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def upsert(self, session_id: str, content: str, tags: list[str], **kwargs) -> str:
        try:
            payload = {"session_id": session_id, "content": content, "tags": tags, **kwargs}
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(f"{self.base_url}/v1/memory/upsert", json=payload)
                resp.raise_for_status()
                return resp.json().get("id", "")
        except Exception as exc:
            logger.warning("memory upsert failed: %s", exc)
            return ""

    def retrieve(self, session_id: str, query: str, top_k: int,
                 workspace_path: str = "", include_cross_session: bool = True,
                 current_phase: str = "") -> list[dict]:
        try:
            payload = {
                "session_id": session_id, "query": query, "top_k": top_k,
                "workspace_path": workspace_path, "include_cross_session": include_cross_session,
                "current_phase": current_phase,
            }
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(f"{self.base_url}/v1/memory/retrieve", json=payload)
                resp.raise_for_status()
                return resp.json().get("items", [])
        except Exception as exc:
            logger.warning("memory retrieve failed: %s", exc)
            return []

    def reflect(self, session_id: str, workspace_path: str,
                task: str, transcript: str, outcome: str) -> str:
        try:
            payload = {
                "session_id": session_id, "workspace_path": workspace_path,
                "task": task, "transcript": transcript, "outcome": outcome,
            }
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(f"{self.base_url}/v1/reflect", json=payload)
                resp.raise_for_status()
                return resp.json().get("lesson", "")
        except Exception as exc:
            logger.warning("reflect failed: %s", exc)
            return ""


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _infer_outcome(output: str) -> str:
    text = output.lower().strip()
    if not text:
        return "unknown"
    strong_s = sum(1 for p in STRONG_SUCCESS if p in text)
    strong_f = sum(1 for p in STRONG_FAILURE if p in text)
    if strong_s > strong_f:
        return "success"
    if strong_f > strong_s:
        return "failure"
    weak_s = min(sum(1 for p in WEAK_SUCCESS if p in text), 3)
    weak_f = min(sum(1 for p in WEAK_FAILURE if p in text), 3)
    if weak_s > 0 and weak_f == 0:
        return "success"
    if weak_f > 0 and weak_s == 0:
        return "failure"
    return "partial"


def _trim_to_context_budget(items: list[dict], budget_tokens: int) -> list[dict]:
    if budget_tokens <= 0 or not items:
        return items
    total = 0
    for i, item in enumerate(items):
        estimated = len(item.get("content", "")) // CHARS_PER_TOKEN
        if total + estimated > budget_tokens:
            return items[:i]
        total += estimated
    return items


def _strip_workspace_echo(rel_path: str, workspace_path: str) -> str:
    """Strip echoed workspace-path prefix components from a model-supplied rel_path.

    Strong models sometimes helpfully prepend the workspace path components to
    the file tag (e.g. workspace `/tmp/foo/task_1` → tag `task_1/string_utils.py`).
    Without stripping, we write the file into a self-nested subdirectory and
    the task runner fails to find it. Only strips a strict PREFIX match to
    avoid altering intentional nested paths like `src/foo/bar.py`.
    """
    if not rel_path or not workspace_path:
        return rel_path
    rel_parts = rel_path.split("/")
    ws_parts = [p for p in os.path.normpath(workspace_path).split(os.sep) if p]
    # Try progressively longer echoed prefixes: e.g. if the workspace ends in
    # ("workspace_test", "task_1"), also try matching just ("task_1",).
    for suffix_len in range(min(len(ws_parts), len(rel_parts) - 1), 0, -1):
        if rel_parts[:suffix_len] == ws_parts[-suffix_len:]:
            return "/".join(rel_parts[suffix_len:])
    return rel_path


def _safe_rel_path(rel_path: str) -> Optional[str]:
    rel_path = rel_path.strip().replace("\\", "/").lstrip("/")
    if not rel_path or ".." in rel_path.split("/"):
        return None
    # Reject Windows drive letters and other colon-scoped paths that would
    # otherwise create a literal "C:" directory under the workspace.
    if ":" in rel_path:
        return None
    # Reject URL schemes / obvious injection attempts.
    if rel_path.startswith(("~", "//")):
        return None
    return rel_path


def _safe_abs_write_path(workspace_path: str, rel_path: str) -> Optional[str]:
    """Resolve rel_path under workspace_path and refuse anything that escapes.

    Symlinks in the workspace can point elsewhere, so we resolve both sides and
    require the target to live under the resolved workspace. Returns None when
    the path is unsafe.
    """
    safe = _safe_rel_path(rel_path)
    if not safe:
        return None
    try:
        base = os.path.realpath(workspace_path)
        target = os.path.realpath(os.path.join(base, safe))
    except OSError:
        return None
    # os.path.commonpath treats "/a" vs "/ab" correctly (unlike startswith).
    try:
        if os.path.commonpath([base, target]) != base:
            return None
    except ValueError:  # e.g. mixed drives on Windows
        return None
    return target


def _iter_file_tags(output: str) -> list[tuple[str, str]]:
    files: list[tuple[str, str]] = []
    seen: set[str] = set()
    for match in _FILE_TAG_RE.finditer(output):
        rel_path = _safe_rel_path(match.group("path"))
        if rel_path and rel_path not in seen:
            seen.add(rel_path)
            files.append((rel_path, match.group("content")))
    # Fenced-code fallback: only fires when the primary <file> tag extraction
    # yielded nothing, so it cannot silently override a correctly-formed run.
    if files:
        return files
    for match in _FENCED_FILE_RE.finditer(output):
        rel_path = _safe_rel_path(match.group("path"))
        if rel_path and rel_path not in seen:
            seen.add(rel_path)
            files.append((rel_path, match.group("content")))
    return files


def _extract_requested_files(task: str) -> list[str]:
    requested: list[str] = []
    for match in _REQUESTED_FILE_RE.finditer(task):
        rel_path = _safe_rel_path(match.group(1))
        if rel_path and rel_path not in requested:
            requested.append(rel_path)
    return requested


def _missing_requested_files(workspace_path: str, requested_files: list[str]) -> list[str]:
    missing: list[str] = []
    for rel_path in requested_files:
        full_path = os.path.join(workspace_path, rel_path)
        if not os.path.exists(full_path):
            missing.append(rel_path)
    return missing


def _build_verification_summary(
    workspace_path: str,
    requested_files: list[str],
    generated_files: list[str],
) -> dict[str, Any]:
    """Return only observable post-run evidence for UI/reporting."""
    unique_generated = sorted(set(generated_files))
    missing_requested = (
        _missing_requested_files(workspace_path, requested_files)
        if workspace_path and requested_files
        else []
    )
    gate_results = (
        _gates.run_gates(workspace_path, unique_generated)
        if workspace_path and unique_generated
        else []
    )
    return {
        "passed": bool(unique_generated)
        and not missing_requested
        and all(gate.ok for _, gate in gate_results),
        "generatedFiles": unique_generated,
        "requestedFiles": requested_files,
        "missingFiles": missing_requested,
        "checks": [
            {"file": rel, **gate.as_dict()}
            for rel, gate in gate_results
        ],
    }


# When the task never names concrete files (e.g. "build me a browser calculator")
# but the implementation phase still wrote zero files, we infer the minimal set
# a professor-facing demo would need so the recovery pass has something to aim
# at. Deliberately narrow: only fires on unambiguous cues so we never invent an
# irrelevant scaffold for a CLI/backend task.
_BROWSER_APP_HINTS = (
    "calculator", "landing page", "web app", "browser", "webpage", "website",
    "html/css/js", "html + css", "html and css", "html, css",
)
_CLI_HINTS = ("cli ", "command line", "command-line", "backend", "server", "api", "script")


def _infer_expected_files(task: str) -> list[str]:
    lowered = task.lower()
    if any(h in lowered for h in _CLI_HINTS) and not any(h in lowered for h in _BROWSER_APP_HINTS):
        return []
    if any(h in lowered for h in _BROWSER_APP_HINTS):
        return ["index.html", "styles.css", "app.js", "README.md"]
    return []


# Total byte budget for the workspace-snapshot block. Kept well below the
# small-model context ceiling so Review and Verification still have room for
# their own analysis on top.
_WORKSPACE_SNAPSHOT_BUDGET = 20_000
_WORKSPACE_SNAPSHOT_PER_FILE_CAP = 8_000


def _build_workspace_snapshot(
    workspace_path: str,
    generated_files: list[str],
    max_total_bytes: int = _WORKSPACE_SNAPSHOT_BUDGET,
) -> str:
    """Read the actual on-disk contents of files this run has generated so far
    and format them for injection into Review / Verification prompts.

    Returns an empty string when there's no workspace or no files. Files are
    de-duplicated preserving first-seen order. Each file is capped at
    _WORKSPACE_SNAPSHOT_PER_FILE_CAP; if the remaining files no longer fit in
    the total budget they are listed by name only so the model still knows
    they exist. This is the mechanism that lets Review and Verification
    grade against reality instead of grading their own memory of what the
    model said it wrote."""
    if not workspace_path or not generated_files:
        return ""
    seen: set[str] = set()
    ordered: list[str] = []
    for f in generated_files:
        if f and f not in seen:
            seen.add(f)
            ordered.append(f)

    parts: list[str] = ["=== WORKSPACE SNAPSHOT (on-disk contents) ==="]
    remaining = max_total_bytes
    truncated: list[str] = []
    for rel in ordered:
        abs_path = os.path.join(workspace_path, rel)
        try:
            size = os.path.getsize(abs_path)
        except OSError:
            continue
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read(_WORKSPACE_SNAPSHOT_PER_FILE_CAP)
        except OSError:
            continue
        note = ""
        if size > _WORKSPACE_SNAPSHOT_PER_FILE_CAP:
            note = f" (truncated at {_WORKSPACE_SNAPSHOT_PER_FILE_CAP} of {size} bytes)"
        block = f"\n--- {rel}{note} ---\n{content}\n"
        if len(block) > remaining:
            truncated.append(rel)
            continue
        parts.append(block)
        remaining -= len(block)
    if truncated:
        parts.append(
            "\n--- files omitted due to snapshot budget (listed but not shown) ---\n"
            + "\n".join(f"  - {p}" for p in truncated)
            + "\n"
        )
    parts.append("=== END WORKSPACE SNAPSHOT ===\n")
    return "".join(parts)


def _extract_and_write_files(workspace_path: str, output: str) -> list[str]:
    written: list[str] = []
    tagged = _iter_file_tags(output)
    for rel_path, content in tagged:
        # If the model echoed workspace path components into the file tag
        # (a common failure mode when the workspace name is in the prompt),
        # strip that prefix before resolving. Prevents writes into
        # <workspace>/<workspace-name>/<...>/<file>.
        cleaned_rel = _strip_workspace_echo(rel_path, workspace_path)
        # Reject path traversal + symlink escape + Windows drive prefixes.
        full_path = _safe_abs_write_path(workspace_path, cleaned_rel)
        if not full_path:
            logger.warning("refused to write unsafe path %r under %s", rel_path, workspace_path)
            continue
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            written.append(cleaned_rel)
        except OSError as exc:
            logger.warning("failed to write %s: %s", full_path, exc)
    if not written:
        # No files emitted — this is the single most common cause of "the agent
        # only wrote a README" complaints. Log it loudly so it shows up in the
        # orchestrator log and in the manifest without silently succeeding.
        logger.warning(
            "phase output produced no <file> tags nor fenced-code fallbacks; "
            "workspace=%s output_length=%d",
            workspace_path,
            len(output),
        )
    return written


def _validate_workspace_references(workspace_path: str, output: str) -> str:
    tagged_paths = [path for path, _content in _iter_file_tags(output)]
    path_pattern = re.compile(
        r"(?<![\w./-])[\w./-]+\.(?:"
        + "|".join(re.escape(ext.lstrip(".")) for ext in _KNOWN_EXTENSIONS)
        + r")(?![\w.-])"
    )
    inline_paths = [
        match.group(0)
        for match in path_pattern.finditer(output)
        if "<file" not in output[max(0, match.start() - 24):match.start()]
    ]
    candidates = tagged_paths + inline_paths
    missing = []
    for rel in set(candidates):
        rel_path = _safe_rel_path(rel)
        if not rel_path:
            continue
        full = os.path.join(workspace_path, rel_path)
        if not os.path.exists(full):
            missing.append(rel_path)
    if not missing:
        return ""
    listed = "\n".join(f"  - {p}" for p in sorted(missing))
    return f"\n\n[WORKSPACE CHECK] The following referenced paths do not exist:\n{listed}"


def _write_run_manifest(
    workspace_path: str,
    state: SessionState,
    result: RunResult,
    generated_files: list[str],
    verification: Optional[dict] = None,
) -> None:
    if not workspace_path:
        return
    os.makedirs(workspace_path, exist_ok=True)
    manifest_path = os.path.join(workspace_path, ".martinus-run.json")
    summary_path = os.path.join(workspace_path, "MARTINUS_SESSION.md")
    payload = {
        "session_id": state.session_id,
        "prompt_version": PROMPT_VERSION,
        "provider": result.provider,
        "model": result.model,
        "hardware_profile": result.hardware_profile,
        "task": state.task,
        "goals": state.goals,
        "status": result.status,
        "success": result.success,
        "failure_reason": result.failure_reason,
        "usage": result.usage,
        "verification": verification or {},
        "budget": {
            "max_seconds": RUN_MAX_SECONDS,
            "max_model_calls": RUN_MAX_MODEL_CALLS,
        },
        "generated_files": sorted(set(generated_files)),
        "phases": [
            {
                "phase": phase.phase,
                "model": phase.model,
                "attempts": phase.attempts,
                "retrieved_count": phase.retrieved_count,
                "duration_seconds": round(phase.finished_at - phase.started_at, 3),
            }
            for phase in result.phase_results
        ],
        "summary": result.summary,
        "reflection": result.reflection,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("# Martinus Session\n\n")
            f.write(f"- Session ID: `{state.session_id}`\n")
            f.write(f"- Prompt version: `{PROMPT_VERSION}`\n")
            f.write(f"- Provider: `{result.provider}`\n")
            f.write(f"- Model: `{result.model}`\n")
            f.write(f"- Hardware profile: `{result.hardware_profile}`\n\n")
            f.write("## Task\n\n")
            f.write(state.task.strip() + "\n\n")
            f.write("## Generated Files\n\n")
            for path in sorted(set(generated_files)):
                f.write(f"- `{path}`\n")
            if not generated_files:
                f.write("- No files were written from `<file>` tags.\n")
            f.write("\n## Phase Trace\n\n")
            for phase in result.phase_results:
                duration = phase.finished_at - phase.started_at
                f.write(f"- `{phase.phase}`: {duration:.1f}s, {phase.attempts} attempt(s), model `{phase.model}`\n")
            if result.summary:
                f.write("\n## Summary\n\n" + result.summary.strip() + "\n")
            if result.reflection:
                f.write("\n## Reflection\n\n" + result.reflection.strip() + "\n")
    except OSError as exc:
        logger.warning("failed to write Martinus run manifest: %s", exc)


def _build_retrieval_query(phase: str, task: str, goals: list[str], entries: list[SessionEntry]) -> str:
    latest = _truncate(entries[-1].content, 200) if entries else ""
    parts = [f"phase={phase}", f"task={task}", f"task={task}"]
    if goals:
        parts.append("goals=" + " ".join(goals))
    if latest:
        parts.append("context=" + latest)
    return _truncate("\n".join(parts), 800)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class Engine:
    def __init__(
        self,
        router: Router,
        memory: Optional[AIMemoryClient] = None,
        routing: Optional[RoleRouting] = None,
    ) -> None:
        self._router = router
        self._memory = memory
        self._routing = routing
        # Per-run state: reset at the top of execute() so one run's failures
        # can never bump temperatures or budgets of an unrelated later run.
        self._temp = TemperatureScheduler()
        self._budget: Optional[RunBudget] = None

    # ---- core retry loop (Algorithm 3.1) -----------------------------------

    def _call_with_retry(
        self,
        provider: str,
        system: str,
        prompt: str,
        temperature: float,
        model_override: str,
        max_retries: int,
        phase: str,
        on_event: Optional[Callable[[dict], None]],
    ) -> tuple[str, str, str, int]:
        """Returns (content, provider_name, model_name, attempts).

        Retry policy (Algorithm 3.1, hardened):
        - budget/cancellation is checked before every attempt;
        - deterministic errors (bad config, HTTP 4xx) are never retried;
        - transient errors back off exponentially (bounded);
        - an empty model response counts as a failed attempt, not success.
        """
        current_prompt = prompt
        for attempt in range(1, max_retries + 1):
            if self._budget:
                self._budget.check(f"phase {phase}, attempt {attempt}")
            def token_cb(tok: str, _phase=phase) -> None:
                if on_event:
                    on_event({"type": "token", "phase": _phase, "token": tok})

            req = ChatRequest(
                messages=[
                    Message(role="system", content=system),
                    Message(role="user", content=current_prompt),
                ],
                temperature=temperature,
                disable_reasoning=True,
                model=model_override,
                on_token=token_cb if on_event else None,
            )
            try:
                if self._budget:
                    self._budget.count_model_call()
                resp = self._router.chat(provider, req)
                content = resp.content.strip()
                if not content:
                    raise ValueError("provider returned an empty response")
                return content, resp.provider, resp.model, attempt
            except (DeterministicProviderError, RunAborted):
                raise
            except Exception as exc:
                if attempt >= max_retries:
                    raise RuntimeError(f"phase {phase}: all {max_retries} attempts failed: {exc}") from exc
                time.sleep(min(RETRY_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)), 15.0))
                current_prompt += FALLBACK_RETRY
        raise RuntimeError(f"phase {phase}: retry loop exhausted")  # unreachable

    def _raw_chat(self, provider: str, req: ChatRequest, phase: str):
        """Single budget-counted model call for auxiliary paths (step-back,
        candidate branches, judges) that bypass _call_with_retry."""
        if self._budget:
            self._budget.check(f"aux call ({phase})")
            self._budget.count_model_call()
        return self._router.chat(provider, req)

    # ---- standard phase execution ------------------------------------------

    def _run_phase(
        self,
        phase: str,
        state: SessionState,
        provider: str,
        model_override: str,
        retrieved: list[dict],
        max_retries: int,
        on_event: Optional[Callable[[dict], None]],
        generated_files: Optional[list[str]] = None,
    ) -> PhaseResult:
        started = time.time()
        system = build_system_instruction(phase, state.max_lines, state.task)
        prompt = build_prompt(
            phase, state.task, state.workspace_path,
            provider, state.max_lines, state.goals,
            [e.to_dict() for e in state.entries], retrieved,
        )
        # Review and Verification grade against ON-DISK reality, not their own
        # memory of what Implementation said. Inject the actual file contents
        # so the model can't hallucinate a function it never wrote.
        if phase in ("Review", "Verification") and state.workspace_path and generated_files:
            snapshot = _build_workspace_snapshot(state.workspace_path, generated_files)
            if snapshot:
                prompt = snapshot + "\n\n" + prompt
        temperature = self._temp.temperature(phase)
        content, prov, model, attempts = self._call_with_retry(
            provider, system, prompt, temperature, model_override,
            max_retries, phase, on_event,
        )
        return PhaseResult(
            phase=phase, output=content, attempts=attempts,
            provider=prov, model=model,
            prompt_version=PROMPT_VERSION,
            retrieved_count=len(retrieved),
            started_at=started, finished_at=time.time(),
        )

    # ---- Algorithm 4.1/4.2: Architecture with step-back / candidates --------

    def _step_back_query(self, state: SessionState, provider: str, model: str) -> str:
        system = "You are a senior software architect. Answer concisely with principle-first conclusions only."
        prompt = (
            "Before designing a specific solution, identify the general architectural principles "
            "that apply to this class of problem.\n\n"
            f"Problem domain: {state.task}\n"
            f"Goals: {'; '.join(state.goals)}\n\n"
            "List 4-6 applicable principles, each with a one-sentence rationale. "
            "Focus on principles most relevant to local-first, constrained, agentic coding systems."
        )
        try:
            req = ChatRequest(
                messages=[Message("system", system), Message("user", prompt)],
                temperature=PHASE_TEMPERATURES["stepback"],
                disable_reasoning=True,
                model=model,
            )
            return self._raw_chat(provider, req, "stepback").content.strip()
        except (DeterministicProviderError, RunAborted):
            raise
        except Exception:
            return ""

    def _run_architecture_with_step_back(
        self, state: SessionState, provider: str, model: str,
        retrieved: list[dict], max_retries: int,
        on_event: Optional[Callable[[dict], None]],
    ) -> PhaseResult:
        started = time.time()
        principles = self._step_back_query(state, provider, model)
        base_prompt = build_prompt(
            "Architecture", state.task, state.workspace_path,
            provider, state.max_lines, state.goals,
            [e.to_dict() for e in state.entries], retrieved,
        )
        if principles:
            base_prompt = (
                f"=== Step-Back: General Architectural Principles ===\n{principles}\n\n"
                f"=== Apply the above principles to the specific design below ===\n\n{base_prompt}"
            )
        system = build_system_instruction("Architecture", state.max_lines, state.task)
        content, prov, model_name, attempts = self._call_with_retry(
            provider, system, base_prompt,
            PHASE_TEMPERATURES["Architecture"], model, max_retries, "Architecture", on_event,
        )
        return PhaseResult(
            phase="Architecture", output=content, attempts=attempts,
            provider=prov, model=model_name,
            prompt_version=PROMPT_VERSION + ".stepback",
            retrieved_count=len(retrieved),
            started_at=started, finished_at=time.time(),
        )

    def _run_architecture_with_candidates(
        self, state: SessionState, provider: str, model: str,
        retrieved: list[dict], profile: HardwareProfile,
        on_event: Optional[Callable[[dict], None]],
    ) -> PhaseResult:
        started = time.time()
        principles = self._step_back_query(state, provider, model) if profile.step_back else ""
        base_prompt = build_prompt(
            "Architecture", state.task, state.workspace_path,
            provider, state.max_lines, state.goals,
            [e.to_dict() for e in state.entries], retrieved,
        )
        if principles:
            base_prompt = (
                f"=== Step-Back: General Architectural Principles ===\n{principles}\n\n"
                f"=== Apply the above principles to each candidate design below ===\n\n{base_prompt}"
            )
        directives = [
            ("A", "CANDIDATE A — Minimal Footprint:\nDesign the smallest possible system. "
             "Prefer flat, procedural code. Zero external libraries unless strictly necessary. "
             "Optimise for read-time simplicity over extensibility."),
            ("B", "CANDIDATE B — Layered Abstractions:\nDesign with clear separation of concerns. "
             "Define explicit interfaces at layer boundaries. Optimise for maintainability."),
            ("C", "CANDIDATE C — Resilience-Optimised:\nDesign for explicit error handling at every boundary. "
             "Add observability hooks. Define graceful-degradation paths. "
             "Optimise for production reliability."),
        ]
        system = build_system_instruction("Architecture", state.max_lines, state.task)
        branches: list[tuple[str, str]] = []
        total_attempts = 0
        for label, directive in directives:
            candidate_prompt = f"{directive}\n\n---\n\n{base_prompt}"
            current = candidate_prompt
            for attempt in range(1, profile.max_retries + 1):
                total_attempts += 1
                try:
                    req = ChatRequest(
                        messages=[Message("system", system), Message("user", current)],
                        temperature=PHASE_TEMPERATURES["Architecture"],
                        disable_reasoning=True, model=model,
                    )
                    resp = self._raw_chat(provider, req, label)
                    branches.append((label, resp.content.strip()))
                    break
                except (DeterministicProviderError, RunAborted):
                    raise
                except Exception:
                    if attempt >= profile.max_retries:
                        break
                    current += FALLBACK_RETRY

        if not branches:
            return self._run_architecture_with_step_back(
                state, provider, model, retrieved, profile.max_retries, on_event)
        if len(branches) == 1:
            return PhaseResult(
                phase="Architecture", output=branches[0][1], attempts=total_attempts,
                provider=provider, model=model,
                prompt_version=PROMPT_VERSION + ".candidate-single",
                retrieved_count=len(retrieved),
                started_at=started, finished_at=time.time(),
            )
        winner = self._run_arch_judge(branches, state, provider, model)
        return PhaseResult(
            phase="Architecture", output=winner or branches[0][1],
            attempts=total_attempts, provider=provider, model=model,
            prompt_version=PROMPT_VERSION + ".candidates",
            retrieved_count=len(retrieved),
            started_at=started, finished_at=time.time(),
        )

    def _run_arch_judge(
        self, candidates: list[tuple[str, str]],
        state: SessionState, provider: str, model: str,
    ) -> str:
        parts = "\n\n".join(f"=== CANDIDATE {lbl} ===\n{out}" for lbl, out in candidates)
        prompt = (
            f"You are evaluating {len(candidates)} alternative architecture designs.\n\n"
            f"Task: {state.task}\nGoals: {'; '.join(state.goals)}\nLine Budget: <= {state.max_lines} lines\n\n"
            f"{parts}\n\n"
            "Evaluate all candidates step by step on five axes:\n"
            "1. Simplicity  2. Correctness  3. Maintainability  4. Resilience  5. Testability\n\n"
            "Output the FULL TEXT of the winning candidate verbatim, "
            "preceded by exactly one line: 'WINNER: Candidate [A|B|C] — [one-sentence reason]'."
        )
        system = (
            "ROLE: You are a principal engineering architect evaluating architecture designs. "
            "Score each candidate on all five axes (1-5) before selecting. Never select on first impression."
        )
        try:
            req = ChatRequest(
                messages=[Message("system", system), Message("user", prompt)],
                temperature=PHASE_TEMPERATURES["judge"], disable_reasoning=True, model=model,
            )
            content = self._raw_chat(provider, req, "judge").content.strip()
            lines = content.split("\n", 1)
            if len(lines) == 2 and lines[0].strip().startswith("WINNER:"):
                return lines[1].strip()
            return content
        except (DeterministicProviderError, RunAborted):
            raise
        except Exception:
            return ""

    # ---- Algorithm 4.3: Tree-of-Code ----------------------------------------

    def _run_tree_of_code(
        self, state: SessionState, provider: str, model: str,
        retrieved: list[dict], max_retries: int,
        on_event: Optional[Callable[[dict], None]],
    ) -> PhaseResult:
        started = time.time()
        base_prompt = build_prompt(
            "Implementation", state.task, state.workspace_path,
            provider, state.max_lines, state.goals,
            [e.to_dict() for e in state.entries], retrieved,
        )
        system = build_system_instruction("Implementation", state.max_lines, state.task)
        branch_specs = [
            ("A", "BRANCH A — Simplicity-First:\n"
             "Prioritize the smallest possible patch set, minimal LOC, and the most direct "
             "implementation path. Avoid abstractions unless they eliminate meaningful duplication. "
             "Prefer standard library over custom code.", 0.1),
            ("B", "BRANCH B — Robustness-First:\n"
             "Prioritize explicit error handling, defensive input validation, comprehensive test "
             "coverage, and clear rollback paths for each step.", 0.2),
        ]
        branches: list[tuple[str, str]] = []
        total_attempts = 0
        for label, directive, temp in branch_specs:
            current = f"{directive}\n\n---\n\n{base_prompt}"
            for attempt in range(1, max_retries + 1):
                total_attempts += 1
                try:
                    def token_cb(tok: str) -> None:
                        if on_event:
                            on_event({"type": "token", "phase": "Implementation", "token": tok})
                    req = ChatRequest(
                        messages=[Message("system", system), Message("user", current)],
                        temperature=temp, disable_reasoning=True, model=model,
                        on_token=token_cb if on_event else None,
                    )
                    resp = self._raw_chat(provider, req, label)
                    branches.append((label, resp.content.strip()))
                    break
                except (DeterministicProviderError, RunAborted):
                    raise
                except Exception:
                    if attempt >= max_retries:
                        break
                    current += FALLBACK_RETRY

        if not branches:
            return self._run_phase("Implementation", state, provider, model, retrieved, max_retries, on_event)
        if len(branches) == 1:
            return PhaseResult(
                phase="Implementation", output=branches[0][1], attempts=total_attempts,
                provider=provider, model=model,
                prompt_version=PROMPT_VERSION + ".toc-single",
                retrieved_count=len(retrieved),
                started_at=started, finished_at=time.time(),
            )
        winner = self._run_toc_judge(branches, state, provider, model)
        return PhaseResult(
            phase="Implementation", output=winner or branches[0][1],
            attempts=total_attempts, provider=provider, model=model,
            prompt_version=PROMPT_VERSION + ".toc",
            retrieved_count=len(retrieved),
            started_at=started, finished_at=time.time(),
        )

    def _run_toc_judge(
        self, branches: list[tuple[str, str]],
        state: SessionState, provider: str, model: str,
    ) -> str:
        parts = "\n\n".join(f"=== BRANCH {lbl} ===\n{out}" for lbl, out in branches)
        prompt = (
            "You are evaluating two alternative implementation plans.\n\n"
            f"Task: {state.task}\nGoals: {'; '.join(state.goals)}\nLine Budget: <= {state.max_lines}\n\n"
            f"{parts}\n\n"
            "Evaluate both branches on: 1. Correctness  2. Safety  3. Testability  4. Maintainability\n\n"
            "Output the FULL TEXT of the winning branch verbatim, "
            "preceded by exactly one line: 'WINNER: Branch [A|B] — [one-sentence reason]'."
        )
        system = (
            "ROLE: You are a principal engineer judging implementation plans. "
            "Score each branch on all four axes (1-5) before selecting."
        )
        try:
            req = ChatRequest(
                messages=[Message("system", system), Message("user", prompt)],
                temperature=PHASE_TEMPERATURES["judge"], disable_reasoning=True, model=model,
            )
            content = self._raw_chat(provider, req, "judge").content.strip()
            lines = content.split("\n", 1)
            if len(lines) == 2 and lines[0].strip().startswith("WINNER:"):
                return lines[1].strip()
            return content
        except (DeterministicProviderError, RunAborted):
            raise
        except Exception:
            return ""

    # ---- Algorithm 4.4: Self-Critique Review --------------------------------

    def _run_review_with_self_critique(
        self, state: SessionState, provider: str, model: str,
        retrieved: list[dict], max_retries: int,
        on_event: Optional[Callable[[dict], None]],
        generated_files: Optional[list[str]] = None,
    ) -> PhaseResult:
        started = time.time()
        initial = self._run_phase(
            "Review", state, provider, model, retrieved, max_retries, on_event,
            generated_files=generated_files,
        )
        critique_prompt = (
            f"You produced the following code review for a coding-agent task.\n\n"
            f"Task: {state.task}\nGoals: {'; '.join(state.goals)}\n\n"
            f"=== Review Draft ===\n{initial.output}\n\n"
            "=== Self-Critique Checklist ===\n"
            "1. Missing defects: list any concrete bugs NOT mentioned.\n"
            "2. Security gaps: are all injection and data-exposure vectors addressed?\n"
            "3. Performance blind spots: were hot paths checked?\n"
            "4. False positives: mark any defects that are incorrect.\n"
            "5. Completeness: is the quality rating (1-10) properly justified?\n\n"
            "After the critique, output a REVISED and COMPLETE review that incorporates all corrections. "
            "The revised review must fully replace the draft."
        )
        system = (
            "ROLE: You are a critical code reviewer improving your own analysis. "
            "REASONING PROTOCOL: For each checklist item, state whether the original addressed it "
            "fully, partially, or missed it. Only emit the revised review — no critique scaffolding."
        )
        try:
            content, prov, model_name, attempts = self._call_with_retry(
                provider, system, critique_prompt,
                PHASE_TEMPERATURES["critique"], model, max_retries, "Review (Critique)", on_event,
            )
            return PhaseResult(
                phase="Review", output=content,
                attempts=initial.attempts + attempts,
                provider=prov, model=model_name,
                prompt_version=PROMPT_VERSION + ".self-critique",
                retrieved_count=len(retrieved),
                started_at=started, finished_at=time.time(),
            )
        except RunAborted:
            raise
        except Exception:
            return initial  # graceful fallback

    # ---- Algorithm 4.8: Hierarchical Summarisation --------------------------

    def _apply_hierarchical_summary(self, state: SessionState) -> None:
        if len(state.entries) < HIERARCHICAL_SUMMARY_THRESHOLD:
            return
        keep_from = len(state.entries) - HISTORY_WINDOW
        if keep_from <= 0:
            return
        older = state.entries[:keep_from]
        recent = state.entries[keep_from:]

        def budget(phase: str) -> int:
            if phase in ("Implementation", "Architecture", "Verification"):
                return 400
            if phase in ("Review", "Clarification"):
                return 250
            return 120

        parts = []
        for entry in older:
            if entry.phase == "Summary":
                parts.append(entry.content)
            else:
                parts.append(f"[{entry.phase}] {_truncate(entry.content, budget(entry.phase))}")

        summary_content = (
            f"=== Hierarchical Summary (compressed {len(older)} entries) ===\n"
            + "\n".join(parts)
        )
        state.entries = [SessionEntry(phase="Summary", content=summary_content)] + list(recent)

    # ---- Memory helpers (Algorithm 4.10) ------------------------------------

    def _retrieve_memories(self, phase: str, state: SessionState) -> list[dict]:
        if not self._memory:
            return []
        query = _build_retrieval_query(phase, state.task, state.goals, state.entries)
        if not query.strip():
            return []
        items = self._memory.retrieve(
            session_id=state.session_id,
            query=query,
            top_k=RETRIEVAL_TOP_K,
            workspace_path=state.workspace_path,
            include_cross_session=True,
            current_phase=phase,
        )
        filtered = [it for it in items if it.get("score", 0.0) >= MIN_RETRIEVAL_SCORE]
        return _trim_to_context_budget(filtered, CONTEXT_BUDGET_TOKENS)

    def _upsert_learning_entry(
        self, phase: str, state: SessionState,
        result: PhaseResult, outcome_label: str,
    ) -> None:
        if not self._memory or not result.output.strip():
            return
        self._memory.upsert(
            session_id=state.session_id,
            content=result.output,
            tags=["phase-output", phase.lower()],
            workspace_path=state.workspace_path,
            source_phase=phase,
            source_provider=result.provider,
            source_model=result.model,
            outcome_label=outcome_label,
            metadata={
                "promptVersion": result.prompt_version,
                "attempts": str(result.attempts),
                "lineBudget": str(state.max_lines),
            },
        )

    # ---- Algorithm 4.6: Post-Run Reflection ---------------------------------

    def _run_reflection(self, state: SessionState) -> str:
        if not self._memory or not state.entries:
            return ""
        lines = [f"[{e.phase}]\n{_truncate(e.content, 800)}" for e in state.entries]
        transcript = "\n\n---\n\n".join(lines)
        outcome = "unknown"
        for entry in state.entries:
            if entry.phase == "Verification":
                outcome = _infer_outcome(entry.content)
                break
        return self._memory.reflect(
            session_id=state.session_id,
            workspace_path=state.workspace_path,
            task=state.task,
            transcript=transcript,
            outcome=outcome,
        )

    def _recover_missing_files(
        self,
        state: SessionState,
        provider: str,
        model: str,
        missing_files: list[str],
        on_event: Optional[Callable[[dict], None]],
    ) -> tuple[str, list[str]]:
        if not missing_files:
            return "", []

        # Build (name, first-2KB-preview) tuples so the model can UPDATE existing
        # files coherently instead of regenerating unrelated content and drifting.
        existing_snapshots: list[tuple[str, str]] = []
        try:
            with os.scandir(state.workspace_path) as it:
                entries = sorted(
                    (e for e in it if e.is_file() and not e.name.startswith(".")),
                    key=lambda e: e.name,
                )
        except OSError:
            entries = []
        for entry in entries[:12]:  # cap to keep the recovery prompt small
            try:
                with open(entry.path, "r", encoding="utf-8", errors="replace") as fh:
                    preview = fh.read(2048)
            except OSError:
                continue
            existing_snapshots.append((entry.name, preview))

        existing_block = "(none)"
        if existing_snapshots:
            existing_block = "\n\n".join(
                f"--- {name} ---\n{preview}" for name, preview in existing_snapshots
            )

        prompt = (
            "A previous generation pass did not materialize every required file.\n\n"
            f"Task: {state.task}\n"
            f"Workspace: {state.workspace_path}\n\n"
            f"=== Existing files (verbatim first 2KB of each) ===\n{existing_block}\n\n"
            f"=== Missing files that MUST be created ===\n"
            + "\n".join(f"- {p}" for p in missing_files)
            + "\n\n"
            "Output ONLY the missing files using complete "
            '<file path="relative/path">...</file> tags.\n'
            "Rules:\n"
            "- Do NOT wrap file content in markdown fences.\n"
            "- Do NOT include explanations, headings, or files that already exist above.\n"
            "- If a missing file depends on an existing file (e.g. index.html referencing styles.css), "
            "make the new file consistent with the existing content shown above.\n"
            "- Each missing file must be complete, self-contained, and working."
        )
        system = (
            "ROLE: You are a surgical file-recovery generator.\n"
            "Return only the missing complete files requested by the user, "
            'each wrapped in <file path="...">...</file> tags with no markdown fences.\n'
            "No commentary. No summaries. No placeholders."
        )
        recovery_output, _prov, _model, _attempts = self._call_with_retry(
            provider=provider,
            system=system,
            prompt=prompt,
            temperature=0.0,
            model_override=model,
            max_retries=2,
            phase="Recovery",
            on_event=on_event,
        )
        written = _extract_and_write_files(state.workspace_path, recovery_output)
        return recovery_output, written

    def _run_recovery_if_needed(
        self,
        phase_result: PhaseResult,
        state: SessionState,
        provider: str,
        model: str,
        requested_files: list[str],
        generated_files: list[str],
        on_event: Optional[Callable[[dict], None]],
        current_phase_wrote: int = 0,
    ) -> None:
        if not state.workspace_path:
            return

        # Two triggers for recovery:
        # 1. Explicit request: user named specific files and one of them is missing.
        # 2. Silent failure: this phase wrote zero files even though it was a
        #    code-producing phase. This is what catches "build me a calculator"
        #    style natural-language tasks where the extractor got nothing.
        missing_files: list[str] = []
        if requested_files:
            missing_files = _missing_requested_files(state.workspace_path, requested_files)

        silent_failure = current_phase_wrote == 0 and not requested_files
        if not missing_files and not silent_failure:
            return

        if silent_failure and not missing_files:
            # Ask the model to at minimum produce the canonical set for the task.
            # We don't know the exact filenames, so we prompt the recovery pass
            # to infer them from the task text and existing workspace.
            inferred = _infer_expected_files(state.task)
            if not inferred:
                logger.warning(
                    "phase %s wrote 0 files and no requested filenames could be "
                    "inferred from the task; skipping recovery",
                    phase_result.phase,
                )
                return
            missing_files = inferred

        if on_event:
            on_event({"type": "phase_started", "phase": "Recovery", "sessionId": state.session_id})
        try:
            recovery_output, recovered_files = self._recover_missing_files(
                state=state,
                provider=provider,
                model=model,
                missing_files=missing_files,
                on_event=on_event,
            )
        except RunAborted:
            raise
        except Exception as exc:
            # A failed recovery pass must not destroy an otherwise-completed
            # run; the missing files still show up in the deterministic
            # verification summary, so the final status stays truthful.
            logger.warning("recovery pass failed: %s", exc)
            recovery_output, recovered_files = f"[RECOVERY FAILED] {exc}", []
        if recovery_output:
            phase_result.output += (
                "\n\n[RECOVERY PASS]\n"
                f"Trigger: {'missing requested files' if requested_files else 'zero files written by ' + phase_result.phase}\n"
                f"Missing files before recovery: {', '.join(missing_files)}\n"
                f"Recovered files: {', '.join(recovered_files) if recovered_files else '(none)'}\n"
            )
        generated_files.extend(recovered_files)
        # Tell the UI which files landed on disk during Recovery.
        if recovered_files and on_event:
            on_event({
                "type": "files_written",
                "phase": "Recovery",
                "files": recovered_files,
                "workspacePath": state.workspace_path,
                "sessionId": state.session_id,
            })
        if on_event:
            on_event({
                "type": "phase_completed",
                "phase": "Recovery",
                "output": recovery_output or "No recovery output.",
                "attempts": 1,
                "model": model,
                "promptVersion": PROMPT_VERSION + ".recovery",
                "retrievedCount": 0,
                "sessionId": state.session_id,
            })

    # ---- Algorithm 4.12: Executable Verification with Automated Repair -----

    def _run_gate_repair_loop(
        self,
        phase_result: PhaseResult,
        state: SessionState,
        provider: str,
        model: str,
        generated_files: list[str],
        on_event: Optional[Callable[[dict], None]],
        max_iterations: int = 2,
    ) -> str:
        """Run per-file static gates (py_compile / node --check / html.parser /
        css brace-balance / json.loads) against every generated file. When any
        file fails its gate, issue a Repair-phase LLM call with the EXACT
        compiler/parser error text and the current file source, then re-check.
        Loop up to `max_iterations` times.

        Returns a compact human-readable verdict block to be appended to the
        phase output (also shown in the run manifest). Emits phase_started/
        phase_completed SSE events on `on_event` so the desktop app renders
        the repair loop visibly.
        """
        if not state.workspace_path or not generated_files:
            return ""

        # Dedup — same file might be listed twice if Implementation and
        # Recovery both touched it.
        seen: set[str] = set()
        target_files: list[str] = []
        for f in generated_files:
            if f not in seen:
                seen.add(f)
                target_files.append(f)

        verdict_lines: list[str] = ["\n\n[EXECUTABLE VERIFICATION — Algorithm 4.12]"]
        # max_iterations is the number of REPAIR attempts allowed. The loop
        # runs one extra iteration for the initial detect pass, so total gate
        # runs = max_iterations + 1 (bounded by profile.repair_iterations).
        total_iterations = max(1, max_iterations) + 1
        for iteration in range(1, total_iterations + 1):
            results = _gates.run_gates(state.workspace_path, target_files)
            fails = _gates.failed(results)

            # Emit a compact per-iteration status line for the manifest.
            verdict_lines.append(
                f"  iteration {iteration}: "
                f"{len(results) - len(fails)}/{len(results)} files pass "
                + ", ".join(
                    f"{rel} [{r.kind}={'OK' if r.ok else 'FAIL'}]" for rel, r in results
                )
            )

            if not fails:
                repaired_passes = iteration - 1
                verdict_lines.append(
                    f"  Runnable-Gate: PASS "
                    f"(after {repaired_passes} repair pass{'es' if repaired_passes != 1 else ''})"
                )
                return "\n".join(verdict_lines)

            if iteration == total_iterations:
                verdict_lines.append("  Runnable-Gate: FAIL (exhausted repair budget)")
                for rel, r in fails:
                    verdict_lines.append(f"    - {rel} ({r.kind}): {r.error[:200]}")
                return "\n".join(verdict_lines)

            # Build the Repair prompt with exact source + exact error.
            failure_payload: list[dict] = []
            for rel, r in fails:
                abs_p = os.path.join(state.workspace_path, rel)
                try:
                    with open(abs_p, "r", encoding="utf-8") as fh:
                        source = fh.read()
                except OSError:
                    source = ""
                failure_payload.append({
                    "path": rel,
                    "kind": r.kind,
                    "error": r.error,
                    "detail": r.detail,
                    "source": source,
                })

            if on_event:
                on_event({
                    "type": "phase_started",
                    "phase": "Repair",
                    "sessionId": state.session_id,
                })

            repair_system = _PHASE_SYSTEM.get("Repair", "You are a code-repair engineer.")
            repair_user = build_repair_prompt(state.task, state.workspace_path, failure_payload)
            try:
                content, prov, model_name, attempts = self._call_with_retry(
                    provider=provider,
                    system=repair_system,
                    prompt=repair_user,
                    temperature=0.0,
                    model_override=model,
                    max_retries=2,
                    phase="Repair",
                    on_event=on_event,
                )
            except RunAborted:
                raise
            except Exception as exc:
                verdict_lines.append(f"  Repair LLM call failed on iteration {iteration}: {exc}")
                verdict_lines.append("  Runnable-Gate: FAIL")
                return "\n".join(verdict_lines)

            repaired = _extract_and_write_files(state.workspace_path, content)
            if repaired and on_event:
                on_event({
                    "type": "files_written",
                    "phase": "Repair",
                    "files": repaired,
                    "workspacePath": state.workspace_path,
                    "sessionId": state.session_id,
                })
            verdict_lines.append(
                f"  iteration {iteration}: repair pass rewrote {len(repaired)} file(s): "
                + ", ".join(repaired) if repaired else f"  iteration {iteration}: repair produced no files"
            )
            # Extend the Implementation phase's generated_files list so the
            # manifest reflects everything actually on disk.
            for r in repaired:
                if r not in generated_files:
                    generated_files.append(r)

            if on_event:
                on_event({
                    "type": "phase_completed",
                    "phase": "Repair",
                    "output": content[:1200],
                    "attempts": attempts,
                    "model": model_name,
                    "promptVersion": PROMPT_VERSION + ".repair",
                    "retrievedCount": 0,
                    "sessionId": state.session_id,
                })
        # Loop exited without a break — should not reach here, but be safe.
        verdict_lines.append("  Runnable-Gate: FAIL (loop exhausted)")
        return "\n".join(verdict_lines)

    # ---- Algorithm 4.11: Main execution loop --------------------------------

    def execute(
        self,
        session_id: str,
        task: str,
        workspace_path: str,
        goals: list[str],
        provider_name: str,
        hardware_profile: str,
        max_lines: int = 2000,
        model_override: str = "",
        on_event: Optional[Callable[[dict], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> RunResult:
        profile_default = os.getenv("HARDWARE_PROFILE_DEFAULT", "medium")
        profile = resolve_profile(hardware_profile, profile_default)

        # Workspace confinement: when the operator pins a workspace root, a
        # run may only write inside it. (Path traversal inside the workspace
        # is rejected separately by _safe_abs_write_path.)
        ws_root = os.getenv("ORCHESTRATOR_WORKSPACE_ROOT", "").strip()
        if workspace_path and ws_root:
            base = os.path.realpath(ws_root)
            target = os.path.realpath(workspace_path)
            if os.path.commonpath([base, target]) != base:
                raise ValueError(
                    f"workspace {workspace_path!r} is outside ORCHESTRATOR_WORKSPACE_ROOT"
                )

        # Fresh per-run state: temperature scheduling and the run budget must
        # never leak between runs.
        self._temp = TemperatureScheduler()
        self._budget = RunBudget(cancel_check=cancel_check)
        usage_before = self._router.usage_totals()

        resolved_provider = self._router.resolve(provider_name)

        state = SessionState(
            session_id=session_id or f"session-{int(time.time())}",
            task=task,
            workspace_path=workspace_path,
            goals=goals,
            provider=resolved_provider,
            max_lines=max_lines if max_lines > 0 else 2000,
        )

        if profile.hierarchical_summary:
            self._apply_hierarchical_summary(state)

        result = RunResult(
            session_id=state.session_id,
            provider=resolved_provider,
            model="",
            hardware_profile=profile.name,
        )

        if on_event:
            on_event({
                "type": "run_started",
                "sessionId": state.session_id,
                "provider": resolved_provider,
                "hardwareProfile": profile.name,
            })

        generated_files: list[str] = []
        requested_files = _extract_requested_files(task) if workspace_path else []
        aborted_reason = ""

        try:
            self._run_phases(
                profile, state, resolved_provider, model_override,
                requested_files, generated_files, result, on_event,
            )
        except RunAborted as exc:
            aborted_reason = str(exc)
            logger.warning("run %s aborted: %s", state.session_id, aborted_reason)
            if on_event:
                on_event({
                    "type": "run_aborted",
                    "sessionId": state.session_id,
                    "reason": aborted_reason,
                })
        except Exception as exc:
            # A hard phase failure (retries exhausted, deterministic provider
            # error) ends the run with a truthful failure status and a manifest
            # instead of an unlogged crash.
            aborted_reason = f"phase execution failed: {exc}"
            logger.error("run %s failed: %s", state.session_id, exc)
            if on_event:
                on_event({
                    "type": "error",
                    "sessionId": state.session_id,
                    "message": str(exc),
                })

        result.summary = self._build_summary(state)

        if profile.reflection and not aborted_reason:
            result.reflection = self._run_reflection(state)

        # Emit a structured, evidence-bounded verification summary for the
        # desktop UI. This deliberately reports observable checks (files on
        # disk and language-specific static gates), not hidden model reasoning.
        verification = _build_verification_summary(
            state.workspace_path,
            requested_files,
            generated_files,
        )
        unique_generated = verification["generatedFiles"]

        # Behavioral evidence (opt-in — executes model-generated tests): when
        # enabled and the workspace contains tests, their result outranks the
        # static gates as final-success evidence.
        if (
            os.getenv("MARTINUS_RUN_WORKSPACE_TESTS", "").strip() in ("1", "true", "yes")
            and state.workspace_path
            and unique_generated
        ):
            behavioral = _gates.run_workspace_tests(state.workspace_path)
            if behavioral is not None:
                verification["behavioralTests"] = behavioral
                if not behavioral["passed"]:
                    verification["passed"] = False

        # Truthful final status: success only when deterministic evidence
        # passed and the run completed. Prose never sets success.
        usage_after = self._router.usage_totals()
        result.usage = {
            k: usage_after.get(k, 0) - usage_before.get(k, 0) for k in usage_after
        }
        result.usage["budget_model_calls"] = self._budget.model_calls if self._budget else 0
        if aborted_reason:
            result.status = "aborted"
            result.success = False
            result.failure_reason = aborted_reason
        elif verification["passed"]:
            result.status = "success"
            result.success = True
        else:
            result.status = "failure"
            result.success = False
            result.failure_reason = (
                "deterministic verification failed: "
                + (f"missing files {verification['missingFiles']}; " if verification["missingFiles"] else "")
                + ("no files were generated; " if not unique_generated else "")
                + "see checks in the run manifest"
            )
        result.summary = (
            f"[status: {result.status}] " + result.summary
            + (f" — {result.failure_reason}" if result.failure_reason else "")
        )

        if on_event:
            on_event({
                "type": "verification_completed",
                "sessionId": state.session_id,
                **verification,
            })

        if state.workspace_path:
            _write_run_manifest(
                state.workspace_path, state, result, generated_files,
                verification=verification,
            )

        if on_event:
            on_event({
                "type": "run_completed",
                "sessionId": state.session_id,
                "summary": result.summary,
                "reflection": result.reflection,
                "provider": result.provider,
                "hardwareProfile": result.hardware_profile,
                "generatedFiles": unique_generated,
                "verification": verification,
                "status": result.status,
                "success": result.success,
                "usage": result.usage,
            })

        return result

    def _run_phases(
        self,
        profile: HardwareProfile,
        state: SessionState,
        resolved_provider: str,
        model_override: str,
        requested_files: list[str],
        generated_files: list[str],
        result: RunResult,
        on_event: Optional[Callable[[dict], None]],
    ) -> None:
        for phase in profile.phases:
            if self._budget:
                self._budget.check(f"before phase {phase}")
            if on_event:
                on_event({"type": "phase_started", "phase": phase, "sessionId": state.session_id})

            retrieved: list[dict] = []
            if profile.memory_retrieval:
                retrieved = self._retrieve_memories(phase, state)

            phase_model = model_override
            if not phase_model and profile.role_routing and self._routing:
                phase_model = self._routing.model_for(resolved_provider, phase)

            try:
                phase_result = self._dispatch_phase(
                    phase, state, resolved_provider, phase_model,
                    retrieved, profile, on_event,
                    generated_files=generated_files,
                )
            except Exception as exc:
                self._temp.record_failure(phase)
                if on_event:
                    on_event({"type": "phase_failed", "phase": phase, "error": str(exc), "sessionId": state.session_id})
                raise

            self._temp.record_success(phase)

            # Auto-write <file> tagged content to workspace.
            # Implementation, Review, and Verification all have write authority
            # (per user contract): Implementation is the primary file-producing
            # phase; Review and Verification MAY emit <file> tags to edit or add
            # files (e.g. tests, small fixes). Last emitter wins for a given
            # path — an intentional trade-off to give the agent maximum ability
            # to self-correct on-disk state.
            if state.workspace_path and phase in ("Implementation", "Review", "Verification"):
                truncated_blocks = _detect_truncated_file_blocks(phase_result.output)
                if truncated_blocks:
                    # A <file> tag without its closing tag means the model hit
                    # a token/context limit; the extractor would silently drop
                    # the file. Surface it as explicit failure evidence.
                    note = (
                        f"\n\n[TRUNCATION WARNING] {truncated_blocks} <file> block(s) "
                        "were opened but never closed — output was likely cut off "
                        "at the provider's token limit; the affected files were NOT written."
                    )
                    phase_result.output += note
                    logger.warning(
                        "phase %s: %d truncated <file> block(s) detected",
                        phase, truncated_blocks,
                    )
                phase_written = _extract_and_write_files(state.workspace_path, phase_result.output)
                generated_files.extend(phase_written)
                # Announce the write so the UI can render "files written by
                # this phase" inline instead of leaving the user guessing.
                if phase_written and on_event:
                    on_event({
                        "type": "files_written",
                        "phase": phase,
                        "files": phase_written,
                        "workspacePath": state.workspace_path,
                        "sessionId": state.session_id,
                    })

                # Fire recovery + Repair only after Implementation or Verification
                # (the phases that carry the primary write budget). Review edits
                # are treated as additive; if they break something, the next
                # user follow-up can regenerate. This prevents Recovery/Repair
                # from re-firing on a Review edit and inflating token cost.
                is_recovery_gate_phase = phase in ("Implementation", "Verification")
                is_last_producing_phase = (
                    phase == "Implementation" and "Verification" not in profile.phases
                ) or phase == "Verification"
                if is_recovery_gate_phase and is_last_producing_phase:
                    self._run_recovery_if_needed(
                        phase_result=phase_result,
                        state=state,
                        provider=resolved_provider,
                        model=phase_model,
                        requested_files=requested_files,
                        generated_files=generated_files,
                        on_event=on_event,
                        current_phase_wrote=len(phase_written) + max(
                            0, len(generated_files) - len(phase_written)
                        ),
                    )

                    # Algorithm 4.12: Executable Verification + Repair loop.
                    # Runs after Recovery so any files created by the recovery
                    # pass are also gate-checked. Extends generated_files if a
                    # repair pass rewrites or adds files.
                    if profile.executable_verification:
                        verdict = self._run_gate_repair_loop(
                            phase_result=phase_result,
                            state=state,
                            provider=resolved_provider,
                            model=phase_model,
                            generated_files=generated_files,
                            on_event=on_event,
                            max_iterations=profile.repair_iterations,
                        )
                        if verdict:
                            phase_result.output += verdict

            # Algorithm 4.9: workspace reference validation after any files in
            # the current phase have been materialized.
            if phase == "Verification" and state.workspace_path:
                warning = _validate_workspace_references(state.workspace_path, phase_result.output)
                if warning:
                    phase_result.output += warning

            state.entries.append(SessionEntry(phase=phase, content=phase_result.output))
            result.phase_results.append(phase_result)
            result.model = phase_result.model

            outcome_label = _infer_outcome(phase_result.output) if phase == "Verification" else "unknown"
            self._upsert_learning_entry(phase, state, phase_result, outcome_label)

            if on_event:
                on_event({
                    "type": "phase_completed",
                    "phase": phase,
                    "output": phase_result.output,
                    "attempts": phase_result.attempts,
                    "model": phase_result.model,
                    "promptVersion": phase_result.prompt_version,
                    "retrievedCount": phase_result.retrieved_count,
                    "sessionId": state.session_id,
                })

    def _dispatch_phase(
        self,
        phase: str,
        state: SessionState,
        provider: str,
        model: str,
        retrieved: list[dict],
        profile: HardwareProfile,
        on_event: Optional[Callable[[dict], None]],
        generated_files: Optional[list[str]] = None,
    ) -> PhaseResult:
        if phase == "Architecture":
            if profile.candidate_designs:
                return self._run_architecture_with_candidates(
                    state, provider, model, retrieved, profile, on_event)
            if profile.step_back:
                return self._run_architecture_with_step_back(
                    state, provider, model, retrieved, profile.max_retries, on_event)
        elif phase == "Implementation" and profile.tree_of_code:
            return self._run_tree_of_code(state, provider, model, retrieved, profile.max_retries, on_event)
        elif phase == "Review" and profile.self_critique:
            return self._run_review_with_self_critique(
                state, provider, model, retrieved, profile.max_retries, on_event, generated_files=generated_files)
        return self._run_phase(
            phase, state, provider, model, retrieved, profile.max_retries, on_event,
            generated_files=generated_files,
        )

    @staticmethod
    def _build_summary(state: SessionState) -> str:
        if not state.entries:
            return "No phases executed"
        latest = state.entries[-1]
        for entry in state.entries:
            if entry.phase == "Verification":
                outcome = _infer_outcome(entry.content)
                return (f"Completed {len(state.entries)} phases. "
                        f"Verification outcome: {outcome}. Latest: {latest.phase}")
        return f"Completed {len(state.entries)} phases. Latest: {latest.phase}"

    def check_provider_health(self, provider: str) -> dict:
        h = self._router.health(provider)
        return {
            "provider": h.provider,
            "reachable": h.reachable,
            "model": h.model,
            "modelAvailable": h.model_available,
            "message": h.message,
        }
