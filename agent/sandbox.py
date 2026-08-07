"""
agent/sandbox.py — Isolated execution environment (LOCALFORGE Rule 2.2 / Node H).

The orchestrator normally writes generated files directly into the user's
workspace. When sandbox isolation is enabled, the whole pipeline instead runs
inside an isolated copy, and results are applied back to the real workspace only
after a verified pass. This satisfies the spec's "never perform autonomous
experimental work directly on the user's primary working tree" and "record the
original repository state before modifications."

Isolation strategy (spec's preferred order):
1. Git worktree on a throwaway branch — when the workspace is a git repo WITH at
   least one commit (a worktree needs a resolvable HEAD).
2. Filesystem copy into a temp dir — fallback for non-git dirs or repos with no
   commits. Vendor / build directories are excluded from the copy.

Apply-back safety (spec Node H "never silently overwrite uncommitted user work"
+ §7 L2 "modify files outside sandbox"):
- Each generated file is resolved into the ORIGINAL workspace with the same
  symlink-escape / traversal guard the orchestrator uses for writes.
- If a target file changed on disk since the sandbox was created (newer-file
  conflict), it is NOT overwritten — reported as a conflict instead.
- The caller passes a PermissionPolicy; an L2 deny blocks apply-back entirely.

Zero new dependencies: `subprocess` for git (already used elsewhere in the repo),
`shutil`/`os`/`hashlib` from the stdlib. Every git call degrades gracefully.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional

from .permissions import PermissionPolicy, default_policy

logger = logging.getLogger(__name__)

_GIT_TIMEOUT = 20.0

# Directories excluded from a filesystem-copy sandbox and from the size precheck.
_COPY_EXCLUDES = (
    ".git", "node_modules", ".venv", "venv", "env", "__pycache__",
    "dist", "build", "target", ".next", ".mypy_cache", ".pytest_cache",
    ".gradle", ".idea", ".vscode", "coverage", ".tox",
)

# Precheck bounds: refuse to sandbox pathologically large workspaces so a copy
# never blows disk or stalls the run.
_MAX_WORKSPACE_FILES = 20_000
_MAX_WORKSPACE_BYTES = 500 * 1024 * 1024  # 500 MB


@dataclass
class SandboxHandle:
    """State for one sandbox lifetime. Opaque to the orchestrator."""
    original_workspace: str
    sandbox_path: str
    kind: str                       # "worktree" | "copy"
    branch: str = ""                # worktree branch name (worktree kind only)
    git_root: str = ""              # repo root for worktree cleanup
    original_revision: str = ""     # HEAD sha at creation, if any
    had_uncommitted: bool = False   # user had uncommitted changes at creation
    # snapshot of pre-run file fingerprints in the ORIGINAL workspace, keyed by
    # rel path → (mtime, size). Used for newer-file conflict detection.
    baseline: dict[str, tuple[float, int]] = field(default_factory=dict)


@dataclass
class ApplyResult:
    """Outcome of applying sandbox results back to the original workspace."""
    applied: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)   # changed by user meanwhile
    refused: list[str] = field(default_factory=list)      # unsafe path / policy deny
    denied_reason: str = ""                               # non-empty if L2 blocked all


# ---------------------------------------------------------------------------
# git helpers (all degrade gracefully)
# ---------------------------------------------------------------------------

def _git(args: list[str], cwd: str) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True,
            timeout=_GIT_TIMEOUT,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, "", str(exc)


def _git_root(path: str) -> str:
    code, out, _ = _git(["rev-parse", "--show-toplevel"], path)
    return out if code == 0 else ""


def _git_head(path: str) -> str:
    code, out, _ = _git(["rev-parse", "HEAD"], path)
    return out if code == 0 else ""


def _git_has_uncommitted(path: str) -> bool:
    code, out, _ = _git(["status", "--porcelain"], path)
    return code == 0 and bool(out)


def is_git_repo_with_commit(path: str) -> bool:
    """True only when a worktree can actually be created: a repo with a HEAD."""
    return bool(_git_root(path)) and bool(_git_head(path))


# ---------------------------------------------------------------------------
# Precheck + baseline
# ---------------------------------------------------------------------------

def _workspace_too_large(path: str) -> bool:
    files = 0
    total = 0
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in _COPY_EXCLUDES]
        for name in filenames:
            files += 1
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
            if files > _MAX_WORKSPACE_FILES or total > _MAX_WORKSPACE_BYTES:
                return True
    return False


def _fingerprint(abs_path: str) -> Optional[tuple[float, int]]:
    try:
        st = os.stat(abs_path)
        return (st.st_mtime, st.st_size)
    except OSError:
        return None


def _snapshot_baseline(workspace: str) -> dict[str, tuple[float, int]]:
    """Fingerprint every existing file so apply-back can detect user edits made
    while the run was in flight."""
    baseline: dict[str, tuple[float, int]] = {}
    for dirpath, dirnames, filenames in os.walk(workspace, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in _COPY_EXCLUDES]
        for name in filenames:
            abs_path = os.path.join(dirpath, name)
            rel = os.path.relpath(abs_path, workspace).replace(os.sep, "/")
            fp = _fingerprint(abs_path)
            if fp is not None:
                baseline[rel] = fp
    return baseline


# ---------------------------------------------------------------------------
# Path safety (mirrors orchestrator._safe_abs_write_path so apply-back honors
# the same symlink-escape / traversal guarantees). Duplicated deliberately to
# avoid a circular import with orchestrator.
# ---------------------------------------------------------------------------

def _safe_abs_write_path(workspace_path: str, rel_path: str) -> Optional[str]:
    rel = rel_path.strip().replace("\\", "/").lstrip("/")
    if not rel or ".." in rel.split("/") or ":" in rel or rel.startswith("~"):
        return None
    base = os.path.realpath(workspace_path)
    target = os.path.realpath(os.path.join(base, rel))
    try:
        if os.path.commonpath([base, target]) != base:
            return None
    except ValueError:
        return None
    return target


def _within_workspace_root(target: str) -> bool:
    """Honor ORCHESTRATOR_WORKSPACE_ROOT on apply-back (this is an L2 write to
    the real workspace, so the operator's pin must still hold)."""
    ws_root = os.getenv("ORCHESTRATOR_WORKSPACE_ROOT", "").strip()
    if not ws_root:
        return True
    base = os.path.realpath(ws_root)
    try:
        return os.path.commonpath([base, os.path.realpath(target)]) == base
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_martinus_repo(workspace: str) -> bool:
    """Detect when the workspace IS the Martinus source checkout (running the
    agent on itself), so the caller can warn."""
    root = _git_root(workspace) or workspace
    return os.path.isdir(os.path.join(root, "agent")) and os.path.isfile(
        os.path.join(root, "agent", "orchestrator.py")
    )


def create_sandbox(
    workspace: str,
    session_id: str = "",
    policy: Optional[PermissionPolicy] = None,
) -> Optional[SandboxHandle]:
    """Create an isolated sandbox for `workspace`. Returns a handle, or None if
    sandboxing is not possible/appropriate (caller then runs in place)."""
    policy = policy or default_policy()
    if not workspace or not os.path.isdir(workspace):
        return None

    # L1 action; allowed under any sane policy, but check for completeness.
    if not policy.decide("create_sandbox").allowed:
        logger.warning("sandbox creation denied by policy; running in place")
        return None

    if _workspace_too_large(workspace):
        logger.warning(
            "workspace exceeds sandbox size limits (%d files / %d bytes); running in place",
            _MAX_WORKSPACE_FILES, _MAX_WORKSPACE_BYTES,
        )
        return None

    baseline = _snapshot_baseline(workspace)
    revision = _git_head(workspace)
    had_uncommitted = _git_has_uncommitted(workspace)
    # Deterministic-ish unique suffix without Math.random / Date.now concerns:
    # session id + pid + a monotonic-ish counter via time.time_ns is fine here
    # (this is runtime, not a workflow script).
    suffix = (session_id or "run").replace("/", "_")[:24] + f"-{os.getpid()}-{time.time_ns()}"

    if is_git_repo_with_commit(workspace):
        handle = _create_worktree_sandbox(workspace, suffix, revision, had_uncommitted, baseline)
        if handle is not None:
            return handle
        logger.warning("git worktree creation failed; falling back to filesystem copy")

    return _create_copy_sandbox(workspace, suffix, revision, had_uncommitted, baseline)


def _create_worktree_sandbox(
    workspace: str, suffix: str, revision: str, had_uncommitted: bool,
    baseline: dict[str, tuple[float, int]],
) -> Optional[SandboxHandle]:
    git_root = _git_root(workspace)
    if not git_root:
        return None
    branch = f"martinus-sandbox-{suffix}"
    sandbox_path = os.path.join(tempfile.gettempdir(), f"martinus-wt-{suffix}")
    code, _out, err = _git(["worktree", "add", "-b", branch, sandbox_path, "HEAD"], git_root)
    if code != 0:
        logger.warning("git worktree add failed: %s", err)
        return None
    return SandboxHandle(
        original_workspace=os.path.realpath(workspace),
        sandbox_path=os.path.realpath(sandbox_path),
        kind="worktree",
        branch=branch,
        git_root=git_root,
        original_revision=revision,
        had_uncommitted=had_uncommitted,
        baseline=baseline,
    )


def _create_copy_sandbox(
    workspace: str, suffix: str, revision: str, had_uncommitted: bool,
    baseline: dict[str, tuple[float, int]],
) -> Optional[SandboxHandle]:
    sandbox_path = os.path.join(tempfile.gettempdir(), f"martinus-copy-{suffix}")
    try:
        shutil.copytree(
            workspace, sandbox_path,
            ignore=shutil.ignore_patterns(*_COPY_EXCLUDES),
            symlinks=True,          # preserve links as links (don't follow them)
            ignore_dangling_symlinks=True,
        )
    except (OSError, shutil.Error) as exc:
        logger.warning("filesystem copy sandbox failed: %s", exc)
        return None
    return SandboxHandle(
        original_workspace=os.path.realpath(workspace),
        sandbox_path=os.path.realpath(sandbox_path),
        kind="copy",
        original_revision=revision,
        had_uncommitted=had_uncommitted,
        baseline=baseline,
    )


def apply_back(
    handle: SandboxHandle,
    generated_files: list[str],
    policy: Optional[PermissionPolicy] = None,
) -> ApplyResult:
    """Copy the given generated files from the sandbox back into the original
    workspace, enforcing L2 policy + hard guards. Never overwrites a file the
    user changed since the sandbox was created."""
    policy = policy or default_policy()
    result = ApplyResult()

    decision = policy.decide("apply_back_to_workspace", detail=handle.kind)
    if not decision.allowed:
        result.denied_reason = decision.reason
        result.refused = list(dict.fromkeys(f for f in generated_files if f))
        return result

    seen: set[str] = set()
    for rel in generated_files:
        if not rel or rel in seen:
            continue
        seen.add(rel)

        # Resolve the destination in the ORIGINAL workspace with the same guard
        # the orchestrator uses (rejects .. / symlink escape / drive prefixes).
        dest = _safe_abs_write_path(handle.original_workspace, rel)
        if dest is None or not _within_workspace_root(dest):
            result.refused.append(rel)
            continue

        # Newer-file conflict: the user edited this file after sandbox creation.
        current_fp = _fingerprint(dest)
        baseline_fp = handle.baseline.get(rel)
        if current_fp is not None and baseline_fp is not None and current_fp != baseline_fp:
            result.conflicts.append(rel)
            continue

        src = os.path.join(handle.sandbox_path, rel)
        if not os.path.isfile(src):
            result.refused.append(rel)
            continue
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copyfile(src, dest)
            result.applied.append(rel)
        except OSError as exc:
            logger.warning("apply-back failed for %s: %s", rel, exc)
            result.refused.append(rel)

    return result


def cleanup(handle: SandboxHandle) -> None:
    """Remove the sandbox. Worktree kind removes the worktree + branch; copy
    kind rmtree's the temp dir. Best-effort; never raises."""
    if handle.kind == "worktree" and handle.git_root:
        _git(["worktree", "remove", "--force", handle.sandbox_path], handle.git_root)
        if handle.branch:
            _git(["branch", "-D", handle.branch], handle.git_root)
        # Hygiene: reap any orphaned worktrees from crashed prior runs.
        _git(["worktree", "prune"], handle.git_root)
        # If the directory still exists (removal partially failed), rmtree it.
        if os.path.isdir(handle.sandbox_path):
            shutil.rmtree(handle.sandbox_path, ignore_errors=True)
    else:
        shutil.rmtree(handle.sandbox_path, ignore_errors=True)
