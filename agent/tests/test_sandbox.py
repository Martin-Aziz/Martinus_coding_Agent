from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from agent.permissions import PermissionPolicy
from agent.sandbox import (
    apply_back,
    cleanup,
    create_sandbox,
    is_git_repo_with_commit,
)


def _git_init_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)
    (root / "seed.txt").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=root, check=True)


def test_worktree_sandbox_created_for_git_repo(tmp_path: Path) -> None:
    _git_init_repo(tmp_path)
    assert is_git_repo_with_commit(str(tmp_path))
    handle = create_sandbox(str(tmp_path), session_id="s1")
    assert handle is not None
    try:
        assert handle.kind == "worktree"
        assert os.path.isdir(handle.sandbox_path)
        assert (Path(handle.sandbox_path) / "seed.txt").exists()
        assert handle.original_revision  # HEAD recorded
    finally:
        cleanup(handle)
    assert not os.path.isdir(handle.sandbox_path)


def test_copy_sandbox_fallback_for_non_git(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    handle = create_sandbox(str(tmp_path), session_id="s2")
    assert handle is not None
    try:
        assert handle.kind == "copy"
        assert (Path(handle.sandbox_path) / "a.py").exists()
    finally:
        cleanup(handle)
    assert not os.path.isdir(handle.sandbox_path)


def test_copy_sandbox_for_git_without_commit(tmp_path: Path) -> None:
    # git init but NO commit → worktree impossible → copy fallback.
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    assert not is_git_repo_with_commit(str(tmp_path))
    handle = create_sandbox(str(tmp_path), session_id="s3")
    assert handle is not None
    try:
        assert handle.kind == "copy"
    finally:
        cleanup(handle)


def test_copy_sandbox_excludes_vendor_dirs(tmp_path: Path) -> None:
    (tmp_path / "keep.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.js").write_text("junk", encoding="utf-8")
    handle = create_sandbox(str(tmp_path), session_id="s4")
    assert handle is not None
    try:
        assert (Path(handle.sandbox_path) / "keep.py").exists()
        assert not (Path(handle.sandbox_path) / "node_modules").exists()
    finally:
        cleanup(handle)


def test_apply_back_copies_generated_files(tmp_path: Path) -> None:
    (tmp_path / "orig.py").write_text("old\n", encoding="utf-8")
    handle = create_sandbox(str(tmp_path), session_id="s5")
    assert handle is not None
    try:
        # Simulate the run writing files into the sandbox.
        (Path(handle.sandbox_path) / "new.py").write_text("generated\n", encoding="utf-8")
        (Path(handle.sandbox_path) / "sub").mkdir(exist_ok=True)
        (Path(handle.sandbox_path) / "sub" / "deep.py").write_text("deep\n", encoding="utf-8")
        res = apply_back(handle, ["new.py", "sub/deep.py"])
        assert sorted(res.applied) == ["new.py", "sub/deep.py"]
        assert (tmp_path / "new.py").read_text(encoding="utf-8") == "generated\n"
        assert (tmp_path / "sub" / "deep.py").exists()
    finally:
        cleanup(handle)


def test_apply_back_refuses_path_escape(tmp_path: Path) -> None:
    handle = create_sandbox(str(tmp_path), session_id="s6")
    assert handle is not None
    try:
        res = apply_back(handle, ["../escape.py", "ok.py"])
        assert "../escape.py" in res.refused
        # ok.py has no source in the sandbox → refused too, but never applied
        # outside the workspace.
        assert not (tmp_path.parent / "escape.py").exists()
    finally:
        cleanup(handle)


def test_apply_back_detects_newer_file_conflict(tmp_path: Path) -> None:
    target = tmp_path / "shared.py"
    target.write_text("v1\n", encoding="utf-8")
    handle = create_sandbox(str(tmp_path), session_id="s7")
    assert handle is not None
    try:
        # Sandbox regenerates shared.py...
        (Path(handle.sandbox_path) / "shared.py").write_text("sandbox-version\n", encoding="utf-8")
        # ...but the user edits the original meanwhile (advance mtime + change size).
        time.sleep(0.01)
        target.write_text("USER EDIT that is longer\n", encoding="utf-8")
        res = apply_back(handle, ["shared.py"])
        assert "shared.py" in res.conflicts
        # Original NOT clobbered.
        assert target.read_text(encoding="utf-8") == "USER EDIT that is longer\n"
    finally:
        cleanup(handle)


def test_apply_back_denied_by_l2_policy(tmp_path: Path) -> None:
    handle = create_sandbox(str(tmp_path), session_id="s8")
    assert handle is not None
    try:
        (Path(handle.sandbox_path) / "new.py").write_text("x\n", encoding="utf-8")
        deny = PermissionPolicy(allow_l2=False)
        res = apply_back(handle, ["new.py"], policy=deny)
        assert res.applied == []
        assert res.denied_reason
        assert not (tmp_path / "new.py").exists()
    finally:
        cleanup(handle)
