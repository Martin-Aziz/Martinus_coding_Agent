"""
agent/repomap.py — Deterministic repository map and symbol index (LOCALFORGE §6.2).

Builds a lightweight, cacheable map of a repository *without* sending the full
codebase to a model. Three layers, per spec §6.2:

- Layer 1 (Manifest): languages, package managers, entry points, and the
  build / test / lint commands declared in Makefile / package.json / pyproject.
- Layer 2 (Architecture): top-level directories with a one-line summary.
- Layer 3 (Symbol index): classes / functions / routes per file.
  Python is parsed with the stdlib `ast`. JavaScript / TypeScript use a
  best-effort regex heuristic — it is NOT a parser and is documented as such.

Design constraints (mirroring agent/gates.py):
- Zero new dependencies. Standard library only. In particular `pyproject.toml`
  is scanned with a regex rather than `tomllib`, because the project's Python
  floor is 3.10 (see mypy.ini) and `tomllib` only exists on 3.11+.
- Never raises on malformed input. A syntactically broken source file (common
  mid-run, since the repair loop exists precisely for that) is skipped, not
  fatal. Every filesystem / parse operation is guarded.
- Deterministic output: directories and symbols are emitted in sorted order so
  the same repository always yields the same map (spec §6.2: "reference cached
  context by identifier").
- Bounded: file, symbol, and byte caps keep the map from blowing the model
  context budget on a large repository.
"""
from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional, cast

# ---------------------------------------------------------------------------
# Bounds. Kept small so the compacted map fits comfortably inside a local
# model's context alongside the task and phase history.
# ---------------------------------------------------------------------------
MAX_FILES_SCANNED = 2000        # hard cap on files walked for the symbol index
MAX_SYMBOLS_PER_FILE = 60       # avoid a generated/minified file flooding the map
MAX_SYMBOLS_TOTAL = 1200        # global symbol ceiling
_COMPACT_BUDGET = 12_000        # byte budget for the compact() string (§6.2 layer cache)

# Directories never worth walking. Shared conceptually with sandbox exclusions.
_EXCLUDED_DIRS: frozenset[str] = frozenset({
    ".git", "node_modules", ".venv", "venv", "env", "__pycache__",
    "dist", "build", "target", ".next", ".mypy_cache", ".pytest_cache",
    ".gradle", ".idea", ".vscode", "coverage", ".tox", "site-packages",
})

# Extension → language label for the manifest.
_LANG_BY_EXT: dict[str, str] = {
    ".py": "Python", ".js": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".jsx": "JavaScript",
    ".rs": "Rust", ".go": "Go", ".java": "Java", ".rb": "Ruby",
    ".c": "C", ".h": "C", ".cpp": "C++", ".cc": "C++", ".hpp": "C++",
    ".cs": "C#", ".php": "PHP", ".swift": "Swift", ".kt": "Kotlin",
    ".html": "HTML", ".htm": "HTML", ".css": "CSS", ".scss": "CSS",
    ".sql": "SQL", ".sh": "Shell", ".yaml": "YAML", ".yml": "YAML",
}

# Common application entry-point filenames worth surfacing.
_ENTRY_POINT_NAMES: frozenset[str] = frozenset({
    "main.py", "app.py", "__main__.py", "manage.py", "wsgi.py", "asgi.py",
    "index.js", "index.ts", "main.js", "main.ts", "server.js", "server.ts",
    "index.html", "main.rs", "main.go", "Main.java", "start.py", "run.py",
})

# Extensions we build a symbol index for. Others are counted but not parsed.
_PY_EXTS: frozenset[str] = frozenset({".py"})
_JS_EXTS: frozenset[str] = frozenset({".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx"})


@dataclass
class Symbol:
    """A single indexed symbol."""
    name: str
    kind: str            # "class" | "function" | "method" | "route" | "export"
    line: int
    detail: str = ""     # e.g. decorator / route path, for context

    def as_dict(self) -> dict[str, object]:
        d: dict[str, object] = {"name": self.name, "kind": self.kind, "line": self.line}
        if self.detail:
            d["detail"] = self.detail
        return d


@dataclass
class FileSymbols:
    """Symbols extracted from one file."""
    path: str            # repo-relative path, forward slashes
    language: str
    symbols: list[Symbol] = field(default_factory=list)
    parse_error: str = ""  # non-empty when the file could not be parsed

    def as_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "path": self.path,
            "language": self.language,
            "symbols": [s.as_dict() for s in self.symbols],
        }
        if self.parse_error:
            d["parseError"] = self.parse_error
        return d


# ---------------------------------------------------------------------------
# Filesystem walk (shared, bounded, symlink-safe)
# ---------------------------------------------------------------------------

def _iter_source_files(root: str) -> list[str]:
    """Return repo-relative paths under `root`, sorted, excluding vendor dirs
    and symlinked directories, capped at MAX_FILES_SCANNED."""
    root = os.path.realpath(root)
    collected: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # Prune excluded / symlinked directories in place (sorted for determinism).
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in _EXCLUDED_DIRS and not os.path.islink(os.path.join(dirpath, d))
        )
        for name in sorted(filenames):
            abs_path = os.path.join(dirpath, name)
            if os.path.islink(abs_path):
                continue
            rel = os.path.relpath(abs_path, root).replace(os.sep, "/")
            collected.append(rel)
            if len(collected) >= MAX_FILES_SCANNED:
                return sorted(collected)
    return sorted(collected)


def _read_text(abs_path: str, cap: int = 200_000) -> Optional[str]:
    """Read a file as UTF-8, capped. Returns None on any OS error."""
    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(cap)
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Layer 1 — Manifest
# ---------------------------------------------------------------------------

# Regexes for command extraction that avoid tomllib / a Makefile parser.
_MAKE_TARGET_RE = re.compile(r"^([A-Za-z0-9_.-]+):(?!=)", re.MULTILINE)
_PYPROJECT_SECTION_RE = re.compile(r"^\[([^\]]+)\]", re.MULTILINE)


def build_manifest(root: str, files: Optional[list[str]] = None) -> dict[str, object]:
    """Layer 1: languages, package managers, entry points, and declared
    build / test / lint commands. Deterministic and never raises."""
    root = os.path.realpath(root)
    if files is None:
        files = _iter_source_files(root)

    lang_counts: dict[str, int] = {}
    entry_points: list[str] = []
    for rel in files:
        ext = os.path.splitext(rel)[1].lower()
        lang = _LANG_BY_EXT.get(ext)
        if lang:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        base = os.path.basename(rel)
        if base in _ENTRY_POINT_NAMES:
            entry_points.append(rel)

    package_managers: list[str] = []
    commands: dict[str, list[str]] = {"build": [], "test": [], "lint": [], "run": []}

    # package.json → npm/yarn scripts.
    pkg_path = os.path.join(root, "package.json")
    if os.path.isfile(pkg_path):
        package_managers.append("npm")
        raw = _read_text(pkg_path)
        if raw:
            try:
                data = json.loads(raw)
                scripts = data.get("scripts", {}) if isinstance(data, dict) else {}
                for key, val in scripts.items():
                    cmd = f"npm run {key}"
                    low = key.lower()
                    if low in ("build", "compile"):
                        commands["build"].append(cmd)
                    elif low in ("test", "tests"):
                        commands["test"].append(cmd)
                    elif low in ("lint", "eslint"):
                        commands["lint"].append(cmd)
                    elif low in ("start", "dev", "serve"):
                        commands["run"].append(cmd)
            except (ValueError, AttributeError):
                pass

    # requirements.txt / pyproject.toml → pip.
    if os.path.isfile(os.path.join(root, "requirements.txt")):
        package_managers.append("pip")
    pyproject = os.path.join(root, "pyproject.toml")
    if os.path.isfile(pyproject):
        if "pip" not in package_managers:
            package_managers.append("pip")
        raw = _read_text(pyproject)
        if raw:
            # Regex-only scan (no tomllib on 3.10). Detect common tool sections
            # so we can hint that pytest / build backends are configured.
            sections = _PYPROJECT_SECTION_RE.findall(raw)
            if any(s.strip().startswith("tool.pytest") for s in sections):
                commands["test"].append("pytest")
            if any(s.strip() == "build-system" for s in sections):
                commands["build"].append("python -m build")

    # Cargo / Go.
    if os.path.isfile(os.path.join(root, "Cargo.toml")):
        package_managers.append("cargo")
        commands["build"].append("cargo build")
        commands["test"].append("cargo test")
    if os.path.isfile(os.path.join(root, "go.mod")):
        package_managers.append("go")
        commands["build"].append("go build ./...")
        commands["test"].append("go test ./...")

    # Makefile → phony-ish targets.
    for mk in ("Makefile", "makefile", "GNUmakefile"):
        mk_path = os.path.join(root, mk)
        if os.path.isfile(mk_path):
            package_managers.append("make")
            raw = _read_text(mk_path)
            if raw:
                for target in _MAKE_TARGET_RE.findall(raw):
                    low = target.lower()
                    cmd = f"make {target}"
                    if "test" in low:
                        commands["test"].append(cmd)
                    elif "build" in low or "compile" in low:
                        commands["build"].append(cmd)
                    elif "lint" in low or "typecheck" in low or "check" in low:
                        commands["lint"].append(cmd)
                    elif low in ("run", "start", "serve", "dev"):
                        commands["run"].append(cmd)
            break

    languages = [lang for lang, _ in sorted(lang_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    return {
        "languages": languages,
        "packageManagers": sorted(set(package_managers)),
        "entryPoints": sorted(set(entry_points)),
        "commands": {k: sorted(set(v)) for k, v in commands.items() if v},
    }


# ---------------------------------------------------------------------------
# Layer 2 — Architecture
# ---------------------------------------------------------------------------

def build_architecture(root: str, files: Optional[list[str]] = None) -> dict[str, object]:
    """Layer 2: top-level directories with a one-line summary (file count +
    dominant language)."""
    root = os.path.realpath(root)
    if files is None:
        files = _iter_source_files(root)

    per_dir: dict[str, dict[str, int]] = {}
    for rel in files:
        top = rel.split("/", 1)[0] if "/" in rel else "(root)"
        ext = os.path.splitext(rel)[1].lower()
        lang = _LANG_BY_EXT.get(ext, "other")
        bucket = per_dir.setdefault(top, {})
        bucket[lang] = bucket.get(lang, 0) + 1

    modules: list[dict[str, object]] = []
    for top in sorted(per_dir):
        langs = per_dir[top]
        total = sum(langs.values())
        dominant = max(langs.items(), key=lambda kv: (kv[1], kv[0]))[0]
        modules.append({
            "name": top,
            "fileCount": total,
            "dominantLanguage": dominant,
        })
    return {"modules": modules}


# ---------------------------------------------------------------------------
# Layer 3 — Symbol index
# ---------------------------------------------------------------------------

def _index_python(rel: str, source: str) -> FileSymbols:
    """Extract classes/functions/methods/routes from Python via ast.
    Never raises: a SyntaxError is captured as parse_error and the file is
    still listed (with zero symbols)."""
    fs = FileSymbols(path=rel, language="Python")
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        fs.parse_error = f"{type(exc).__name__}: {exc}"
        return fs

    def decorator_detail(node: ast.AST) -> str:
        # Surface a route decorator like @app.get("/x") as detail — spec §6.2
        # calls out "routes" explicitly.
        details: list[str] = []
        for dec in getattr(node, "decorator_list", []) or []:
            try:
                if isinstance(dec, ast.Call):
                    func = dec.func
                    method = getattr(func, "attr", "") or getattr(func, "id", "")
                    arg = ""
                    if dec.args and isinstance(dec.args[0], ast.Constant):
                        arg = str(dec.args[0].value)
                    if method:
                        details.append(f"@{method}({arg})" if arg else f"@{method}")
                elif isinstance(dec, ast.Attribute):
                    details.append(f"@{dec.attr}")
                elif isinstance(dec, ast.Name):
                    details.append(f"@{dec.id}")
            except Exception:  # pragma: no cover - defensive
                continue
        return " ".join(details)

    for node in ast.walk(tree):
        if len(fs.symbols) >= MAX_SYMBOLS_PER_FILE:
            break
        if isinstance(node, ast.ClassDef):
            fs.symbols.append(Symbol(node.name, "class", node.lineno, decorator_detail(node)))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            detail = decorator_detail(node)
            kind = "route" if ("@get" in detail or "@post" in detail or "@put" in detail
                               or "@delete" in detail or "@route" in detail
                               or "@patch" in detail) else "function"
            fs.symbols.append(Symbol(node.name, kind, node.lineno, detail))
    fs.symbols.sort(key=lambda s: s.line)
    return fs


# Best-effort JS/TS heuristics. Documented as lossy — arrow-function edge cases,
# JSX components, and default exports are only partially covered.
_JS_FUNC_RE = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)", re.MULTILINE)
_JS_CONST_FN_RE = re.compile(r"^\s*(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(", re.MULTILINE)
_JS_CLASS_RE = re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_$][\w$]*)", re.MULTILINE)
_JS_EXPORT_RE = re.compile(r"^\s*export\s+(?:default\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)", re.MULTILINE)


def _index_javascript(rel: str, source: str) -> FileSymbols:
    """Best-effort JS/TS symbol extraction via regex. Heuristic, not a parser."""
    fs = FileSymbols(path=rel, language="JavaScript")
    seen: set[str] = set()

    def add(name: str, kind: str, offset: int) -> None:
        if name in seen or len(fs.symbols) >= MAX_SYMBOLS_PER_FILE:
            return
        seen.add(name)
        line = source.count("\n", 0, offset) + 1
        fs.symbols.append(Symbol(name, kind, line))

    for m in _JS_CLASS_RE.finditer(source):
        add(m.group(1), "class", m.start())
    for m in _JS_FUNC_RE.finditer(source):
        add(m.group(1), "function", m.start())
    for m in _JS_CONST_FN_RE.finditer(source):
        add(m.group(1), "function", m.start())
    for m in _JS_EXPORT_RE.finditer(source):
        add(m.group(1), "export", m.start())
    fs.symbols.sort(key=lambda s: s.line)
    return fs


def build_symbol_index(root: str, files: Optional[list[str]] = None) -> list[FileSymbols]:
    """Layer 3: per-file symbols for Python (ast) and JS/TS (regex). Sorted by
    path, globally capped at MAX_SYMBOLS_TOTAL."""
    root = os.path.realpath(root)
    if files is None:
        files = _iter_source_files(root)

    out: list[FileSymbols] = []
    total = 0
    for rel in files:
        ext = os.path.splitext(rel)[1].lower()
        if ext not in _PY_EXTS and ext not in _JS_EXTS:
            continue
        source = _read_text(os.path.join(root, rel))
        if source is None:
            continue
        fs = _index_python(rel, source) if ext in _PY_EXTS else _index_javascript(rel, source)
        if not fs.symbols and not fs.parse_error:
            continue
        out.append(fs)
        total += len(fs.symbols)
        if total >= MAX_SYMBOLS_TOTAL:
            break
    out.sort(key=lambda f: f.path)
    return out


# ---------------------------------------------------------------------------
# Assembly + compaction
# ---------------------------------------------------------------------------

def build_repomap(root: str, layers: tuple[int, ...] = (1, 2, 3)) -> dict[str, object]:
    """Assemble the requested layers into one map. Walks the tree once and
    shares the file list across layers."""
    if not root or not os.path.isdir(root):
        return {"root": root, "manifest": {}, "architecture": {}, "symbols": []}
    real = os.path.realpath(root)
    files = _iter_source_files(real)
    out: dict[str, object] = {"root": real}
    if 1 in layers:
        out["manifest"] = build_manifest(real, files)
    if 2 in layers:
        out["architecture"] = build_architecture(real, files)
    if 3 in layers:
        out["symbols"] = [fs.as_dict() for fs in build_symbol_index(real, files)]
    return out


def compact(repomap: dict[str, object], budget: int = _COMPACT_BUDGET) -> str:
    """Render a compact, human/model-readable string under a byte budget.
    Mirrors the byte-budget approach of _build_workspace_snapshot rather than
    reusing the orchestrator's list[dict] token trimmer."""
    manifest = cast("dict[str, object]", repomap.get("manifest", {}) or {})
    architecture = cast("dict[str, object]", repomap.get("architecture", {}) or {})
    symbols = cast("list[dict[str, object]]", repomap.get("symbols", []) or [])

    parts: list[str] = ["=== REPOSITORY MAP ==="]
    langs = cast("list[str]", manifest.get("languages") or [])
    if langs:
        parts.append("Languages: " + ", ".join(langs))
    pms = cast("list[str]", manifest.get("packageManagers") or [])
    if pms:
        parts.append("Package managers: " + ", ".join(pms))
    eps = cast("list[str]", manifest.get("entryPoints") or [])
    if eps:
        parts.append("Entry points: " + ", ".join(eps))
    commands = cast("dict[str, list[str]]", manifest.get("commands") or {})
    for kind in ("build", "test", "lint", "run"):
        if commands.get(kind):
            parts.append(f"{kind.capitalize()} commands: " + "; ".join(commands[kind]))

    modules = cast("list[dict[str, object]]", architecture.get("modules") or [])
    if modules:
        parts.append("\nModules:")
        for m in modules:
            parts.append(f"  {m['name']}/  ({m['fileCount']} files, {m['dominantLanguage']})")

    if symbols:
        parts.append("\nSymbols:")
        for fs in symbols:
            fs_symbols = cast("list[dict[str, object]]", fs.get("symbols", []))[:12]
            names = ", ".join(
                f"{s['name']}" + (f"[{s['kind']}]" if s['kind'] not in ("function",) else "")
                for s in fs_symbols
            )
            line = f"  {fs['path']}: {names}"
            if fs.get("parseError"):
                line = f"  {fs['path']}: (unparseable)"
            parts.append(line)

    parts.append("=== END REPOSITORY MAP ===")

    # Enforce the byte budget by dropping trailing symbol lines if needed.
    text = "\n".join(parts)
    if len(text) <= budget:
        return text
    # Trim from the end (symbols are least critical) until under budget.
    while len(parts) > 1 and len("\n".join(parts)) > budget:
        parts.pop()
    parts.append("=== END REPOSITORY MAP (truncated to budget) ===")
    return "\n".join(parts)
