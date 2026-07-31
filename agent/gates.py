"""
agent/gates.py — Executable verification gates for generated files.

Algorithm 4.12: Executable Verification with Automated Repair Loop
------------------------------------------------------------------
After Implementation has written files to the workspace, run a per-language
static gate on each written file. If any file fails its gate, feed the exact
compiler / parser error into a Repair phase (see prompts._PHASE_SYSTEM['Repair'])
and let the model regenerate ONLY the broken file. Repeat up to K iterations.
Emit a Runnable-Gate: PASS|FAIL verdict.

Grounding in the thesis literature:
- Self-Debug (Chen et al., 2023 — Kap.2 §Self-Debug Using Execution Feedback):
  execution feedback is a stronger repair signal than free-form critique.
- ThinkPRM (Khalifa et al., 2025 — Kap.2 §Calibrated PRMs): grading intermediate
  artifacts against ground truth catches more errors than end-to-end judging.
- ReAct (Yao et al., 2023): observations from real tools inform the next action.

Design constraints:
- Zero new Python dependencies. All checks use the standard library.
- Node.js is used opportunistically for JavaScript syntax checks; when
  unavailable the JS gate skips (does not fail).
- Every gate returns quickly (< 2 s per file on typical inputs) so the loop
  cannot stall an evaluation run.
"""
from __future__ import annotations

import ast
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Optional

logger = logging.getLogger(__name__)

# Per-gate execution cap. Keeps a hostile / infinite-recursion payload from
# stalling the orchestrator. Node in particular can hang if invoked wrong.
_GATE_TIMEOUT_SECONDS = 8.0


@dataclass
class GateResult:
    """Outcome of running a single-file gate."""
    ok: bool                    # True if the file passes the gate
    kind: str                   # gate name: "py_compile", "node_check", "html_parse", "css_brace_balance", "skipped"
    error: str = ""             # exact compiler/parser error text; empty on success
    detail: str = ""            # extra context (line number, offset) for the Repair prompt

    def as_dict(self) -> dict:
        return {"ok": self.ok, "kind": self.kind, "error": self.error, "detail": self.detail}


# ---------------------------------------------------------------------------
# Language dispatch
# ---------------------------------------------------------------------------

# Extension → gate function. Anything not listed skips gracefully (ok=True,
# kind="skipped") — we never fail a file just because we don't know how to
# check it.
_EXTENSION_MAP: dict[str, str] = {
    ".py":   "python",
    ".js":   "javascript",
    ".mjs":  "javascript",
    ".cjs":  "javascript",
    ".html": "html",
    ".htm":  "html",
    ".css":  "css",
    ".json": "json",
}


def gate_for(path: str) -> str:
    """Return the gate kind for a given file path, or 'skipped'."""
    ext = os.path.splitext(path)[1].lower()
    return _EXTENSION_MAP.get(ext, "skipped")


def check_file(abs_path: str) -> GateResult:
    """Run the appropriate gate for `abs_path`. Never raises."""
    try:
        kind = gate_for(abs_path)
        if kind == "python":
            return _gate_python(abs_path)
        if kind == "javascript":
            return _gate_javascript(abs_path)
        if kind == "html":
            return _gate_html(abs_path)
        if kind == "css":
            return _gate_css(abs_path)
        if kind == "json":
            return _gate_json(abs_path)
        return GateResult(ok=True, kind="skipped")
    except Exception as exc:  # last-resort safety net — never crash the orchestrator
        logger.warning("gate crashed on %s: %s", abs_path, exc)
        return GateResult(ok=True, kind="skipped", detail=f"gate exception: {exc}")


# ---------------------------------------------------------------------------
# Python: `ast.parse` (stdlib, always available, exact error location)
# ---------------------------------------------------------------------------

def _gate_python(abs_path: str) -> GateResult:
    """SyntaxError catcher via `ast.parse`. Faster and more precise than
    subprocess py_compile — gives us line/offset for the Repair prompt."""
    try:
        with open(abs_path, "r", encoding="utf-8") as fh:
            source = fh.read()
    except OSError as exc:
        return GateResult(ok=False, kind="py_compile",
                          error=f"could not read {abs_path}: {exc}")

    try:
        ast.parse(source, filename=abs_path)
    except SyntaxError as exc:
        # Reconstruct a copy-pasteable error snippet so the Repair prompt has
        # the same signal the professor would see in their terminal.
        line = exc.lineno or 1
        col = (exc.offset or 1) - 1
        snippet_lines = source.splitlines()
        context = ""
        if 1 <= line <= len(snippet_lines):
            src_line = snippet_lines[line - 1]
            marker = " " * max(0, col) + "^"
            context = f"\n{src_line}\n{marker}"
        detail = f"line {line}, column {col + 1}"
        return GateResult(
            ok=False,
            kind="py_compile",
            error=f"SyntaxError: {exc.msg} at {detail}{context}",
            detail=detail,
        )
    return GateResult(ok=True, kind="py_compile")


# ---------------------------------------------------------------------------
# JavaScript: `node --check <file>` if node is on PATH
# ---------------------------------------------------------------------------

def _run_node_check(node: str, source: str, module_mode: bool) -> subprocess.CompletedProcess:
    """Run `node --check` with the source piped on stdin.

    Deliberately NOT `node --check <path>`: with a path argument, node first
    resolves the argument through the CommonJS loader (Module._resolveFilename
    -> _findPath -> toRealPath). That resolution walks node_modules and calls
    realpath on ancestor directories, and it can fail for reasons that have
    nothing to do with the file's syntax — on Windows it raised
    "EISDIR: illegal operation on a directory, lstat 'F:'" for a perfectly
    valid file sitting on drive F:. Node then exits non-zero and the gate
    reported a syntax failure for a file that parses fine.

    Reading from stdin skips module resolution entirely, so the exit code
    reflects the parser and nothing else.
    """
    args = [node]
    if module_mode:
        args.append("--input-type=module")
    args.append("--check")
    return subprocess.run(
        args,
        input=source,
        capture_output=True,
        text=True,
        # Pin the pipe encoding. With text=True Python otherwise encodes stdin
        # with the locale codec, which on a default Windows console is cp1252 —
        # any non-Latin-1 character the model emits (box drawing, arrows, emoji,
        # accented comments) raises UnicodeEncodeError and the file never gets
        # checked at all. Generated source is written as UTF-8, so read it back
        # out the same way.
        encoding="utf-8",
        errors="replace",
        timeout=_GATE_TIMEOUT_SECONDS,
    )


# Node reports ESM-only syntax in a classic-script parse as one of these. They
# mean "wrong mode", not "broken file", so we re-check the source as a module.
_ESM_HINTS = (
    "Failed to load the ES module",
    "Cannot use import statement outside a module",
    "Unexpected token 'export'",
    "await is only valid in async functions",
    'set "type": "module"',
)


def _gate_javascript(abs_path: str) -> GateResult:
    node = shutil.which("node")
    if not node:
        return GateResult(
            ok=True,
            kind="skipped",
            detail="node binary not found on PATH; JS syntax gate skipped",
        )

    try:
        with open(abs_path, "r", encoding="utf-8") as fh:
            source = fh.read()
    except OSError as exc:
        return GateResult(ok=False, kind="node_check",
                          error=f"could not read {abs_path}: {exc}")

    try:
        proc = _run_node_check(node, source, module_mode=False)
        if proc.returncode == 0:
            return GateResult(ok=True, kind="node_check")

        err = (proc.stderr or proc.stdout or "").strip()

        # Classic-script parse rejected ESM syntax — re-check as a module
        # before calling the file broken.
        if any(hint in err for hint in _ESM_HINTS):
            module_proc = _run_node_check(node, source, module_mode=True)
            if module_proc.returncode == 0:
                return GateResult(ok=True, kind="node_check", detail="checked as ES module")
            err = (module_proc.stderr or module_proc.stdout or "").strip()

        # Only a real parser diagnostic counts as a failure. Anything else is
        # the gate failing to run, not the file being wrong — and per this
        # module's contract we never fail a file we could not actually check.
        if "SyntaxError" not in err:
            return GateResult(
                ok=True,
                kind="skipped",
                detail=f"node could not run the syntax check: {err[:300]}",
            )

        # Node prints file:line syntax errors; keep the first ~600 chars so the
        # Repair prompt has the exact message but doesn't balloon.
        return GateResult(
            ok=False,
            kind="node_check",
            error=err[:600] or "node --check failed with no output",
        )
    except subprocess.TimeoutExpired:
        return GateResult(
            ok=True, kind="skipped",
            detail=f"node --check timed out after {_GATE_TIMEOUT_SECONDS}s; JS gate skipped",
        )
    except OSError as exc:
        return GateResult(
            ok=True, kind="skipped",
            detail=f"could not launch node: {exc}",
        )


# ---------------------------------------------------------------------------
# HTML: html.parser strict mode
# ---------------------------------------------------------------------------

class _StrictHTMLValidator(HTMLParser):
    """Track unclosed tags — Python's html.parser doesn't raise on mismatched
    close tags by default, so we do it manually. Void elements are ignored."""

    VOID = frozenset({
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    })

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.errors: list[str] = []
        self.stack: list[tuple[str, int]] = []

    def error(self, message: str) -> None:  # abstract in older Pythons
        self.errors.append(message)

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag.lower() not in self.VOID:
            self.stack.append((tag.lower(), self.getpos()[0]))

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self.VOID:
            return
        if not self.stack:
            self.errors.append(f"line {self.getpos()[0]}: stray </{tag}> — no opening tag")
            return
        opened, opened_line = self.stack[-1]
        if opened != tag.lower():
            self.errors.append(
                f"line {self.getpos()[0]}: </{tag}> does not match "
                f"<{opened}> opened at line {opened_line}"
            )
            # heuristic recovery: pop until we find a match or empty the stack
            while self.stack and self.stack[-1][0] != tag.lower():
                self.stack.pop()
            if self.stack:
                self.stack.pop()
            return
        self.stack.pop()


def _gate_html(abs_path: str) -> GateResult:
    try:
        with open(abs_path, "r", encoding="utf-8") as fh:
            source = fh.read()
    except OSError as exc:
        return GateResult(ok=False, kind="html_parse",
                          error=f"could not read {abs_path}: {exc}")

    parser = _StrictHTMLValidator()
    try:
        parser.feed(source)
        parser.close()
    except Exception as exc:
        return GateResult(ok=False, kind="html_parse",
                          error=f"html parse crashed: {exc}")

    unclosed = [f"<{tag}> opened at line {ln}" for tag, ln in parser.stack]
    problems = list(parser.errors) + [f"unclosed tag {u}" for u in unclosed]
    if problems:
        return GateResult(
            ok=False,
            kind="html_parse",
            error="\n".join(problems[:10]),
            detail=f"{len(problems)} HTML issue(s) detected",
        )
    return GateResult(ok=True, kind="html_parse")


# ---------------------------------------------------------------------------
# CSS: brace + parenthesis balance
# ---------------------------------------------------------------------------

def _gate_css(abs_path: str) -> GateResult:
    """Cheap CSS sanity check — brace balance + no stray semicolons after `}`.
    A real CSS parser would be heavier; brace balance catches ~90% of the
    'model forgot the closing }' failures that break the whole stylesheet."""
    try:
        with open(abs_path, "r", encoding="utf-8") as fh:
            source = fh.read()
    except OSError as exc:
        return GateResult(ok=False, kind="css_brace_balance",
                          error=f"could not read {abs_path}: {exc}")

    # Strip /* ... */ comments so { or } inside comments don't count.
    stripped = _strip_css_comments(source)
    depth = 0
    paren_depth = 0
    line = 1
    for ch in stripped:
        if ch == "\n":
            line += 1
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return GateResult(
                    ok=False,
                    kind="css_brace_balance",
                    error=f"line {line}: '}}' with no matching '{{'",
                )
        elif ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth -= 1
            if paren_depth < 0:
                return GateResult(
                    ok=False,
                    kind="css_brace_balance",
                    error=f"line {line}: ')' with no matching '('",
                )
    if depth != 0:
        return GateResult(
            ok=False,
            kind="css_brace_balance",
            error=f"unbalanced braces at EOF: depth={depth} ({'open' if depth > 0 else 'unmatched close'})",
        )
    if paren_depth != 0:
        return GateResult(
            ok=False,
            kind="css_brace_balance",
            error=f"unbalanced parentheses at EOF: depth={paren_depth}",
        )
    return GateResult(ok=True, kind="css_brace_balance")


def _strip_css_comments(source: str) -> str:
    """Remove /* ... */ blocks. Doesn't handle // comments (not standard CSS)."""
    out: list[str] = []
    i = 0
    n = len(source)
    while i < n:
        if i + 1 < n and source[i] == "/" and source[i + 1] == "*":
            end = source.find("*/", i + 2)
            if end == -1:
                # Unterminated comment — treat rest of file as comment
                break
            # Preserve newlines so line numbers still match
            comment = source[i:end + 2]
            out.append("".join(ch for ch in comment if ch == "\n"))
            i = end + 2
        else:
            out.append(source[i])
            i += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# JSON: json.loads
# ---------------------------------------------------------------------------

def _gate_json(abs_path: str) -> GateResult:
    import json
    try:
        with open(abs_path, "r", encoding="utf-8") as fh:
            source = fh.read()
    except OSError as exc:
        return GateResult(ok=False, kind="json_parse",
                          error=f"could not read {abs_path}: {exc}")
    try:
        json.loads(source)
    except json.JSONDecodeError as exc:
        return GateResult(
            ok=False, kind="json_parse",
            error=f"JSON parse error: {exc.msg} at line {exc.lineno}, column {exc.colno}",
            detail=f"line {exc.lineno}, column {exc.colno}",
        )
    return GateResult(ok=True, kind="json_parse")


# ---------------------------------------------------------------------------
# Batch entry point used by the orchestrator
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Cross-file DOM coherence: catches "the JS references markup nobody wrote"
# ---------------------------------------------------------------------------
#
# Every gate above is single-file and syntax-only. That leaves a real hole:
# app.js can be perfectly valid JavaScript, index.html perfectly valid HTML,
# and the app still be a dead blank page — because app.js does
#   document.querySelector("#display")  ->  null
#   if (!displayEl) return;
# and nothing in the workspace ever defines id="display". Every gate above
# reports PASS for that exact case; nothing checks whether the files agree
# with each other. This gate closes that gap for the browser-app path
# (index.html + a JS file), which is the common case for the desktop app's
# calculator/landing-page demos.
_ID_ATTR_RE = re.compile(r"""\bid\s*=\s*["']([\w-]+)["']""")
_CLASS_ATTR_RE = re.compile(r"""\bclass(?:Name)?\s*=\s*["']([^"']+)["']""")
_SET_ATTR_RE = re.compile(r"""setAttribute\(\s*["'](id|class)["']\s*,\s*["']([^"']+)["']""")
_CLASSLIST_ADD_RE = re.compile(r"classList\.add\(([^)]*)\)")
_QUOTED_TOKEN_RE = re.compile(r"""["']([\w-]+)["']""")

_GET_BY_ID_RE = re.compile(r"""getElementById\(\s*["']([\w-]+)["']\s*\)""")
_QUERY_SELECTOR_RE = re.compile(r"""querySelector(?:All)?\(\s*["']([^"']+)["']\s*\)""")
_SELECTOR_TOKEN_RE = re.compile(r"([#.])([\w-]+)")


def _collect_defined_tokens(text: str) -> tuple[set[str], set[str]]:
    """ids/classes a source file DEFINES: HTML attributes, or JS that assigns
    .id / .className / classList.add / setAttribute — OR embeds them in an
    innerHTML-style template string, which the same attribute regexes catch
    since `id="x"` / `class="x"` look the same whether they sit in a .html
    file or inside a JS template literal."""
    ids = set(_ID_ATTR_RE.findall(text))
    classes: set[str] = set()
    for value in _CLASS_ATTR_RE.findall(text):
        classes.update(value.split())
    for kind, value in _SET_ATTR_RE.findall(text):
        (ids if kind == "id" else classes).update(
            [value] if kind == "id" else value.split()
        )
    for args in _CLASSLIST_ADD_RE.findall(text):
        classes.update(_QUOTED_TOKEN_RE.findall(args))
    return ids, classes


def _collect_referenced_tokens(text: str) -> tuple[set[str], set[str]]:
    """ids/classes a JS file READS via getElementById/querySelector(All)."""
    ids = set(_GET_BY_ID_RE.findall(text))
    classes: set[str] = set()
    for selector in _QUERY_SELECTOR_RE.findall(text):
        for prefix, name in _SELECTOR_TOKEN_RE.findall(selector):
            (ids if prefix == "#" else classes).add(name)
    return ids, classes


_ARROW_BEFORE_BRACE_RE = re.compile(r"=>\s*$")
_FUNCTION_HEAD_BEFORE_BRACE_RE = re.compile(r"\bfunction\b[^{}()]*\([^()]*\)\s*$")


def _is_function_body_open(text_before_brace: str) -> bool:
    """Does this `{` open a FUNCTION body, as opposed to an object literal,
    an `if`/`for`/`while`/`try` block, or any other brace? Only function
    bodies are deferred until something calls them later; everything else
    (including an object literal like `{ a: foo(), b: bar() }`) executes
    inline, in source order, at the same point as the surrounding code."""
    tail = text_before_brace[-200:]
    return bool(_ARROW_BEFORE_BRACE_RE.search(tail) or _FUNCTION_HEAD_BEFORE_BRACE_RE.search(tail))


def _brace_depth_profile(source: str) -> list[int]:
    """FUNCTION-BODY nesting depth at every character offset (not raw brace
    depth — an object literal's own `{ }` must NOT count, see
    `_is_function_body_open`), skipping strings/template-literals/comments so
    braces inside them don't miscount.

    Used as a cheap proxy for "does this statement run immediately when the
    script loads, or only later when some function is called". Top-level
    statements in an IIFE — including object literals and if/for/try blocks
    written directly in that top level — all sit at the same (shallowest)
    depth and execute in source order; that shared depth and that ordering
    assumption are exactly what the stale-reference check below relies on."""
    depth = 0
    is_function_brace_stack: list[bool] = []
    n = len(source)
    profile = [0] * n
    i = 0
    while i < n:
        ch = source[i]
        if ch == "/" and i + 1 < n and source[i + 1] == "/":
            end = source.find("\n", i)
            end = n if end == -1 else end
            for j in range(i, end):
                profile[j] = depth
            i = end
            continue
        if ch == "/" and i + 1 < n and source[i + 1] == "*":
            end = source.find("*/", i + 2)
            end = n if end == -1 else end + 2
            for j in range(i, min(end, n)):
                profile[j] = depth
            i = end
            continue
        if ch in ("'", '"', "`"):
            quote = ch
            j = i + 1
            while j < n and source[j] != quote:
                j += 2 if source[j] == "\\" else 1
            j = min(j + 1, n)
            for k in range(i, j):
                profile[k] = depth
            i = j
            continue
        profile[i] = depth
        if ch == "{":
            is_func = _is_function_body_open(source[:i])
            is_function_brace_stack.append(is_func)
            if is_func:
                depth += 1
        elif ch == "}":
            if is_function_brace_stack:
                was_func = is_function_brace_stack.pop()
                if was_func:
                    depth = max(0, depth - 1)
        i += 1
    return profile


def _find_stale_dom_references(js_rel: str, js_source: str) -> Optional[GateResult]:
    """Catch the "captured a DOM reference before building the markup" bug:

        const display = { result: app.querySelector(".result") };  // null here
        app.innerHTML = `...<div class="result">...`;              // built LATER
        function render() { if (!display.result) return; ... }     // dead forever

    Every id/class involved genuinely exists in the file — my other coherence
    check (which only asks "does this appear ANYWHERE") sees nothing wrong.
    The bug is purely about WHEN: the query ran before the element existed and
    the result was cached, so it stays null for the file's whole lifetime.

    Heuristic, restricted to statements at the shallowest brace depth in the
    file (roughly: the IIFE's own top level, not inside any function) —
    those execute once, in source order, at script-load time. A query
    re-issued fresh inside a function called after the build (the correct,
    common pattern) sits one level deeper and is deliberately left alone."""
    profile = _brace_depth_profile(js_source)
    if not profile:
        return None

    query_events: list[tuple[int, str, str]] = []  # (offset, kind, name)
    for m in _GET_BY_ID_RE.finditer(js_source):
        query_events.append((m.start(), "id", m.group(1)))
    for m in _QUERY_SELECTOR_RE.finditer(js_source):
        offset = m.start()
        for pm in _SELECTOR_TOKEN_RE.finditer(m.group(1)):
            query_events.append((offset, "id" if pm.group(1) == "#" else "class", pm.group(2)))

    creation_events: list[tuple[int, str, str]] = []
    for m in _ID_ATTR_RE.finditer(js_source):
        creation_events.append((m.start(), "id", m.group(1)))
    for m in _CLASS_ATTR_RE.finditer(js_source):
        offset = m.start()
        for cls in m.group(1).split():
            creation_events.append((offset, "class", cls))
    for m in _SET_ATTR_RE.finditer(js_source):
        offset = m.start()
        kind, value = m.group(1), m.group(2)
        if kind == "id":
            creation_events.append((offset, "id", value))
        else:
            for cls in value.split():
                creation_events.append((offset, "class", cls))
    for m in _CLASSLIST_ADD_RE.finditer(js_source):
        offset = m.start()
        for tok in _QUOTED_TOKEN_RE.finditer(m.group(1)):
            creation_events.append((offset, "class", tok.group(1)))

    if not query_events and not creation_events:
        return None
    # Derive the baseline from where the actual DOM activity sits, NOT from
    # `min(profile)` over the raw source: depth 0 is "outside every function",
    # which for an IIFE-wrapped script (function () { ... })() is only the
    # handful of characters before/after the wrapper itself — everything the
    # script actually does lives one level deeper, inside the IIFE's own
    # function body. Using the file-wide minimum silently excluded every real
    # event from the "top level" bucket and made this check a no-op.
    top_level_depth = min(profile[offset] for offset, _, _ in query_events + creation_events)

    # Earliest TOP-LEVEL creation offset per (kind, name).
    earliest_creation: dict[tuple[str, str], int] = {}
    for offset, kind, name in creation_events:
        if profile[offset] != top_level_depth:
            continue
        key = (kind, name)
        if key not in earliest_creation or offset < earliest_creation[key]:
            earliest_creation[key] = offset

    stale: list[str] = []
    seen: set[tuple[str, str]] = set()
    for offset, kind, name in query_events:
        if profile[offset] != top_level_depth:
            continue  # queried inside a function — runs later, timing unknown but not this bug
        key = (kind, name)
        creation_offset = earliest_creation.get(key)
        if creation_offset is not None and creation_offset > offset and key not in seen:
            seen.add(key)
            marker = f"#{name}" if kind == "id" else f".{name}"
            stale.append(marker)

    if not stale:
        return None

    message = (
        f"{js_rel} queries {', '.join(stale)} at the top level BEFORE the line that builds "
        "that markup (an innerHTML/createElement/classList call further down in the same "
        "top-level scope). The captured reference is null at that point and stays null "
        "forever — nothing re-queries it after the build — so any code using it (typically "
        "the render/update function) silently does nothing. Move the query to AFTER the "
        "markup is built, or re-query fresh inside the function that uses it instead of "
        "caching the result at the top of the file."
    )
    return GateResult(
        ok=False,
        kind="dom_coherence",
        error=message,
        detail=f"{len(stale)} DOM reference(s) captured before their markup exists",
    )


def check_browser_app_coherence(
    workspace_path: str, files: list[str]
) -> list[tuple[str, GateResult]]:
    """Flag ids/classes a JS file queries but that no file in the workspace
    ever defines. Never raises; skips (returns []) when there's no HTML+JS
    pair to check, matching every other gate's "don't fail what we can't
    check" contract."""
    html_files = [f for f in files if f.lower().endswith((".html", ".htm"))]
    js_files = [f for f in files if f.lower().endswith((".js", ".mjs", ".cjs"))]
    if not html_files or not js_files:
        return []

    html_rel = html_files[0]
    try:
        with open(os.path.join(workspace_path, html_rel), "r", encoding="utf-8") as fh:
            html_ids, html_classes = _collect_defined_tokens(fh.read())
    except OSError:
        return []

    findings: list[tuple[str, GateResult]] = []
    for js_rel in js_files:
        try:
            with open(os.path.join(workspace_path, js_rel), "r", encoding="utf-8") as fh:
                js_source = fh.read()
        except OSError:
            continue

        created_ids, created_classes = _collect_defined_tokens(js_source)
        referenced_ids, referenced_classes = _collect_referenced_tokens(js_source)
        missing_ids = sorted(referenced_ids - html_ids - created_ids)
        missing_classes = sorted(referenced_classes - html_classes - created_classes)
        if missing_ids or missing_classes:
            parts = []
            if missing_ids:
                parts.append("id(s) " + ", ".join(f"#{i}" for i in missing_ids))
            if missing_classes:
                parts.append("class(es) " + ", ".join(f".{c}" for c in missing_classes))
            message = (
                f"{js_rel} looks up {' and '.join(parts)} via getElementById/querySelector, "
                f"but {html_rel} never defines them and {js_rel} never creates them "
                "(no id=/class= attribute, no classList.add/setAttribute, no innerHTML markup "
                "containing them). Code depending on these elements will find null/empty and "
                "silently do nothing at runtime — the page will render without that part of "
                "the UI, even though every file parses as syntactically valid. Add the missing "
                "markup to the HTML, or build it dynamically in the script, so every id/class "
                "the script queries actually exists somewhere."
            )
            result = GateResult(
                ok=False,
                kind="dom_coherence",
                error=message,
                detail=f"{len(missing_ids) + len(missing_classes)} missing selector(s)",
            )
            findings.append((html_rel, result))
            findings.append((js_rel, result))

        # Every id/class above can genuinely exist somewhere and STILL be dead:
        # captured via querySelector before the line that builds it. That bug
        # is invisible to the "does it exist anywhere" check above.
        stale_result = _find_stale_dom_references(js_rel, js_source)
        if stale_result is not None:
            findings.append((js_rel, stale_result))
    return findings


def run_gates(workspace_path: str, files: list[str]) -> list[tuple[str, GateResult]]:
    """Check every file in `files` (relative paths under `workspace_path`).
    Returns [(rel_path, GateResult)] in order. Files that fail to read are
    reported as gate failures; files whose extension is unknown are reported
    as skipped (ok=True)."""
    results: list[tuple[str, GateResult]] = []
    for rel in files:
        abs_path = os.path.join(workspace_path, rel)
        results.append((rel, check_file(abs_path)))
    results.extend(check_browser_app_coherence(workspace_path, files))
    return results


def failed(results: list[tuple[str, GateResult]]) -> list[tuple[str, GateResult]]:
    """Return only the failed gate results, preserving order."""
    return [(rel, r) for rel, r in results if not r.ok]


# ---------------------------------------------------------------------------
# Behavioral evidence: run project-supplied Python tests inside the workspace
# ---------------------------------------------------------------------------

_TEST_RUN_TIMEOUT_SECONDS = 120.0


def run_workspace_tests(workspace_path: str, timeout: float = _TEST_RUN_TIMEOUT_SECONDS) -> Optional[dict]:
    """Run Python tests found in the workspace (allow-listed command only:
    `<this interpreter> -m pytest -q` or unittest discovery fallback), with
    cwd pinned to the workspace and a hard timeout.

    Returns None when there are no test files (nothing to run), otherwise
    {"passed": bool, "command": str, "returncode": int, "output": str}.

    NOTE: this executes model-generated code. It is only invoked when the
    operator opts in via MARTINUS_RUN_WORKSPACE_TESTS=1.
    """
    import glob
    import sys

    if not workspace_path or not os.path.isdir(workspace_path):
        return None
    patterns = ("test_*.py", "*_test.py", "tests/test_*.py", "tests/*_test.py")
    has_tests = any(glob.glob(os.path.join(workspace_path, p)) for p in patterns)
    if not has_tests:
        return None

    try:
        import pytest  # noqa: F401
        cmd = [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"]
    except ImportError:
        cmd = [sys.executable, "-m", "unittest", "discover", "-v"]

    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "LANG": os.environ.get("LANG", "en_US.UTF-8"),
    }
    # Windows needs a few variables that are not optional: without SYSTEMROOT
    # the loader cannot initialise Winsock, and even `import pytest` fails with
    # "the requested service provider could not be loaded or initialized".
    # Keep the allow-list narrow — these are environment plumbing, not secrets.
    if os.name == "nt":
        for key in ("SYSTEMROOT", "SystemDrive", "TEMP", "TMP", "PATHEXT", "COMSPEC", "USERPROFILE"):
            value = os.environ.get(key)
            if value:
                env[key] = value
    try:
        proc = subprocess.run(
            cmd,
            cwd=workspace_path,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        return {
            "passed": proc.returncode == 0,
            "command": " ".join(cmd),
            "returncode": proc.returncode,
            "output": output[-4000:],
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "command": " ".join(cmd),
            "returncode": -1,
            "output": f"test run timed out after {timeout}s",
        }
    except OSError as exc:
        return {
            "passed": False,
            "command": " ".join(cmd),
            "returncode": -1,
            "output": f"could not launch test runner: {exc}",
        }
