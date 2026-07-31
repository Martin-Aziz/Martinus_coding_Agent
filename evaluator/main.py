#!/usr/bin/env python3
"""
Evaluator Agent Benchmark Pipeline
-----------------------------------
Tests the Martinus orchestrator across 10 tasks of increasing
difficulty, validates outputs via subprocess execution or Playwright, and
stores results in the episodic memory service.

Usage:
    python evaluator/main.py [--tasks 1,2,3] [--stress] [--profile medium] [--provider lmstudio]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Configuration — override via environment variables
# ---------------------------------------------------------------------------
# NOTE: The orchestrator exposes /v1/tasks/stream (SSE), not /v1/tasks/execute.
# The old constant was one of the P1 defects the audit flagged. The stream URL
# is what this evaluator actually POSTs to; the phase outputs are drained from
# the SSE `phase_completed` events.
ORCHESTRATOR_STREAM_URL = os.getenv(
    "ORCHESTRATOR_STREAM_URL",
    "http://127.0.0.1:8080/v1/tasks/stream",
)
OLLAMA_URL       = os.getenv("OLLAMA_URL",        "http://127.0.0.1:11434").rstrip("/")
EVALUATOR_MODEL  = os.getenv("EVALUATOR_MODEL",   "llama3.2")
AI_SERVICE_URL   = os.getenv("AI_SERVICE_URL",    "http://127.0.0.1:8090")
EVAL_PROVIDER    = os.getenv("EVALUATOR_PROVIDER", "lmstudio")
EVAL_PROFILE     = os.getenv("EVALUATOR_PROFILE",  "medium")
MAX_RETRIES      = int(os.getenv("MAX_RETRIES",    "3"))

# When JUDGE_PROVIDER=claude, evaluate_with_ollama() shells out to the Claude
# Code CLI instead of hitting Ollama. Useful when the audit machine has no
# local models loaded. Set to "ollama" (the default) to keep original behaviour.
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "ollama").strip().lower()
CLAUDE_JUDGE_BIN = (
    os.getenv("MARTINUS_CLAUDE_BINARY", "").strip() or "claude"
)

PYTHON_BIN = sys.executable  # reuse the interpreter running this script

# ---------------------------------------------------------------------------
# Master evaluator system prompt
# ---------------------------------------------------------------------------
MASTER_EVALUATOR_PROMPT = """You are an expert AI coding evaluator agent.

Your task is to rigorously evaluate another AI coding agent's output.

Evaluate the output for:
1. Correctness — does it solve the task correctly?
2. Code Quality — readability, structure, naming conventions
3. Edge Cases — are edge cases handled?
4. Performance — is the solution efficient?

STRICT RULES:
- Code that "looks right" but doesn't run → FAIL
- No error handling when required → penalise
- Hardcoded values instead of parameters → penalise
- Poor naming / unreadable code → penalise
- Missing async handling when needed → penalise

You will also receive execution context (subprocess output or browser test
results) where available. Factor this into your evaluation.

Output format (follow EXACTLY):

EVALUATION:
- Correctness: <your assessment>
- Code Quality: <your assessment>
- Edge Cases: <your assessment>
- Performance: <your assessment>

RESULT: PASS
SCORE: X/10

OR if failing:

RESULT: FAIL
SCORE: X/10

HINT: <one short, actionable hint — do NOT give the full solution>
"""

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------
TASKS: list[dict] = [
    {
        "id": 1,
        "title": "Palindrome Checker",
        "difficulty": "Easy",
        "category": "python_function",
        "description": (
            "Write a Python function `is_palindrome(s: str) -> bool` in a file called "
            "`string_utils.py`. The function must check whether a string is a palindrome, "
            "ignoring spaces and letter casing. "
            "Example: `is_palindrome('A man a plan a canal Panama')` → True."
        ),
        "target_file": "string_utils.py",
        "test_inputs": [
            ("is_palindrome('racecar')", True),
            ("is_palindrome('hello')", False),
            ("is_palindrome('A man a plan a canal Panama')", True),
            ("is_palindrome('')", True),
            ("is_palindrome('  ')", True),
        ],
    },
    {
        "id": 2,
        "title": "Two-Sum",
        "difficulty": "Easy",
        "category": "python_function",
        "description": (
            "Write a Python function `two_sum(nums: list, target: int) -> list` in a file called "
            "`two_sum.py`. The function must return the indices (as a list of two ints) of the two "
            "numbers in `nums` that add up to `target`. Assume exactly one solution exists. "
            "Example: `two_sum([2, 7, 11, 15], 9)` → [0, 1]."
        ),
        "target_file": "two_sum.py",
        "test_inputs": [
            ("two_sum([2, 7, 11, 15], 9)", [0, 1]),
            ("two_sum([3, 2, 4], 6)", [1, 2]),
            ("two_sum([3, 3], 6)", [0, 1]),
        ],
    },
    {
        "id": 3,
        "title": "CSV File Parser",
        "difficulty": "Medium",
        "category": "python_csv",
        "description": (
            "Write a Python function `parse_csv(filepath: str, column: str) -> dict` in a file "
            "called `file_parser.py`. The function must read a CSV file and return a dict with keys "
            "`mean`, `min`, `max` (as floats) for the specified numeric column. "
            "Raise a `ValueError` for missing files or non-numeric column values."
        ),
        "target_file": "file_parser.py",
        "csv_fixture": "sample.csv",
        "csv_content": "name,score,age\nAlice,85,30\nBob,92,25\nCarol,78,35\nDave,95,28\nEve,60,40\n",
        "test_inputs": [
            ("parse_csv('sample.csv', 'score')", {"mean": 82.0, "min": 60.0, "max": 95.0}),
        ],
    },
    {
        "id": 4,
        "title": "API Fetch with Error Handling",
        "difficulty": "Medium",
        "category": "python_api",
        "description": (
            "Write a Python function `fetch_data(url: str, timeout: int = 5) -> dict` in a file "
            "called `api_client.py`. The function must:\n"
            "- Fetch JSON from the given URL using the `requests` library\n"
            "- Return `{'success': True, 'data': <parsed json>}` on success\n"
            "- Return `{'success': False, 'error': '<message>'}` on network errors or timeouts\n"
            "- Return `{'success': False, 'error': 'invalid json'}` when the response body is not "
            "  valid JSON\n"
            "Do NOT let any exception propagate to the caller."
        ),
        "target_file": "api_client.py",
    },
    {
        "id": 5,
        "title": "CLI Todo App",
        "difficulty": "Medium",
        "category": "python_cli",
        "description": (
            "Build a CLI todo app in a file called `todo_cli.py`. It must support:\n"
            "  python todo_cli.py add <text>      — add a new todo item\n"
            "  python todo_cli.py list            — list all todos with 1-based index numbers\n"
            "  python todo_cli.py delete <index>  — delete the todo at the given 1-based index\n"
            "Store todos persistently in a local JSON file called `todos.json` in the same "
            "directory. Print a confirmation message for each add/delete operation."
        ),
        "target_file": "todo_cli.py",
    },
    {
        "id": 6,
        "title": "Mini Web App",
        "difficulty": "Medium",
        "category": "browser",
        "description": (
            "Create a self-contained webpage in a file called `app.html` that includes:\n"
            "- An input field for entering todo items\n"
            "- An 'Add' button that appends the entered text to a list below\n"
            "- Each list item has a 'Delete' button that removes it from the list\n"
            "- The list updates in-place without reloading the page\n"
            "Use only vanilla HTML, CSS, and JavaScript — no external libraries or frameworks."
        ),
        "target_file": "app.html",
    },
    {
        "id": 7,
        "title": "Bug Fix: sum()",
        "difficulty": "Easy",
        "category": "python_bugfix",
        "description": (
            "The following JavaScript function has two bugs. Identify and fix them, then write an "
            "equivalent Python function `bugfix_sum(arr: list) -> int` in a file called `bugfix.py`:\n\n"
            "```javascript\n"
            "function sum(arr) {\n"
            "  let total;\n"
            "  for (let i = 0; i <= arr.length; i++) {\n"
            "    total += arr[i];\n"
            "  }\n"
            "  return total;\n"
            "}\n"
            "```\n\n"
            "Bug 1: `total` is uninitialised (causes NaN — should be initialised to 0).\n"
            "Bug 2: The loop condition is `i <= arr.length` (off-by-one — should be `i < arr.length`).\n"
            "In `bugfix.py`, include a brief comment explaining each bug."
        ),
        "target_file": "bugfix.py",
        "test_inputs": [
            ("bugfix_sum([1, 2, 3, 4, 5])", 15),
            ("bugfix_sum([])", 0),
            ("bugfix_sum([-1, 1])", 0),
        ],
    },
    {
        "id": 8,
        "title": "Refactoring",
        "difficulty": "Hard",
        "category": "llm_judge_only",
        "description": (
            "Refactor the following messy Python function into clean, well-named, modular code. "
            "Write the result in a file called `refactored.py`. Use descriptive variable names, "
            "add a docstring, and split the logic into at least two functions:\n\n"
            "```python\n"
            "def do_stuff(x, y, z, a, b):\n"
            "    r = []\n"
            "    for i in range(len(x)):\n"
            "        if x[i] > a and x[i] < b:\n"
            "            r.append(x[i] * y + z)\n"
            "    return r\n"
            "```"
        ),
        "target_file": "refactored.py",
    },
    {
        "id": 9,
        "title": "Performance: Duplicate Detection",
        "difficulty": "Hard",
        "category": "python_perf",
        "description": (
            "Write a Python function `has_duplicates(items: list) -> bool` in a file called "
            "`duplicates.py`. The function must return True if the list contains any duplicate "
            "values, False otherwise. It must run in O(n) time complexity and handle a list of "
            "1,000,000 items in under 1 second."
        ),
        "target_file": "duplicates.py",
        "perf_threshold_seconds": 1.0,
    },
    {
        "id": 10,
        "title": "TTL Cache",
        "difficulty": "Hard",
        "category": "python_class",
        "description": (
            "Design a Python class `TTLCache` in a file called `ttl_cache.py` with the following "
            "methods:\n"
            "  `set(key, value, ttl_seconds: float)` — store a value with a time-to-live\n"
            "  `get(key)` — return the value if not expired, or `None` if expired or missing\n"
            "  `delete(key)` — remove a key\n"
            "  `clear()` — remove all keys\n"
            "Expired entries must not be returned by `get`. Thread safety is not required."
        ),
        "target_file": "ttl_cache.py",
    },
]

STRESS_TASK: dict = {
    "id": 99,
    "title": "Stress: CLI Todo with TTL Cache",
    "difficulty": "Hard+",
    "category": "python_cli",
    "description": (
        "Build a CLI todo app in a file called `todo_cached.py` that:\n"
        "  python todo_cached.py add <text>    — add a todo with a 60-second TTL\n"
        "  python todo_cached.py list          — list non-expired todos with 1-based indices\n"
        "  python todo_cached.py delete <index> — delete by 1-based index\n"
        "Instead of a plain JSON file, use an in-memory TTL cache class `TTLCache` (define it "
        "in the same file) as the storage layer. Persist the cache state to `todos_cache.json` "
        "between runs. Expired items must not appear in list."
    ),
    "target_file": "todo_cached.py",
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ExecutionResult:
    passed: bool
    output: str
    error: str = ""
    timing: Optional[float] = None
    details: str = ""


@dataclass
class TaskResult:
    task_id: int
    title: str
    difficulty: str
    score: int
    result: str          # "PASS" | "FAIL"
    attempts: int
    execution: ExecutionResult
    raw_evaluation: str
    hints_used: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Ollama judge
# ---------------------------------------------------------------------------


def evaluate_with_ollama(
    task_description: str,
    candidate_output: str,
    execution_context: str = "",
) -> str:
    """Grade candidate_output. Backed by Ollama by default; when
    JUDGE_PROVIDER=claude we shell out to the local `claude` CLI instead.
    Function name is kept for API compatibility with call sites."""
    user_content = (
        f"TASK:\n{task_description}\n\nCANDIDATE OUTPUT:\n{candidate_output}"
    )
    if execution_context:
        user_content += (
            f"\n\nEXECUTION CONTEXT (subprocess / browser test results):\n"
            f"{execution_context}"
        )

    if JUDGE_PROVIDER == "claude":
        # Fold the master evaluator prompt into a single stdin blob so we can
        # use `claude -p --output-format json`, same shape the Martinus
        # adapter uses.
        merged = (
            f"[SYSTEM INSTRUCTIONS]\n{MASTER_EVALUATOR_PROMPT}\n\n"
            f"[USER REQUEST]\n{user_content}"
        )
        try:
            proc = subprocess.run(
                [CLAUDE_JUDGE_BIN, "-p", "--output-format", "json"],
                input=merged,
                capture_output=True,
                text=True,
                timeout=180,
            )
            if proc.returncode != 0:
                return f"Evaluator error (claude exit {proc.returncode}): {proc.stderr.strip()[:400]}"
            raw = proc.stdout.strip()
            try:
                envelope = json.loads(raw)
                if isinstance(envelope, dict):
                    return str(
                        envelope.get("result") or envelope.get("text") or envelope.get("content") or raw
                    )
                return raw
            except json.JSONDecodeError:
                return raw
        except subprocess.TimeoutExpired:
            return "Evaluator error: claude judge timed out after 180s"
        except FileNotFoundError:
            return f"Evaluator error: claude binary not found at {CLAUDE_JUDGE_BIN!r}"
        except Exception as exc:
            return f"Evaluator error: {exc}"

    # Original Ollama path
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": EVALUATOR_MODEL,
                "messages": [
                    {"role": "system", "content": MASTER_EVALUATOR_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                "temperature": 0.0,
                "stream":      False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "No evaluation returned.")
    except Exception as exc:
        return f"Evaluator error: {exc}"


def parse_evaluation(raw: str) -> dict:
    """Extract RESULT, SCORE, and HINT from judge free-text response."""
    result_match = re.search(r"RESULT\s*:\s*(PASS|FAIL)", raw, re.IGNORECASE)
    score_match  = re.search(r"SCORE\s*:\s*(\d+)\s*/\s*10", raw, re.IGNORECASE)
    hint_match   = re.search(
        r"HINT\s*:\s*(.+?)(?:\n\n|\Z)", raw, re.IGNORECASE | re.DOTALL
    )
    return {
        "result": result_match.group(1).upper() if result_match else "FAIL",
        "score":  int(score_match.group(1)) if score_match else 0,
        "hint":   hint_match.group(1).strip() if hint_match else "",
        "raw":    raw,
    }


# ---------------------------------------------------------------------------
# Candidate agent
# ---------------------------------------------------------------------------


def run_candidate_agent(
    task_desc: str,
    workspace_path: str,
    session_id: str,
    hint: str = "",
) -> str:
    """Send a task (with optional hint) to the orchestrator's SSE stream and
    collect phase output. Falls back to a friendly error string on any error
    so the judge can still evaluate."""
    full_desc = task_desc
    if hint:
        full_desc = (
            f"{task_desc}\n\n"
            f"⚠️  Previous attempt failed. Hint for this retry:\n{hint}"
        )

    print(f"  → Streaming orchestrator (session={session_id}) …")
    payload = {
        "sessionId":       session_id,
        "task":            full_desc,
        "workspacePath":   workspace_path,
        "goals":           ["produce clean, runnable code using <file> tags"],
        "provider":        EVAL_PROVIDER,
        "hardwareProfile": EVAL_PROFILE,
    }

    phase_outputs: list[str] = []
    error_lines: list[str] = []
    try:
        with requests.post(
            ORCHESTRATOR_STREAM_URL,
            json=payload,
            stream=True,
            timeout=1200,
            headers={"Accept": "text/event-stream"},
        ) as resp:
            if resp.status_code != 200:
                return f"Orchestrator error {resp.status_code}: {resp.text[:400]}"
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data: "):
                    continue
                try:
                    event = json.loads(raw_line[6:])
                except json.JSONDecodeError:
                    continue
                etype = event.get("type", "")
                if etype == "phase_started":
                    print(f"    → {event.get('phase', '?')} started")
                elif etype == "phase_completed":
                    phase = event.get("phase", "?")
                    output = event.get("output", "") or ""
                    print(f"    ✓ {phase} completed ({event.get('model', '?')}, {len(output)} chars)")
                    phase_outputs.append(f"--- Phase: {phase} ---\n{output}")
                elif etype == "phase_failed":
                    err = event.get("error", "?")
                    print(f"    ✗ {event.get('phase', '?')} failed: {err[:120]}")
                    error_lines.append(f"phase_failed {event.get('phase')}: {err}")
                elif etype == "error":
                    msg = event.get("message", "?")
                    print(f"    !! run error: {msg[:200]}")
                    error_lines.append(f"run_error: {msg}")
                elif etype == "run_completed":
                    print(f"    ✓ run_completed")
    except requests.exceptions.RequestException as exc:
        return f"Orchestrator connection error: {exc}"

    if not phase_outputs and error_lines:
        return "No phase output returned.\n" + "\n".join(error_lines)
    if not phase_outputs:
        return "No phase output returned."
    if error_lines:
        phase_outputs.append("--- Errors ---\n" + "\n".join(error_lines))
    return "\n\n".join(phase_outputs)


# ---------------------------------------------------------------------------
# Code execution helpers
# ---------------------------------------------------------------------------


def _run_script(
    script_path: str,
    args: Optional[list] = None,
    cwd: Optional[str] = None,
    timeout: int = 15,
) -> tuple[int, str, str]:
    """Run a Python script; return (exit_code, stdout, stderr)."""
    cmd = [PYTHON_BIN, script_path] + (args or [])
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd or os.path.dirname(script_path),
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Execution timed out after {timeout}s"
    except Exception as exc:
        return -1, "", str(exc)


def _write_and_run_driver(
    workspace: str,
    driver_name: str,
    driver_code: str,
    timeout: int = 15,
) -> ExecutionResult:
    """Write a test-driver script to workspace and execute it."""
    driver_path = os.path.join(workspace, driver_name)
    with open(driver_path, "w") as fh:
        fh.write(driver_code)
    code, out, err = _run_script(driver_path, cwd=workspace, timeout=timeout)
    passed = code == 0 and "FAIL" not in out.upper() and not err.strip()
    return ExecutionResult(passed=passed, output=out, error=err)


def execute_python_function_task(task: dict, workspace: str) -> ExecutionResult:
    """Run assertions for a simple Python-function task."""
    target = os.path.join(workspace, task["target_file"])
    if not os.path.exists(target):
        return ExecutionResult(
            passed=False, output="",
            error=f"{task['target_file']} not found in workspace",
        )

    module = task["target_file"].replace(".py", "")
    lines = [
        "import sys, os; sys.path.insert(0, '.')",
        f"from {module} import *",
        "failures = []",
    ]
    for expr, expected in task.get("test_inputs", []):
        # Use double-quoted f-string with %r-formatting on the raw expression so
        # single quotes inside `expr` (e.g. is_palindrome('racecar')) do not
        # collide with the f-string's own quoting.
        lines += [
            f"result = {expr}",
            f"expected = {repr(expected)}",
            f"if result != expected:",
            f"    failures.append(%r + f' -> {{result!r}} (expected {{expected!r}})')" % expr,
        ]
    lines += [
        "if failures:",
        "    print('FAIL')",
        "    print('\\n'.join(failures))",
        "    sys.exit(1)",
        "else:",
        "    print('PASS -- all test cases passed')",
    ]
    return _write_and_run_driver(workspace, "_eval_driver.py", "\n".join(lines))


def execute_csv_task(task: dict, workspace: str) -> ExecutionResult:
    """Write fixture CSV, then test the parse_csv function."""
    csv_path = os.path.join(workspace, task["csv_fixture"])
    with open(csv_path, "w") as fh:
        fh.write(task["csv_content"])

    target = os.path.join(workspace, task["target_file"])
    if not os.path.exists(target):
        return ExecutionResult(
            passed=False, output="",
            error=f"{task['target_file']} not found in workspace",
        )

    lines = [
        "import sys, os; sys.path.insert(0, '.')",
        "from file_parser import parse_csv",
        "failures = []",
        "result = parse_csv('sample.csv', 'score')",
        "expected = {'mean': 82.0, 'min': 60.0, 'max': 95.0}",
        "for k in expected:",
        "    if abs(float(result.get(k, 0)) - expected[k]) > 0.01:",
        "        failures.append(f'  {k}: {result.get(k)!r} != {expected[k]!r}')",
        "if failures:",
        "    print('FAIL')",
        "    print('\\n'.join(failures))",
        "    sys.exit(1)",
        "else:",
        "    print('PASS — statistics match expected values')",
    ]
    return _write_and_run_driver(workspace, "_eval_driver.py", "\n".join(lines))


def execute_api_task(task: dict, workspace: str) -> ExecutionResult:
    """Test fetch_data error-handling against httpbin.org."""
    target = os.path.join(workspace, task["target_file"])
    if not os.path.exists(target):
        return ExecutionResult(
            passed=False, output="",
            error=f"{task['target_file']} not found in workspace",
        )

    lines = [
        "import sys; sys.path.insert(0, '.')",
        "from api_client import fetch_data",
        "failures = []",
        "",
        "# Test 1: successful JSON fetch",
        "r = fetch_data('https://httpbin.org/json', timeout=10)",
        "if not r.get('success'):",
        "    failures.append(f'  success case failed: {r}')",
        "",
        "# Test 2: timeout",
        "r = fetch_data('https://httpbin.org/delay/10', timeout=1)",
        "if r.get('success') or not r.get('error'):",
        "    failures.append(f'  timeout case should fail but got: {r}')",
        "",
        "# Test 3: connection error (nothing listening on this port)",
        "r = fetch_data('http://127.0.0.1:19999/no-such-server', timeout=3)",
        "if r.get('success') or not r.get('error'):",
        "    failures.append(f'  connection error case should fail but got: {r}')",
        "",
        "if failures:",
        "    print('FAIL')",
        "    print('\\n'.join(failures))",
        "    sys.exit(1)",
        "else:",
        "    print('PASS — all error-handling cases passed')",
    ]
    return _write_and_run_driver(workspace, "_eval_driver.py", "\n".join(lines), timeout=60)


def execute_cli_task(task: dict, workspace: str) -> ExecutionResult:
    """Run the CLI todo app through an add/list/delete/list sequence."""
    target = os.path.join(workspace, task["target_file"])
    if not os.path.exists(target):
        return ExecutionResult(
            passed=False, output="",
            error=f"{task['target_file']} not found in workspace",
        )

    # Remove stale state from any previous attempt
    for stale in ("todos.json", "todos_cache.json"):
        stale_path = os.path.join(workspace, stale)
        if os.path.exists(stale_path):
            os.remove(stale_path)

    outputs, failures = [], []

    for text in ("Buy milk", "Learn Python"):
        code, out, err = _run_script(target, args=["add", text], cwd=workspace)
        outputs.append(f"add '{text}' → {out.strip()}")
        if code != 0:
            failures.append(f"  'add {text}' failed: {err.strip()}")

    code, list_out, err = _run_script(target, args=["list"], cwd=workspace)
    outputs.append(f"list → {list_out.strip()}")
    if code != 0 or "buy milk" not in list_out.lower():
        failures.append(f"  'list' didn't show added items: {list_out!r}")

    code, out, err = _run_script(target, args=["delete", "1"], cwd=workspace)
    outputs.append(f"delete 1 → {out.strip()}")
    if code != 0:
        failures.append(f"  'delete 1' failed: {err.strip()}")

    code, list_out2, err = _run_script(target, args=["list"], cwd=workspace)
    outputs.append(f"list after delete → {list_out2.strip()}")
    if code != 0:
        failures.append(f"  final 'list' failed: {err.strip()}")

    item_lines = [ln for ln in list_out2.splitlines() if ln.strip()]
    if len(item_lines) < 1:
        failures.append("  expected at least 1 item remaining after deleting one of two")

    combined = "\n".join(outputs)
    if failures:
        return ExecutionResult(passed=False, output=combined, error="\n".join(failures))
    return ExecutionResult(passed=True, output=combined)


def execute_browser_task(task: dict, workspace: str) -> ExecutionResult:
    """Open the HTML file with Playwright and validate DOM interactions."""
    html_path = os.path.join(workspace, task["target_file"])
    if not os.path.exists(html_path):
        return ExecutionResult(
            passed=False, output="",
            error=f"{task['target_file']} not found in workspace",
        )

    try:
        from playwright.sync_api import TimeoutError as PWTimeout  # noqa: F401
        from playwright.sync_api import sync_playwright
    except ImportError:
        return ExecutionResult(
            passed=False, output="",
            error=(
                "playwright not installed. "
                "Run: pip install playwright && playwright install chromium"
            ),
        )

    details: list[str] = []
    failures: list[str] = []
    file_url = f"file://{os.path.abspath(html_path)}"

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(file_url, timeout=10_000)

            # Locate input
            input_sel = "input[type=text], input:not([type]), textarea"
            try:
                page.wait_for_selector(input_sel, timeout=5_000)
                page.fill(input_sel, "Test Item One")
                details.append("Filled input with 'Test Item One'")
            except Exception:
                failures.append("Could not find or fill an input field")
                browser.close()
                return ExecutionResult(
                    passed=False, output="\n".join(details), error="\n".join(failures)
                )

            # Click Add button (prefer button whose text contains "add")
            buttons = page.query_selector_all("button")
            add_btn = next(
                (b for b in buttons if "add" in (b.inner_text() or "").lower()),
                buttons[0] if buttons else None,
            )
            if add_btn:
                add_btn.click()
                details.append("Clicked Add button")
            else:
                failures.append("No button found on page")

            page.wait_for_timeout(400)

            # Assert at least 1 list item appeared
            item_sel = "li, [data-item], .todo-item, .item"
            list_items = page.query_selector_all(item_sel)
            details.append(f"List items after add: {len(list_items)}")
            if len(list_items) < 1:
                failures.append("No list items appeared after clicking Add")

            # Click Delete on first item
            del_btn = page.query_selector(
                "button:has-text('Delete'), button:has-text('Remove'), li button, .item button"
            )
            if del_btn:
                del_btn.click()
                page.wait_for_timeout(400)
                details.append("Clicked Delete on first item")
                remaining = page.query_selector_all(item_sel)
                details.append(f"List items after delete: {len(remaining)}")
                if len(remaining) >= len(list_items):
                    failures.append("Delete button did not remove the item")
            else:
                failures.append("No delete button found on list item")

            browser.close()

    except Exception as exc:
        return ExecutionResult(
            passed=False, output="\n".join(details), error=f"Playwright error: {exc}"
        )

    combined = "\n".join(details)
    if failures:
        return ExecutionResult(passed=False, output=combined, error="\n".join(failures))
    return ExecutionResult(passed=True, output=combined)


def execute_perf_task(task: dict, workspace: str) -> ExecutionResult:
    """Run has_duplicates on a 1 M-item list and assert it finishes under threshold."""
    target = os.path.join(workspace, task["target_file"])
    if not os.path.exists(target):
        return ExecutionResult(
            passed=False, output="",
            error=f"{task['target_file']} not found in workspace",
        )

    threshold = task.get("perf_threshold_seconds", 1.0)
    lines = [
        "import sys, time; sys.path.insert(0, '.')",
        "from duplicates import has_duplicates",
        "# correctness",
        "assert has_duplicates([1,2,3,1]) is True",
        "assert has_duplicates([1,2,3,4]) is False",
        "assert has_duplicates([]) is False",
        "# performance: 1M unique ints (no duplicates = worst case)",
        "big = list(range(1_000_000))",
        "start = time.perf_counter()",
        "result = has_duplicates(big)",
        "elapsed = time.perf_counter() - start",
        f"print(f'1M items: {{elapsed:.3f}}s, result={{result}}')",
        f"if elapsed > {threshold}:",
        f"    print(f'FAIL — took {{elapsed:.3f}}s, limit is {threshold}s')",
        f"    sys.exit(1)",
        "else:",
        "    print('PASS')",
    ]
    result = _write_and_run_driver(workspace, "_eval_driver.py", "\n".join(lines), timeout=30)
    m = re.search(r"(\d+\.\d+)s", result.output)
    if m:
        result.timing = float(m.group(1))
    return result


def execute_ttl_task(task: dict, workspace: str) -> ExecutionResult:
    """Test TTLCache for basic correctness and expiry."""
    target = os.path.join(workspace, task["target_file"])
    if not os.path.exists(target):
        return ExecutionResult(
            passed=False, output="",
            error=f"{task['target_file']} not found in workspace",
        )

    lines = [
        "import sys, time; sys.path.insert(0, '.')",
        "from ttl_cache import TTLCache",
        "cache = TTLCache()",
        "# basic set / get",
        "cache.set('a', 1, ttl_seconds=10)",
        "assert cache.get('a') == 1, 'get after set should return value'",
        "# missing key returns None",
        "assert cache.get('missing') is None, 'missing key should be None'",
        "# expiry",
        "cache.set('b', 2, ttl_seconds=0.05)",
        "time.sleep(0.1)",
        "assert cache.get('b') is None, 'expired key should be None'",
        "# delete",
        "cache.set('c', 3, ttl_seconds=60)",
        "cache.delete('c')",
        "assert cache.get('c') is None, 'deleted key should be None'",
        "# clear",
        "cache.set('d', 4, ttl_seconds=60)",
        "cache.clear()",
        "assert cache.get('d') is None, 'cleared cache should be empty'",
        "print('PASS — all TTLCache tests passed')",
    ]
    return _write_and_run_driver(workspace, "_eval_driver.py", "\n".join(lines), timeout=15)


def execute_task(task: dict, workspace: str) -> ExecutionResult:
    """Dispatch to the correct executor based on task category."""
    cat = task.get("category", "")
    dispatch = {
        "python_function": execute_python_function_task,
        "python_csv":      execute_csv_task,
        "python_api":      execute_api_task,
        "python_cli":      execute_cli_task,
        "browser":         execute_browser_task,
        "python_bugfix":   execute_python_function_task,  # same test mechanism
        "python_perf":     execute_perf_task,
        "python_class":    execute_ttl_task,
        "llm_judge_only":  lambda t, w: ExecutionResult(
            passed=True,
            output="No execution — assessed by LLM judge only",
        ),
    }
    handler = dispatch.get(cat)
    if handler:
        return handler(task, workspace)
    return ExecutionResult(passed=False, output="", error=f"Unknown category: {cat}")


# ---------------------------------------------------------------------------
# Retry loop
# ---------------------------------------------------------------------------


def run_task_with_retries(
    task: dict,
    base_workspace: str,
    max_retries: int = MAX_RETRIES,
) -> TaskResult:
    """Run one task through the candidate agent, retrying up to max_retries times."""
    task_id   = task["id"]
    workspace = os.path.join(base_workspace, f"task_{task_id}")
    os.makedirs(workspace, exist_ok=True)

    hint        = ""
    hints_used: list[str] = []
    exec_result = ExecutionResult(passed=False, output="", error="Not executed")
    last_eval:  dict = {}

    for attempt in range(1, max_retries + 1):
        session_id = f"eval-task{task_id}-a{attempt}-{int(time.time())}"
        print(f"\n  Attempt {attempt}/{max_retries}")

        candidate_output = run_candidate_agent(
            task_desc=task["description"],
            workspace_path=workspace,
            session_id=session_id,
            hint=hint,
        )

        exec_result = execute_task(task, workspace)

        exec_ctx = (
            f"Execution result: {'PASS' if exec_result.passed else 'FAIL'}\n"
            f"Output:\n{exec_result.output[:600]}"
        )
        if exec_result.error:
            exec_ctx += f"\nError:\n{exec_result.error[:400]}"
        if exec_result.timing is not None:
            exec_ctx += f"\nTiming: {exec_result.timing:.3f}s"

        raw_eval  = evaluate_with_ollama(task["description"], candidate_output, exec_ctx)
        last_eval = parse_evaluation(raw_eval)

        icon      = "✅" if last_eval["result"] == "PASS" else "❌"
        exec_icon = "✅" if exec_result.passed else "❌"
        print(
            f"  {icon} {last_eval['result']} — Score: {last_eval['score']}/10  "
            f"(execution: {exec_icon})"
        )

        if last_eval["result"] == "PASS":
            break

        if last_eval["hint"]:
            hint = last_eval["hint"]
            hints_used.append(hint)
            print(f"  💡 Hint: {hint[:120]}")

    return TaskResult(
        task_id=task_id,
        title=task["title"],
        difficulty=task["difficulty"],
        score=last_eval.get("score", 0),
        result=last_eval.get("result", "FAIL"),
        attempts=attempt,
        execution=exec_result,
        raw_evaluation=last_eval.get("raw", ""),
        hints_used=hints_used,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def store_result_to_memory(result: TaskResult, session_id: str) -> None:
    """Persist evaluation result to the episodic memory service."""
    content = (
        f"Evaluation Task {result.task_id}: {result.title}\n"
        f"Difficulty: {result.difficulty}\n"
        f"Score: {result.score}/10\n"
        f"Result: {result.result}\n"
        f"Attempts: {result.attempts}\n"
        f"Execution passed: {result.execution.passed}"
    )
    payload = {
        "session_id":    session_id,
        "content":       content,
        "tags":          [
            "evaluation",
            f"task-{result.task_id}",
            result.difficulty.lower(),
            result.result.lower(),
        ],
        "source_phase":  "Evaluation",
        "outcome_label": "success" if result.result == "PASS" else "failure",
        "metadata": {
            "task_id":  str(result.task_id),
            "score":    str(result.score),
            "attempts": str(result.attempts),
        },
    }
    try:
        resp = requests.post(
            f"{AI_SERVICE_URL}/v1/memory/upsert", json=payload, timeout=15
        )
        if resp.status_code == 200:
            print(f"  💾 Stored to memory (id={resp.json().get('id')})")
        else:
            print(f"  ⚠️  Memory upsert failed: {resp.status_code} {resp.text[:100]}")
    except Exception as exc:
        print(f"  ⚠️  Memory upsert skipped: {exc}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_task_summary(result: TaskResult) -> None:
    line = "─" * 60
    print(f"\n{line}")
    print(f"TASK {result.task_id}: {result.title}  [{result.difficulty}]")
    print(line)
    print(f"RESULT:    {result.result}")
    print(f"SCORE:     {result.score}/10")
    print(f"ATTEMPTS:  {result.attempts}")
    exec_label = "PASS" if result.execution.passed else "FAIL"
    print(f"EXECUTION: {exec_label}")
    if result.execution.output:
        print(f"  Output: {result.execution.output[:300]}")
    if result.execution.error:
        print(f"  Error:  {result.execution.error[:200]}")
    if result.hints_used:
        print(f"HINTS USED ({len(result.hints_used)}):")
        for i, h in enumerate(result.hints_used, 1):
            print(f"  {i}. {h[:120]}")
    if result.raw_evaluation:
        print(f"\nEVALUATION:\n{result.raw_evaluation[:800]}")


def print_final_summary(results: list[TaskResult]) -> None:
    line = "═" * 60
    print(f"\n{line}")
    print("FINAL BENCHMARK SUMMARY")
    print(line)

    if not results:
        print("No results to summarise.")
        return

    avg_score = sum(r.score for r in results) / len(results)
    passed    = [r for r in results if r.result == "PASS"]

    print(f"Tasks run:      {len(results)}")
    print(f"Passed:         {len(passed)}/{len(results)}")
    print(f"Average score:  {avg_score:.1f}/10")
    print()

    for diff in ("Easy", "Medium", "Hard", "Hard+"):
        tier = [r for r in results if r.difficulty == diff]
        if not tier:
            continue
        tier_avg  = sum(r.score for r in tier) / len(tier)
        tier_pass = sum(1 for r in tier if r.result == "PASS")
        print(f"  {diff:8s}: {tier_pass}/{len(tier)} passed, avg {tier_avg:.1f}/10")

    print()

    strengths = [r for r in results if r.score >= 8]
    if strengths:
        print("STRENGTHS (score ≥ 8):")
        for r in strengths:
            print(f"  ✅ Task {r.task_id} — {r.title}: {r.score}/10")

    weaknesses = [r for r in results if r.score < 5]
    if weaknesses:
        print("WEAKNESSES (score < 5):")
        for r in weaknesses:
            print(f"  ❌ Task {r.task_id} — {r.title}: {r.score}/10")

    print()
    if avg_score >= 7.5:
        print("RECOMMENDATION: ✅  PRODUCTION-READY")
        print("  The agent consistently produces correct, high-quality code.")
    elif avg_score >= 5.0:
        print("RECOMMENDATION: ⚠️   PARTIALLY READY")
        print("  The agent handles easy/medium tasks but struggles with hard ones.")
    else:
        print("RECOMMENDATION: ❌  NOT PRODUCTION-READY")
        print("  The agent produces too many incorrect or incomplete solutions.")
    print(line)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluator Agent Benchmark Pipeline")
    p.add_argument(
        "--tasks",
        default="",
        help="Comma-separated task IDs to run (e.g. 1,2,7). Default: all 10.",
    )
    p.add_argument(
        "--stress",
        action="store_true",
        help="Append the combined stress task (id 99) after the main run.",
    )
    p.add_argument(
        "--profile",
        default=EVAL_PROFILE,
        help=f"Hardware profile for the candidate agent (default: {EVAL_PROFILE})",
    )
    p.add_argument(
        "--provider",
        default=EVAL_PROVIDER,
        help=f"LLM provider for the candidate agent (default: {EVAL_PROVIDER})",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=MAX_RETRIES,
        help=f"Max retries per task (default: {MAX_RETRIES})",
    )
    p.add_argument(
        "--no-memory",
        action="store_true",
        help="Skip storing results to the episodic memory service.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Apply CLI overrides
    global EVAL_PROFILE, EVAL_PROVIDER, MAX_RETRIES
    EVAL_PROFILE  = args.profile
    EVAL_PROVIDER = args.provider
    MAX_RETRIES   = args.retries

    # Select tasks
    if args.tasks:
        ids = {int(x.strip()) for x in args.tasks.split(",")}
        tasks_to_run = [t for t in TASKS if t["id"] in ids]
    else:
        tasks_to_run = list(TASKS)

    if args.stress:
        tasks_to_run.append(STRESS_TASK)

    print("🚀 Evaluator Agent Benchmark Pipeline")
    print(f"   Judge:     {EVALUATOR_MODEL} @ {OLLAMA_URL}")
    print(f"   Candidate: {EVAL_PROVIDER} / {EVAL_PROFILE}")
    print(f"   Tasks:     {[t['id'] for t in tasks_to_run]}")
    print(f"   Retries:   {MAX_RETRIES}")

    base_workspace = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "workspace_test")
    )
    os.makedirs(base_workspace, exist_ok=True)

    eval_session = f"eval-{int(time.time())}"
    results: list[TaskResult] = []

    for task in tasks_to_run:
        print(f"\n{'=' * 60}")
        print(f"🛠️  Task {task['id']} ({task['difficulty']}): {task['title']}")
        print("=" * 60)

        result = run_task_with_retries(task, base_workspace, max_retries=MAX_RETRIES)
        results.append(result)
        print_task_summary(result)

        if not args.no_memory:
            store_result_to_memory(result, eval_session)

        # List generated (non-driver) files
        ws = os.path.join(base_workspace, f"task_{task['id']}")
        gen_files = [
            os.path.join(root, fname)
            for root, _, files in os.walk(ws)
            for fname in files
            if not fname.startswith("_eval_driver")
        ]
        if gen_files:
            print("\n  Generated files:")
            for fp in gen_files:
                print(f"    {os.path.relpath(fp, base_workspace)}")

    print_final_summary(results)


if __name__ == "__main__":
    main()
