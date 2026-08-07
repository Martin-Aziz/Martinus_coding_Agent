"""
Martinus vs Claude Code CLI benchmark.

Runs the same task through both systems and measures:
  - Time (wall-clock seconds)
  - Token usage (prompt + completion where available)
  - Output correctness (syntax, tests pass, files produced)
  - Model calls made

Usage:
  python3 _benchmark.py [--task TASK]

No side effects on the repo: all workspaces are in /tmp/.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

# ------------------------------------------------------------------
# Add project root to sys.path so we can import agent directly.
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.orchestrator import Engine
from agent.providers import ChatResponse, from_env as providers_from_env


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
TASK = (
    "Create a Python module calc.py with four functions: add(a, b), "
    "subtract(a, b), multiply(a, b), divide(a, b). divide should raise "
    "ValueError for division by zero. Also create test_calc.py with pytest "
    "tests covering all four operations and the division-by-zero edge case."
)

MARTINUS_ENV = {
    "DEFAULT_PROVIDER": "claude",
    "MARTINUS_ENABLE_CLAUDE_ADAPTER": "1",
    "MARTINUS_RUN_WORKSPACE_TESTS": "1",
    "MARTINUS_INDEPENDENT_VERIFIER": "1",
    "MARTINUS_GRAPH_ENGINE": "1",
}

CLAUDE_PROMPT = (
    "Create a Python module calc.py with four functions: add(a, b), "
    "subtract(a, b), multiply(a, b), divide(a, b). divide should raise "
    "ValueError for division by zero. Also create test_calc.py with pytest "
    "tests covering all four operations and the division-by-zero edge case. "
    "Output only the file contents and nothing else. Write the files to disk."
)


# ------------------------------------------------------------------
# Result containers
# ------------------------------------------------------------------
@dataclass
class BenchResult:
    name: str
    wall_seconds: float = 0.0
    model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    files_produced: list[str] = field(default_factory=list)
    syntax_ok: dict[str, bool] = field(default_factory=dict)
    tests_pass: bool = False
    test_count: int = 0
    status: str = "unknown"
    notes: list[str] = field(default_factory=list)
    error: str = ""


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _check_syntax(path: Path) -> bool:
    try:
        ast.parse(path.read_text())
        return True
    except SyntaxError:
        return False


def _run_pytest(workspace: Path) -> tuple[bool, int]:
    """Returns (passed, count). Runs pytest in workspace."""
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", ".", "-q", "--tb=no", "--no-header"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = r.stdout + r.stderr
        passed = r.returncode == 0
        # parse "N passed" from output
        count = 0
        for part in output.split():
            if part.isdigit():
                # next token might be "passed"
                idx = output.split().index(part)
                rest = output.split()
                if idx + 1 < len(rest) and "passed" in rest[idx + 1]:
                    count = int(part)
                    break
        return passed, count
    except Exception:
        return False, 0


def _collect_py_files(workspace: Path) -> list[str]:
    return sorted(
        str(p.relative_to(workspace))
        for p in workspace.rglob("*.py")
        if not p.name.startswith(".") and "__pycache__" not in str(p)
    )


# ------------------------------------------------------------------
# Runner A: Martinus
# ------------------------------------------------------------------
def run_martinus(task: str, profile: str = "medium") -> BenchResult:
    result = BenchResult(name="Martinus (graph+verifier+behavioral-tests)")

    # Apply env overrides for this run.
    old_env = {k: os.environ.get(k) for k in MARTINUS_ENV}
    os.environ.update(MARTINUS_ENV)

    workspace = Path(tempfile.mkdtemp(prefix="martinus_bench_"))
    try:
        events: list[dict] = []
        t0 = time.perf_counter()

        engine = Engine(router=providers_from_env(), memory=None, routing=None)
        run_result = engine.execute(
            session_id="bench-martinus",
            task=task,
            workspace_path=str(workspace),
            goals=[],
            provider_name="claude",
            hardware_profile=profile,
            on_event=events.append,
        )

        result.wall_seconds = round(time.perf_counter() - t0, 2)
        result.status = run_result.status

        # Usage from manifest (most accurate)
        manifest_path = workspace / ".martinus-run.json"
        if manifest_path.exists():
            m = json.loads(manifest_path.read_text())
            usage = m.get("usage", {})
            result.model_calls = usage.get("model_requests", 0)
            result.prompt_tokens = usage.get("prompt_tokens", 0)
            result.completion_tokens = usage.get("completion_tokens", 0)
            result.total_tokens = usage.get("total_tokens", 0)
            # fallback: count phase events
            if result.model_calls == 0:
                result.model_calls = sum(
                    1 for e in events if e.get("type") == "phase_completed"
                )

        result.files_produced = _collect_py_files(workspace)

        for fname in result.files_produced:
            p = workspace / fname
            result.syntax_ok[fname] = _check_syntax(p)

        result.tests_pass, result.test_count = _run_pytest(workspace)

        # Pull independent verifier verdict from events
        for e in events:
            if e.get("type") == "independent_verifier_result":
                result.notes.append(
                    f"IndependentVerifier: {e.get('verdict','?')} "
                    f"(confidence {e.get('confidence','?')})"
                )

        # Graph trace summary
        if manifest_path.exists():
            m = json.loads(manifest_path.read_text())
            g = m.get("verification", {}).get("graph", {})
            if g:
                completed = g.get("completed", [])
                result.notes.append(f"Graph nodes completed: {', '.join(completed)}")

    except Exception as exc:
        result.error = str(exc)
        result.status = "error"
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
        # Restore env
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    return result


# ------------------------------------------------------------------
# Runner B: Claude Code CLI  (claude -p "..." --output-format json)
# ------------------------------------------------------------------
def run_claude_code(task: str) -> BenchResult:
    result = BenchResult(name="Claude Code CLI (claude -p)")

    workspace = Path(tempfile.mkdtemp(prefix="claude_bench_"))

    prompt = (
        f"{task}\n\n"
        "Write the files directly to disk in the current directory. "
        "Do not print code blocks — just create the files."
    )

    try:
        t0 = time.perf_counter()
        proc = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "json"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "CLAUDE_CODE_DISABLE_TELEMETRY": "1"},
        )
        result.wall_seconds = round(time.perf_counter() - t0, 2)

        # Parse JSON output for token usage
        output = proc.stdout.strip()
        try:
            data = json.loads(output)
            usage = data.get("usage", {})
            result.prompt_tokens = usage.get("input_tokens", 0)
            result.completion_tokens = usage.get("output_tokens", 0)
            result.total_tokens = result.prompt_tokens + result.completion_tokens
            result.model_calls = 1  # single turn for baseline
            # If it ran tool calls, those are sub-calls inside the one turn
            tool_uses = data.get("total_cost", None)
            if "num_turns" in data:
                result.model_calls = data["num_turns"]
            result.status = "success" if proc.returncode == 0 else "error"
        except (json.JSONDecodeError, AttributeError):
            # Non-JSON output
            result.status = "success" if proc.returncode == 0 else "error"
            result.notes.append("Non-JSON output — token counts unavailable")

        if proc.returncode != 0:
            result.error = proc.stderr[:400]
            result.status = "error"

        result.files_produced = _collect_py_files(workspace)
        for fname in result.files_produced:
            p = workspace / fname
            result.syntax_ok[fname] = _check_syntax(p)

        result.tests_pass, result.test_count = _run_pytest(workspace)

    except subprocess.TimeoutExpired:
        result.error = "timeout (300s)"
        result.status = "timeout"
    except Exception as exc:
        result.error = str(exc)
        result.status = "error"
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    return result


# ------------------------------------------------------------------
# Report printer
# ------------------------------------------------------------------
def print_report(results: list[BenchResult]) -> None:
    SEP = "─" * 72
    print(f"\n{SEP}")
    print("  MARTINUS vs CLAUDE CODE CLI  —  Benchmark Report")
    print(SEP)

    for r in results:
        print(f"\n{'━'*72}")
        print(f"  {r.name}")
        print(f"{'━'*72}")
        print(f"  Status          : {r.status}")
        print(f"  Wall clock      : {r.wall_seconds:.1f}s")
        print(f"  Model calls     : {r.model_calls}")
        if r.total_tokens:
            print(f"  Prompt tokens   : {r.prompt_tokens:,}")
            print(f"  Completion tok  : {r.completion_tokens:,}")
            print(f"  Total tokens    : {r.total_tokens:,}")
        else:
            print(f"  Tokens          : (not available)")
        print(f"  Files produced  : {r.files_produced or '(none)'}")
        if r.syntax_ok:
            ok = all(r.syntax_ok.values())
            print(f"  Syntax check    : {'ALL OK' if ok else 'FAILURES: ' + str([k for k,v in r.syntax_ok.items() if not v])}")
        print(f"  Tests pass      : {r.tests_pass}  ({r.test_count} collected)")
        if r.notes:
            for note in r.notes:
                print(f"  Note            : {note}")
        if r.error:
            print(f"  Error           : {r.error[:120]}")

    # --- Comparison table ---
    if len(results) == 2:
        a, b = results
        print(f"\n{SEP}")
        print("  COMPARISON")
        print(SEP)
        print(f"  {'Metric':<26}  {'Martinus':>14}  {'Claude Code':>14}")
        print(f"  {'------':<26}  {'--------':>14}  {'-----------':>14}")
        print(f"  {'Wall clock (s)':<26}  {a.wall_seconds:>14.1f}  {b.wall_seconds:>14.1f}")
        print(f"  {'Model calls':<26}  {a.model_calls:>14}  {b.model_calls:>14}")
        if a.total_tokens or b.total_tokens:
            print(f"  {'Total tokens':<26}  {a.total_tokens:>14,}  {b.total_tokens:>14,}")
            if a.total_tokens and b.total_tokens:
                ratio = a.total_tokens / b.total_tokens
                print(f"  {'Token ratio (M/CC)':<26}  {ratio:>14.2f}x")
        print(f"  {'Files produced':<26}  {len(a.files_produced):>14}  {len(b.files_produced):>14}")
        print(f"  {'All syntax OK':<26}  {str(all(a.syntax_ok.values()) if a.syntax_ok else 'n/a'):>14}  {str(all(b.syntax_ok.values()) if b.syntax_ok else 'n/a'):>14}")
        print(f"  {'Tests pass':<26}  {str(a.tests_pass):>14}  {str(b.tests_pass):>14}")
        print(f"  {'Test count':<26}  {a.test_count:>14}  {b.test_count:>14}")

        print(f"\n  Interpretation:")
        # Token savings
        if a.total_tokens and b.total_tokens:
            saved = b.total_tokens - a.total_tokens
            if saved > 0:
                pct = 100 * saved / b.total_tokens
                print(f"  - Martinus used {saved:,} fewer tokens ({pct:.0f}% savings vs Claude Code).")
            elif saved < 0:
                pct = 100 * abs(saved) / b.total_tokens
                print(f"  - Martinus used {abs(saved):,} MORE tokens ({pct:.0f}% overhead vs Claude Code).")
            else:
                print(f"  - Token usage was identical.")
        # Quality
        q_m = a.tests_pass
        q_c = b.tests_pass
        if q_m and not q_c:
            print(f"  - Martinus produced VERIFIED passing tests; Claude Code's tests did NOT pass.")
        elif q_c and not q_m:
            print(f"  - Claude Code produced passing tests; Martinus verification FAILED.")
        elif q_m and q_c:
            print(f"  - Both produced working, passing tests.")
        else:
            print(f"  - Neither system's tests passed in this run.")
        # Speed
        if a.wall_seconds and b.wall_seconds:
            if a.wall_seconds > b.wall_seconds * 1.5:
                print(f"  - Martinus was {a.wall_seconds/b.wall_seconds:.1f}x slower (multi-phase pipeline overhead).")
            elif b.wall_seconds > a.wall_seconds * 1.5:
                print(f"  - Martinus was {b.wall_seconds/a.wall_seconds:.1f}x faster.")
            else:
                print(f"  - Wall-clock time was comparable ({a.wall_seconds:.0f}s vs {b.wall_seconds:.0f}s).")

    print(f"\n{SEP}\n")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--skip-claude-code", action="store_true",
                        help="Only run Martinus (skip Claude Code CLI run)")
    parser.add_argument("--profile", default="medium",
                        help="Martinus hardware profile (desktop/medium/high)")
    args = parser.parse_args()

    print(f"\nTask: {args.task[:120]}{'...' if len(args.task) > 120 else ''}")
    print(f"Profile: {args.profile}")
    print("\n[1/2] Running Martinus ...", flush=True)
    r_martinus = run_martinus(args.task, profile=args.profile)
    print(f"      done in {r_martinus.wall_seconds:.1f}s — {r_martinus.status}", flush=True)

    results = [r_martinus]

    if not args.skip_claude_code:
        print("\n[2/2] Running Claude Code CLI ...", flush=True)
        r_claude = run_claude_code(args.task)
        print(f"      done in {r_claude.wall_seconds:.1f}s — {r_claude.status}", flush=True)
        results.append(r_claude)

    print_report(results)


if __name__ == "__main__":
    main()
