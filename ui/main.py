# flake8: noqa: E501
"""
apps/ui-python/main.py -- NiceGUI developer/reviewer dashboard for Martinus.

Chat-first interface (ChatGPT/Claude-style) with:
  * Unified chat area -- phase output as collapsible cards, run history preserved
  * Mode toggle -- Run / Auto-Eval pill switcher in topbar
  * Settings drawer -- slides in from right, keeps config out of the way
  * Phase progress stepper -- horizontal dots showing pipeline progress
  * Fixed input bar -- always visible at bottom
  * Dynamic model selector -- fetches available models from Ollama / LM Studio

Environment variables
---------------------
  ORCHESTRATOR_URL      Orchestrator base URL   (default http://127.0.0.1:8080)
  ORCHESTRATOR_API_KEY  Optional API key        -> X-API-Key header
  OLLAMA_URL            Ollama base URL         (default http://127.0.0.1:11434)
  LMSTUDIO_URL          LM Studio base URL      (default http://127.0.0.1:1234)
  UI_HOST               Bind address            (default 0.0.0.0)
  UI_PORT               Bind port               (default 8091)
"""

import json
import os
import re
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional
import asyncio

import httpx
from nicegui import app, ui  # type: ignore[import-not-found]

# -- Configuration -------------------------------------------------------------
ORCHESTRATOR_URL     = os.getenv("ORCHESTRATOR_URL",     "http://127.0.0.1:8080").rstrip("/")
ORCHESTRATOR_API_KEY = os.getenv("ORCHESTRATOR_API_KEY", "").strip()
OLLAMA_URL           = os.getenv("OLLAMA_URL",           "http://127.0.0.1:11434").rstrip("/")
LMSTUDIO_URL         = os.getenv("LMSTUDIO_URL",         "http://127.0.0.1:1234").rstrip("/")
UI_HOST              = os.getenv("UI_HOST", "0.0.0.0")
UI_PORT              = int(os.getenv("UI_PORT", "8091"))
ASSET_DIR            = Path(__file__).resolve().parents[1] / "desktop" / "src-tauri" / "icons"

PHASE_COLORS: Dict[str, str] = {
    "Discovery":          "#2563eb",
    "Clarification":      "#7c3aed",
    "Architecture":       "#4f46e5",
    "Implementation": "#059669",
    "Review":             "#d97706",
    "Verification":       "#dc2626",
}
_FALLBACK_COLOR = "#64748b"

ALL_PHASES = ["Discovery", "Clarification", "Architecture", "Implementation", "Review", "Verification"]

# -- Prompt engineering system --------------------------------------------------
SYSTEM_TASK_PROMPT = """ROLE: You are an elite senior software engineer with 20+ years of professional experience in production systems. You write clean, type-annotated, test-covered, and highly maintainable code. You are responsible for both technical correctness AND visual quality of any output users will see.

REASONING PROTOCOL: Before writing any code, explicitly state:
(1) What the task requires — the actual deliverable, not a restatement of the prompt
(2) Which files will be created or modified and why
(3) The algorithm or design pattern you will use and why it is the best choice for this context

GENERATION RULE: Do NOT produce the first solution that comes to mind. Internally consider 3 candidate approaches, evaluate each on correctness / simplicity / maintainability, then implement the best one. State which approach won and why (one sentence).

## Architecture & Design
- Clean architecture: strict separation of concerns (transport / domain / persistence)
- SOLID principles applied rigorously — especially Single Responsibility and Dependency Inversion
- Choose optimal, well-supported libraries; avoid unnecessary dependencies
- Design for testability: dependency-injectable components, no global mutable state
- Appropriate design patterns (Factory, Strategy, Observer) only when they eliminate real complexity

## Code Quality
- Type annotations on every function signature (Python: full typing module)
- Google-style docstrings on every public function and class
- Maximum function length: 40 lines. Decompose if longer.
- Maximum file length: 400 lines. Split into modules if longer.
- No magic numbers or strings — constants only, in a dedicated constants block
- All configuration via environment variables or config file — nothing hardcoded

## Second Pass (mandatory)
After writing the implementation, re-read each file as a senior code reviewer. Apply at least 3 concrete improvements:
- Remove redundant code paths or duplicated logic
- Add missing error handling at system boundaries
- Improve any name that requires a comment to understand

## Self-Critique Gate
After the second pass, rate your output 1-10 on:
(a) Correctness — does it fully satisfy the spec?
(b) Design quality — is it minimal, readable, and well-structured?
(c) Algorithmic efficiency — are data structures and algorithms appropriate?
State: "Quality-Gate: X/10" before finalizing. Revise if below 8.

## UI/UX Design Standards (mandatory for any visual output)
Token File First: emit a design-token file (tokens.css for web, theme.py for desktop) BEFORE any component code. Define:
  - Color palette: neutral-first, single accent for CTAs, OKLCH color space preferred
  - Typography: fluid clamp() scale (body min 16px), max 2 font families, display font >=24px only
  - Spacing: strict 4px grid -- 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96px
  - Borders: alpha-blended only -- e.g. oklch(0% 0 0 / 0.12) -- never solid gray
  - Shadows: tone-matched to surface color, layered for depth
  - Motion: 180ms cubic-bezier(0.16, 1, 0.3, 1) for all transitions
  - Dark mode: data-theme attribute + prefers-color-scheme fallback
  - Mobile: 375px-first, touch targets >=44px, no hover-only interactions

PROHIBITED: gradient buttons, icon-in-circle decorations, 3-column feature grids, centered-everything layouts, colored card side-borders, hardcoded hex colors, magic pixel values in components.

REQUIRED: semantic HTML5 (nav, main, article, section), WCAG AA contrast (4.5:1 body, 3:1 large text >=18px bold), :hover and :focus-visible on all interactive elements, aria-label on icon-only and landmark elements.

Design Quality Gate: Before finalizing any UI, explicitly verify:
(a) No hardcoded hex/px in component files -- tokens only
(b) All interactive elements have :focus-visible states
(c) Contrast ratios meet WCAG AA
State: "Design-Gate: PASS" or list violations.

## Communication
- If the task is ambiguous, ask one focused clarifying question before coding
- Briefly explain architectural decisions and the alternative you rejected
- Flag risks or trade-offs as [RISK: ...]
"""

MASTER_EVALUATOR_PROMPT = """ROLE: You are a principal engineer acting as an expert AI coding evaluator. You have deep experience in code review, testing, and quality assurance. Your evaluations are evidence-based and actionable.

REASONING PROTOCOL: Before scoring, explicitly reason through:
(1) Does the output satisfy every acceptance criterion stated or implied by the task?
(2) What is the most critical defect, if any?
(3) Is the score justified by concrete evidence from the output — not by general impressions?

TASK: Rigorously evaluate the Candidate Agent's output against the following dimensions:

1. Correctness (0-10): Does the output do exactly what the task requires? List deviations.
2. Code Quality (0-10): Is the code readable, well-named, type-annotated, and structured? Flag violations.
3. Edge Case Coverage (0-10): Are error paths, boundary inputs, and failure modes handled?
4. Performance (0-10): Are appropriate algorithms and data structures used? Flag O(n^2) or worse in hot paths.
5. Completeness (0-10): Is the implementation complete and independently runnable?

EVALUATION FORMAT (output exactly this structure):

REASONING:
- Correctness evidence: [specific examples from the output]
- Critical defect (if any): [file/line/function reference]
- Score justification: [why these numbers, not just assertions]

EVALUATION:
- Correctness: X/10
- Code Quality: X/10
- Edge Cases: X/10
- Performance: X/10
- Completeness: X/10

RESULT: PASS / FAIL
SCORE: X/10

If FAIL:
HINT: <one specific, actionable correction — not a restatement of what failed>
"""


# -- Utility helpers -----------------------------------------------------------
def _headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if ORCHESTRATOR_API_KEY:
        h["X-API-Key"] = ORCHESTRATOR_API_KEY
    return h


async def fetch_available_models(provider: str) -> List[str]:
    """Fetch available model names from Ollama, LM Studio, or the Claude
    Code provider."""
    if provider == "claude":
        # The Claude Code adapter picks its own model; the dashboard shows a
        # single logical model id so the dropdown is never empty.
        return ["claude-code"]
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            if provider == "ollama":
                resp = await client.get(f"{OLLAMA_URL}/api/tags")
                resp.raise_for_status()
                return sorted(m["name"] for m in resp.json().get("models", []))
            else:  # lmstudio
                resp = await client.get(f"{LMSTUDIO_URL}/v1/models")
                resp.raise_for_status()
                return sorted(m["id"] for m in resp.json().get("data", []))
    except Exception:
        return []


# Hardcoded professional goals — always injected, never shown to user
_DEFAULT_GOALS = [
    "clean architecture",
    "SOLID principles",
    "comprehensive testing",
    "production-grade code quality",
    "clear documentation and comments",
    "optimal performance",
    "strong verification",
    "memory-driven quality",
]


def _goals(_raw: str = "") -> List[str]:
    return list(_DEFAULT_GOALS)


def _default_session_id() -> str:
    return "run-" + secrets.token_hex(4)


def _phase_color(phase: str) -> str:
    return PHASE_COLORS.get(phase, _FALLBACK_COLOR)


def _parse_eval_scores(text: str) -> Dict[str, Optional[int]]:
    """Extract numeric scores from evaluator response text."""
    dims: Dict[str, Any] = {}
    for dim in ("Correctness", "Code Quality", "Edge Cases", "Performance"):
        m = re.search(rf"{dim}\s*[:]\s*(\d+)\s*/\s*10", text, re.IGNORECASE)
        dims[dim] = int(m.group(1)) if m else None
    m = re.search(r"SCORE\s*[:]\s*(\d+)\s*/\s*10", text, re.IGNORECASE)
    dims["_overall"] = int(m.group(1)) if m else None
    m = re.search(r"RESULT\s*[:]\s*(PASS|FAIL)", text, re.IGNORECASE)
    dims["_result"] = m.group(1).upper() if m else None
    m = re.search(r"HINT\s*[:]\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    dims["_hint"] = m.group(1).strip() if m else None
    return dims


# -- Network helpers -----------------------------------------------------------
async def stream_task_execution(
    task: str,
    session_id: str,
    workspace_path: str,
    goals: List[str],
    provider: str,
    hardware_profile: str,
    max_lines: int,
    model: str = "",
) -> AsyncGenerator[Dict[str, Any], None]:
    payload: Dict[str, Any] = {
        "sessionId":       session_id,
        "workspacePath":   workspace_path,
        "task":            task,
        "goals":           goals,
        "provider":        provider,
        "hardwareProfile": hardware_profile,
        "maxLines":        max_lines,
    }
    if model:
        payload["model"] = model
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            async with client.stream(
                "POST",
                f"{ORCHESTRATOR_URL}/v1/tasks/stream",
                json=payload,
                headers=_headers(),
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        try:
                            yield json.loads(data_str)
                        except json.JSONDecodeError:
                            pass
    except Exception as exc:
        yield {"type": "error", "message": str(exc)}


async def evaluate_with_ollama(
    task_description: str,
    candidate_output: str,
    model: str,
) -> str:
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model":    model,
                    "messages": [
                        {"role": "system", "content": MASTER_EVALUATOR_PROMPT},
                        {
                            "role":    "user",
                            "content": f"TASK:\n{task_description}\n\nCANDIDATE OUTPUT:\n{candidate_output}",
                        },
                    ],
                    "temperature": 0.0,
                    "stream":      False,
                },
            )
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "No evaluation returned.")
    except Exception as exc:
        return f"Evaluator error: {exc}"


# -- Global styles (chat-first layout) ----------------------------------------
def build_css() -> str:
    """Return the full <style> block (and font imports) as an HTML string."""
    return """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:          #f1f5f9;
    --bg-card:     #ffffff;
    --bg-input:    #f8fafc;
    --border:      rgba(0,0,0,0.08);
    --border-hi:   rgba(0,0,0,0.15);
    --primary:     #6366f1;
    --primary-low: rgba(99,102,241,0.12);
    --success:     #059669;
    --warning:     #d97706;
    --error:       #dc2626;
    --text:        #1e293b;
    --text-muted:  #475569;
    --text-dim:    #94a3b8;
    --font-sans:   "Inter", system-ui, sans-serif;
    --font-head:   "Space Grotesk", system-ui, sans-serif;
    --font-mono:   "IBM Plex Mono", "Fira Code", monospace;
    --s1:4px; --s2:8px; --s3:12px; --s4:16px; --s5:20px; --s6:24px;
    --radius-sm: 6px; --radius-md: 10px; --radius-lg: 16px;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0f172a; --bg-card: #1e293b; --bg-input: #0f172a;
      --border: rgba(255,255,255,0.08); --border-hi: rgba(255,255,255,0.15);
      --text: #f1f5f9; --text-muted: #94a3b8; --text-dim: #475569;
    }
  }
  *, *::before, *::after { box-sizing: border-box; }
  body { font-family:var(--font-sans); background:var(--bg); color:var(--text); margin:0; }
  .q-page,.q-page-container,.q-layout { background:var(--bg) !important; }
  .q-drawer { background:var(--bg-card) !important; }
  .q-drawer__backdrop { background:rgba(0,0,0,0.3) !important; }

  /* -- Full-height app shell -- */
  .app-shell {
    display:flex; flex-direction:column; height:100vh; max-width:100%;
    margin:0; padding:0; width:100%; align-items:stretch !important;
  }
  .nicegui-column.app-shell { align-items:stretch !important; }
  @media (max-width:640px)  { .app-shell { padding:0; } }

  /* -- Topbar -- */
  .topbar {
    display:flex; align-items:center; gap:var(--s3); padding:var(--s3) var(--s5);
    border-bottom:1px solid var(--border); background:var(--bg-card);
    flex-shrink:0; z-index:10; width:100%;
  }
  .topbar-title {
    font-family:var(--font-head); font-size:1.1rem; font-weight:700;
    color:var(--text); white-space:nowrap;
  }
  .topbar-spacer { flex:1; }
  .mode-pill {
    display:inline-flex; align-items:center; padding:4px 14px; border-radius:99px;
    font-family:var(--font-head); font-size:0.78rem; font-weight:600;
    cursor:pointer; border:1px solid var(--border-hi); color:var(--text-muted);
    background:transparent; transition:all 0.15s;
  }
  .mode-pill:hover { background:var(--primary-low); }
  .mode-pill.active {
    background:var(--primary); color:white; border-color:var(--primary);
  }
  .health-dot {
    display:inline-block; width:9px; height:9px; border-radius:50%;
    margin-right:4px; vertical-align:middle;
  }
  .health-pill {
    display:inline-flex; align-items:center; padding:3px 10px; border-radius:99px;
    font-family:var(--font-mono); font-size:0.7rem; font-weight:500;
    border:1px solid var(--border-hi); cursor:pointer; background:transparent;
    color:var(--text-muted); transition:all 0.15s;
  }
  .health-pill:hover { background:var(--bg-input); }
  .status-badge {
    font-family:var(--font-mono); font-size:0.72rem; color:var(--text-dim);
  }

  /* -- Progress stepper -- */
  .stepper {
    display:flex; align-items:center; justify-content:center; gap:0;
    padding:var(--s3) var(--s5); border-bottom:1px solid var(--border);
    background:var(--bg-card); flex-shrink:0; flex-wrap:wrap; width:100%;
  }
  .step-item {
    display:flex; align-items:center; gap:0; font-family:var(--font-mono);
    font-size:0.7rem; white-space:nowrap;
  }
  .step-dot {
    width:12px; height:12px; border-radius:50%; border:2px solid var(--text-dim);
    display:inline-flex; align-items:center; justify-content:center;
    font-size:0.5rem; color:white; flex-shrink:0; transition:all 0.2s;
  }
  .step-dot.completed {
    background:var(--success); border-color:var(--success);
  }
  .step-dot.active {
    background:var(--primary); border-color:var(--primary);
    animation:pulse 1.2s ease-in-out infinite;
  }
  .step-dot.pending { background:transparent; }
  .step-label {
    margin:0 2px; color:var(--text-dim); font-size:0.68rem; cursor:pointer;
  }
  .step-label.completed { color:var(--success); font-weight:500; }
  .step-label.active { color:var(--primary); font-weight:600; }
  .step-line {
    width:20px; height:2px; background:var(--border-hi); margin:0 2px;
  }
  .step-line.completed { background:var(--success); }
  @keyframes pulse {
    0%,100% { box-shadow:0 0 0 0 rgba(99,102,241,0.4); }
    50%     { box-shadow:0 0 0 5px rgba(99,102,241,0); }
  }

  /* -- Chat area -- */
  .chat-area {
    flex:1; overflow-y:auto; padding:var(--s4) var(--s5);
    display:flex; flex-direction:column; gap:var(--s3);
  }

  /* -- Phase cards -- */
  .card {
    background:var(--bg-card); border:1px solid var(--border);
    border-radius:var(--radius-md); padding:var(--s4);
    transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1);
  }
  .phase-card {
    background:var(--bg-card); border:1px solid var(--border);
    border-radius:var(--radius-md); overflow:hidden;
    animation: fadeUp 0.2s ease-out both;
    transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1);
  }
  @keyframes fadeUp { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }

  .phase-header {
    display:flex; align-items:center; gap:var(--s2);
    padding:var(--s3) var(--s4); cursor:pointer; user-select:none;
    transition:background 0.1s;
  }
  .phase-header:hover { background:var(--bg-input); }
  .phase-dot {
    width:10px; height:10px; border-radius:50%; flex-shrink:0;
  }
  .phase-body {
    padding:var(--s4); font-family:var(--font-mono); font-size:0.78rem;
    line-height:1.65; color:var(--text); white-space:pre-wrap;
    word-break:break-word; overflow-y:auto; max-height:320px;
    background:var(--bg-input); border-top:1px solid var(--border);
  }
  .phase-actions {
    display:flex; gap:var(--s2); padding:var(--s2) var(--s4);
    border-top:1px solid var(--border);
  }
  .phase-action-btn {
    font-family:var(--font-mono); font-size:0.68rem; color:var(--text-muted);
    cursor:pointer; padding:2px 8px; border-radius:var(--radius-sm); border:1px solid var(--border);
    background:transparent; transition:all 0.1s;
  }
  .phase-action-btn:hover { background:var(--bg-input); color:var(--text); }

  .expand-arrow {
    margin-left:auto; font-size:0.7rem; color:var(--text-dim);
    transition:transform 0.2s;
  }
  .expand-arrow.open { transform:rotate(180deg); }

  .spinner {
    display:inline-block; width:13px; height:13px;
    border:2px solid var(--border-hi); border-top-color:var(--primary);
    border-radius:50%; animation:spin 0.65s linear infinite; flex-shrink:0;
  }
  @keyframes spin { to { transform:rotate(360deg) } }

  .pill {
    display:inline-flex; align-items:center; padding:2px 9px; border-radius:99px;
    font-family:var(--font-mono); font-size:0.68rem; font-weight:500;
    background:var(--bg-input); color:var(--text-muted);
    border:1px solid var(--border-hi); white-space:nowrap;
  }

  .stream-bar {
    flex:1; height:4px; border-radius:2px; background:var(--border);
    overflow:hidden; margin:0 var(--s2);
  }
  .stream-bar-inner {
    height:100%; border-radius:2px; background:var(--primary);
    animation:stream-pulse 1.5s ease-in-out infinite;
  }
  @keyframes stream-pulse {
    0%   { width:10%; }
    50%  { width:70%; }
    100% { width:10%; }
  }

  /* -- Summary & reflection cards -- */
  .summary-card {
    background:linear-gradient(135deg,rgba(99,102,241,.10),rgba(99,102,241,.03));
    border:1px solid rgba(99,102,241,.28); border-radius:var(--radius-md); padding:var(--s4);
    animation:fadeUp 0.2s ease-out both;
  }
  .reflection-card {
    background:linear-gradient(135deg,rgba(52,211,153,.09),rgba(52,211,153,.02));
    border:1px solid rgba(52,211,153,.24); border-radius:var(--radius-md); padding:var(--s4);
    animation:fadeUp 0.2s ease-out both;
  }

  /* -- Evaluation card -- */
  .eval-card {
    background:var(--bg-card); border:1px solid var(--border);
    border-radius:var(--radius-md); padding:var(--s4); animation:fadeUp 0.2s ease-out both;
    transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1);
  }
  .eval-dim-row {
    display:flex; align-items:center; gap:var(--s2); margin:var(--s2) 0;
    font-family:var(--font-mono); font-size:0.78rem;
  }
  .eval-dim-label { width:110px; color:var(--text-muted); flex-shrink:0; }
  .eval-bar {
    flex:1; height:8px; border-radius:var(--radius-sm); background:var(--border);
    overflow:hidden;
  }
  .eval-bar-fill {
    height:100%; border-radius:var(--radius-sm); transition:width 0.4s ease-out;
  }
  .eval-score { width:40px; text-align:right; font-weight:600; color:var(--text); }

  /* -- Input bar -- */
  .input-bar {
    border-top:1px solid var(--border); background:var(--bg-card);
    padding:var(--s4) var(--s5) var(--s5); flex-shrink:0; width:100%;
  }
  .input-bar .task-textarea-wrap {
    width:100% !important; display:flex; flex-direction:column;
  }
  .input-bar .task-textarea-wrap > * {
    width:100% !important;
  }
  .input-bar .task-textarea-wrap .q-field {
    width:100% !important;
  }
  .input-bar .task-textarea-wrap .q-field__control,
  .input-bar .task-textarea-wrap .q-field--filled .q-field__control {
    background:var(--bg-input) !important;
    border:1.5px solid var(--border-hi) !important;
    border-radius:var(--radius-lg) !important;
    padding:var(--s3) var(--s4) !important;
    min-height:100px !important;
    transition:border-color 0.15s;
  }
  .input-bar .task-textarea-wrap .q-field__control:focus-within {
    border-color:var(--primary) !important;
    box-shadow:0 0 0 3px var(--primary-low) !important;
  }
  .input-bar .task-textarea-wrap .q-field__control.textarea-warn {
    border-color:var(--warning) !important;
  }
  .input-bar .task-textarea-wrap textarea {
    font-size:0.95rem !important;
    line-height:1.6 !important;
    min-height:80px !important;
    width:100% !important;
  }
  .char-counter {
    font-family:var(--font-mono); font-size:0.68rem; color:var(--text-dim);
    text-align:right; margin-top:2px;
  }
  .char-counter.warn { color:var(--warning); font-weight:600; }
  .input-actions {
    display:flex; align-items:center; justify-content:flex-end;
    gap:var(--s2); margin-top:var(--s2);
  }
  @media (max-width:640px) {
    .input-bar { padding:var(--s3) var(--s3) var(--s4); }
    .input-bar .task-textarea-wrap textarea { font-size:0.88rem !important; min-height:50px !important; }
    .topbar { padding:var(--s2) var(--s3); gap:var(--s2); flex-wrap:wrap; }
    .topbar-title { font-size:0.95rem; }
    .stepper { padding:var(--s2) var(--s3); }
    .chat-area { padding:var(--s3); }
  }
  .input-bar .q-field__control,.input-bar .q-field--filled .q-field__control {
    background:var(--bg-input) !important;
  }
  .input-bar .q-field__native,.input-bar .q-field__input { color:var(--text) !important; }

  .ref-panel {
    padding:var(--s3); margin-bottom:var(--s2); background:var(--bg-input);
    border:1px solid var(--border); border-radius:var(--radius-sm);
  }

  .auto-eval-row {
    display:flex; gap:var(--s2); align-items:center; padding:var(--s2) 0;
    flex-wrap:wrap;
  }

  /* -- NiceGUI field overrides -- */
  .q-field__control,.q-field--filled .q-field__control { background:var(--bg-input) !important; }
  .q-field__native,.q-field__input { color:var(--text) !important; }
  .q-field__label { color:var(--text-muted) !important; }

  .font-head  { font-family:var(--font-head); }
  .mono       { font-family:var(--font-mono); }
  .text-muted { color:var(--text-muted); }
  .section-title {
    font-family:var(--font-head); font-size:0.75rem; font-weight:600;
    text-transform:uppercase; letter-spacing:0.07em;
    color:var(--text-dim); margin-bottom:var(--s2);
  }

  /* -- Hardware profile subtitles -- */
  .profile-option-title {
    font-family:var(--font-mono); font-size:0.78rem; font-weight:600; color:var(--text);
  }
  .profile-option-sub {
    font-family:var(--font-mono); font-size:0.68rem; color:var(--text-dim); margin-top:1px;
  }

  /* Hide NiceGUI default padding */
  .q-page { padding:0 !important; }
  .nicegui-content { padding:0 !important; }
</style>
"""


app.add_static_files("/martinus-assets", str(ASSET_DIR))
ui.add_head_html(
    '<link rel="icon" type="image/svg+xml" href="/martinus-assets/martinus.svg">'
    + build_css()
)
ui.page_title("Martinus")
ui.dark_mode(value=False)

# -- State ---------------------------------------------------------------------
current_mode = {"value": "run"}
# NiceGUI runs on a single async event loop; no lock needed for health_state dict
health_state = {"status": "unknown"}
is_running   = {"value": False}
run_control: Dict[str, Any] = {"task": None, "cancel_requested": False}
stepper_phases: Dict[str, Any] = {}
ref_panel_visible = {"value": False}


# -- Settings Drawer (right) --------------------------------------------------
drawer = ui.right_drawer(value=False, fixed=True, bordered=True, elevated=True).style(
    "width:320px; padding:0"
).classes("q-pa-none")

with drawer:
    with ui.column().classes("w-full gap-4").style("padding:20px"):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Settings").style(
                "font-family:var(--font-head);font-size:1rem;font-weight:700;color:var(--text)"
            )
            close_drawer_btn = ui.button(icon="close").props("flat dense round").style("color:var(--text-muted)")

        ui.separator()

        ui.label("Session ID").classes("section-title")
        with ui.row().classes("w-full items-center gap-1"):
            session_input = ui.input(value=_default_session_id()).classes("flex-1").props("dense")
            regen_session_btn = ui.button(icon="refresh").props("flat dense round").style(
                "color:var(--text-muted)"
            )

        ui.label("Workspace Path").classes("section-title")
        workspace_input = ui.input(
            value=os.path.join(os.path.expanduser("~"), "agent-workspace")
        ).classes("w-full").props("dense")

        ui.label("Provider").classes("section-title")
        # All supported providers remain visible so reviewers can switch
        # runtimes without restarting the UI. Health checks explain when a
        # provider is not installed or not currently running.
        _claude_adapter_on = os.getenv("MARTINUS_ENABLE_CLAUDE_ADAPTER", "").strip() in ("1", "true", "yes")
        _provider_options = {
            "ollama": "Ollama",
            "lmstudio": "LM Studio",
            "claude": "Claude Code",
        }
        provider_select = ui.radio(
            _provider_options,
            value="claude" if _claude_adapter_on else "ollama",
        ).classes("w-full")

        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Model").classes("section-title").style("margin-bottom:0")
            refresh_models_btn = ui.button(icon="refresh").props("flat dense round").style(
                "color:var(--text-muted)"
            )
        model_select = ui.select(
            options={},
            value=None,
            label="Auto (hardware profile)",
        ).classes("w-full").props("dense clearable")

        ui.label("Hardware Profile").classes("section-title")
        profile_select = ui.slider(min=0, max=2, step=1, value=1).classes("w-full").style(
            "color:var(--primary)"
        )
        ui.html(
            '<div style="display:flex;justify-content:space-between;font-family:var(--font-mono);'
            'font-size:0.7rem;color:var(--text-muted);margin-top:-4px">'
            '<span>Low</span><span>Medium</span><span>High</span></div>'
        )
        ui.html(
            '<div style="margin-top:var(--s2);display:flex;flex-direction:column;gap:4px">'
            '<div class="profile-option-title">Profile descriptions:</div>'
            '<div><span class="profile-option-title">Low</span>'
            '&nbsp;<span class="profile-option-sub">8GB RAM — 3 essential phases</span></div>'
            '<div><span class="profile-option-title">Medium</span>'
            '&nbsp;<span class="profile-option-sub">16GB RAM — all 6 phases</span></div>'
            '<div><span class="profile-option-title">High</span>'
            '&nbsp;<span class="profile-option-sub">32GB+ RAM — all phases + advanced techniques</span></div>'
            '</div>'
        )

        ui.label("Line Budget").classes("section-title")
        max_lines_input = ui.number(value=2000, min=100, max=2000).classes("w-full").props("dense")

        ui.separator()

        save_defaults_btn = ui.button("Save as defaults").props("flat").style(
            "font-family:var(--font-head);color:var(--primary)"
        )

_PROFILE_MAP = {0: "low", 1: "medium", 2: "high"}

# -- Workspace Explorer (left) ------------------------------------------------
explorer_drawer = ui.left_drawer(value=False, fixed=True, bordered=True, elevated=True).style(
    "width:420px; padding:0"
).classes("q-pa-none")

with explorer_drawer:
    with ui.column().classes("w-full gap-2").style("padding:12px"):
        ui.label("Workspace Explorer").style("font-family:var(--font-head);font-weight:700")
        explorer_path_input = ui.input(value='.', label='Path').classes("w-full").props("dense")
        # NOTE: Manual file-creation dialog removed (May 4, 2026).
        # File creation now happens automatically via the agent's LLM output.
        # The orchestrator's ExtractAndWriteFiles() function parses <file path="..."> tags
        # from phase outputs and writes them directly to the workspace.
        # Workflow: Submit task → Agent runs phases → LLM wraps content in tags → 
        # Files auto-created in workspace. No manual UI dialog needed.
        explorer_refresh = ui.button("Refresh").props("flat").classes("q-mb-sm")
        entries_container = ui.column().classes("w-full gap-1").style("max-height:70vh;overflow:auto;padding-right:6px")
        ui.separator()
        file_viewer = ui.code("", language="text").style("height:40vh; width:100%;")



async def _fetch_workspace_list(path: str):
    """
    Fetch and display workspace files/folders.
    
    NOTE: Files displayed here now include those auto-created by the agent.
    When the agent completes a task, it wraps file content in <file path="..."> tags,
    which the orchestrator's ExtractAndWriteFiles() automatically writes to disk.
    This function simply lists what the agent has created.
    """
    wp = workspace_input.value.strip()
    if not wp:
        entries_container.clear()
        ui.label("Set Workspace Path in Settings first.", parent=entries_container)
        return
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(f"{ORCHESTRATOR_URL}/v1/workspace/list", params={"workspacePath": wp, "path": path}, headers=_headers())
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        entries_container.clear()
        ui.label(f"Error: {exc}", parent=entries_container)
        return

    entries_container.clear()
    dirs = [e for e in data if e.get("is_dir")]
    files = [e for e in data if not e.get("is_dir")]
    for e in sorted(dirs, key=lambda x: x.get("name")) + sorted(files, key=lambda x: x.get("name")):
        name = e.get("name")
        relp = e.get("path")
        if e.get("is_dir"):
            def _open_dir(_, p=relp):
                explorer_path_input.set_value(p)
                asyncio.create_task(_fetch_workspace_list(p))

            ui.button(f"📁 {name}", on_click=_open_dir, parent=entries_container).props("flat")
        else:
            async def _open_file(_, p=relp):
                await _read_workspace_file(p)

            ui.button(f"📄 {name}", on_click=_open_file, parent=entries_container).props("flat")

async def _read_workspace_file(rel_path: str):
    wp = workspace_input.value.strip()
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(f"{ORCHESTRATOR_URL}/v1/workspace/read", params={"workspacePath": wp, "path": rel_path}, headers=_headers())
            resp.raise_for_status()
            j = resp.json()
    except Exception as exc:
        entries_container.clear()
        ui.label(f"Error: {exc}", parent=entries_container)
        return

    content = j.get("content", "")
    file_viewer.set_text(content)

def _open_explorer():
    """Open workspace explorer to view files.
    
    Files shown here are created by the agent automatically when tasks complete.
    The agent's phase outputs contain <file path="...">content</file> tags
    which are parsed by the orchestrator and written to disk via ExtractAndWriteFiles().
    """
    explorer_drawer.value = True
    # load root
    asyncio.create_task(_fetch_workspace_list(explorer_path_input.value or "."))

# Wire refresh and an Explore button in settings
explore_btn = ui.button("Explore workspace").props("flat").style("margin-top:6px").on("click", lambda _: _open_explorer())
explorer_refresh.on("click", lambda _: asyncio.create_task(_fetch_workspace_list(explorer_path_input.value or ".")))
def _get_profile() -> str:
    return _PROFILE_MAP.get(int(profile_select.value), "medium")


def _regen_session():
    session_input.set_value(_default_session_id())


def _close_drawer():
    drawer.toggle()


async def _refresh_models() -> None:
    """Fetch available models and populate the model selectors."""
    models = await fetch_available_models(provider_select.value)
    options: Dict[str, str] = {"": "Auto (hardware profile)"}
    for m in models:
        options[m] = m
    model_select.set_options(options)
    auto_evaluator_model.set_options(options)
    quick_model_select.set_options(options)
    # Only reconcile the current selection when we actually got a model list.
    # This runs on every browser connect, and an empty list means "the provider
    # is not reachable yet", not "your model is gone" — clearing the selection
    # there would silently discard a saved default before it could be used.
    if not models:
        return
    # Keep current value if still available, otherwise reset to auto
    if model_select.value not in options:
        model_select.set_value(None)
    if auto_evaluator_model.value not in options:
        auto_evaluator_model.set_value(None)
    if quick_model_select.value not in options:
        quick_model_select.set_value(None)


# -- Saved defaults ------------------------------------------------------------
# `Save as defaults` was rendered but never connected to a handler, so clicking
# it did nothing at all. Persist through app.storage.general (a JSON file on
# disk) so the choice survives a restart, matching the desktop app's behaviour.
_DEFAULTS_KEY = "martinus:defaults"


def _save_defaults() -> None:
    try:
        app.storage.general[_DEFAULTS_KEY] = {
            "provider": provider_select.value,
            "model": model_select.value,
            "workspace": workspace_input.value,
            "profile": profile_select.value,
            "max_lines": max_lines_input.value,
        }
    except Exception as exc:  # storage is best-effort; never break the UI
        ui.notify(f"Could not save defaults: {exc}", type="negative")
        return
    label = model_select.value or "auto"
    ui.notify(f"Saved — {provider_select.value} · {label} will be selected next launch.",
              type="positive")


def _apply_saved_defaults() -> None:
    try:
        saved = app.storage.general.get(_DEFAULTS_KEY) or {}
    except Exception:
        return
    if not isinstance(saved, dict):
        return
    if saved.get("provider") in (provider_select.options or {}):
        provider_select.set_value(saved["provider"])
    if saved.get("workspace"):
        workspace_input.set_value(saved["workspace"])
    if saved.get("profile") is not None:
        profile_select.set_value(saved["profile"])
    if saved.get("max_lines"):
        max_lines_input.set_value(saved["max_lines"])
    # The model list is fetched asynchronously; register the saved value as an
    # option up front so _refresh_models sees it as still-valid and keeps it.
    model = saved.get("model")
    if model:
        options = dict(model_select.options or {})
        options.setdefault(model, model)
        model_select.set_options(options)
        model_select.set_value(model)


save_defaults_btn.on_click(_save_defaults)


async def _on_provider_change() -> None:
    if quick_provider_select.value != provider_select.value:
        quick_provider_select.set_value(provider_select.value)
    await _refresh_models()


async def _on_quick_provider_change() -> None:
    if provider_select.value != quick_provider_select.value:
        provider_select.set_value(quick_provider_select.value)
    else:
        await _refresh_models()


def _on_model_change() -> None:
    if quick_model_select.value != model_select.value:
        quick_model_select.set_value(model_select.value)


def _on_quick_model_change() -> None:
    if model_select.value != quick_model_select.value:
        model_select.set_value(quick_model_select.value)


regen_session_btn.on_click(_regen_session)
close_drawer_btn.on_click(_close_drawer)
refresh_models_btn.on_click(_refresh_models)
provider_select.on_value_change(_on_provider_change)
model_select.on_value_change(_on_model_change)


# -- Page layout: App shell ----------------------------------------------------
with ui.column().classes("app-shell w-full").style("align-items:stretch"):

    # ---- Topbar ----
    with ui.element("div").classes("topbar"):
        ui.image("/martinus-assets/martinus.svg").style(
            "width:34px;height:34px;min-width:34px;border-radius:9px"
        ).props('fit="contain" alt="Martinus logo"')
        ui.label("Martinus").classes("topbar-title")
        ui.element("div").classes("topbar-spacer")

        quick_provider_select = ui.select(
            _provider_options,
            value=provider_select.value,
            label="Provider",
        ).props("dense outlined options-dense").style("min-width:135px;max-width:150px")

        quick_model_select = ui.select(
            {"": "Auto (profile)"},
            value=model_select.value,
            label="Model",
        ).props("dense outlined options-dense clearable").style("min-width:175px;max-width:240px")

        mode_run_btn = ui.html(
            '<span class="mode-pill active" id="mode-run">Run</span>'
        )
        mode_eval_btn = ui.html(
            '<span class="mode-pill" id="mode-eval">Auto-Eval</span>'
        )

        settings_btn = ui.button(icon="settings").props("flat dense round").style(
            "color:var(--text-muted)"
        )

        health_pill_el = ui.html(
            '<span class="health-pill" id="health-pill">'
            '<span class="health-dot" style="background:var(--text-dim)"></span>Health</span>'
        )

        status_badge_el = ui.html(
            '<span class="status-badge">\u25cf idle</span>'
        )

    quick_provider_select.on_value_change(_on_quick_provider_change)
    quick_model_select.on_value_change(_on_quick_model_change)

    # ---- Progress stepper ----
    stepper_container = ui.element("div").classes("stepper")
    with stepper_container:
        for idx, phase in enumerate(ALL_PHASES):
            if idx > 0:
                line_el = ui.html('<div class="step-line"></div>')
            else:
                line_el = None
            dot_el   = ui.html('<div class="step-dot pending"></div>')
            label_el = ui.html(f'<span class="step-label">{phase}</span>')
            stepper_phases[phase] = {
                "dot_el": dot_el, "label_el": label_el,
                "line_el": line_el, "state": "pending",
            }

    # ---- Chat area (scrollable) ----
    chat_area = ui.column().classes("chat-area w-full")
    chat_area.style("min-height:0")

    # ---- Input bar (fixed bottom) ----
    with ui.element("div").classes("input-bar w-full").style("width:100% !important"):
        # Reference document panel (hidden by default)
        ref_panel_container = ui.column().classes("w-full").style("display:none")
        with ref_panel_container:
            with ui.element("div").classes("ref-panel"):
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label("Reference Document").style(
                        "font-family:var(--font-head);font-size:0.75rem;font-weight:600;color:var(--text-muted)"
                    )
                    close_ref_btn = ui.button(icon="close").props("flat dense round size=sm").style(
                        "color:var(--text-muted)"
                    )
                topic_input = ui.textarea(
                    placeholder="Paste reference text to inject into the task prompt\u2026",
                ).props("autogrow rows=3 dense").classes("w-full")

        # Auto-eval extra row (hidden by default)
        auto_eval_row = ui.element("div").classes("auto-eval-row").style("display:none")
        with auto_eval_row:
            ui.label("Evaluator:").style(
                "font-family:var(--font-mono);font-size:0.75rem;color:var(--text-muted)"
            )
            auto_evaluator_model = ui.select(
                options={"": "Auto"},
                value=None,
                label="Select model",
            ).props("dense").style("min-width:160px;max-width:200px")
            ui.label("Workspace:").style(
                "font-family:var(--font-mono);font-size:0.75rem;color:var(--text-muted)"
            )
            auto_workspace_input = ui.input(
                value="/workspace/auto-eval",
            ).props("dense").style("max-width:180px")

        # Main textarea — large and prominent
        with ui.element("div").classes("task-textarea-wrap"):
            task_input = ui.textarea(
                placeholder="Describe what you want to build\u2026\n\nBe specific about features, tech stack, and requirements. "
                "The agent will ask clarifying questions if anything is unclear.",
            ).props("autogrow rows=4 maxRows=14").classes("w-full").style(
                "font-size:0.95rem"
            )
            char_counter_label = ui.label("0 / 2048").classes("char-counter")

        def _update_char_counter():
            n = len(task_input.value)
            char_counter_label.set_text(f"{n} / 2048")
            if n > 1800:
                char_counter_label.classes(add="warn")
                ui.run_javascript(
                    "var el=document.querySelector('.task-textarea-wrap .q-field__control');"
                    "if(el)el.classList.add('textarea-warn');"
                )
            else:
                char_counter_label.classes(remove="warn")
                ui.run_javascript(
                    "var el=document.querySelector('.task-textarea-wrap .q-field__control');"
                    "if(el)el.classList.remove('textarea-warn');"
                )

        task_input.on_value_change(lambda _: _update_char_counter())

        # Hidden goals — always injected via prompt engineering, not visible
        goals_input = ui.input(value="").style("display:none")
        goals_input.set_value(", ".join(_DEFAULT_GOALS))

        # Action row: attach, profile, run
        with ui.element("div").classes("input-actions"):
            attach_btn = ui.button(icon="attach_file").props("flat dense round").style(
                "color:var(--text-muted)"
            )

            profile_pill = ui.select(
                {"low": "Low", "medium": "Medium", "high": "High"},
                value="medium",
            ).props("dense borderless").style(
                "font-family:var(--font-mono);font-size:0.75rem;max-width:100px"
            )

            run_button = ui.button("\u25b6 Run").props("unelevated").style(
                "background:var(--primary);color:white;font-family:var(--font-head);"
                "font-weight:600;border-radius:10px;padding:8px 28px;font-size:0.9rem"
            )
            cancel_button = ui.button("\u25a0 Stop").props("outline").style(
                "color:var(--error);font-family:var(--font-head);font-weight:600;"
                "border-radius:10px;padding:8px 20px;font-size:0.9rem"
            )
            cancel_button.set_visibility(False)


# -- Sync profile pill <-> settings slider ------------------------------------
def _sync_profile_to_slider():
    mapping = {"low": 0, "medium": 1, "high": 2}
    profile_select.set_value(mapping.get(profile_pill.value, 1))


def _sync_slider_to_pill():
    mapping = {0: "low", 1: "medium", 2: "high"}
    profile_pill.set_value(mapping.get(int(profile_select.value), "medium"))


profile_pill.on_value_change(_sync_profile_to_slider)
profile_select.on_value_change(_sync_slider_to_pill)


# -- Mode toggle ---------------------------------------------------------------
def _set_mode_run():
    current_mode["value"] = "run"
    mode_run_btn.set_content(
        '<span class="mode-pill active" id="mode-run">Run</span>'
    )
    mode_eval_btn.set_content(
        '<span class="mode-pill" id="mode-eval">Auto-Eval</span>'
    )
    auto_eval_row.style("display:none")
    run_button.set_text("\u25b6 Run")


def _set_mode_eval():
    current_mode["value"] = "auto-eval"
    mode_run_btn.set_content(
        '<span class="mode-pill" id="mode-run">Run</span>'
    )
    mode_eval_btn.set_content(
        '<span class="mode-pill active" id="mode-eval">Auto-Eval</span>'
    )
    auto_eval_row.style("display:flex")
    run_button.set_text("\u25b6 Start Auto-Eval")


mode_run_btn.on("click", _set_mode_run)
mode_eval_btn.on("click", _set_mode_eval)


# -- Attachment toggle ---------------------------------------------------------
def _toggle_ref_panel():
    ref_panel_visible["value"] = not ref_panel_visible["value"]
    ref_panel_container.style(
        "display:block" if ref_panel_visible["value"] else "display:none"
    )


def _close_ref_panel():
    ref_panel_visible["value"] = False
    ref_panel_container.style("display:none")


attach_btn.on_click(_toggle_ref_panel)
close_ref_btn.on_click(_close_ref_panel)


# -- Settings drawer open -----------------------------------------------------
settings_btn.on_click(lambda: drawer.toggle())


# -- Health check --------------------------------------------------------------
async def check_provider_health() -> None:
    try:
        params = {"provider": provider_select.value}
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(
                f"{ORCHESTRATOR_URL}/v1/providers/health",
                params=params,
                headers={k: v for k, v in _headers().items() if k != "Content-Type"},
            )
            resp.raise_for_status()
            data = resp.json()

        reachable = data.get("reachable", False)
        model_ok  = data.get("modelAvailable", False)
        if reachable and model_ok:
            health_state["status"] = "healthy"
            dot_color = "var(--success)"
        elif reachable:
            health_state["status"] = "degraded"
            dot_color = "var(--warning)"
        else:
            health_state["status"] = "down"
            dot_color = "var(--error)"

        health_pill_el.set_content(
            f'<span class="health-pill" id="health-pill">'
            f'<span class="health-dot" style="background:{dot_color}"></span>'
            f'{data.get("provider", "?")} \u00b7 {data.get("model", "?")}</span>'
        )
    except Exception:
        health_state["status"] = "down"
        health_pill_el.set_content(
            '<span class="health-pill" id="health-pill">'
            '<span class="health-dot" style="background:var(--error)"></span>Offline</span>'
        )


health_pill_el.on("click", check_provider_health)

# Restore saved defaults before the first model fetch, then fetch models on
# every new browser connection.
ui.on("connect", _apply_saved_defaults)
ui.on("connect", _refresh_models)


# -- Stepper helpers -----------------------------------------------------------
def _reset_stepper():
    for phase in ALL_PHASES:
        s = stepper_phases[phase]
        s["state"] = "pending"
        s["dot_el"].set_content('<div class="step-dot pending"></div>')
        s["label_el"].set_content(f'<span class="step-label">{phase}</span>')
        if s["line_el"]:
            s["line_el"].set_content('<div class="step-line"></div>')


def _set_stepper_active(phase_name: str):
    if phase_name not in stepper_phases:
        return
    s = stepper_phases[phase_name]
    s["state"] = "active"
    s["dot_el"].set_content('<div class="step-dot active"></div>')
    s["label_el"].set_content(f'<span class="step-label active">{phase_name}</span>')


def _set_stepper_completed(phase_name: str):
    if phase_name not in stepper_phases:
        return
    s = stepper_phases[phase_name]
    s["state"] = "completed"
    s["dot_el"].set_content(
        '<div class="step-dot completed" style="font-size:8px;color:white">\u2713</div>'
    )
    s["label_el"].set_content(f'<span class="step-label completed">{phase_name}</span>')
    if stepper_phases[phase_name]["line_el"]:
        stepper_phases[phase_name]["line_el"].set_content(
            '<div class="step-line completed"></div>'
        )


def _set_stepper_failed(phase_name: str):
    if phase_name not in stepper_phases:
        return
    s = stepper_phases[phase_name]
    s["state"] = "failed"
    s["dot_el"].set_content(
        '<div class="step-dot" style="background:var(--error);border-color:var(--error);'
        'font-size:8px;color:white">\u2715</div>'
    )
    s["label_el"].set_content(
        f'<span class="step-label" style="color:var(--error);font-weight:600">{phase_name}</span>'
    )


# -- Auto-scroll chat to bottom -----------------------------------------------
async def _scroll_chat():
    try:
        await ui.run_javascript(
            'document.querySelector(".chat-area").scrollTop = '
            'document.querySelector(".chat-area").scrollHeight'
        )
    except Exception:
        pass


# -- Phase card builder --------------------------------------------------------
def _build_phase_card_streaming(phase_name: str, color: str, container) -> Dict[str, Any]:
    """Create a phase card in streaming state (animated bar, no output yet)."""
    with container:
        outer = (
            ui.element("div")
            .classes("phase-card w-full")
            .style(f"border-left:3px solid {color}")
        )
        with outer:
            header_el = ui.element("div").classes("phase-header")
            with header_el:
                ui.html(f'<div class="phase-dot" style="background:{color}"></div>')
                spinner_el = ui.html('<div class="spinner"></div>')
                ui.label(phase_name).style(
                    f"font-family:var(--font-head);font-weight:600;"
                    f"font-size:0.85rem;color:{color}"
                )
                ui.html('<div class="stream-bar"><div class="stream-bar-inner"></div></div>')
                stream_label = ui.label("streaming\u2026").style(
                    "font-family:var(--font-mono);font-size:0.7rem;color:var(--text-dim)"
                )
            # Files-written strip \u2014 hidden until the phase emits a files_written
            # event. Makes it obvious WHICH phase produced each file on disk.
            files_el = (
                ui.element("div")
                .classes("phase-files")
                .style(
                    "display:none;flex-wrap:wrap;gap:var(--s2);"
                    "padding:var(--s2) var(--s3);border-top:1px solid var(--border);"
                    "background:var(--surface-2);font-family:var(--font-mono);font-size:0.72rem"
                )
            )
            body_el = ui.element("div").classes("phase-body w-full").style("display:none")
            actions_el = ui.element("div").classes("phase-actions").style("display:none")

    return {
        "outer": outer, "header_el": header_el, "body_el": body_el,
        "spinner_el": spinner_el, "stream_label": stream_label,
        "actions_el": actions_el, "content": "",
        "files_el": files_el, "files_written": [],
        "color": color, "phase": phase_name, "expanded": False, "pinned": False,
    }


def _finalize_phase_card(blk: Dict[str, Any], output: str, attempts: int,
                         retrieved: int, model_name: str, prompt_version: str,
                         duration_s: float):
    """Convert a streaming card into its collapsed-done state."""
    phase_name = blk["phase"]
    color      = blk["color"]

    # Replace header content with collapsed summary
    blk["header_el"].clear()
    with blk["header_el"]:
        ui.html(f'<div class="phase-dot" style="background:{color}"></div>')
        ui.label(f"\u2713  {phase_name}").style(
            f"font-family:var(--font-head);font-weight:600;"
            f"font-size:0.85rem;color:{color}"
        )
        if duration_s > 0:
            ui.html(f'<span class="pill">{duration_s:.1f}s</span>')
        if attempts:
            ui.html(f'<span class="pill">{attempts} attempt{"s" if attempts != 1 else ""}</span>')
        if model_name:
            ui.html(f'<span class="pill">{model_name}</span>')
        arrow_el = ui.html('<span class="expand-arrow">\u25bc</span>')

    # Set body text
    blk["body_el"].clear()
    with blk["body_el"]:
        ui.label(output or "(empty phase output)").style(
            "font-family:var(--font-mono);font-size:0.78rem;line-height:1.65;"
            "color:var(--text);white-space:pre-wrap;word-break:break-word"
        )

    # Set actions
    blk["actions_el"].clear()
    with blk["actions_el"]:
        ui.html(
            '<button class="phase-action-btn" onclick="'
            "navigator.clipboard.writeText(this.closest('.phase-card').querySelector('.phase-body').innerText)"
            '">Copy output</button>'
        )
        if retrieved:
            ui.html(f'<span class="pill" style="margin-left:auto">retrieved {retrieved}</span>')
        if prompt_version:
            ui.html(f'<span class="pill">{prompt_version}</span>')

    # Wire expand/collapse on header click
    _expanded_state = {"v": False}

    def toggle_expand():
        _expanded_state["v"] = not _expanded_state["v"]
        if _expanded_state["v"]:
            blk["body_el"].style("display:block")
            blk["actions_el"].style("display:flex")
            arrow_el.set_content('<span class="expand-arrow open">\u25bc</span>')
        else:
            blk["body_el"].style("display:none")
            blk["actions_el"].style("display:none")
            arrow_el.set_content('<span class="expand-arrow">\u25bc</span>')

    blk["header_el"].on("click", toggle_expand)
    blk["toggle"] = toggle_expand


# -- Render evaluation card ----------------------------------------------------
def _render_eval_card(raw_text: str, container):
    """Parse evaluation scores and render a structured card + raw text."""
    scores = _parse_eval_scores(raw_text)
    result = scores.get("_result")
    overall = scores.get("_overall")
    hint = scores.get("_hint")

    dim_colors = {
        "Correctness": "#6366f1",
        "Code Quality": "#8b5cf6",
        "Edge Cases": "#d97706",
        "Performance": "#059669",
    }

    with container:
        with ui.element("div").classes("eval-card w-full"):
            ui.label("\U0001f9d1\u200d\u2696\ufe0f  Evaluation").style(
                "font-family:var(--font-head);font-weight:700;font-size:0.95rem;"
                "color:var(--text);margin-bottom:var(--s3)"
            )

            for dim in ("Correctness", "Code Quality", "Edge Cases", "Performance"):
                val = scores.get(dim)
                pct = (val * 10) if val is not None else 0
                bar_color = dim_colors.get(dim, "var(--primary)")
                score_text = f"{val}/10" if val is not None else "—"
                ui.html(
                    f'<div class="eval-dim-row">'
                    f'  <span class="eval-dim-label">{dim}</span>'
                    f'  <div class="eval-bar"><div class="eval-bar-fill" style="width:{pct}%;background:{bar_color}"></div></div>'
                    f'  <span class="eval-score">{score_text}</span>'
                    f'</div>'
                )

            ui.separator().style("margin:var(--s3) 0")

            with ui.row().classes("items-center gap-3"):
                if result == "PASS":
                    ui.html(
                        '<span style="font-family:var(--font-head);font-weight:700;font-size:1rem;'
                        'color:var(--success)">\u2705 PASS</span>'
                    )
                elif result == "FAIL":
                    ui.html(
                        '<span style="font-family:var(--font-head);font-weight:700;font-size:1rem;'
                        'color:var(--error)">\u274c FAIL</span>'
                    )
                else:
                    ui.html(
                        '<span style="font-family:var(--font-head);font-weight:700;font-size:1rem;'
                        'color:var(--text-muted)">Result: unknown</span>'
                    )
                if overall is not None:
                    ui.html(
                        f'<span style="font-family:var(--font-head);font-weight:700;font-size:1rem;'
                        f'color:var(--text)">SCORE: {overall}/10</span>'
                    )

            if result == "FAIL" and hint:
                ui.html(
                    f'<div style="margin-top:var(--s3);padding:var(--s2) var(--s3);'
                    f'background:rgba(217,119,6,0.1);border:1px solid rgba(217,119,6,0.3);'
                    f'border-radius:6px;font-family:var(--font-mono);font-size:0.78rem;color:var(--warning)">'
                    f'\U0001f4a1 {hint}</div>'
                )

            ui.separator().style("margin:var(--s3) 0")
            ui.label("Raw evaluator output").style(
                "font-family:var(--font-head);font-size:0.72rem;font-weight:600;"
                "color:var(--text-dim);text-transform:uppercase;letter-spacing:0.05em"
            )
            ui.label(raw_text).style(
                "font-family:var(--font-mono);font-size:0.75rem;line-height:1.6;"
                "white-space:pre-wrap;color:var(--text-muted);margin-top:var(--s2);"
                "max-height:200px;overflow-y:auto"
            )


# -- Main run handler ----------------------------------------------------------
async def run_agent_loop() -> None:
    is_running["value"] = True
    run_button.disable()
    _reset_stepper()
    status_badge_el.set_content(
        '<span class="status-badge" style="color:var(--primary)">\u25cf live</span>'
    )

    ref_text  = topic_input.value.strip() if ref_panel_visible["value"] else ""
    task_text = task_input.value.strip()
    if not task_text:
        run_button.enable()
        is_running["value"] = False
        return

    # Inject prompt engineering system + reference material
    enriched_task = f"{SYSTEM_TASK_PROMPT}\n---\nUSER TASK:\n{task_text}"
    if ref_text:
        enriched_task = f"{enriched_task}\n\nReference Material:\n{ref_text}"

    with chat_area:
        ui.html(
            f'<div style="font-family:var(--font-head);font-size:0.78rem;font-weight:600;'
            f'color:var(--text-dim);padding:var(--s2) 0">'
            f'\u25b6 New Run \u00b7 {datetime.now().strftime("%H:%M:%S")} \u00b7 '
            f'{_get_profile()} profile</div>'
        )

    active_blocks: Dict[str, Any] = {}
    phase_start_times: Dict[str, float] = {}

    try:
        async for event in stream_task_execution(
            task             = enriched_task,
            session_id       = session_input.value.strip(),
            workspace_path   = workspace_input.value.strip(),
            goals            = _goals(),
            provider         = provider_select.value,
            hardware_profile = _get_profile(),
            max_lines        = int(max_lines_input.value or 2000),
            model            = model_select.value or "",
        ):
            etype = event.get("type", "")

            if etype == "error":
                status_badge_el.set_content(
                    '<span class="status-badge" style="color:var(--error)">\u25cf error</span>'
                )
                with chat_area:
                    ui.html(
                        f'<div style="color:var(--error);font-family:var(--font-mono);font-size:0.8rem;'
                        f'padding:var(--s3)">Stream error: {event.get("message", "?")}</div>'
                    )
                break

            elif etype == "run_started":
                pass

            elif etype == "phase_started":
                phase_name = event.get("phase", "Unknown")
                color      = _phase_color(phase_name)
                phase_start_times[phase_name] = datetime.now().timestamp()
                _set_stepper_active(phase_name)
                blk = _build_phase_card_streaming(phase_name, color, chat_area)
                active_blocks[phase_name] = blk
                await _scroll_chat()

            elif etype == "token":
                phase_name = event.get("phase", "Unknown")
                token      = event.get("token", "")
                if phase_name in active_blocks:
                    active_blocks[phase_name]["content"] += token

            elif etype == "files_written":
                phase_name = event.get("phase", "Unknown")
                files      = event.get("files", []) or []
                # If the phase card doesn't exist yet (Repair emits before
                # phase_started in some paths), create a stub so the chip has
                # a home.
                if phase_name not in active_blocks:
                    color = _phase_color(phase_name)
                    active_blocks[phase_name] = _build_phase_card_streaming(phase_name, color, chat_area)
                blk = active_blocks[phase_name]
                blk["files_written"].extend(files)
                blk["files_el"].style("display:flex")
                with blk["files_el"]:
                    ui.html(
                        f'<span style="color:var(--text-dim)">wrote →</span>'
                    )
                    for path in files:
                        ui.html(
                            f'<span class="pill" style="background:var(--surface);'
                            f'border:1px solid {blk["color"]};color:{blk["color"]};'
                            f'padding:2px 8px">{path}</span>'
                        )
                await _scroll_chat()

            elif etype == "phase_completed":
                phase_name     = event.get("phase", "Unknown")
                output         = event.get("output", "")
                attempts       = event.get("attempts", 0)
                retrieved      = event.get("retrievedCount", 0)
                model_name     = event.get("model", "")
                prompt_version = event.get("promptVersion", "")
                start_ts       = phase_start_times.get(phase_name, 0)
                duration_s     = (datetime.now().timestamp() - start_ts) if start_ts else 0

                _set_stepper_completed(phase_name)

                if phase_name in active_blocks:
                    _finalize_phase_card(
                        active_blocks[phase_name], output, attempts,
                        retrieved, model_name, prompt_version, duration_s,
                    )
                else:
                    color = _phase_color(phase_name)
                    blk = _build_phase_card_streaming(phase_name, color, chat_area)
                    _finalize_phase_card(blk, output, attempts, retrieved, model_name, prompt_version, duration_s)
                await _scroll_chat()

            elif etype == "phase_failed":
                phase_name = event.get("phase", "Unknown")
                err_msg    = event.get("error", "unknown error")
                _set_stepper_failed(phase_name)
                if phase_name in active_blocks:
                    blk = active_blocks[phase_name]
                    blk["header_el"].clear()
                    with blk["header_el"]:
                        ui.html(f'<div class="phase-dot" style="background:var(--error)"></div>')
                        ui.label(f"\u2717  {phase_name}").style(
                            "font-family:var(--font-head);font-weight:600;"
                            "font-size:0.85rem;color:var(--error)"
                        )
                    blk["body_el"].style("display:block")
                    blk["body_el"].clear()
                    with blk["body_el"]:
                        ui.label(f"[FAILED] {err_msg}").style(
                            "font-family:var(--font-mono);font-size:0.78rem;color:var(--error)"
                        )

            elif etype == "run_completed":
                summary    = event.get("summary",    "")
                reflection = event.get("reflection", "")
                hw_profile = event.get("hardwareProfile", "")
                provider   = event.get("provider", "")

                status_badge_el.set_content(
                    '<span class="status-badge" style="color:var(--success)">\u25cf done</span>'
                )
                with chat_area:
                    if summary:
                        with ui.element("div").classes("summary-card w-full"):
                            ui.label("\u2705 Run Complete").style(
                                "font-family:var(--font-head);font-weight:600;"
                                "font-size:0.88rem;color:var(--primary)"
                            )
                            ui.label(summary).style(
                                "font-size:0.85rem;color:var(--text);margin-top:6px;line-height:1.55"
                            )
                            with ui.row().classes("gap-2 flex-wrap").style("margin-top:8px"):
                                if hw_profile:
                                    ui.html(f'<span class="pill">{hw_profile} profile</span>')
                                if provider:
                                    ui.html(f'<span class="pill">{provider}</span>')
                    if reflection:
                        with ui.element("div").classes("reflection-card w-full"):
                            ui.label("\U0001f4a1 Lesson Learned").style(
                                "font-family:var(--font-head);font-weight:600;"
                                "font-size:0.85rem;color:var(--success)"
                            )
                            ui.label(reflection).style(
                                "font-size:0.85rem;color:var(--text);margin-top:6px;line-height:1.55"
                            )
                await _scroll_chat()
                break

    except Exception as exc:
        status_badge_el.set_content(
            '<span class="status-badge" style="color:var(--error)">\u25cf error</span>'
        )
        with chat_area:
            ui.html(
                f'<div style="color:var(--error);font-family:var(--font-mono);font-size:0.8rem;'
                f'padding:var(--s3)">Execution failed: {exc}</div>'
            )
    finally:
        run_button.enable()
        is_running["value"] = False
        status_badge_el.set_content('<span class="status-badge">\u25cf idle</span>')


# -- Auto-eval handler ---------------------------------------------------------
async def run_auto_evaluation() -> None:
    is_running["value"] = True
    run_button.disable()
    _reset_stepper()
    status_badge_el.set_content(
        '<span class="status-badge" style="color:var(--primary)">\u25cf evaluating</span>'
    )

    task_desc = task_input.value.strip()
    if not task_desc:
        run_button.enable()
        is_running["value"] = False
        return

    # Inject prompt engineering system
    enriched_task = f"{SYSTEM_TASK_PROMPT}\n---\nUSER TASK:\n{task_desc}"

    session_id      = f"auto-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    workspace_path  = auto_workspace_input.value.strip() or "/workspace/auto-eval"
    provider        = provider_select.value
    evaluator_model = auto_evaluator_model.value or model_select.value or "gemma4:latest"
    accumulated     = ""

    with chat_area:
        ui.html(
            f'<div style="font-family:var(--font-head);font-size:0.78rem;font-weight:600;'
            f'color:var(--text-dim);padding:var(--s2) 0">'
            f'\U0001f9d1\u200d\u2696\ufe0f Auto-Eval \u00b7 '
            f'{datetime.now().strftime("%H:%M:%S")} \u00b7 {evaluator_model}</div>'
        )

    active_blocks: Dict[str, Any] = {}
    phase_start_times: Dict[str, float] = {}

    try:
        async for event in stream_task_execution(
            task             = enriched_task,
            session_id       = session_id,
            workspace_path   = workspace_path,
            goals            = _goals(),
            provider         = provider,
            hardware_profile = _get_profile(),
            max_lines        = int(max_lines_input.value or 2000),
            model            = model_select.value or "",
        ):
            etype = event.get("type", "")

            if etype == "error":
                with chat_area:
                    ui.html(
                        f'<div style="color:var(--error);font-family:var(--font-mono);font-size:0.8rem">'
                        f'Stream error: {event.get("message", "")}</div>'
                    )
                break

            elif etype == "phase_started":
                phase_name = event.get("phase", "Unknown")
                color = _phase_color(phase_name)
                phase_start_times[phase_name] = datetime.now().timestamp()
                _set_stepper_active(phase_name)
                blk = _build_phase_card_streaming(phase_name, color, chat_area)
                active_blocks[phase_name] = blk
                await _scroll_chat()

            elif etype == "token":
                phase_name = event.get("phase", "Unknown")
                token      = event.get("token", "")
                if phase_name in active_blocks:
                    active_blocks[phase_name]["content"] += token

            elif etype == "files_written":
                phase_name = event.get("phase", "Unknown")
                files      = event.get("files", []) or []
                # If the phase card doesn't exist yet (Repair emits before
                # phase_started in some paths), create a stub so the chip has
                # a home.
                if phase_name not in active_blocks:
                    color = _phase_color(phase_name)
                    active_blocks[phase_name] = _build_phase_card_streaming(phase_name, color, chat_area)
                blk = active_blocks[phase_name]
                blk["files_written"].extend(files)
                blk["files_el"].style("display:flex")
                with blk["files_el"]:
                    ui.html(
                        f'<span style="color:var(--text-dim)">wrote →</span>'
                    )
                    for path in files:
                        ui.html(
                            f'<span class="pill" style="background:var(--surface);'
                            f'border:1px solid {blk["color"]};color:{blk["color"]};'
                            f'padding:2px 8px">{path}</span>'
                        )
                await _scroll_chat()

            elif etype == "phase_completed":
                phase_name     = event.get("phase", "Unknown")
                output         = event.get("output", "")
                attempts       = event.get("attempts", 0)
                retrieved      = event.get("retrievedCount", 0)
                model_name     = event.get("model", "")
                prompt_version = event.get("promptVersion", "")
                start_ts       = phase_start_times.get(phase_name, 0)
                duration_s     = (datetime.now().timestamp() - start_ts) if start_ts else 0
                accumulated += f"\n\n--- {phase_name} ---\n{output}"

                _set_stepper_completed(phase_name)

                if phase_name in active_blocks:
                    _finalize_phase_card(
                        active_blocks[phase_name], output, attempts,
                        retrieved, model_name, prompt_version, duration_s,
                    )
                await _scroll_chat()

            elif etype == "run_completed":
                with chat_area:
                    eval_status_el = ui.label(
                        f"Pipeline complete. Evaluating with {evaluator_model}\u2026"
                    ).style(
                        "font-family:var(--font-mono);font-size:0.8rem;color:var(--primary);"
                        "padding:var(--s2) 0"
                    )
                await _scroll_chat()

                evaluation_result = await evaluate_with_ollama(
                    task_description = task_desc,
                    candidate_output = accumulated,
                    model            = evaluator_model,
                )

                eval_status_el.style("display:none")
                _render_eval_card(evaluation_result, chat_area)
                await _scroll_chat()
                break

    except Exception as exc:
        with chat_area:
            ui.html(
                f'<div style="color:var(--error);font-family:var(--font-mono);font-size:0.8rem">'
                f'Auto-evaluation failed: {exc}</div>'
            )
    finally:
        run_button.enable()
        is_running["value"] = False
        status_badge_el.set_content('<span class="status-badge">\u25cf idle</span>')


# -- Unified run dispatch -------------------------------------------------------
async def _on_run_click():
    run_control["task"] = asyncio.current_task()
    run_control["cancel_requested"] = False
    cancel_button.set_visibility(True)
    try:
        if current_mode["value"] == "auto-eval":
            await run_auto_evaluation()
        else:
            await run_agent_loop()
    except asyncio.CancelledError:
        if run_control["cancel_requested"]:
            status_badge_el.set_content(
                '<span class="status-badge" style="color:var(--warning)">\u25cf stopped</span>'
            )
            with chat_area:
                ui.html(
                    '<div style="color:var(--warning);font-family:var(--font-mono);font-size:0.8rem;'
                    'padding:var(--s3)">Run stopped by the user.</div>'
                )
        else:
            raise
    finally:
        run_control["task"] = None
        cancel_button.set_visibility(False)


def _cancel_current_run() -> None:
    task = run_control.get("task")
    if task is None or task.done():
        return
    run_control["cancel_requested"] = True
    cancel_button.disable()
    status_badge_el.set_content(
        '<span class="status-badge" style="color:var(--warning)">\u25cf stopping</span>'
    )
    task.cancel()
    cancel_button.enable()


run_button.on_click(_on_run_click)
cancel_button.on_click(_cancel_current_run)


# -- Enter-to-submit keyboard shortcut ----------------------------------------
ui.add_head_html("""
<script>
document.addEventListener('DOMContentLoaded', () => {
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      const el = e.target;
      if (el && el.closest && el.closest('.input-bar') && el.tagName === 'TEXTAREA') {
        e.preventDefault();
        const runBtn = document.querySelector('.input-bar button[class*="q-btn"]');
        if (runBtn) runBtn.click();
      }
    }
  });
});
</script>
""")


# -- Launch --------------------------------------------------------------------
ui.run(
    host   = UI_HOST,
    port   = UI_PORT,
    title  = "Martinus",
    reload = False,
    show   = False,
)
