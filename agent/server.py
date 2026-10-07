"""
agent/server.py — FastAPI HTTP server: SSE task streaming, workspace explorer, health.

Replaces the Go Fiber server. Same API surface — the NiceGUI UI needs no changes.
Endpoints:
  GET  /healthz
  GET  /v1/providers/health?provider=ollama|lmstudio
  GET  /v1/models?provider=ollama|lmstudio
  POST /v1/tasks/stream          (Server-Sent Events)
  GET  /v1/workspace/list
  GET  /v1/workspace/read
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import secrets
import time
from pathlib import PureWindowsPath
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from .orchestrator import AIMemoryClient, Engine
from .gates import run_workspace_tests as _run_workspace_tests
from .profiles import load_role_routing
from .providers import Router, from_env
from .repomap import build_repomap

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Settings from environment
# ---------------------------------------------------------------------------

_ALLOWED_ORIGINS = os.getenv(
    "ORCHESTRATOR_ALLOWED_ORIGINS",
    # NiceGUI dev/prototype (8091) + Tauri dev/production origins.
    "http://localhost:8091,http://127.0.0.1:8091,"
    "http://127.0.0.1:1420,http://localhost:1420,"
    "tauri://localhost,http://tauri.localhost",
).split(",")
_API_KEY = os.getenv("ORCHESTRATOR_API_KEY", "").strip()
_REQUIRE_API_KEY = os.getenv("ORCHESTRATOR_REQUIRE_API_KEY", "false").strip().lower() == "true"
_WORKSPACE_ROOT = os.getenv("ORCHESTRATOR_WORKSPACE_ROOT", "").strip()
_MAX_PREVIEW_BYTES = 512 * 1024
_MAX_LIVE_PREVIEW_BYTES = 5 * 1024 * 1024
_WORKSPACE_TEST_TIMEOUT_SECONDS = 60.0
_AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "http://127.0.0.1:8090")
_ROUTING_CONFIG = os.getenv("MODEL_ROUTING_CONFIG", "config/model-routing.yaml")
_PREVIEW_SESSIONS: dict[str, pathlib.Path] = {}
_PREVIEW_EXTENSIONS = {
    ".html", ".htm", ".css", ".js", ".mjs", ".json", ".svg", ".png",
    ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".txt", ".woff2",
}
_PREVIEW_MIME_TYPES = {
    ".html": "text/html; charset=utf-8", ".htm": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8", ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml; charset=utf-8", ".txt": "text/plain; charset=utf-8",
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".ico": "image/x-icon",
    ".woff2": "font/woff2",
}
_PREVIEW_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "sandbox allow-scripts; default-src 'none'; script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self' data:; "
        "connect-src 'none'; form-action 'none'; base-uri 'none'; frame-ancestors "
        + " ".join(origin.strip() for origin in _ALLOWED_ORIGINS if origin.strip())
    ),
    "X-Content-Type-Options": "nosniff",
    "Cache-Control": "no-store",
    "Referrer-Policy": "no-referrer",
}

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="martinus", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Origin", "Content-Type", "Accept", "X-API-Key"],
)


@app.middleware("http")
async def preview_response_headers(request: Request, call_next: Any) -> Response:
    response = await call_next(request)
    if request.url.path.startswith("/preview/"):
        for name, value in _PREVIEW_SECURITY_HEADERS.items():
            response.headers[name] = value
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    headers = dict(exc.headers or {})
    if request.url.path.startswith("/preview/"):
        headers.update(_PREVIEW_SECURITY_HEADERS)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=headers,
    )


def _build_engine() -> Engine:
    router = from_env()
    memory = AIMemoryClient(_AI_SERVICE_URL)
    routing = load_role_routing(_ROUTING_CONFIG)
    return Engine(router=router, memory=memory, routing=routing)


_engine = _build_engine()


def _check_api_key(request: Request) -> None:
    if request.url.path in ("/healthz",) or request.method == "OPTIONS":
        return
    if not _API_KEY:
        if _REQUIRE_API_KEY:
            # Fail-closed: operator explicitly asked for auth but forgot to configure it.
            raise HTTPException(
                status_code=503,
                detail="ORCHESTRATOR_REQUIRE_API_KEY=true but ORCHESTRATOR_API_KEY is empty",
            )
        return  # dev / local mode
    key = request.headers.get("X-API-Key", "").strip()
    import hmac
    if not hmac.compare_digest(key, _API_KEY):
        raise HTTPException(status_code=401, detail="missing or invalid API key")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ExecuteTaskRequest(BaseModel):
    sessionId: str = ""
    task: str = ""
    workspacePath: str = ""
    goals: list[str] = []
    provider: str = "ollama"
    hardwareProfile: str = "medium"
    maxLines: int = 2000
    model: str = ""


class WorkspacePathRequest(BaseModel):
    workspacePath: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/healthz")
def health() -> dict:
    return {"ok": True, "service": "orchestrator-python"}


@app.get("/v1/providers/health")
async def provider_health(request: Request, provider: str = Query("")) -> dict:
    _check_api_key(request)
    if provider and provider not in ("ollama", "lmstudio", "claude"):
        raise HTTPException(status_code=400, detail="provider must be ollama, lmstudio, or claude")
    return _engine.check_provider_health(provider or "ollama")


@app.get("/v1/models")
async def list_models(request: Request, provider: str = Query("ollama")) -> dict:
    _check_api_key(request)
    models = _engine._router.list_models(provider)
    return {"models": models, "provider": provider}


@app.post("/v1/tasks/stream")
async def stream_task(request: Request, req: ExecuteTaskRequest) -> StreamingResponse:
    _check_api_key(request)

    if not req.task.strip():
        raise HTTPException(status_code=400, detail="task is required")

    queue: asyncio.Queue[Optional[dict]] = asyncio.Queue()
    # Set when the client disconnects; the on_event callback drops subsequent
    # events, and the executor thread checks it between phases to short-circuit.
    import threading
    cancel_event = threading.Event()

    def on_event(event: dict) -> None:
        if cancel_event.is_set():
            return
        queue.put_nowait(event)

    async def run_in_thread() -> None:
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: _engine.execute(
                    session_id=req.sessionId,
                    task=req.task,
                    workspace_path=req.workspacePath,
                    goals=req.goals,
                    provider_name=req.provider,
                    hardware_profile=req.hardwareProfile,
                    max_lines=req.maxLines,
                    model_override=req.model,
                    on_event=on_event,
                    cancel_check=cancel_event.is_set,
                ),
            )
        except Exception as exc:
            if not cancel_event.is_set():
                queue.put_nowait({"type": "error", "message": str(exc)})
        finally:
            queue.put_nowait(None)  # sentinel

    async def event_stream() -> AsyncIterator[str]:
        task = asyncio.create_task(run_in_thread())
        try:
            while True:
                if await request.is_disconnected():
                    cancel_event.set()
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # No event in 15s; keep-alive comment for reverse proxies and check disconnect.
                    yield ": keepalive\n\n"
                    continue
                if event is None:
                    break
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            cancel_event.set()
            task.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Workspace routes
# ---------------------------------------------------------------------------

def _safe_abs(workspace_path: str, rel: str) -> pathlib.Path:
    base = pathlib.Path(workspace_path).resolve()
    target = (base / rel).resolve()
    # Reject path traversal
    try:
        target.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=400, detail="path escapes workspace root")
    if _WORKSPACE_ROOT:
        root = pathlib.Path(_WORKSPACE_ROOT).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            raise HTTPException(status_code=400, detail="path escapes workspace root")
    return target


def _validated_workspace(workspace_path: str) -> pathlib.Path:
    if not workspace_path:
        raise HTTPException(status_code=400, detail="workspacePath is required")
    base = pathlib.Path(workspace_path).resolve()
    if not base.is_dir():
        raise HTTPException(status_code=400, detail="workspacePath is not a directory")
    if _WORKSPACE_ROOT:
        root = pathlib.Path(_WORKSPACE_ROOT).resolve()
        try:
            base.relative_to(root)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="workspacePath is outside allowed workspace root",
            )
    return base


@app.post("/v1/preview/session")
async def create_preview_session(request: Request, req: WorkspacePathRequest) -> dict[str, Any]:
    _check_api_key(request)
    workspace = _validated_workspace(req.workspacePath)
    for old_token, old_workspace in list(_PREVIEW_SESSIONS.items()):
        if old_workspace == workspace:
            _PREVIEW_SESSIONS.pop(old_token, None)
    token = secrets.token_urlsafe(16)
    _PREVIEW_SESSIONS[token] = workspace
    entries: list[str] = []
    for entry in workspace.rglob("*"):
        try:
            resolved = entry.resolve()
            resolved.relative_to(workspace)
        except (OSError, ValueError):
            continue
        if not resolved.is_file():
            continue
        if resolved.suffix.lower() == ".py":
            entries.append(resolved.relative_to(workspace).as_posix())
            continue
        if resolved.suffix.lower() not in _PREVIEW_EXTENSIONS:
            continue
        try:
            if resolved.stat().st_size <= _MAX_LIVE_PREVIEW_BYTES:
                entries.append(resolved.relative_to(workspace).as_posix())
        except OSError:
            continue
    return {
        "token": token,
        "entry": "index.html" if (workspace / "index.html").is_file() else None,
        "files": sorted(entries),
    }


@app.get("/preview/{token}/{path:path}")
async def serve_preview(token: str, path: str = "") -> Response:
    workspace = _PREVIEW_SESSIONS.get(token)
    if workspace is None:
        raise HTTPException(status_code=404, detail="preview session not found")
    rel = path or "index.html"
    requested = pathlib.PurePosixPath(rel)
    windows_path = PureWindowsPath(rel)
    if (
        requested.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in requested.parts
        or "\\" in rel
    ):
        raise HTTPException(status_code=404, detail="preview path not found")
    candidate = workspace.joinpath(*requested.parts)
    try:
        target = candidate.resolve(strict=True)
        target.relative_to(workspace)
    except (OSError, ValueError):
        raise HTTPException(status_code=404, detail="preview path not found")
    if target.is_dir():
        target = target / "index.html"
        try:
            target = target.resolve(strict=True)
            target.relative_to(workspace)
        except (OSError, ValueError):
            raise HTTPException(status_code=404, detail="preview path not found")
    if not target.is_file() or target.suffix.lower() not in _PREVIEW_EXTENSIONS:
        raise HTTPException(status_code=404, detail="preview path not found")
    try:
        with target.open("rb") as preview_file:
            content = preview_file.read(_MAX_LIVE_PREVIEW_BYTES + 1)
        if len(content) > _MAX_LIVE_PREVIEW_BYTES:
            raise HTTPException(status_code=413, detail="preview file exceeds 5 MB limit")
    except OSError:
        raise HTTPException(status_code=404, detail="preview path not found")
    return Response(
        content=content,
        media_type=_PREVIEW_MIME_TYPES.get(
            target.suffix.lower(), "application/octet-stream"
        ),
        headers=_PREVIEW_SECURITY_HEADERS,
    )


@app.post("/v1/workspace/tests")
async def run_workspace_tests_route(request: Request, req: WorkspacePathRequest) -> dict[str, Any]:
    _check_api_key(request)
    workspace = _validated_workspace(req.workspacePath)
    started = time.monotonic()
    result = await asyncio.to_thread(
        _run_workspace_tests,
        str(workspace),
        timeout=_WORKSPACE_TEST_TIMEOUT_SECONDS,
    )
    duration_ms = int((time.monotonic() - started) * 1000)
    if result is None:
        return {
            "ran": False,
            "passed": None,
            "timedOut": False,
            "output": "No Python test files were found in this workspace.",
            "framework": None,
            "durationMs": duration_ms,
        }
    command = result.get("command", "")
    timed_out = result.get("returncode") == -1 and "timed out" in result.get("output", "")
    return {
        "ran": True,
        "passed": bool(result.get("passed")),
        "timedOut": timed_out,
        "output": str(result.get("output", ""))[-4000:],
        "framework": "pytest" if "pytest" in command else "unittest",
        "durationMs": duration_ms,
    }


@app.get("/v1/workspace/list")
async def workspace_list(
    request: Request,
    workspacePath: str = Query(""),
    path: str = Query("."),
) -> list[dict]:
    _check_api_key(request)
    if not workspacePath:
        raise HTTPException(status_code=400, detail="workspacePath is required")
    target = _safe_abs(workspacePath, path)
    if not target.exists():
        raise HTTPException(status_code=400, detail="path does not exist")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="path is not a directory")
    entries = []
    for entry in sorted(target.iterdir(), key=lambda e: (not e.is_dir(), e.name)):
        rel = str(entry.relative_to(pathlib.Path(workspacePath).resolve()))
        stat = entry.stat()
        entries.append({
            "name": entry.name,
            "path": rel,
            "is_dir": entry.is_dir(),
            "size": stat.st_size,
            "mod_time": stat.st_mtime,
        })
    return entries


@app.get("/v1/workspace/read")
async def workspace_read(
    request: Request,
    workspacePath: str = Query(""),
    path: str = Query(""),
) -> dict:
    _check_api_key(request)
    if not workspacePath:
        raise HTTPException(status_code=400, detail="workspacePath is required")
    if not path:
        raise HTTPException(status_code=400, detail="path is required")
    target = _safe_abs(workspacePath, path)
    if not target.exists():
        raise HTTPException(status_code=400, detail="path does not exist")
    if target.is_dir():
        raise HTTPException(status_code=400, detail="path is a directory")
    size = target.stat().st_size
    with open(target, "rb") as f:
        raw = f.read(_MAX_PREVIEW_BYTES)
    content = raw.decode("utf-8", errors="replace")
    return {
        "path": path,
        "size": size,
        "truncated": size > _MAX_PREVIEW_BYTES,
        "content": content,
    }


@app.get("/v1/workspace/ping")
async def workspace_ping() -> dict:
    return {"ok": True, "routes": ["list", "read", "ping"]}


# ---------------------------------------------------------------------------
# Repository map (LOCALFORGE §6.2) — deterministic, read-only (risk level L0).
# ---------------------------------------------------------------------------

@app.get("/v1/repomap")
async def repomap(
    request: Request,
    workspacePath: str = Query(""),
    layers: str = Query("1,2,3"),
) -> dict:
    """Return a deterministic Layer 1-3 repository map for the workspace.

    Read-only and dependency-free; reuses the same path-confinement as the
    workspace explorer routes. `layers` is a comma-separated subset of 1,2,3.
    """
    _check_api_key(request)
    if not workspacePath:
        raise HTTPException(status_code=400, detail="workspacePath is required")
    target = _safe_abs(workspacePath, ".")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="workspacePath is not a directory")
    try:
        wanted = tuple(sorted({int(x) for x in layers.split(",") if x.strip() in ("1", "2", "3")}))
    except ValueError:
        raise HTTPException(status_code=400, detail="layers must be a comma-separated subset of 1,2,3")
    if not wanted:
        wanted = (1, 2, 3)
    return build_repomap(str(target), layers=wanted)
