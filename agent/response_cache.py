"""
agent/response_cache.py — Disk-backed semantic response cache.

Caches deterministic model responses keyed on (provider, model, temperature,
messages). On a cache hit the model call is skipped entirely — zero tokens,
near-zero latency. This is the largest single lever for the "iterate on the
same task" and "retry after a transient failure" workflows, where the same
prompt is issued repeatedly.

Design (deterministic, stdlib-only, no new deps):
- OFF by default. Opt in with MARTINUS_RESPONSE_CACHE=1. Caching model output
  is a behavior change (a stale cache could mask a fixed bug), so the user
  chooses it explicitly.
- Only caches LOW-temperature requests (temperature <= _MAX_CACHE_TEMPERATURE).
  A high-temperature request is asking for variety; replaying one answer defeats
  the point.
- Key = sha256 over a canonical JSON of the request's semantically-relevant
  fields. Model name and temperature are part of the key, so switching models
  or temperature never returns a stale answer.
- Entries live under MARTINUS_CACHE_DIR (default: <cwd>/.martinus-cache/responses),
  one JSON file per key. A simple count-based cap evicts oldest-by-mtime when
  the directory grows past _MAX_ENTRIES.
- Best-effort: any filesystem error degrades to a cache miss and a live call.
  The cache is a performance optimization, never a correctness dependency.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from typing import Optional

logger = logging.getLogger("martinus.response_cache")

# Only cache near-deterministic requests. Above this temperature, callers want
# variety and a replayed answer would be wrong.
_MAX_CACHE_TEMPERATURE = float(os.getenv("MARTINUS_CACHE_MAX_TEMP", "0.2"))
# Soft cap on stored entries; oldest-by-mtime are evicted past this.
_MAX_ENTRIES = int(os.getenv("MARTINUS_CACHE_MAX_ENTRIES", "500"))

# Matches OS temp-dir workspace paths that leak into prompts (repo-map headers,
# snapshot blocks). These are environmental, not semantic — two runs of the same
# task in different temp dirs must produce the SAME key. We normalize any such
# absolute path to a stable placeholder before hashing.
_TMP_PATH_RE = re.compile(
    r"(?:/private)?/(?:var|tmp)/[\w./\-]*?"
    r"(?:martinus|bench|kb|lin|cache|T)[\w./\-]*",
    re.IGNORECASE,
)
_WORKSPACE_PLACEHOLDER = "<WORKSPACE>"


def _normalize_paths(text: str, workspace_path: str = "") -> str:
    """Replace run-specific absolute paths with a stable placeholder so the same
    task in a different workspace hashes identically. The explicit workspace_path
    (when known) is normalized precisely; a general temp-dir heuristic catches
    the rest."""
    if workspace_path:
        # Normalize the exact workspace path and its realpath form.
        for wp in {workspace_path, os.path.realpath(workspace_path)}:
            if wp:
                text = text.replace(wp, _WORKSPACE_PLACEHOLDER)
    return _TMP_PATH_RE.sub(_WORKSPACE_PLACEHOLDER, text)


def is_enabled() -> bool:
    return os.getenv("MARTINUS_RESPONSE_CACHE", "").strip() in ("1", "true", "yes")


def _cache_dir() -> str:
    base = os.getenv("MARTINUS_CACHE_DIR", "").strip()
    if not base:
        base = os.path.join(os.getcwd(), ".martinus-cache", "responses")
    return base


def cacheable(temperature: float) -> bool:
    """A request is cacheable only when near-deterministic."""
    return temperature <= _MAX_CACHE_TEMPERATURE


def make_key(provider: str, model: str, temperature: float,
             messages: list[dict[str, str]], workspace_path: str = "") -> str:
    """Stable key over the semantically-relevant request fields.

    `messages` is a list of {"role", "content"} dicts. We quantize temperature
    to 2 decimals so 0.100001 and 0.1 collide (they are the same request).
    Run-specific absolute workspace paths are normalized to a placeholder so the
    same task in a different temp dir hashes identically (the whole point of a
    cross-run cache)."""
    payload = {
        "provider": provider,
        "model": model,
        "temp": round(temperature, 2),
        "messages": [
            {"role": m.get("role", ""),
             "content": _normalize_paths(m.get("content", ""), workspace_path)}
            for m in messages
        ],
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _path_for(key: str) -> str:
    return os.path.join(_cache_dir(), key + ".json")


def get(key: str) -> Optional[dict[str, object]]:
    """Return the cached response dict, or None on miss / any error."""
    path = _path_for(key)
    try:
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "content" in data:
            # Touch mtime so LRU-ish eviction keeps hot entries.
            try:
                os.utime(path, None)
            except OSError:
                pass
            return data
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("cache get failed for %s (non-fatal): %s", key[:12], exc)
    return None


def put(key: str, response: dict[str, object]) -> None:
    """Persist a response dict. Best-effort; failures are swallowed."""
    d = _cache_dir()
    path = _path_for(key)
    try:
        os.makedirs(d, exist_ok=True)
        # Atomic-ish write: temp file + rename.
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(response, f, ensure_ascii=False)
        os.replace(tmp, path)
        _maybe_evict(d)
    except OSError as exc:
        logger.debug("cache put failed for %s (non-fatal): %s", key[:12], exc)


def _maybe_evict(directory: str) -> None:
    """Evict oldest-by-mtime entries when over the cap. Cheap: only runs the
    listdir when we just wrote, and only sorts when actually over cap."""
    try:
        entries = [
            os.path.join(directory, n)
            for n in os.listdir(directory)
            if n.endswith(".json")
        ]
        if len(entries) <= _MAX_ENTRIES:
            return
        entries.sort(key=lambda p: _safe_mtime(p))
        # Remove the oldest until we're back under the cap.
        for p in entries[: len(entries) - _MAX_ENTRIES]:
            try:
                os.remove(p)
            except OSError:
                pass
    except OSError:
        pass


def _safe_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return time.time()
