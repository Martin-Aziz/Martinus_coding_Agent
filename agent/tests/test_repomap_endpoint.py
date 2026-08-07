from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from agent.server import app

client = TestClient(app)


def _seed(root: Path) -> None:
    (root / "main.py").write_text(
        "class App:\n    pass\n\ndef start():\n    return App()\n", encoding="utf-8"
    )
    (root / "requirements.txt").write_text("fastapi\n", encoding="utf-8")


def test_repomap_endpoint_returns_layers(tmp_path: Path) -> None:
    _seed(tmp_path)
    resp = client.get("/v1/repomap", params={"workspacePath": str(tmp_path)})
    assert resp.status_code == 200
    data = resp.json()
    assert "Python" in data["manifest"]["languages"]
    assert data["architecture"]["modules"]
    assert any(fs["path"] == "main.py" for fs in data["symbols"])


def test_repomap_endpoint_requires_workspace_path() -> None:
    resp = client.get("/v1/repomap", params={"workspacePath": ""})
    assert resp.status_code == 400


def test_repomap_endpoint_rejects_non_directory(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("x", encoding="utf-8")
    resp = client.get("/v1/repomap", params={"workspacePath": str(f)})
    assert resp.status_code == 400


def test_repomap_endpoint_layer_subset(tmp_path: Path) -> None:
    _seed(tmp_path)
    resp = client.get("/v1/repomap", params={"workspacePath": str(tmp_path), "layers": "1"})
    assert resp.status_code == 200
    data = resp.json()
    assert "manifest" in data
    assert "symbols" not in data  # layer 3 not requested
