from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agent import server


@pytest.fixture
def client() -> TestClient:
    return TestClient(server.app)


def make_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


def session(client: TestClient, workspace: Path) -> dict[str, Any]:
    response = client.post(
        "/v1/preview/session", json={"workspacePath": str(workspace)}
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_preview_serves_files_with_security_headers(
    client: TestClient, tmp_path: Path
) -> None:
    workspace = make_workspace(tmp_path)
    (workspace / "index.html").write_text(
        '<link rel="stylesheet" href="style.css">'
        '<script src="script.js"></script>',
        encoding="utf-8",
    )
    (workspace / "style.css").write_text(
        "body { color: red }", encoding="utf-8"
    )
    (workspace / "script.js").write_text(
        "window.ready = true", encoding="utf-8"
    )
    preview = session(client, workspace)

    assets = (
        ("index.html", "text/html"),
        ("style.css", "text/css"),
        ("script.js", "text/javascript"),
    )
    for name, expected_type in assets:
        response = client.get(f"/preview/{preview['token']}/{name}")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith(expected_type)
        assert response.headers["content-security-policy"].startswith(
            "sandbox allow-scripts; default-src 'none'"
        )
        assert "connect-src 'none'" in response.headers[
            "content-security-policy"
        ]
        assert "frame-ancestors" in response.headers["content-security-policy"]
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["referrer-policy"] == "no-referrer"


def test_unknown_token_is_404(client: TestClient) -> None:
    response = client.get("/preview/unknown/index.html")
    assert response.status_code == 404
    assert "content-security-policy" in response.headers


@pytest.mark.parametrize(
    "path", ["../secret.txt", "%2e%2e/secret.txt", "%2Fetc/passwd"]
)
def test_preview_rejects_traversal_and_absolute_paths(
    client: TestClient, tmp_path: Path, path: str
) -> None:
    workspace = make_workspace(tmp_path)
    (workspace / "index.html").write_text("ok", encoding="utf-8")
    preview = session(client, workspace)
    response = client.get(f"/preview/{preview['token']}/{path}")
    assert response.status_code in (400, 404)


def test_preview_rejects_symlinks_outside_workspace(
    client: TestClient, tmp_path: Path
) -> None:
    workspace = make_workspace(tmp_path)
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    (workspace / "index.html").write_text("ok", encoding="utf-8")
    (workspace / "escape.txt").symlink_to(outside)
    preview = session(client, workspace)
    response = client.get(f"/preview/{preview['token']}/escape.txt")
    assert response.status_code == 404


@pytest.mark.parametrize("name", ["secret.py", ".env", "data.db"])
def test_preview_rejects_disallowed_extensions(
    client: TestClient, tmp_path: Path, name: str
) -> None:
    workspace = make_workspace(tmp_path)
    (workspace / name).write_text("secret", encoding="utf-8")
    preview = session(client, workspace)
    assert client.get(f"/preview/{preview['token']}/{name}").status_code == 404


def test_preview_directory_without_index_and_oversized_file(
    client: TestClient, tmp_path: Path
) -> None:
    workspace = make_workspace(tmp_path)
    (workspace / "empty").mkdir()
    (workspace / "large.txt").write_bytes(
        b"x" * (5 * 1024 * 1024 + 1)
    )
    preview = session(client, workspace)
    assert client.get(f"/preview/{preview['token']}/empty").status_code == 404
    response = client.get(f"/preview/{preview['token']}/large.txt")
    assert response.status_code == 413


def test_workspace_tests_pass_fail_and_no_tests(
    client: TestClient, tmp_path: Path
) -> None:
    workspace = make_workspace(tmp_path)
    no_tests = client.post(
        "/v1/workspace/tests", json={"workspacePath": str(workspace)}
    ).json()
    assert no_tests["ran"] is False
    test_source = (
        "import unittest\n"
        "class T(unittest.TestCase):\n"
        " def test_ok(self): self.assertEqual(1, 1)\n"
    )
    (workspace / "test_sample.py").write_text(test_source, encoding="utf-8")
    passed = client.post(
        "/v1/workspace/tests", json={"workspacePath": str(workspace)}
    ).json()
    assert passed["ran"] is True and passed["passed"] is True
    failing_source = test_source.replace(
        "self.assertEqual(1, 1)", 'self.fail("expected failure")'
    )
    (workspace / "test_sample.py").write_text(failing_source, encoding="utf-8")
    failed = client.post(
        "/v1/workspace/tests", json={"workspacePath": str(workspace)}
    ).json()
    assert failed["ran"] is True and failed["passed"] is False


def test_workspace_test_timeout_is_bounded(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = make_workspace(tmp_path)
    (workspace / "test_hang.py").write_text(
        "while True: pass\n", encoding="utf-8"
    )
    monkeypatch.setattr(server, "_WORKSPACE_TEST_TIMEOUT_SECONDS", 0.2)
    result = client.post(
        "/v1/workspace/tests", json={"workspacePath": str(workspace)}
    ).json()
    assert result["timedOut"] is True


def test_workspace_root_confinement(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    monkeypatch.setattr(server, "_WORKSPACE_ROOT", str(allowed))
    preview = client.post(
        "/v1/preview/session", json={"workspacePath": str(outside)}
    )
    tests = client.post(
        "/v1/workspace/tests", json={"workspacePath": str(outside)}
    )
    assert preview.status_code == 400
    assert tests.status_code == 400
