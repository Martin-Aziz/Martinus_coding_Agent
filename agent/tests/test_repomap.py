from __future__ import annotations

from pathlib import Path

from agent.repomap import (
    build_architecture,
    build_manifest,
    build_repomap,
    build_symbol_index,
    compact,
)


def _seed_repo(root: Path) -> None:
    (root / "agent").mkdir()
    (root / "agent" / "server.py").write_text(
        "import fastapi\n"
        "app = fastapi.FastAPI()\n"
        "\n"
        "class Engine:\n"
        "    def run(self):\n"
        "        return 1\n"
        "\n"
        "@app.get('/healthz')\n"
        "def healthz():\n"
        "    return {}\n",
        encoding="utf-8",
    )
    (root / "web").mkdir()
    (root / "web" / "app.js").write_text(
        "export function boot() {}\n"
        "const helper = () => {}\n"
        "class Widget {}\n",
        encoding="utf-8",
    )
    (root / "package.json").write_text(
        '{"scripts": {"build": "webpack", "test": "jest", "lint": "eslint ."}}',
        encoding="utf-8",
    )
    (root / "Makefile").write_text(
        "test:\n\tpytest\nbuild:\n\techo build\ntypecheck:\n\tmypy .\n",
        encoding="utf-8",
    )
    (root / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
    (root / "main.py").write_text("def main():\n    pass\n", encoding="utf-8")


def test_manifest_extracts_languages_entrypoints_and_commands(tmp_path: Path) -> None:
    _seed_repo(tmp_path)
    manifest = build_manifest(str(tmp_path))

    assert "Python" in manifest["languages"]
    assert "JavaScript" in manifest["languages"]
    assert "pip" in manifest["packageManagers"]
    assert "npm" in manifest["packageManagers"]
    assert "make" in manifest["packageManagers"]
    assert "main.py" in manifest["entryPoints"]
    # Commands come from both package.json and Makefile.
    assert "npm run test" in manifest["commands"]["test"]
    assert "make test" in manifest["commands"]["test"]
    assert "npm run build" in manifest["commands"]["build"]
    assert "make typecheck" in manifest["commands"]["lint"]


def test_pyproject_pytest_section_detected_without_tomllib(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[build-system]\nrequires = [\"setuptools\"]\n\n"
        "[tool.pytest.ini_options]\naddopts = \"-q\"\n",
        encoding="utf-8",
    )
    (tmp_path / "main.py").write_text("def main():\n    pass\n", encoding="utf-8")
    manifest = build_manifest(str(tmp_path))
    assert "pip" in manifest["packageManagers"]
    assert "pytest" in manifest["commands"]["test"]
    assert "python -m build" in manifest["commands"]["build"]


def test_architecture_lists_top_level_modules(tmp_path: Path) -> None:
    _seed_repo(tmp_path)
    arch = build_architecture(str(tmp_path))
    names = {m["name"] for m in arch["modules"]}
    assert "agent" in names
    assert "web" in names
    agent_mod = next(m for m in arch["modules"] if m["name"] == "agent")
    assert agent_mod["dominantLanguage"] == "Python"


def test_symbol_index_python_finds_class_function_and_route(tmp_path: Path) -> None:
    _seed_repo(tmp_path)
    index = build_symbol_index(str(tmp_path))
    server = next(fs for fs in index if fs.path == "agent/server.py")
    kinds = {(s.name, s.kind) for s in server.symbols}
    assert ("Engine", "class") in kinds
    assert ("healthz", "route") in kinds  # detected via @app.get decorator


def test_symbol_index_javascript_heuristic(tmp_path: Path) -> None:
    _seed_repo(tmp_path)
    index = build_symbol_index(str(tmp_path))
    app_js = next(fs for fs in index if fs.path == "web/app.js")
    names = {s.name for s in app_js.symbols}
    assert "boot" in names
    assert "Widget" in names
    assert "helper" in names


def test_broken_python_file_is_skipped_not_fatal(tmp_path: Path) -> None:
    _seed_repo(tmp_path)
    (tmp_path / "broken.py").write_text("def oops(:\n    this is not python\n", encoding="utf-8")
    # Must not raise.
    index = build_symbol_index(str(tmp_path))
    broken = next(fs for fs in index if fs.path == "broken.py")
    assert broken.parse_error != ""
    # Other files still indexed.
    assert any(fs.path == "agent/server.py" for fs in index)


def test_excluded_dirs_not_walked(tmp_path: Path) -> None:
    _seed_repo(tmp_path)
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.js").write_text("function junk(){}", encoding="utf-8")
    index = build_symbol_index(str(tmp_path))
    assert not any("node_modules" in fs.path for fs in index)


def test_build_repomap_and_compact_under_budget(tmp_path: Path) -> None:
    _seed_repo(tmp_path)
    repomap = build_repomap(str(tmp_path))
    assert set(repomap.keys()) >= {"root", "manifest", "architecture", "symbols"}
    text = compact(repomap, budget=8000)
    assert len(text) <= 8000
    assert "REPOSITORY MAP" in text
    assert "Languages:" in text


def test_build_repomap_missing_dir_returns_empty(tmp_path: Path) -> None:
    missing = str(tmp_path / "does-not-exist")
    repomap = build_repomap(missing)
    assert repomap["symbols"] == []
    assert repomap["manifest"] == {}


def test_compact_respects_tight_budget(tmp_path: Path) -> None:
    _seed_repo(tmp_path)
    repomap = build_repomap(str(tmp_path))
    text = compact(repomap, budget=200)
    assert len(text) <= 200 + len("=== END REPOSITORY MAP (truncated to budget) ===") + 1
