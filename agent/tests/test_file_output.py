from __future__ import annotations

from pathlib import Path

from agent.orchestrator import (
    _build_verification_summary,
    _extract_requested_files,
    _extract_and_write_files,
    _iter_file_tags,
    _missing_requested_files,
    _validate_workspace_references,
)


def test_file_tags_accept_straight_single_and_curly_quotes() -> None:
    output = """
<file path="README.md">readme</file>
<file path='styles.css'>css</file>
<file path=“index.html">html</file>
"""

    assert _iter_file_tags(output) == [
        ("README.md", "readme"),
        ("styles.css", "css"),
        ("index.html", "html"),
    ]


def test_extract_and_write_files_writes_requested_structure(tmp_path: Path) -> None:
    output = """
<file path="README.md"># Demo</file>
<file path="index.html"><!doctype html></file>
<file path="src/calculator.py">class Calculator: pass</file>
<file path="tests/test_calculator.py">def test_ok(): assert True</file>
"""

    written = _extract_and_write_files(str(tmp_path), output)

    assert sorted(written) == [
        "README.md",
        "index.html",
        "src/calculator.py",
        "tests/test_calculator.py",
    ]
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "# Demo"
    assert (tmp_path / "src" / "calculator.py").exists()


def test_workspace_validation_does_not_report_partial_extensions(tmp_path: Path) -> None:
    (tmp_path / "tokens.css").write_text(":root {}", encoding="utf-8")
    (tmp_path / "index.html").write_text("<!doctype html>", encoding="utf-8")

    output = """
<file path="tokens.css">:root {}</file>
<file path="index.html"><link rel="stylesheet" href="tokens.css"></file>
"""

    assert _validate_workspace_references(str(tmp_path), output) == ""


def test_extract_requested_files_preserves_unique_requested_order() -> None:
    task = """
Create a browser calculator with README.md, index.html, styles.css, and app.js.
Return files using <file path="index.html">...</file> tags.
"""

    assert _extract_requested_files(task) == [
        "README.md",
        "index.html",
        "styles.css",
        "app.js",
    ]


def test_missing_requested_files_reports_only_absent_files(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Demo", encoding="utf-8")
    (tmp_path / "index.html").write_text("<!doctype html>", encoding="utf-8")

    missing = _missing_requested_files(
        str(tmp_path),
        ["README.md", "index.html", "styles.css", "app.js"],
    )

    assert missing == ["styles.css", "app.js"]


def test_verification_summary_reports_real_files_and_static_checks(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html><title>Demo</title>", encoding="utf-8")
    (tmp_path / "app.js").write_text("const ready = true;\n", encoding="utf-8")

    summary = _build_verification_summary(
        str(tmp_path),
        ["index.html", "app.js"],
        ["app.js", "index.html", "app.js"],
    )

    assert summary["passed"] is True
    assert summary["generatedFiles"] == ["app.js", "index.html"]
    assert summary["missingFiles"] == []
    assert {check["kind"] for check in summary["checks"]} == {"node_check", "html_parse"}
    assert all(check["ok"] for check in summary["checks"])


def test_verification_summary_fails_when_requested_file_is_missing(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html><title>Demo</title>", encoding="utf-8")

    summary = _build_verification_summary(
        str(tmp_path),
        ["index.html", "styles.css"],
        ["index.html"],
    )

    assert summary["passed"] is False
    assert summary["missingFiles"] == ["styles.css"]
