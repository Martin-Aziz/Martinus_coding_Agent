from __future__ import annotations

from agent.prompts import PROMPT_VERSION, build_prompt, build_system_instruction


def test_frontend_prompt_requires_polished_complete_ui() -> None:
    task = """
Create a tiny browser calculator project.
<file path="index.html">...</file>
<file path="styles.css">...</file>
<file path="app.js">...</file>
"""

    prompt = build_prompt(
        phase="Implementation",
        task=task,
        workspace_path="/tmp/martinus-demo",
        provider="ollama",
        max_lines=2000,
        goals=["Create inspectable local files"],
        entries=[],
        retrieved=[],
    )

    assert PROMPT_VERSION == "v10.implementation-with-file-editing"
    assert "Frontend Polish Contract" in prompt
    assert "Small scope means compact code, NOT plain visuals" in prompt
    assert "Elevated card surfaces" in prompt
    assert "keyboard shortcuts" in prompt
    assert "each file must contain meaningful complete content" in prompt
    assert "Files First" in prompt
    assert "Code Implementation" in prompt
    assert prompt.index("Code Implementation") < prompt.index("Brief Review Note")


def test_frontend_system_instruction_mentions_demo_ready_design() -> None:
    instruction = build_system_instruction(
        phase="Implementation",
        max_lines=2000,
        task="Create a responsive HTML CSS JavaScript calculator",
    )

    assert "FRONTEND SPECIALIST MODE ACTIVE" in instruction
    assert "willing to demo" in instruction
    assert "GLASS / MODERN SURFACE RECIPE" in instruction
    assert "working, complete implementation" in instruction
    assert "No dead controls" in instruction
