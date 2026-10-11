from __future__ import annotations

import unittest
from unittest.mock import patch
from types import SimpleNamespace
import os

from agent.providers import (
    ChatRequest,
    ClaudeCodeClient,
    DeterministicProviderError,
    Message,
    Router,
)


class ClaudeProviderTests(unittest.TestCase):
    def test_chat_uses_system_prompt_text_output_no_tools_and_neutral_cwd(self) -> None:
        client = ClaudeCodeClient()
        proc = SimpleNamespace(returncode=0, stdout='<file path="a.txt">ok</file>', stderr="")
        with patch("subprocess.run", return_value=proc) as run:
            response = client.chat(ChatRequest(messages=[
                Message(role="system", content="Return file blocks."),
                Message(role="user", content="Create a file."),
            ]))

        args, kwargs = run.call_args
        command = args[0]
        self.assertEqual(response.content, '<file path="a.txt">ok</file>')
        self.assertEqual(kwargs["input"], "Create a file.")
        self.assertEqual(command[command.index("--tools") + 1], "")
        self.assertIn("--safe-mode", command)
        self.assertIn("--strict-mcp-config", command)
        self.assertEqual(command[command.index("--system-prompt") + 1], "Return file blocks.")
        self.assertEqual(command[command.index("--output-format") + 1], "text")
        self.assertEqual(command[command.index("--permission-prompts") + 1], "none")
        self.assertTrue(os.path.basename(kwargs["cwd"]).startswith("martinus-claude-"))

    def test_streaming_chat_still_uses_plain_text_transport(self) -> None:
        client = ClaudeCodeClient()
        content = '<file path="a.txt">ok</file>'
        proc = SimpleNamespace(returncode=0, stdout=content, stderr="")
        tokens: list[str] = []
        with patch("subprocess.run", return_value=proc) as run:
            response = client.chat(ChatRequest(
                messages=[
                    Message(role="system", content="Return file blocks."),
                    Message(role="user", content="Create a file."),
                ],
                on_token=tokens.append,
            ))

        command = run.call_args.args[0]
        self.assertEqual(response.content, content)
        self.assertEqual(tokens, [content])
        self.assertEqual(command[command.index("--output-format") + 1], "text")

    def test_disabled_claude_does_not_fall_back_to_ollama(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            router = Router()

        self.assertFalse(router.health("claude").reachable)
        self.assertEqual(router.list_models("claude"), [])
        with self.assertRaises(DeterministicProviderError):
            router.chat(
                "claude",
                ChatRequest(messages=[Message(role="user", content="hello")]),
            )

    def test_enabled_claude_exposes_health_and_model(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "MARTINUS_ENABLE_CLAUDE_ADAPTER": "1",
                "MARTINUS_CLAUDE_BINARY": "/bin/echo",
            },
            clear=True,
        ):
            router = Router()

        self.assertTrue(router.health("claude").reachable)
        self.assertEqual(router.list_models("claude"), ["claude-code"])


if __name__ == "__main__":
    unittest.main()
