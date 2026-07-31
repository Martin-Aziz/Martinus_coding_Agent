from __future__ import annotations

import unittest
from unittest.mock import patch

from agent.providers import (
    ChatRequest,
    DeterministicProviderError,
    Message,
    Router,
)


class ClaudeProviderTests(unittest.TestCase):
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
