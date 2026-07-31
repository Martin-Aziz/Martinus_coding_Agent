import logging
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# CoT-aligned system prompt applied to all requests through this client.
# Instructs the model to: reason before answering, identify the key constraint,
# and produce a concise conclusion — reducing hallucinations on planning tasks.
_COT_SYSTEM_PROMPT = (
    "ROLE: You are a concise senior software engineer assisting a local coding agent.\n\n"
    "REASONING PROTOCOL: Before answering, explicitly identify: "
    "(1) what the question actually asks, "
    "(2) the single most important constraint, "
    "(3) whether a simpler answer exists. "
    "State your reasoning, then give a focused conclusion. "
    "Avoid restating the question, avoid padding, avoid qualifiers without evidence."
)

# Task-type temperature routing.
# Source: Google Prompt Engineering Whitepaper (2024).
TEMPERATURE_BY_TASK: Dict[str, float] = {
    "summarize": 0.1,
    "plan": 0.1,
    "explain": 0.2,
    "default": 0.1,
}


class LocalModelClient:
    def __init__(self) -> None:
        self.default_provider = (
            os.getenv("AI_PROVIDER", "ollama").strip().lower()
        )
        self.ollama_url = os.getenv(
            "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
        ).rstrip("/")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
        self.lmstudio_url = os.getenv(
            "LMSTUDIO_BASE_URL", "http://127.0.0.1:1234"
        ).rstrip("/")
        self.lmstudio_model = os.getenv("LMSTUDIO_MODEL", "local-model")
        self.timeout = float(os.getenv("AI_HTTP_TIMEOUT_SECONDS", "90"))  # seconds; set via AI_HTTP_TIMEOUT_SECONDS env var (default 90)

    async def chat(
        self,
        prompt: str,
        task_type: str = "default",
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """Send a chat request to the configured local provider.

        Args:
            prompt: The user message.
            task_type: Controls temperature when not explicitly set.
                One of 'summarize', 'plan', 'explain', 'default'.
            temperature: Explicit override, clamped to [0.0, 1.0].
            system_prompt: Override for system prompt. Defaults to CoT.

        Returns:
            Tuple of (provider_name, model_name, response_text).
        """
        temp = self._resolve_temperature(task_type, temperature)
        sys_prompt = system_prompt or _COT_SYSTEM_PROMPT

        if self.default_provider == "lmstudio":
            output = await self._chat_lmstudio(prompt, sys_prompt, temp)
            return "lmstudio", self.lmstudio_model, output

        output = await self._chat_ollama(prompt, sys_prompt, temp)
        return "ollama", self.ollama_model, output

    async def self_consistency_chat(
        self,
        prompt: str,
        n: int = 3,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """Sample the model N times and return the most common response.

        Uses majority vote for short outputs; falls back to the
        median-length response when no duplicates are found.

        Source: Wang et al. (2023), arXiv:2203.11171.
        Recommended temperature: 0.7-0.9 for diverse sampling paths.
        """
        n = max(1, min(n, 7))
        sys_prompt = system_prompt or _COT_SYSTEM_PROMPT

        responses: List[str] = []
        provider_name = "unknown"
        model_name = "unknown"

        for _ in range(n):
            try:
                if self.default_provider == "lmstudio":
                    out = await self._chat_lmstudio(
                        prompt, sys_prompt, temperature
                    )
                    provider_name = "lmstudio"
                    model_name = self.lmstudio_model
                else:
                    out = await self._chat_ollama(
                        prompt, sys_prompt, temperature
                    )
                    provider_name = "ollama"
                    model_name = self.ollama_model

                if out:
                    responses.append(out)
            except Exception as exc:
                logger.debug("self-consistency sample %d failed: %s", len(responses), exc)
                continue

        if not responses:
            return provider_name, model_name, ""

        counts = Counter(responses)
        most_common, freq = counts.most_common(1)[0]
        if freq > 1:
            return provider_name, model_name, most_common

        # No duplicates — return median-length response.
        sorted_by_len = sorted(responses, key=len)
        median = sorted_by_len[len(sorted_by_len) // 2]
        return provider_name, model_name, median

    def _resolve_temperature(
        self,
        task_type: str,
        explicit: Optional[float],
    ) -> float:
        if explicit is not None:
            return max(0.0, min(1.0, explicit))
        return TEMPERATURE_BY_TASK.get(
            task_type, TEMPERATURE_BY_TASK["default"]
        )

    async def _chat_ollama(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
    ) -> str:
        payload: Dict[str, object] = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.ollama_url}/api/chat", json=payload
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(f"ollama returned HTTP {exc.response.status_code}") from exc
            except httpx.ConnectError:
                raise RuntimeError(f"ollama unreachable at {self.ollama_url}") from None
            data = response.json()
            return str(
                data.get("message", {}).get("content", "")
            ).strip()

    async def _chat_lmstudio(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
    ) -> str:
        payload: Dict[str, object] = {
            "model": self.lmstudio_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            # LM Studio's OpenAI-compatible endpoint returns a chunked stream by
            # default on some builds; force non-streaming JSON so json() works.
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.lmstudio_url}/v1/chat/completions", json=payload
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(f"lmstudio returned HTTP {exc.response.status_code}") from exc
            except httpx.ConnectError:
                raise RuntimeError(f"lmstudio unreachable at {self.lmstudio_url}") from None
            data = response.json()
            choices: List[Dict[str, object]] = data.get("choices", [])
            if not choices:
                return ""
            message = choices[0].get("message", {})
            return str(message.get("content", "")).strip()
