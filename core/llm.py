"""
llm.py
------
Async LLM wrapper supporting OpenAI or Anthropic via LLM_PROVIDER env var.

Defaults:
  LLM_PROVIDER=openai  (matches your Sprint 1 BWIM agent conventions)
  OPENAI_MODEL=o4-mini
  ANTHROPIC_MODEL=claude-sonnet-4-20250514
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_REASONING_MODEL_TAGS = ("o1", "o3", "o4")


class LLM:
    def __init__(self):
        self._provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        self._max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))
        self._temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))

        if self._provider == "anthropic":
            from anthropic import AsyncAnthropic
            api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set.")
            self._anthropic = AsyncAnthropic(api_key=api_key)
            self._model = os.getenv(
                "ANTHROPIC_MODEL", "claude-sonnet-4-20250514"
            ).strip()
            logger.info("LLM provider=anthropic model=%s", self._model)
        else:
            from openai import AsyncOpenAI
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set.")
            base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
            self._openai = AsyncOpenAI(api_key=api_key, base_url=base_url)
            self._model = os.getenv("OPENAI_MODEL", "o4-mini").strip()
            logger.info("LLM provider=openai model=%s", self._model)

    def _is_openai_reasoning(self) -> bool:
        return any(tag in self._model for tag in _REASONING_MODEL_TAGS)

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        if self._provider == "anthropic":
            resp = await self._anthropic.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            parts = []
            for block in resp.content:
                if getattr(block, "type", None) == "text":
                    parts.append(block.text)
            return "".join(parts).strip()

        # OpenAI
        params: dict = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self._is_openai_reasoning():
            params["max_completion_tokens"] = self._max_tokens
        else:
            params["max_tokens"] = self._max_tokens
            params["temperature"] = self._temperature
        resp = await self._openai.chat.completions.create(**params)
        return (resp.choices[0].message.content or "").strip()
