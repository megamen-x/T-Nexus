"""
OpenAI-compatible chat client.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from collections import deque
from threading import Lock
from time import monotonic, sleep

from openai import OpenAI, RateLimitError, OpenAIError
from openai.types.chat import ChatCompletion
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.t_nexus.ml.config.schema import LLMSettings
from src.t_nexus.ml.llm.base import BaseLLM


class OpenAILLM(BaseLLM):
    """Thin wrapper over :class:`openai.OpenAI` chat completions."""

    def __init__(self, settings: LLMSettings) -> None:
        """Initialize the OpenAI chat client."""
        super().__init__(settings)
        self.client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
        self._rpm = settings.requests_per_minute or 0
        self._rate_window = 60.0
        self._recent = deque()
        self._rate_lock = Lock()

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Tuple[str, Dict]:
        """
        Invoke the chat completion endpoint.
        """
        return self._call(messages=messages, **kwargs)

    @retry(
        retry=retry_if_exception_type((RateLimitError, OpenAIError)),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
    )
    def _call(self, messages: List[Dict[str, str]], **kwargs) -> Tuple[str, Dict]:
        self._wait_for_slot()
        response: ChatCompletion = self.client.chat.completions.create(
            model=self.settings.model,
            messages=messages,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_output_tokens,
            response_format={"type": self.settings.response_format},
            **kwargs,
        )
        message = response.choices[0].message.content or ""
        metadata = {
            "finish_reason": response.choices[0].finish_reason,
            "usage": response.usage.model_dump() if hasattr(response.usage, "model_dump") else {},
        }
        return message, metadata

    def _wait_for_slot(self) -> None:
        """Block until the next request slot is available."""
        if self._rpm <= 0:
            return
        with self._rate_lock:
            now = monotonic()
            while self._recent and now - self._recent[0] >= self._rate_window:
                self._recent.popleft()
            if len(self._recent) >= self._rpm:
                wait = self._rate_window - (now - self._recent[0])
                if wait > 0:
                    sleep(wait)
                now = monotonic()
                while self._recent and now - self._recent[0] >= self._rate_window:
                    self._recent.popleft()
            self._recent.append(now)
