"""
OpenAI-compatible chat client.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from openai import OpenAI, RateLimitError, OpenAIError
from openai.types.chat import ChatCompletion
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.ml.config.schema import LLMSettings
from src.ml.llm.base import BaseLLM


class OpenAILLM(BaseLLM):
    """Thin wrapper over :class:`openai.OpenAI` chat completions."""

    def __init__(self, settings: LLMSettings) -> None:
        """Initialize the OpenAI chat client."""
        super().__init__(settings)
        self.client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)

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
