"""
OpenAI-compatible chat client with proper rate limiting and concurrency control.
"""

from __future__ import annotations

import asyncio
import json
import logging

from typing import Dict, List, Tuple

from openai import AsyncOpenAI, RateLimitError, APIError, APIConnectionError, APITimeoutError
from openai.types.chat import ChatCompletion
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    after_log,
)

from src.t_nexus.ml.config.schema import LLMSettings
from src.t_nexus.ml.llm.base import BaseLLM, JSONParseError
from src.t_nexus.ml.utils import AsyncRateLimiter

logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    """
    Async OpenAI client with proper rate limiting and concurrency control.
    """

    RETRYABLE_ERRORS = (
        RateLimitError,
        APIError,
        APIConnectionError,
        APITimeoutError,
        JSONParseError,
    )

    def __init__(self, settings: LLMSettings) -> None:
        super().__init__(settings)
        
        self.client = AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.request_timeout or 60.0,
            max_retries=0,
        )
        
        self._rate_limiter = AsyncRateLimiter(
            rpm=settings.requests_per_minute or 0
        )
        
        self._max_concurrent = settings.max_concurrent or 10
        self._semaphore: asyncio.Semaphore | None = None
        
        self._max_retries = settings.max_retries or 3
        self._retry_min_wait = settings.retry_min_wait or 1
        self._retry_max_wait = settings.retry_max_wait or 30

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """Lazy initialization of semaphore."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
        return self._semaphore

    async def generate(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Tuple[str, Dict]:
        """
        Invoke the chat completion endpoint with rate limiting and concurrency control.
        """
        async with self.semaphore:
            await self._rate_limiter.acquire()
            return await self._call_with_retry(messages, **kwargs)

    async def _call_with_retry(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Tuple[str, Dict]:
        """Call with configurable retry logic."""
        
        @retry(
            retry=retry_if_exception_type(self.RETRYABLE_ERRORS),
            wait=wait_exponential(
                multiplier=1,
                min=self._retry_min_wait,
                max=self._retry_max_wait,
            ),
            stop=stop_after_attempt(self._max_retries),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            after=after_log(logger, logging.DEBUG),
            reraise=True,
        )
        async def _do_call():
            return await self._call(messages, **kwargs)
        
        return await _do_call()

    async def _call(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Tuple[str, Dict]:
        """Execute single API call."""
        response: ChatCompletion = await self.client.chat.completions.create(
            model=self.settings.model,
            messages=messages,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_output_tokens,
            response_format={"type": self.settings.response_format},
            **kwargs,
        )

        message = response.choices[0].message.content or ""
        payload = self._parse_payload(message)
        
        metadata = {
            "finish_reason": response.choices[0].finish_reason,
            "usage": response.usage.model_dump() if response.usage else {},
            "json_payload": payload,
            "model": response.model,
        }
        
        if response.usage:
            logger.debug(
                f"API call: {response.usage.prompt_tokens} prompt + "
                f"{response.usage.completion_tokens} completion tokens"
            )
        
        return message, metadata

    def _parse_payload(self, content: str) -> dict | None:
        """Parse JSON response if required."""
        if self.settings.response_format != "json_object":
            return None
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            logger.warning(f"Invalid JSON response: {content}...")
            raise JSONParseError("LLM returned invalid JSON payload") from exc

    async def close(self) -> None:
        """Close the client connection."""
        await self.client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    @property
    def stats(self) -> Dict:
        """Current client statistics."""
        return {
            "rate_limit_usage": f"{self._rate_limiter.current_usage}/{self._rate_limiter.rpm}",
            "concurrent_slots": f"{self._max_concurrent - self.semaphore._value}/{self._max_concurrent}",
        }