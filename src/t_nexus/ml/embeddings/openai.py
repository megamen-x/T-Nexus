"""
OpenAI-compatible embedding backend with async support and rate limiting.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List

import numpy as np
from openai import AsyncOpenAI, RateLimitError, APIError, APIConnectionError, APITimeoutError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    after_log,
)

from src.t_nexus.ml.config.schema import EmbeddingSettings
from src.t_nexus.ml.embeddings.base import BaseEmbeddingModel
from src.t_nexus.ml.utils import AsyncRateLimiter

logger = logging.getLogger(__name__)


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """
    Embedding wrapper that talks to any OpenAI-compatible service.
    Refactored to support async concurrency, retries, and rate limiting.
    """

    RETRYABLE_ERRORS = (
        RateLimitError,
        APIError,
        APIConnectionError,
        APITimeoutError,
    )

    def __init__(self, settings: EmbeddingSettings) -> None:
        """Initialize the AsyncOpenAI client."""
        super().__init__(settings)
        
        self.client = AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            max_retries=0,
        )

        self._rate_limiter = AsyncRateLimiter(
            rpm=getattr(settings, "requests_per_minute", 0)
        )
        
        self._max_concurrent = getattr(settings, "max_concurrent", 20)
        self._semaphore: asyncio.Semaphore | None = None
        
        self._max_retries = 3
        self._retry_min_wait = 1.0
        self._retry_max_wait = 30.0

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """Lazy initialization of semaphore."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
        return self._semaphore

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings if the config requests it.
        """
        if not self.settings.normalize_embeddings:
            return embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        return embeddings / norms

    async def embed(self, texts: List[str]) -> np.ndarray:
        """
        Encode *texts* concurrently with batching and rate limiting.
        """
        
        batch_size = self.settings.batch_size
        batches = [
            texts[i : i + batch_size] 
            for i in range(0, len(texts), batch_size)
        ]

        tasks = [self._process_batch(batch) for batch in batches]
        results_list = await asyncio.gather(*tasks)
        if not results_list:
             return np.empty((0, self.settings.dim), dtype=np.float32)
             
        embeddings = np.concatenate(results_list, axis=0)
        return self._normalize(embeddings)

    async def _process_batch(self, batch: List[str]) -> np.ndarray:
        """Process a single batch with semaphore and rate limiter."""
        async with self.semaphore:
            await self._rate_limiter.acquire()
            return await self._call_with_retry(batch)

    async def _call_with_retry(self, batch: List[str]) -> np.ndarray:
        """Call API with exponential backoff."""
        
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
            response = await self.client.embeddings.create(
                model=self.settings.model,
                input=batch,
                encoding_format="float"
            )
            
            if response.usage:
                logger.debug(f"Embedding batch usage: {response.usage.total_tokens} tokens")

            data = [row.embedding for row in response.data]
            return np.array(data, dtype=np.float32)
        
        return await _do_call()

    async def close(self) -> None:
        """Close the client connection."""
        await self.client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()