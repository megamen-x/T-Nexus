"""
OpenAI-compatible embedding backend.
"""

from __future__ import annotations

from typing import List

import numpy as np
from openai import OpenAI

from src.ml.config.schema import EmbeddingSettings
from src.ml.embeddings.base import BaseEmbeddingModel


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """
    Embedding wrapper that talks to any OpenAI-compatible service.
    """

    def __init__(self, settings: EmbeddingSettings) -> None:
        """Initialize the OpenAI client."""
        super().__init__(settings)
        self.client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
        )

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings if the config requests it.
        """
        if not self.settings.normalize_embeddings:
            return embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        return embeddings / norms

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Encode *texts* and return an ndarray.
        """
        cleaned = [text.replace("\n", " ").strip() or " " for text in texts]
        embeddings = []
        batch_size = self.settings.batch_size
        for start in range(0, len(cleaned), batch_size):
            batch = cleaned[start : start + batch_size]
            response = self.client.embeddings.create(
                model=self.settings.model,
                input=batch,
            )
            embeddings.extend([row.embedding for row in response.data])
        if not embeddings:
            return np.empty((0, self.settings.dim), dtype=np.float32)
        return self._normalize(np.array(embeddings, dtype=np.float32))
