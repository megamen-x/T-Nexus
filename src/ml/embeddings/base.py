"""
Abstract embedding interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List

import numpy as np

from src.ml.config.schema import EmbeddingSettings


class BaseEmbeddingModel(ABC):
    """Base contract for embedding backends."""

    def __init__(self, settings: EmbeddingSettings) -> None:
        """Persist settings for downstream use."""
        self.settings = settings

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Encode the provided texts into embedding vectors.

        :param texts: List of texts.
        :return: 2-D array ``(len(texts), dim)``.
        """

    def embed_one(self, text: str) -> np.ndarray:
        """
        Convenience helper to encode a single string.
        """
        return self.embed([text])[0]
