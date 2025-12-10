"""
Embedding backends.
"""

from src.ml.embeddings.base import BaseEmbeddingModel
from src.ml.embeddings.openai import OpenAIEmbeddingModel

__all__ = ["BaseEmbeddingModel", "OpenAIEmbeddingModel"]
