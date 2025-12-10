"""
Factory that instantiates vector stores based on configuration.
"""

from __future__ import annotations

from typing import Dict

from src.ml.config.schema import VectorStoreSettings
from src.ml.vectorstores.base import VectorStore
from src.ml.vectorstores.memory import MemoryVectorStore
from src.ml.vectorstores.milvus import MilvusVectorStore
from src.ml.vectorstores.qdrant import QdrantVectorStore


class VectorStoreFactory:
    """Create vector store instances per namespace."""

    def __init__(self, settings: VectorStoreSettings) -> None:
        """Store the factory settings."""
        self.settings = settings

    def create(self, namespace: str) -> VectorStore:
        """
        Instantiate the configured backend for a namespace.
        """
        collection = self.settings.namespaces.get(
            namespace, f"{self.settings.collection_prefix}_{namespace}"
        )
        backend = self.settings.backend.lower()
        if backend == "memory":
            return MemoryVectorStore(collection=collection, dim=self.settings.dim)
        if backend == "milvus":
            return MilvusVectorStore(
                collection=collection,
                dim=self.settings.dim,
                connection=self.settings.connection,
            )
        if backend == "qdrant":
            return QdrantVectorStore(
                collection=collection,
                dim=self.settings.dim,
                connection=self.settings.connection,
            )
        raise ValueError(f"Unsupported vector store backend '{self.settings.backend}'.")
