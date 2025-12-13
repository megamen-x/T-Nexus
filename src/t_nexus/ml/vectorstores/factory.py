"""
Factory that instantiates vector stores based on configuration.
"""

from __future__ import annotations

from src.t_nexus.ml.config.schema import VectorStoreSettings
from src.t_nexus.ml.vectorstores.base import VectorStore
from src.t_nexus.ml.vectorstores.memory import MemoryVectorStore
from src.t_nexus.ml.vectorstores.milvus import MilvusVectorStore
from src.t_nexus.ml.vectorstores.qdrant import QdrantVectorStore


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
