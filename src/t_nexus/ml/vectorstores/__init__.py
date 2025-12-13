"""
Vector database abstractions and factory helpers.
"""

from src.t_nexus.ml.vectorstores.base import VectorRecord, VectorSearchResult, VectorStore, VectorStoreError
from src.t_nexus.ml.vectorstores.factory import VectorStoreFactory
from src.t_nexus.ml.vectorstores.memory import MemoryVectorStore

__all__ = [
    "VectorRecord",
    "VectorSearchResult",
    "VectorStore",
    "VectorStoreError",
    "VectorStoreFactory",
    "MemoryVectorStore",
]
