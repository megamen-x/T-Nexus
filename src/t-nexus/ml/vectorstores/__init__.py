"""
Vector database abstractions and factory helpers.
"""

from src.ml.vectorstores.base import VectorRecord, VectorSearchResult, VectorStore, VectorStoreError
from src.ml.vectorstores.factory import VectorStoreFactory
from src.ml.vectorstores.memory import MemoryVectorStore

__all__ = [
    "VectorRecord",
    "VectorSearchResult",
    "VectorStore",
    "VectorStoreError",
    "VectorStoreFactory",
    "MemoryVectorStore",
]
