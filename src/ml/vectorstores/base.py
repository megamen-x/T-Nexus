"""
Core vector store interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List

import numpy as np


@dataclass
class VectorRecord:
    """Encapsulates a single vector entry."""

    record_id: str
    vector: np.ndarray
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class VectorSearchResult:
    """Result returned by :meth:`VectorStore.query`."""

    record_id: str
    score: float
    text: str
    metadata: Dict[str, str]


class VectorStoreError(RuntimeError):
    """Generic vector store exception."""


class VectorStore(ABC):
    """Abstract contract implemented by each backend."""

    def __init__(self, collection: str, dim: int) -> None:
        self.collection = collection
        self.dim = dim

    @abstractmethod
    def upsert(self, records: Iterable[VectorRecord]) -> List[str]:
        """Insert or update *records* and return their ids."""

    @abstractmethod
    def delete(self, record_ids: Iterable[str]) -> None:
        """Delete the specified records."""

    @abstractmethod
    def fetch(self, record_ids: Iterable[str]) -> List[VectorRecord]:
        """Fetch vector records by id."""

    @abstractmethod
    def query(self, vector: np.ndarray, top_k: int) -> List[VectorSearchResult]:
        """Query top-k most similar vectors."""

    def iter_records(self, batch_size: int = 512) -> Iterator[VectorRecord]:
        """
        Iterate through records for admin tasks. Defaults to fetching everything.
        """
        raise NotImplementedError("iter_records is not implemented for this backend.")
