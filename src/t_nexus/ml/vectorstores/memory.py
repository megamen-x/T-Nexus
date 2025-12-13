"""
In-memory vector store useful for tests and local development.
"""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, List

import numpy as np

from src.t_nexus.ml.vectorstores.base import VectorRecord, VectorSearchResult, VectorStore


class MemoryVectorStore(VectorStore):
    """
    Simple NumPy-based store that mirrors the real interface.
    """

    def __init__(self, collection: str, dim: int) -> None:
        """Initialize the ephemeral store."""
        super().__init__(collection, dim)
        self._records: Dict[str, VectorRecord] = {}

    def upsert(self, records: Iterable[VectorRecord]) -> List[str]:
        """Insert or update *records*."""
        ids = []
        for record in records:
            if record.vector.shape[-1] != self.dim:
                raise ValueError(
                    f"Vector dim mismatch for {record.record_id}: "
                    f"{record.vector.shape[-1]} != {self.dim}"
                )
            self._records[record.record_id] = record
            ids.append(record.record_id)
        return ids

    def delete(self, record_ids: Iterable[str]) -> None:
        """Remove records from the store."""
        for record_id in record_ids:
            self._records.pop(record_id, None)

    def fetch(self, record_ids: Iterable[str]) -> List[VectorRecord]:
        """Fetch specific records."""
        return [self._records[rid] for rid in record_ids if rid in self._records]

    def query(self, vector: np.ndarray, top_k: int) -> List[VectorSearchResult]:
        """Return the top-k approximate nearest neighbors."""
        if vector.shape[-1] != self.dim:
            raise ValueError("Query vector dimension mismatch.")
        query_norm = np.linalg.norm(vector)
        scores: List[VectorSearchResult] = []
        for record in self._records.values():
            denom = np.linalg.norm(record.vector) * query_norm or 1.0
            score = float(np.dot(record.vector, vector) / denom)
            scores.append(
                VectorSearchResult(
                    record_id=record.record_id,
                    score=score,
                    text=record.text,
                    metadata=record.metadata,
                )
            )
        scores.sort(key=lambda item: item.score, reverse=True)
        return scores[:top_k]

    def iter_records(self, batch_size: int = 512) -> Iterator[VectorRecord]:
        """Yield all stored records in batches."""
        items = list(self._records.values())
        for start in range(0, len(items), batch_size):
            for record in items[start : start + batch_size]:
                yield record
