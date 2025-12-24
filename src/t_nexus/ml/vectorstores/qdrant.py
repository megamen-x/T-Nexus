"""
Qdrant-backed vector store.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from src.t_nexus.ml.vectorstores.base import VectorRecord, VectorSearchResult, VectorStore, VectorStoreError

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest
except ImportError:  # pragma: no cover - optional dependency
    QdrantClient = None  # type: ignore[assignment]
    rest = None  # type: ignore[assignment]


class QdrantVectorStore(VectorStore):
    """Vector store implementation layered on top of Qdrant."""

    def __init__(self, collection: str, dim: int, connection: Dict[str, str]) -> None:
        """Create the Qdrant client and bootstrap the collection."""
        if QdrantClient is None:  # pragma: no cover - import guard
            raise VectorStoreError(
                "qdrant-client is not installed. Install it to use the Qdrant backend."
            )
        super().__init__(collection, dim)
        url = connection.get("url", "http://qdrant:6333")
        api_key = connection.get("api_key")
        prefer_grpc = connection.get("prefer_grpc", "false").lower() == "true"
        self.client = QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection when missing."""
        if self.client.collection_exists(self.collection):
            return
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=rest.VectorParams(size=self.dim, distance=rest.Distance.COSINE),
        )

    def upsert(self, records: Iterable[VectorRecord]) -> List[str]:
        """Insert or update vectors."""
        points = []
        for record in records:
            points.append(
                rest.PointStruct(
                    id=record.record_id,
                    vector=record.vector.tolist(),
                    payload={"text": record.text, **record.metadata},
                )
            )
        if points:
            self.client.upsert(collection_name=self.collection, points=points)
        return [point.id for point in points]

    def delete(self, record_ids: Iterable[str]) -> None:
        """Remove points from the collection."""
        ids = list(record_ids)
        if not ids:
            return
        self.client.delete(
            collection_name=self.collection,
            points_selector=rest.PointIdsList(points=ids),
        )

    def fetch(self, record_ids: Iterable[str]) -> List[VectorRecord]:
        """Retrieve specific points along with their payload."""
        ids = list(record_ids)
        if not ids:
            return []
        rows = self.client.retrieve(
            collection_name=self.collection,
            ids=ids,
            with_vectors=True,
            with_payload=True,
        )
        records: List[VectorRecord] = []
        for row in rows:
            vector = np.array(row.vector, dtype=np.float32)
            metadata = dict(row.payload or {})
            text = metadata.pop("text", "")
            records.append(
                VectorRecord(
                    record_id=str(row.id),
                    vector=vector,
                    text=text,
                    metadata=metadata,
                )
            )
        return records

    def query(self, vector: np.ndarray, top_k: int) -> List[VectorSearchResult]:
        """Search for approximate nearest neighbors."""
        if hasattr(self.client, "query_points"):
            res = self.client.query_points(
                collection_name=self.collection,
                query=vector.tolist(),
                limit=top_k,
                with_payload=True,
            )
            hits = res.points
        else:
            # fallback for old qdrant-client
            hits = self.client.search(
                collection_name=self.collection,
                query_vector=vector.tolist(),
                limit=top_k,
                with_payload=True,
            )

        results: List[VectorSearchResult] = []
        for hit in hits:
            payload = dict(hit.payload or {})
            text = payload.pop("text", "")
            results.append(
                VectorSearchResult(
                    record_id=str(hit.id),
                    score=float(hit.score),
                    text=text,
                    metadata=payload,
                )
            )
        return results
