"""
Milvus-backed vector store.
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, List

import numpy as np

from src.t_nexus.ml.vectorstores.base import VectorRecord, VectorSearchResult, VectorStore, VectorStoreError

try:
    from pymilvus import (
        CollectionSchema,
        DataType,
        FieldSchema,
        MilvusClient,
    )
except ImportError:  # pragma: no cover - optional dependency
    MilvusClient = None  # type: ignore[assignment]


class MilvusVectorStore(VectorStore):
    """
    Vector store implementation backed by Milvus.
    """

    def __init__(self, collection: str, dim: int, connection: Dict[str, str]) -> None:
        """Create the Milvus client and ensure the target collection exists."""
        if MilvusClient is None:  # pragma: no cover - import guard
            raise VectorStoreError(
                "pymilvus is not installed. Install 'pymilvus' to use the Milvus backend."
            )

        super().__init__(collection, dim)
        uri = connection.get("uri", "http://localhost:19530")
        token = connection.get("token")
        self.client = MilvusClient(uri=uri, token=token)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection if missing."""
        if self.client.has_collection(self.collection):
            return

        schema = CollectionSchema(
            fields=[
                FieldSchema(
                    name="pk",
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    auto_id=False,
                    max_length=64,
                ),
                FieldSchema(
                    name="vector",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.dim,
                ),
                FieldSchema(
                    name="text",
                    dtype=DataType.VARCHAR,
                    max_length=8192,
                ),
                FieldSchema(
                    name="metadata",
                    dtype=DataType.VARCHAR,
                    max_length=2048,
                ),
            ]
        )
        self.client.create_collection(
            collection_name=self.collection,
            schema=schema,
            consistency_level="Strong",
        )

    def upsert(self, records: Iterable[VectorRecord]) -> List[str]:
        """Insert or update Milvus rows."""
        payload = []
        for record in records:
            payload.append(
                {
                    "pk": record.record_id,
                    "vector": record.vector.tolist(),
                    "text": record.text,
                    "metadata": json.dumps(record.metadata),
                }
            )
        if payload:
            self.client.upsert(collection_name=self.collection, data=payload)
        return [row["pk"] for row in payload]

    def delete(self, record_ids: Iterable[str]) -> None:
        """Delete the specified Milvus rows."""
        ids = list(record_ids)
        if not ids:
            return
        self.client.delete(collection_name=self.collection, ids=ids)

    def fetch(self, record_ids: Iterable[str]) -> List[VectorRecord]:
        """Fetch vectors by id."""
        ids = list(record_ids)
        if not ids:
            return []
        filter_expr = f'pk in {json.dumps(ids)}'
        rows = self.client.query(
            collection_name=self.collection,
            filter=filter_expr,
            output_fields=["pk", "text", "metadata", "vector"],
        )
        records: List[VectorRecord] = []
        for row in rows:
            metadata = json.loads(row.get("metadata") or "{}")
            vector = np.array(row["vector"], dtype=np.float32)
            records.append(
                VectorRecord(
                    record_id=row["pk"],
                    vector=vector,
                    text=row.get("text", ""),
                    metadata=metadata,
                )
            )
        return records

    def query(self, vector: np.ndarray, top_k: int) -> List[VectorSearchResult]:
        """Return the ANN results for *vector*."""
        response = self.client.search(
            collection_name=self.collection,
            data=[vector.tolist()],
            anns_field="vector",
            output_fields=["pk", "text", "metadata"],
            limit=top_k,
        )
        matches = response[0] if response else []
        results: List[VectorSearchResult] = []
        for match in matches:
            metadata = json.loads(match.entity.get("metadata") or "{}")
            results.append(
                VectorSearchResult(
                    record_id=match.id,
                    score=float(match.score),
                    text=match.entity.get("text", ""),
                    metadata=metadata,
                )
            )
        return results
