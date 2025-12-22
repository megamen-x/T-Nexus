"""
Typed configuration objects for HippoRAG.

All knobs converge into :class:`HippoRAGSettings` so downstream modules do not
have to touch YAML or dictionaries directly.
"""

from __future__ import annotations

from typing import Dict, Optional
from pydantic import BaseModel, Field


class LLMSettings(BaseModel):
    """Extended LLM settings with all controls."""
    
    api_key: str
    base_url: str | None = None
    model: str = "gpt-4"
    temperature: float = 0.0
    max_output_tokens: int = 4096
    response_format: str = "json_object"
    
    requests_per_minute: int = Field(default=60, ge=0)
    tokens_per_minute: int = Field(default=0, ge=0)
    max_concurrent: int = Field(default=10, ge=1, le=100)
    
    request_timeout: float = Field(default=120.0, ge=1.0)
    connect_timeout: float = Field(default=10.0, ge=1.0)
    
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_min_wait: float = Field(default=1.0, ge=0.1)
    retry_max_wait: float = Field(default=30.0, ge=1.0)


class EmbeddingSettings(BaseModel):
    """Embedding backend configuration."""
    
    model: str = "text-embedding-3-small"
    dim: int = Field(default=1536, ge=1)
    base_url: str | None = None
    api_key: str | None = None
    batch_size: int = Field(default=16, ge=1)
    normalize_embeddings: bool = True

    requests_per_minute: int = Field(default=0, ge=0)
    max_concurrent: int = Field(default=20, ge=1)
    request_timeout: float = Field(default=60.0, ge=1.0)

    max_retries: int = Field(default=3, ge=0)
    retry_min_wait: float = Field(default=1.0, ge=0.1)
    retry_max_wait: float = Field(default=30.0, ge=1.0)

    model_config = {"extra": "forbid"}


class VectorStoreSettings(BaseModel):
    """Vector database configuration."""
    
    backend: str = "memory"
    collection_prefix: str = "hipporag"
    url: str = "url"
    dim: int = Field(default=1536, ge=1)
    recreate: bool = False
    namespaces: Dict[str, str] = Field(
        default_factory=lambda: {
            "passages": "hipporag_passages",
            "entities": "hipporag_entities",
            "facts": "hipporag_facts",
        }
    )
    connection: Dict[str, str] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class Neo4jSettings(BaseModel):
    """Neo4j connection parameters and graph projection metadata."""

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "neo4j"
    database: Optional[str] = None
    graph_projection: str = "hipporag_graph"
    weight_property: str = "weight"
    create_constraints: bool = True


class RabbitMQSettings(BaseModel):
    """RabbitMQ publication settings."""

    enabled: bool = False
    url: str = "amqp://guest:guest@localhost:5672/%2F"
    exchange: str = "hipporag"
    routing_key_indexed: str = "hipporag.documents.indexed"
    routing_key_deleted: str = "hipporag.documents.deleted"
    queue: str = "hipporag"


class ConversationSettings(BaseModel):
    """Controls how chat histories are condensed into a retrieval query."""

    mode: str = "last_message"
    max_messages: int = 20


class RetrievalSettings(BaseModel):
    """Retrieval-loop knobs."""

    top_k: int = 5
    fact_top_k: int = 5
    damping: float = 0.5
    passage_weight: float = 0.05
    synonym_top_k: int = 50
    synonym_threshold: float = 0.8
    save_openie: bool = True
    engine: str = "neo4j"


class HippoRAGSettings(BaseModel):
    """Aggregated settings tree consumed by :class:`HippoRAG`."""

    llm: LLMSettings
    
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    rabbitmq: RabbitMQSettings = Field(default_factory=RabbitMQSettings)
    conversation: ConversationSettings = Field(default_factory=ConversationSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    
    dataset: str = "musique"
    save_dir: str = "hipporag_testing_dir"