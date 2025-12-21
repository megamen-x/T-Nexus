"""
Typed configuration objects for HippoRAG.

All knobs converge into :class:`HippoRAGSettings` so downstream modules do not
have to touch YAML or dictionaries directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class LLMSettings:
    """LLM invocation parameters."""

    model: str = "gpt-5-mini"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_output_tokens: int = 512
    prompt_name: str = "rag_qa_musique"
    response_format: str = "json_object"
    requests_per_minute: int | None = None


@dataclass
class EmbeddingSettings:
    """Embedding backend configuration."""

    model: str = "text-embedding-3-small"
    dim: int = 1536
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    batch_size: int = 16
    normalize_embeddings: bool = True


@dataclass
class VectorStoreSettings:
    """Descriptor for the vector database factory."""

    backend: str = "memory"
    collection_prefix: str = "hipporag"
    url: str = "url"
    dim: int = 1536
    recreate: bool = False
    namespaces: Dict[str, str] = field(
        default_factory=lambda: {
            "passages": "hipporag_passages",
            "entities": "hipporag_entities",
            "facts": "hipporag_facts",
        }
    )
    connection: Dict[str, str] = field(default_factory=dict)


@dataclass
class Neo4jSettings:
    """Neo4j connection parameters and graph projection metadata."""

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "neo4j"
    database: Optional[str] = None
    graph_projection: str = "hipporag_graph"
    weight_property: str = "weight"
    create_constraints: bool = True


@dataclass
class RabbitMQSettings:
    """RabbitMQ publication settings."""

    enabled: bool = False
    url: str = "amqp://guest:guest@localhost:5672/%2F"
    exchange: str = "hipporag"
    routing_key_indexed: str = "hipporag.documents.indexed"
    routing_key_deleted: str = "hipporag.documents.deleted"
    queue: str = "hipporag"


@dataclass
class ConversationSettings:
    """Controls how chat histories are condensed into a retrieval query."""

    mode: str = "last_message"
    max_messages: int = 20


@dataclass
class RetrievalSettings:
    """Retrieval-loop knobs."""

    top_k: int = 5
    fact_top_k: int = 5
    damping: float = 0.5
    passage_weight: float = 0.05
    synonym_top_k: int = 50
    synonym_threshold: float = 0.8
    save_openie: bool = True
    engine: str = "neo4j"


@dataclass
class HippoRAGSettings:
    """Aggregated settings tree consumed by :class:`HippoRAG`."""

    llm: LLMSettings = field(default_factory=LLMSettings)
    embeddings: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    vector_store: VectorStoreSettings = field(default_factory=VectorStoreSettings)
    neo4j: Neo4jSettings = field(default_factory=Neo4jSettings)
    rabbitmq: RabbitMQSettings = field(default_factory=RabbitMQSettings)
    conversation: ConversationSettings = field(default_factory=ConversationSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    prompts_path: Optional[str] = None
    openie_mode: str = "online"
    dataset: str = "musique"
    save_dir: str = "hipporag_testing_dir"

    @classmethod
    def from_dict(cls, raw: Dict) -> "HippoRAGSettings":
        """
        Build a :class:`HippoRAGSettings` instance from a plain dictionary.

        :param raw: Parsed YAML data.
        """
        def _build(sub_cls, key: str):
            return sub_cls(**raw.get(key, {}))

        return cls(
            llm=_build(LLMSettings, "llm"),
            embeddings=_build(EmbeddingSettings, "embeddings"),
            vector_store=_build(VectorStoreSettings, "vector_store"),
            neo4j=_build(Neo4jSettings, "neo4j"),
            rabbitmq=_build(RabbitMQSettings, "rabbitmq"),
            conversation=_build(ConversationSettings, "conversation"),
            retrieval=_build(RetrievalSettings, "retrieval"),
            prompts_path=raw.get("prompts_path"),
            openie_mode=raw.get("openie_mode", "online"),
            dataset=raw.get("dataset", "musique"),
            save_dir=raw.get("save_dir", ".hipporag"),
        )
