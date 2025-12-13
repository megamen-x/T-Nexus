# HippoRAG 2 Migration Plan

This document captures the redesigned HippoRAG stack that will live inside `src/ml`.
It highlights the modules we have to build, how they interact, and how each new
requirement from Minerva’s shortcomings is addressed.

## Top-Level Modules

| Module | Responsibility |
| --- | --- |
| `config` | Typed YAML configuration parser exposed through `HippoRAGSettings`. |
| `embeddings` | Backend-agnostic embedding client(s) (OpenAI-compatible by default). |
| `extraction` | Interfaces for chunking text and extracting triples/entities via LLM prompts. |
| `graph` | Neo4j connector plus helpers to insert/read phrase/passage/fact nodes. |
| `vectorstores` | Factory-driven abstraction for Milvus, Qdrant, or an in-memory fallback. |
| `hipporag` | Orchestration layer (indexing, retrieval, QA, chat-history policy). |
| `messaging` | RabbitMQ publisher/consumer helpers for indexing and deletion events. |
| `utils` | Shared helpers (hash ids, normalization, dataclasses for triples, etc.). |

## Configuration Surface

All runtime knobs are centralized in `config/hipporag.yaml`. The loader maps YAML into
`HippoRAGSettings`, which is later injected into every subsystem. Key sections:

- `llm`: base URL, API key, temperature, and default QA prompt names.
- `embeddings`: model family, batching, optional adapters.
- `vector_store`: factory descriptor (`backend`, `collection`, credentials, approximate search options).
- `neo4j`: URI, auth, and toggles for constraint/ index bootstrapping.
- `conversation`: `mode` can be `last_message` or `full_history`.
- `rabbitmq`: optional broker URI + queues.
- `retrieval`: numerical knobs (top_k, damping factor, recognition-memory prompt ids).

## Graph Layer (Neo4j)

`graph/neo4j_store.py` exposes:

- `Neo4jGraphStore`: context-managed driver wrapper with retry + health checks.
- `GraphIndexer`: batch upserts for phrase, passage, and fact nodes plus the `CONTAINS`,
  `RELATES_TO`, `SYNONYM_OF`, and `DERIVED_FROM` relationships.
- `GraphRetriever`: utilities to compute Personalized PageRank using Neo4j’s built-in
  GDS `gds.pageRank.stream` procedure and to materialize seed weights supplied by
  HippoRAG.

All graph operations are executed inside Neo4j transactions, removing the prior
dependency on `igraph` and supporting distributed deployments.

## Vector Store Factory

`vectorstores/base.py` defines the `VectorStoreProtocol` plus records for upserts and
query responses. `vectorstores/factory.py` instantiates concrete drivers:

- `MilvusVectorStore` (requires `pymilvus`), handles collection/partition bootstrap.
- `QdrantVectorStore` (requires `qdrant-client`), wraps HTTP/gRPC client.
- `MemoryVectorStore` only exists for unit tests and mirrors the protocol in pure NumPy.

Each namespace (passages, entities, facts) gets its own collection and can leverage the
native approximate search without touching pandas/fireducks/sqlite.

## Messaging (RabbitMQ)

`messaging/rabbitmq.py` introduces:

- `RabbitMQPublisher`: fire-and-forget fan-out for document lifecycle events.
- `RabbitMQWorker`: optional consumer helper when indexing should happen asynchronously.

Both rely on `pika` and accept declarative queue definitions mounted from YAML. The
HippoRAG pipeline emits events after document add/delete so external systems can hydrate
monitoring dashboards or trigger cache invalidations.

## Conversation Strategy

`hipporag/conversation.py` implements simple data classes describing chat turns and a
`ConversationReducer` that collapses full histories or last messages depending on
`HippoRAGSettings.conversation.mode`. Retrieval calls always route through this reducer
so the behavior is reproducible via config or per-call overrides.

## HippoRAG Pipeline

`hipporag/pipeline.py` ties everything together:

1. **Indexing:** chunk documents, extract OpenIE triples, store embeddings via the chosen
   vector DB, add nodes/edges into Neo4j, connect synonyms via ANN search, publish a
   `document.indexed` event.
2. **Retrieval:** embed query/history, run recognition memory + ANN search for phrases
   and passages, seed Neo4j Personalized PageRank with phrase/passage weights, blend
   dense and graph scores, and return ranked passages.
3. **QA:** feed the ranked passages to the configured LLM prompt template.
4. **Deletion:** remove embeddings, detach nodes/edges in Neo4j, and emit a
   `document.deleted` event.

Every public method carries English docstrings and returns structured dataclasses, which
keeps the API predictable for downstream services.

## Additional Functions to Implement

- Vector DB health probes (to surface connectivity issues early).
- Periodic synonym refresh jobs using RabbitMQ workers.
- Persistence of recognition-memory prompts so non-English locales can be supported.
- Automatic back-pressure handling for RabbitMQ consumers (optional but recommended).

This structure keeps Minerva’s algorithmic intent but replaces each problematic
component with scalable, configurable infrastructure ready for distributed deployments.
