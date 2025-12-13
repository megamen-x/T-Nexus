"""
Modernized HippoRAG pipeline that wires together all infrastructure pieces.
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm
import logging

from src.t_nexus.ml.config import load_settings
from src.t_nexus.ml.embeddings import BaseEmbeddingModel, OpenAIEmbeddingModel
from src.t_nexus.ml.extraction import OpenIEExtractor
from src.t_nexus.ml.graph import (
    GraphIndexer,
    Neo4jGraphStore,
    PassageNode,
    PhraseNode,
    create_ppr_retriever,
)
from src.t_nexus.ml.prompts.linking import get_query_instruction
from src.t_nexus.ml.llm import BaseLLM, OpenAILLM
from src.t_nexus.ml.messaging import RabbitMQPublisher
from src.t_nexus.ml.prompts import PromptTemplateManager
from src.t_nexus.ml.utils import (
    DocumentChunk,
    DocumentSource,
    ExtractionOutput,
    GraphSeedWeights,
    RetrievalResult,
    TripleRecord,
    chunk_text,
    compute_uuid5,
    normalize_text,
)
from src.t_nexus.ml.vectorstores import VectorRecord, VectorStoreFactory
from src.t_nexus.ml.vectorstores.base import VectorSearchResult
from src.t_nexus.ml.hipporag.conversation import ChatHistory, ConversationReducer

logger = logging.getLogger(__name__)


class HippoRAG:
    """
    Orchestrates indexing, retrieval, and QA for the HippoRAG algorithm.
    """

    def __init__(self, config_path: str | Path) -> None:
        self.settings = load_settings(config_path)
        self.state_dir = Path(self.settings.save_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.state_dir / "chunk_manifest.json"
        self.manifest = self._load_manifest()

        logger.info("Initializing HippoRAG with config: %s", config_path)

        self.embedding_model: BaseEmbeddingModel = OpenAIEmbeddingModel(self.settings.embeddings)
        self.llm: BaseLLM = OpenAILLM(self.settings.llm)
        self.prompt_manager = PromptTemplateManager()
        self.openie = OpenIEExtractor(self.llm, self.prompt_manager)
        self.vector_factory = VectorStoreFactory(self.settings.vector_store)
        self.passage_store = self.vector_factory.create("passages")
        self.entity_store = self.vector_factory.create("entities")
        self.fact_store = self.vector_factory.create("facts")

        self.graph_store = Neo4jGraphStore(self.settings.neo4j)
        self.graph_indexer = GraphIndexer(self.graph_store)
        self.graph_retriever = create_ppr_retriever(
            self.graph_store, self.settings.neo4j, self.settings.retrieval.engine
        )
        self._graph_ready = False
        self.query_embeddings: Dict[str, Dict[str, np.ndarray]] = {
            "fact": {},
            "passage": {},
        }
        self._executor = ThreadPoolExecutor(max_workers=16)

        self.conversation_reducer = ConversationReducer(self.settings.conversation)
        self.publisher = RabbitMQPublisher(self.settings.rabbitmq)

    # ------------------------------------------------------------------ #
    # Manifest helpers
    # ------------------------------------------------------------------ #
    def _load_manifest(self) -> Dict[str, Dict]:
        """
        Load the local chunk manifest if it exists.
        """
        if self.manifest_path.exists():
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        return {}

    def _save_manifest(self) -> None:
        """Persist the chunk manifest to disk."""
        with self.manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(self.manifest, handle, indent=2)
        self.manifest = self.manifest or {}

    async def _run_in_executor(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, partial(func, *args, **kwargs))

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #
    def _normalize_documents(self, documents: Sequence[str | DocumentSource]) -> List[DocumentSource]:
        normalized: List[DocumentSource] = []
        for doc in documents:
            if isinstance(doc, DocumentSource):
                normalized.append(doc)
            elif isinstance(doc, str):
                normalized.append(DocumentSource(text=doc))
            elif isinstance(doc, dict):
                normalized.append(DocumentSource(
                    text=str(doc.get("text") or doc.get("content") or ""),
                    source=doc.get("source"),
                    url=doc.get("url"),
                ))
            else:
                normalized.append(DocumentSource(text=str(doc)))
        return normalized

    def _chunk_documents(self, documents: Sequence[str | DocumentSource]) -> List[DocumentChunk]:
        """
        Split documents into chunks and assign deterministic IDs.
        """
        normalized_documents = self._normalize_documents(documents)
        chunks: List[DocumentChunk] = []
        for doc in normalized_documents:
            doc_id = compute_uuid5(doc.text)
            for idx, text in enumerate(chunk_text(doc.text)):
                chunk_id = compute_uuid5(f"{doc_id}-{idx}")
                metadata = {}
                if doc.source:
                    metadata["source"] = doc.source
                if doc.url:
                    metadata["url"] = doc.url
                chunks.append(DocumentChunk(chunk_id=chunk_id, text=text, source_id=doc_id, metadata=metadata))
        return chunks

    async def _extract_async(self, chunks: List[DocumentChunk]) -> List[ExtractionOutput]:
        """
        Run the OpenIE extractor for every chunk asynchronously.
        """
        if not chunks:
            return []
        return await self._run_in_executor(
            partial(self.openie.process_chunks, chunks, self.settings.retrieval.fact_top_k or None)
        )

    async def _encode_query_async(self, query: str, target: Literal["fact", "passage"]) -> np.ndarray:
        cache = self.query_embeddings[target]
        if query in cache:
            return cache[query]

        instruction = get_query_instruction(
            "query_to_fact" if target == "fact" else "query_to_passage"
        )
        text = f"{instruction}\n\n{query}"
        embeddings = await self._run_in_executor(self.embedding_model.embed, [text])
        embedding = embeddings[0]
        cache[query] = embedding
        return embedding

    async def _embed_records_async(self, texts: List[str]) -> np.ndarray:
        """
        Encode text lists using the embedding model asynchronously.
        """
        if not texts:
            return np.empty((0, self.settings.embeddings.dim))
        return await self._run_in_executor(self.embedding_model.embed, texts)

    def _embed_records(
        self, texts: List[str]
    ) -> np.ndarray:
        """Encode texts into embeddings."""
        if not texts:
            return np.empty((0, self.settings.embeddings.dim))
        batch_size = max(1, self.settings.embeddings.batch_size)
        chunks = []
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            chunk = texts[start : start + batch_size]
            chunk_embeddings = self.embedding_model.embed(chunk)
            chunks.append(chunk_embeddings)
        return np.vstack(chunks)

    def _prepare_entity_records(
        self, extractions: List[ExtractionOutput]
    ) -> Tuple[List[str], List[str]]:
        """
        Collect deduplicated entity texts and their deterministic IDs.
        """
        entities: List[str] = []
        for result in extractions:
            entities.extend([normalize_text(ent) for ent in result.entities if ent.strip()])
        deduped = list(dict.fromkeys(entities))
        ids = [compute_uuid5(ent) for ent in deduped]
        return ids, deduped

    def _prepare_fact_records(self, extractions: List[ExtractionOutput]) -> Tuple[List[str], List[str]]:
        """
        Collect deduplicated fact strings and deterministic IDs.
        """
        fact_texts: List[str] = []
        for result in extractions:
            for triple in result.triples:
                fact_texts.append(self._format_fact_text(triple))
        deduped = list(dict.fromkeys(fact_texts))
        ids = [compute_uuid5(fact) for fact in deduped]
        return ids, deduped

    def _format_fact_text(self, triple: TripleRecord) -> str:
        """Render a TripleRecord as canonical text."""
        return f"{normalize_text(triple.subject)}|||{normalize_text(triple.predicate)}|||{normalize_text(triple.object)}"

    def _parse_fact_text(self, text: str) -> Optional[TripleRecord]:
        """Parse canonical fact text back into a TripleRecord."""
        parts = text.split("|||")
        if len(parts) != 3:
            return None
        return TripleRecord(subject=parts[0], predicate=parts[1], object=parts[2])

    async def index_documents(self, documents: Sequence[str | DocumentSource]) -> List[str]:
        """
        Async indexer for FastAPI environments.
        """
        logger.info("Indexing %d documents", len(documents))
        chunks = self._chunk_documents(documents)

        existing_chunk_ids = set(self.manifest.keys())
        new_chunks = [chunk for chunk in chunks if chunk.chunk_id not in existing_chunk_ids]

        if not new_chunks:
            logger.info("All chunks already indexed, nothing to do.")
            return []

        extractions = await self._extract_async(new_chunks)
        chunk_ids = [chunk.chunk.chunk_id for chunk in extractions]

        passage_texts = [result.chunk.text for result in extractions]
        passage_embeddings = await self._embed_records_async(passage_texts)
        passage_records = [
            VectorRecord(
                record_id=result.chunk.chunk_id,
                vector=passage_embeddings[idx],
                text=result.chunk.text,
                metadata={"source_id": result.chunk.source_id or ""},
            )
            for idx, result in enumerate(extractions)
        ]
        self.passage_store.upsert(passage_records)

        entity_ids, entity_texts = self._prepare_entity_records(extractions)
        entity_embeddings = await self._embed_records_async(entity_texts)
        entity_records = [
            VectorRecord(record_id=entity_ids[idx], vector=entity_embeddings[idx], text=entity_texts[idx])
            for idx in range(len(entity_ids))
        ]
        if entity_records:
            self.entity_store.upsert(entity_records)

        fact_ids, fact_texts = self._prepare_fact_records(extractions)
        fact_embeddings = await self._embed_records_async(fact_texts)
        fact_records = [
            VectorRecord(record_id=fact_ids[idx], vector=fact_embeddings[idx], text=fact_texts[idx])
            for idx in range(len(fact_ids))
        ]
        if fact_records:
            self.fact_store.upsert(fact_records)

        for result in extractions:
            passage_node = PassageNode(
                node_id=result.chunk.chunk_id,
                text=result.chunk.text,
                source_id=result.chunk.source_id,
            )
            phrase_nodes = [
                PhraseNode(node_id=compute_uuid5(normalize_text(entity)), text=entity)
                for entity in result.entities
            ]
            triple_payload = []
            for triple in result.triples:
                subject_node = PhraseNode(
                    node_id=compute_uuid5(normalize_text(triple.subject)),
                    text=triple.subject,
                )
                object_node = PhraseNode(
                    node_id=compute_uuid5(normalize_text(triple.object)),
                    text=triple.object,
                )
                triple_payload.append((subject_node, triple.predicate, object_node))
            self.graph_indexer.index_chunk(passage_node, phrase_nodes, triple_payload)
            self.manifest[result.chunk.chunk_id] = {
                "text": result.chunk.text,
                "entities": [node.node_id for node in phrase_nodes],
                "facts": [
                    compute_uuid5(self._format_fact_text(triple)) for triple in result.triples
                ],
            }
            self.publisher.document_indexed(result.chunk.chunk_id, {"source_id": result.chunk.source_id})

        logger.info("Indexed %d chunks; saving manifest and refreshing graph", len(chunk_ids))
        self._save_manifest()
        self._graph_ready = False
        self._link_synonyms(entity_records)
        await self._run_in_executor(self._refresh_projection_if_needed)
        return chunk_ids

    def _link_synonyms(self, entity_records: List[VectorRecord]) -> None:
        """
        Use the vector store to find synonym pairs for the new batch of entities.
        """
        if not entity_records:
            return
        threshold = self.settings.retrieval.synonym_threshold
        top_k = self.settings.retrieval.synonym_top_k
        synonym_pairs = []
        for record in entity_records:
            results = self.entity_store.query(record.vector, top_k=top_k)
            for hit in results:
                if hit.record_id == record.record_id:
                    continue
                if hit.score < threshold:
                    continue
                synonym_pairs.append(
                    (
                        PhraseNode(node_id=record.record_id, text=record.text),
                        PhraseNode(node_id=hit.record_id, text=hit.text),
                        hit.score,
                    )
                )
        if synonym_pairs:
            self.graph_indexer.add_synonym_edges(synonym_pairs)
            self._graph_ready = False

    # ------------------------------------------------------------------ #
    # Deletion
    # ------------------------------------------------------------------ #
    def delete_documents(self, chunk_ids: Sequence[str]) -> None:
        """
        Delete chunks and related embeddings/graph entries.
        """
        logger.info("Deleting %d chunks", len(chunk_ids))
        for chunk_id in chunk_ids:
            chunk_data = self.manifest.pop(chunk_id, None)
            if not chunk_data:
                continue
            self.passage_store.delete([chunk_id])
            self.publisher.document_deleted(chunk_id)

            # Delete orphan entities
            for entity_id in chunk_data.get("entities", []):
                still_used = any(entity_id in entry.get("entities", []) for entry in self.manifest.values())
                if not still_used:
                    self.entity_store.delete([entity_id])

            # Delete orphan facts
            for fact_id in chunk_data.get("facts", []):
                still_used = any(fact_id in entry.get("facts", []) for entry in self.manifest.values())
                if not still_used:
                    self.fact_store.delete([fact_id])

        self.graph_indexer.delete_passages(chunk_ids)
        self._graph_ready = False
        self._save_manifest()
        logger.info("Deleted documents; refreshing graph projection")
        self._refresh_projection_if_needed()

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def _refresh_projection_if_needed(self) -> None:
        """Rebuild the Neo4j projection if the flag instructs us to."""
        if not self._graph_ready:
            self.graph_retriever.refresh_projection()
            self._graph_ready = True

    def _build_phrase_weights(self, fact_hits: List[VectorSearchResult]) -> Dict[str, float]:
        """Translate fact hits into phrase weights for Personalized PageRank."""
        weights: Dict[str, float] = {}
        for hit in fact_hits:
            triple = self._parse_fact_text(hit.text)
            if not triple:
                continue
            for phrase in (triple.subject, triple.object):
                node_id = compute_uuid5(phrase)
                weights[node_id] = weights.get(node_id, 0.0) + hit.score
        return weights

    async def retrieve(
        self, history: ChatHistory, *, top_k: Optional[int] = None, mode: Optional[str] = None
    ) -> RetrievalResult:
        """
        Retrieve passages for the next answer using graph + vector search.
        """
        logger.info("Retrieving for query with mode=%s", mode or self.settings.conversation.mode)
        bundle = self.conversation_reducer.build_query(history, override_mode=mode)
        fact_vector = await self._encode_query_async(bundle.question, "fact")
        passage_vector = await self._encode_query_async(bundle.question, "passage")

        top_k = top_k or self.settings.retrieval.top_k
        fact_hits_future = self._run_in_executor(
            self.fact_store.query, fact_vector, self.settings.retrieval.fact_top_k
        )
        passage_hits_future = self._run_in_executor(self.passage_store.query, passage_vector, top_k)

        fact_hits, passage_hits = await asyncio.gather(fact_hits_future, passage_hits_future)
        passages = [hit.text for hit in passage_hits]
        scores = [hit.score for hit in passage_hits]

        phrase_weights = self._build_phrase_weights(fact_hits)
        seed = GraphSeedWeights()
        for node_id, weight in phrase_weights.items():
            seed.phrase_weights[node_id] = weight
        for hit in passage_hits:
            seed.passage_weights[hit.record_id] = hit.score * self.settings.retrieval.passage_weight
        seed.normalize()

        await self._run_in_executor(self._refresh_projection_if_needed)
        pagerank_results = await self._run_in_executor(
            partial(
                self.graph_retriever.personalized_pagerank,
                seeds=seed,
                damping=self.settings.retrieval.damping,
                top_k=top_k,
            )
        )

        if not pagerank_results:
            logger.info("Falling back to dense DPR results (%d passages)", len(passages))
            return RetrievalResult(query=bundle, passages=passages, scores=scores)

        passages = [row[2] for row in pagerank_results]
        scores = [row[1] for row in pagerank_results]
        logger.info("Retrieved %d passages via graph", len(passages))
        return RetrievalResult(query=bundle, passages=passages, scores=scores)

    # ------------------------------------------------------------------ #
    # QA
    # ------------------------------------------------------------------ #
    async def answer(self, retrieval: RetrievalResult, max_context: Optional[int] = None) -> str:
        """
        Generate an answer using the retrieved passages.
        """
        logger.info("Generating QA response for query '%s'", retrieval.query.question)
        max_context = max_context or self.settings.retrieval.top_k
        prompt_user = ""
        for passage in retrieval.passages[:max_context]:
            prompt_user += f"Wikipedia Title: {passage}\n\n"
        prompt_user += f"Question: {retrieval.query.question}\nThought: "
        template_name = f"rag_qa_{self.settings.dataset}"
        if not self.prompt_manager.is_template_name_valid(template_name):
            template_name = "rag_qa_musique"
        messages = self.prompt_manager.render(template_name, prompt_user=prompt_user)
        response, _ = await self._run_in_executor(self.llm.generate, messages)
        return response

    async def rag(self, history: ChatHistory, *, top_k: Optional[int] = None) -> Tuple[str, RetrievalResult]:
        """
        Convenience helper that runs retrieval and QA in one call.
        """
        retrieval = await self.retrieve(history, top_k=top_k)
        answer = await self.answer(retrieval)
        return answer, retrieval

    # ------------------------------------------------------------------ #
    # Shutdown
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        """Release open connections and files."""
        self.graph_store.close()
        self.publisher.close()
