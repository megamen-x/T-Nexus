"""
LLM-based Open Information Extraction helpers.
"""

from __future__ import annotations

import json
import logging
from typing import List, Sequence
import asyncio
from tqdm.asyncio import tqdm_asyncio

from src.t_nexus.ml.llm import BaseLLM, JSONParseError
from src.t_nexus.ml.prompts import PromptTemplateManager
from src.t_nexus.ml.utils import (
    DocumentChunk,
    ExtractionOutput,
    TripleRecord,
    filter_invalid_triples,
)

logger = logging.getLogger(__name__)


class OpenIEExtractor:
    """
    Prompt-template driven OpenIE pipeline backed by a chat model.
    """

    def __init__(self, llm: BaseLLM, prompt_manager: PromptTemplateManager | None = None) -> None:
        """Store the LLM backend used for extraction."""
        self.llm = llm
        self.prompts = prompt_manager or PromptTemplateManager()

    async def _run_ner(self, passage: str) -> List[str]:
        """Call the LLM to extract entities."""
        messages = self.prompts.render("ner", passage=passage)
        _, metadata = await self.llm.generate(messages)
        payload = metadata.get("json_payload")
        entities = payload.get("named_entities") if isinstance(payload, dict) else None
        if not isinstance(entities, list):
            logger.warning("NER response missing named_entities field")
            raise JSONParseError("NER missing named_entities field")

        seen = set()
        ordered = []
        for entity in entities:
            if entity not in seen:
                ordered.append(entity)
                seen.add(entity)
        return ordered

    async def _run_triples(self, passage: str, entities: List[str]) -> List[TripleRecord]:
        """Call the LLM to extract triples."""
        entity_json = json.dumps({"named_entities": entities})
        messages = self.prompts.render("triple_extraction", passage=passage, named_entity_json=entity_json)
        _, metadata = await self.llm.generate(messages)
        payload = metadata.get("json_payload")
        values = payload.get("triples") if isinstance(payload, dict) else None
        if not isinstance(values, list):
            logger.warning("Triple extraction response missing JSON payload")
            raise JSONParseError("triple extraction missing triples field")

        triples = filter_invalid_triples(values)
        triple_records: List[TripleRecord] = []
        for triple in triples:
            triple_records.append(
                TripleRecord(subject=triple[0], predicate=triple[1], object=triple[2])
            )
        return triple_records

    async def process_chunk(self, chunk: DocumentChunk) -> ExtractionOutput:
        """
        Run NER + triple extraction for a single chunk.
        """
        entities = await self._run_ner(chunk.text)
        triples = await self._run_triples(chunk.text, entities)
        return ExtractionOutput(chunk=chunk, entities=entities, triples=triples)

    async def process_chunks(
        self,
        chunks: Sequence[DocumentChunk],
    ) -> List[ExtractionOutput]:
        """
        Run OpenIE sequentially over each chunk asynchronously.

        :param chunks: Iterable of chunks to process.
        """
        if not chunks:
            return []

        tasks = [self.process_chunk(chunk) for chunk in chunks]
        return await tqdm_asyncio.gather(
            *tasks, 
            desc="OpenIE Extraction", 
            total=len(tasks)
        )
