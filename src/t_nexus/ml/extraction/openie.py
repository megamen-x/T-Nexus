"""
LLM-based Open Information Extraction helpers.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Sequence

from tqdm import tqdm

from src.t_nexus.ml.llm import BaseLLM
from src.t_nexus.ml.prompts import PromptTemplateManager
from src.t_nexus.ml.utils import (
    DocumentChunk,
    ExtractionOutput,
    TripleRecord,
    extract_json_field,
    filter_invalid_triples,
)

logger = logging.getLogger(__name__)


class JSONParseError(ValueError):
    """Raised when an LLM response could not be parsed."""


class OpenIEExtractor:
    """
    Prompt-template driven OpenIE pipeline backed by a chat model.
    """

    def __init__(self, llm: BaseLLM, prompt_manager: PromptTemplateManager | None = None) -> None:
        """Store the LLM backend used for extraction."""
        self.llm = llm
        self.prompts = prompt_manager or PromptTemplateManager()

    def _run_ner(self, passage: str) -> List[str]:
        """Call the LLM to extract entities."""
        messages = self.prompts.render("ner", passage=passage)
        response, _ = self.llm.generate(messages)
        entities = extract_json_field(response, "named_entities")
        if entities is None:
            logger.warning("NER response missing JSON payload")
            raise JSONParseError("NER missing named_entities field")

        seen = set()
        ordered = []
        for entity in entities:
            if entity not in seen:
                ordered.append(entity)
                seen.add(entity)
        return ordered

    def _run_triples(self, passage: str, entities: List[str]) -> List[TripleRecord]:
        """Call the LLM to extract triples."""
        entity_json = json.dumps({"named_entities": entities})
        messages = self.prompts.render("triple_extraction", passage=passage, named_entity_json=entity_json)
        response, _ = self.llm.generate(messages)
        values = extract_json_field(response, "triples")
        if values is None:
            logger.warning("Triple extraction response missing JSON payload")
            raise JSONParseError("triple extraction missing triples field")

        triples = filter_invalid_triples(values)
        triple_records: List[TripleRecord] = []
        for triple in triples:
            triple_records.append(
                TripleRecord(subject=triple[0], predicate=triple[1], object=triple[2])
            )
        return triple_records

    def process_chunk(self, chunk: DocumentChunk) -> ExtractionOutput:
        """
        Run NER + triple extraction for a single chunk.
        """
        entities = self._run_ner(chunk.text)
        triples = self._run_triples(chunk.text, entities)
        return ExtractionOutput(chunk=chunk, entities=entities, triples=triples)

    def process_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        max_workers: int | None = None,
    ) -> List[ExtractionOutput]:
        """
        Run OpenIE over many chunks concurrently using a thread pool.

        :param chunks: Iterable of chunks to process.
        :param max_workers: Optional override for worker count.
        """
        if not chunks:
            return []

        max_workers = max_workers or min(4, len(chunks))
        results: List[ExtractionOutput | None] = [None] * len(chunks)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}
            for idx, chunk in enumerate(tqdm(chunks, desc="Submitting OpenIE tasks")):
                future = executor.submit(self.process_chunk, chunk)
                future_to_idx[future] = idx

            for future in tqdm(
                as_completed(future_to_idx),
                total=len(future_to_idx),
                desc="Collecting OpenIE results",
            ):
                idx = future_to_idx[future]
                results[idx] = future.result()

        return [res for res in results if res is not None]
