"""
Dataclasses shared across modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


@dataclass
class DocumentChunk:
    """Lightweight representation of a chunked passage."""

    chunk_id: str
    text: str
    source_id: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class DocumentSource:
    """Normalized representation of an incoming document for indexing."""

    text: str
    source: str | None = None
    url: str | None = None


@dataclass
class TripleRecord:
    """Subject-predicate-object triple extracted from text."""

    subject: str
    predicate: str
    object: str
    confidence: float | None = None

    def as_tuple(self) -> tuple[str, str, str]:
        """Return the triple as a plain tuple."""
        return (self.subject, self.predicate, self.object)


@dataclass
class QueryBundle:
    """
    Query plus optional conversation history that the retrieval stack consumes.
    """

    question: str
    history: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Container returned by :meth:`HippoRAG.retrieve`."""

    query: QueryBundle
    passages: List[str]
    scores: List[float]

    def top_passages(self, limit: int) -> Iterable[str]:
        """Yield up to *limit* passages preserving ranking order."""
        return self.passages[:limit]


@dataclass
class ExtractionOutput:
    """Result of running OpenIE on a chunk."""

    chunk: DocumentChunk
    entities: List[str]
    triples: List[TripleRecord]


@dataclass
class GraphSeedWeights:
    """Reset probabilities for Personalized PageRank."""

    phrase_weights: Dict[str, float] = field(default_factory=dict)
    passage_weights: Dict[str, float] = field(default_factory=dict)

    def normalize(self) -> "GraphSeedWeights":
        """
        Normalize all weights to sum to one.
        """
        total = sum(self.phrase_weights.values()) + sum(self.passage_weights.values())
        if total <= 0:
            return self
        factor = 1.0 / total
        self.phrase_weights = {k: v * factor for k, v in self.phrase_weights.items()}
        self.passage_weights = {k: v * factor for k, v in self.passage_weights.items()}
        return self
