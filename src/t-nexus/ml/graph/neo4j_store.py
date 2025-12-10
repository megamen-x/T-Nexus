"""
Neo4j-backed graph utilities for HippoRAG plus alternative PPR retrievers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from neo4j import GraphDatabase, basic_auth
from dotenv import load_dotenv

load_dotenv()

try:
    import networkx as nx
    nx.config.warnings_to_ignore.add("cache")
except ImportError:  # pragma: no cover - optional dependency
    nx = None

try:
    import cudf
    import cugraph
except ImportError:
    cudf = None
    cugraph = None

from src.ml.config.schema import Neo4jSettings
from src.ml.utils import GraphSeedWeights


@dataclass
class PassageNode:
    """Represents a passage node in Neo4j."""

    node_id: str
    text: str
    source_id: str | None = None


@dataclass
class PhraseNode:
    """Represents a phrase node."""

    node_id: str
    text: str


class Neo4jGraphStore:
    """
    Low-level connection wrapper for Neo4j.
    """

    def __init__(self, settings: Neo4jSettings) -> None:
        """Create the Neo4j driver and optional constraints."""
        self.settings = settings
        self.driver = GraphDatabase.driver(
            settings.uri,
            auth=basic_auth(settings.user, settings.password),
        )
        if settings.create_constraints:
            self._ensure_constraints()

    def close(self) -> None:
        """Close underlying driver."""
        self.driver.close()

    def _ensure_constraints(self) -> None:
        """Create uniqueness constraints if they do not exist."""
        constraints = [
            ("Passage", "id"),
            ("Phrase", "id"),
        ]
        query = "CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
        with self.driver.session(database=self.settings.database) as session:
            for label, prop in constraints:
                session.execute_write(lambda tx: list(tx.run(query.format(label=label, prop=prop))))

    def run(self, query: str, *, read_only: bool = False, **params):
        """Execute a Cypher query and consume its result before closing the transaction."""

        def _consume(tx):
            result = tx.run(query, **params)
            return list(result)

        with self.driver.session(database=self.settings.database) as session:
            if read_only:
                return session.execute_read(_consume)
            return session.execute_write(_consume)


class BasePPRRetriever(ABC):
    """Abstract base for all PPR retrievers."""

    @abstractmethod
    def refresh_projection(self) -> None:
        """Ensure the engine is ready for PageRank."""

    @abstractmethod
    def personalized_pagerank(
        self, seeds: GraphSeedWeights, damping: float, top_k: int
    ) -> List[Tuple[str, float, str]]:
        """Return top-k passages for the current seeds."""


class GraphIndexer:
    """Helper that writes passages, phrases, and edges into Neo4j."""

    def __init__(self, store: Neo4jGraphStore) -> None:
        """Bind the indexer to a graph store."""
        self.store = store

    def index_chunk(
        self,
        passage: PassageNode,
        phrases: List[PhraseNode],
        triples: List[Tuple[PhraseNode, str, PhraseNode]],
    ) -> None:
        """
        Create/Update the passage node plus all phrase nodes and edges that
        originate from a chunk.
        """
        query = query = """
        MERGE (p:Passage {id: $passage.node_id})
        SET p.text = $passage.text,
            p.source_id = $passage.source_id
        WITH p
        UNWIND $phrases AS phrase
            MERGE (ph:Phrase {id: phrase.node_id})
            SET ph.text = phrase.text
            MERGE (p)-[:CONTAINS {weight: 1.0}]->(ph)
        WITH p
        UNWIND $triples AS triple
            MERGE (s:Phrase {id: triple.subject.node_id})
            SET s.text = triple.subject.text
            MERGE (o:Phrase {id: triple.object.node_id})
            SET o.text = triple.object.text
            MERGE (s)-[r:RELATES_TO {predicate: triple.predicate}]->(o)
            ON CREATE SET r.weight = 1.0, r.passage_ids = [$passage.node_id]
            ON MATCH SET r.weight = r.weight + 1.0,
                r.passage_ids = CASE 
                    WHEN $passage.node_id IN r.passage_ids THEN r.passage_ids
                    ELSE r.passage_ids + $passage.node_id
                END
        """
        formatted_triples = [
            {
                "subject": {"node_id": s.node_id, "text": s.text},
                "predicate": predicate,
                "object": {"node_id": o.node_id, "text": o.text},
            }
            for s, predicate, o in triples
        ]
        self.store.run(
            query,
            passage=passage.__dict__,
            phrases=[phrase.__dict__ for phrase in phrases],
            triples=formatted_triples,
        )

    def add_synonym_edges(self, pairs: List[Tuple[PhraseNode, PhraseNode, float]]) -> None:
        """Create synonym edges between phrase nodes."""
        if not pairs:
            return
        query = """
        UNWIND $pairs AS entry
            MERGE (a:Phrase {id: entry.a.node_id})
            SET a.text = entry.a.text
            MERGE (b:Phrase {id: entry.b.node_id})
            SET b.text = entry.b.text
            MERGE (a)-[rel:SYNONYM_OF]->(b)
            SET rel.weight = entry.weight
            MERGE (b)-[rel2:SYNONYM_OF]->(a)
            SET rel2.weight = entry.weight
        """
        payload = [
            {"a": pair[0].__dict__, "b": pair[1].__dict__, "weight": pair[2]} for pair in pairs
        ]
        self.store.run(query, pairs=payload)

    def delete_passages(self, passage_ids: Iterable[str]) -> None:
        """Remove passage nodes and detach orphan phrase nodes."""
        ids = list(passage_ids)
        if not ids:
            return
        delete_passages = """
        MATCH (p:Passage)
        WHERE p.id IN $ids
        DETACH DELETE p
        """
        self.store.run(delete_passages, ids=ids)
        cleanup_phrases = """
        MATCH (ph:Phrase)
        WHERE NOT (ph)--()
        DELETE ph
        """
        self.store.run(cleanup_phrases)


class GraphRetriever(BasePPRRetriever):
    """Execute graph algorithms using the Neo4j GDS library."""

    def __init__(self, store: Neo4jGraphStore, settings: Neo4jSettings) -> None:
        """Bind the retriever to a graph store and settings."""
        self.store = store
        self.settings = settings

    def refresh_projection(self) -> None:
        """Recreate the projection used by the GDS pipeline."""
        drop_query = """
        CALL gds.graph.exists($name) YIELD exists
        WITH exists
        WHERE exists
        CALL gds.graph.drop($name) YIELD graphName
        RETURN graphName
        """
        self.store.run(drop_query, name=self.settings.graph_projection)
        project_query = """
        CALL gds.graph.project(
            $name,
            ['Passage', 'Phrase'],
            {
                CONTAINS: {type: 'CONTAINS', orientation: 'UNDIRECTED', properties: 'weight'},
                RELATES_TO: {type: 'RELATES_TO', orientation: 'UNDIRECTED', properties: 'weight'},
                SYNONYM_OF: {type: 'SYNONYM_OF', orientation: 'UNDIRECTED', properties: 'weight'}
            }
        )
        """
        self.store.run(project_query, name=self.settings.graph_projection)

    def personalized_pagerank(
        self, seeds: GraphSeedWeights, damping: float, top_k: int
    ) -> List[Tuple[str, float, str]]:
        """
        Run Personalized PageRank and return the top passages.
        """
        seed_ids = list(seeds.phrase_weights.keys()) + list(seeds.passage_weights.keys())

        if not seed_ids:
            return []

        query = """
        MATCH (n)
        WHERE n.id IN $seed_ids
        WITH collect(n) AS sourceNodes

        CALL gds.pageRank.stream($name, {
            dampingFactor: $damping,
            sourceNodes: sourceNodes
        })
        YIELD nodeId, score
        WITH gds.util.asNode(nodeId) AS node, score
        WHERE node:Passage
        RETURN node.id AS passage_id, node.text AS text, score
        ORDER BY score DESC
        LIMIT $top_k
        """

        rows = self.store.run(
            query,
            name=self.settings.graph_projection,
            damping=damping,
            seed_ids=seed_ids,
            top_k=top_k,
            read_only=True,
        )
        return [(row["passage_id"], row["score"], row.get("text", "")) for row in rows]


def _load_graph_data(store: Neo4jGraphStore) -> Tuple[List[Dict[str, any]], List[Dict[str, any]]]:
    """Helper to fetch nodes and edges."""
    nodes_query = """
    MATCH (n)
    WHERE n:Passage OR n:Phrase
    RETURN n.id AS id, labels(n) AS labels, n.text AS text
    """
    nodes = list(store.run(nodes_query, read_only=True))

    edges_query = """
    MATCH (a)-[r]->(b)
    WHERE (a:Passage OR a:Phrase) AND (b:Passage OR b:Phrase)
    RETURN a.id AS source, b.id AS target, coalesce(r.weight, 1.0) AS weight
    """
    edges = list(store.run(edges_query, read_only=True))
    return nodes, edges


class NetworkXPPRRetriever(BasePPRRetriever):
    """NetworkX-based PPR retriever. Supports nx-cugraph backend if configured."""

    def __init__(self, store: Neo4jGraphStore) -> None:
        if nx is None:
            raise RuntimeError("NetworkX is required for NetworkXPPRRetriever.")
        self.store = store
        self._graph: nx.DiGraph | None = None
        self._passage_texts: Dict[str, str] = {}
        self._passage_ids: set = set()

    def refresh_projection(self) -> None:
        """Build the NetworkX graph from Neo4j."""
        nodes, edges = _load_graph_data(self.store)
        self._graph = nx.DiGraph()
        self._passage_texts = {}
        for node in nodes:
            self._graph.add_node(node["id"], labels=node["labels"], text=node.get("text", ""))
            if "Passage" in node["labels"]:
                self._passage_texts[node["id"]] = node.get("text", "")
                self._passage_ids.add(node["id"])

        for edge in edges:
            self._graph.add_edge(edge["source"], edge["target"], weight=edge["weight"])

    def personalized_pagerank(
        self, seeds: GraphSeedWeights, damping: float, top_k: int
    ) -> List[Tuple[str, float, str]]:
        """Run NetworkX (possibly GPU-accelerated) PageRank."""
        personalization = {**seeds.phrase_weights, **seeds.passage_weights}

        if not personalization:
            return []

        if self._graph is None:
            self.refresh_projection()

        valid_personalization = {
            node_id: weight
            for node_id, weight in personalization.items()
            if node_id in self._graph
        }

        if not valid_personalization:
            return []

        total = sum(valid_personalization.values())
        normalized = {node_id: weight / total for node_id, weight in valid_personalization.items()}

        result = nx.pagerank(
            self._graph,
            alpha=damping,
            personalization=normalized,
            weight="weight",
        )

        passage_scores = [
            (node_id, score, self._passage_texts.get(node_id, ""))
            for node_id, score in result.items()
            if node_id in self._passage_ids
        ]
        passage_scores.sort(key=lambda item: item[1], reverse=True)
        return passage_scores[:top_k]


class CuGraphPPRRetriever(BasePPRRetriever):
    """GPU-only cuGraph retriever."""

    def __init__(self, store: Neo4jGraphStore) -> None:
        if cugraph is None or cudf is None:
            raise RuntimeError("cuGraph/cuDF are required for CuGraphPPRRetriever.")
        self.store = store
        self._graph: cugraph.Graph | None = None
        self._passage_texts: Dict[str, str] = {}
        self._passage_ids: set = set()
        self._node_ids: set = set()

    def refresh_projection(self) -> None:
        """Build the cuGraph representation using Neo4j data."""
        nodes, edges = _load_graph_data(self.store)
        
        self._passage_texts = {}
        self._passage_ids = set()
        self._node_ids = set()
        
        for node in nodes:
            node_id = node["id"]
            self._node_ids.add(node_id)
            if "Passage" in node["labels"]:
                self._passage_texts[node_id] = node.get("text", "")
                self._passage_ids.add(node_id)

        if not edges:
            self._graph = None
            return

        edge_df = cudf.DataFrame({
            "src": [e["source"] for e in edges],
            "dst": [e["target"] for e in edges],
            "weight": [e["weight"] for e in edges],
        })
        
        self._graph = cugraph.Graph(directed=True)
        self._graph.from_cudf_edgelist(
            edge_df, source="src", destination="dst", edge_attr="weight"
        )

    def personalized_pagerank(
        self, seeds: GraphSeedWeights, damping: float, top_k: int
    ) -> List[Tuple[str, float, str]]:
        """Run cuGraph PageRank on GPU."""
        personalization = {**seeds.phrase_weights, **seeds.passage_weights}

        if not personalization:
            return []

        if self._graph is None:
            self.refresh_projection()

        if self._graph is None:
            return []

        # Фильтруем валидные seeds (O(1) lookup)
        valid_personalization = {
            node_id: weight
            for node_id, weight in personalization.items()
            if node_id in self._node_ids
        }

        if not valid_personalization:
            return []

        # Нормализуем веса
        total = sum(valid_personalization.values())
        vertices = list(valid_personalization.keys())
        values = [weight / total for weight in valid_personalization.values()]
        
        personalization_df = cudf.DataFrame({
            "vertex": vertices, 
            "values": values
        })

        result = cugraph.pagerank(
            self._graph,
            alpha=damping,
            personalization=personalization_df,
        )

        # Фильтруем только Passage узлы
        df = result.to_pandas()
        df = df[df["vertex"].isin(self._passage_ids)]
        df = df.nlargest(top_k, "pagerank")

        output = []
        for _, row in df.iterrows():
            node_id = row["vertex"]
            score = float(row["pagerank"])
            text = self._passage_texts.get(node_id, "")
            output.append((node_id, score, text))
        
        return output


def create_ppr_retriever(
    store: Neo4jGraphStore, settings: Neo4jSettings, engine: str
) -> BasePPRRetriever:
    """
    Factory that instantiates a retriever based on the requested engine.
    """
    engine = engine.lower()
    if engine == "neo4j":
        return GraphRetriever(store, settings)
    if engine == "networkx":
        return NetworkXPPRRetriever(store)
    if engine == "cugraph":
        return CuGraphPPRRetriever(store)
    raise ValueError(f"Unknown retrieval engine '{engine}'.")
