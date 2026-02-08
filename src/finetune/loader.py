"""
KG Data Loader

Loads GraphML graph and all JSON KV stores into a unified interface.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Parsed entity with split descriptions."""
    entity_id: str
    name: str
    entity_type: str
    descriptions: List[str]
    source_ids: List[str]
    chunk_ids: List[str] = field(default_factory=list)


@dataclass
class Relationship:
    """Parsed relationship."""
    src: str
    tgt: str
    description: str
    keywords: List[str]
    weight: float = 1.0


@dataclass
class TextChunk:
    """Parsed text chunk."""
    chunk_id: str
    content: str
    tokens: int
    full_doc_id: str = ""
    order_index: int = 0


class KGLoader:
    """
    Unified loader for all KG data sources.

    Loads:
    - GraphML graph (entity-relation structure)
    - kv_store_text_chunks.json (semantic text chunks)
    - kv_store_full_docs.json (full document texts)
    - kv_store_entity_chunks.json (entity-to-chunk mappings)
    - vdb_entities.json (entity metadata with descriptions)
    - vdb_relationships.json (relationship metadata)
    """

    def __init__(self, kg_dir: str | Path):
        self.kg_dir = Path(kg_dir)
        self._graph: Optional[nx.Graph] = None
        self._entities: Dict[str, Entity] = {}
        self._relationships: List[Relationship] = []
        self._chunks: Dict[str, TextChunk] = {}
        self._full_docs: Dict[str, str] = {}
        self._loaded = False

    def load(self) -> "KGLoader":
        """Load all KG data sources."""
        if self._loaded:
            return self

        self._load_graph()
        self._load_entities()
        self._load_relationships()
        self._load_chunks()
        self._load_full_docs()
        self._load_entity_chunks()
        self._loaded = True

        logger.info(
            f"KG loaded: {len(self._entities)} entities, "
            f"{len(self._relationships)} relationships, "
            f"{len(self._chunks)} chunks, "
            f"{len(self._full_docs)} docs"
        )
        return self

    @property
    def graph(self) -> nx.Graph:
        self._ensure_loaded()
        return self._graph

    @property
    def entities(self) -> Dict[str, Entity]:
        self._ensure_loaded()
        return self._entities

    @property
    def relationships(self) -> List[Relationship]:
        self._ensure_loaded()
        return self._relationships

    @property
    def chunks(self) -> Dict[str, TextChunk]:
        self._ensure_loaded()
        return self._chunks

    @property
    def full_docs(self) -> Dict[str, str]:
        self._ensure_loaded()
        return self._full_docs

    def _ensure_loaded(self):
        if not self._loaded:
            self.load()

    def _load_graph(self):
        """Load GraphML graph."""
        path = self.kg_dir / "graph_chunk_entity_relation.graphml"
        if not path.exists():
            logger.warning(f"GraphML not found: {path}")
            self._graph = nx.Graph()
            return

        self._graph = nx.read_graphml(str(path))
        logger.info(f"Graph: {self._graph.number_of_nodes()} nodes, {self._graph.number_of_edges()} edges")

    def _load_entities(self):
        """Load entities from vdb_entities.json."""
        path = self.kg_dir / "vdb_entities.json"
        if not path.exists():
            logger.warning(f"vdb_entities not found: {path}")
            return

        data = json.loads(path.read_text(encoding="utf-8"))
        entity_list = data.get("data", [])

        for item in entity_list:
            name = item.get("entity_name", "")
            if not name:
                continue

            content = item.get("content", "")
            # content format: "EntityName\ndescription1<SEP>description2<SEP>..."
            # Split off the entity name line, then split descriptions by <SEP>
            lines = content.split("\n", 1)
            desc_text = lines[1] if len(lines) > 1 else ""
            descriptions = [d.strip() for d in desc_text.split("<SEP>") if d.strip()]

            source_ids = [s.strip() for s in item.get("source_id", "").split("<SEP>") if s.strip()]

            self._entities[name] = Entity(
                entity_id=item.get("__id__", ""),
                name=name,
                entity_type=self._get_node_type(name),
                descriptions=descriptions,
                source_ids=source_ids,
            )

        logger.info(f"Entities: {len(self._entities)}")

    def _get_node_type(self, name: str) -> str:
        """Get entity type from graph node attributes."""
        if self._graph and name in self._graph.nodes:
            return self._graph.nodes[name].get("entity_type", "unknown")
        return "unknown"

    def _load_relationships(self):
        """Load relationships from vdb_relationships.json."""
        path = self.kg_dir / "vdb_relationships.json"
        if not path.exists():
            logger.warning(f"vdb_relationships not found: {path}")
            return

        data = json.loads(path.read_text(encoding="utf-8"))
        rel_list = data.get("data", [])

        for item in rel_list:
            src = item.get("src_id", "")
            tgt = item.get("tgt_id", "")
            if not src or not tgt:
                continue

            content = item.get("content", "")
            if not content:
                continue

            # content format: "keywords\tSrc\nTgt\nDescription"
            parts = content.split("\t", 1)
            if len(parts) > 1:
                keywords_str = parts[0]
                desc_part = parts[1]
            else:
                keywords_str = ""
                desc_part = content

            # Description is after the two entity name lines
            desc_lines = desc_part.split("\n")
            if len(desc_lines) > 2:
                description = desc_lines[2]
            elif len(desc_lines) > 0:
                description = desc_lines[-1]
            else:
                description = desc_part

            keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]

            self._relationships.append(Relationship(
                src=src,
                tgt=tgt,
                description=description,
                keywords=keywords,
            ))

        logger.info(f"Relationships: {len(self._relationships)}")

    def _load_chunks(self):
        """Load text chunks from kv_store_text_chunks.json."""
        path = self.kg_dir / "kv_store_text_chunks.json"
        if not path.exists():
            logger.warning(f"kv_store_text_chunks not found: {path}")
            return

        data = json.loads(path.read_text(encoding="utf-8"))

        for chunk_id, chunk_data in data.items():
            if not isinstance(chunk_data, dict):
                continue
            content = chunk_data.get("content", "")
            if not content or len(content.strip()) < 10:
                continue

            self._chunks[chunk_id] = TextChunk(
                chunk_id=chunk_id,
                content=content,
                tokens=chunk_data.get("tokens", 0),
                full_doc_id=chunk_data.get("full_doc_id", ""),
                order_index=chunk_data.get("chunk_order_index", 0),
            )

        logger.info(f"Chunks: {len(self._chunks)}")

    def _load_full_docs(self):
        """Load full documents from kv_store_full_docs.json."""
        path = self.kg_dir / "kv_store_full_docs.json"
        if not path.exists():
            logger.warning(f"kv_store_full_docs not found: {path}")
            return

        data = json.loads(path.read_text(encoding="utf-8"))

        for doc_id, doc_data in data.items():
            if isinstance(doc_data, dict):
                content = doc_data.get("content", "")
            else:
                content = str(doc_data)
            if content:
                self._full_docs[doc_id] = content

        logger.info(f"Full docs: {len(self._full_docs)}")

    def _load_entity_chunks(self):
        """Load entity-to-chunk mappings from kv_store_entity_chunks.json."""
        path = self.kg_dir / "kv_store_entity_chunks.json"
        if not path.exists():
            logger.warning(f"kv_store_entity_chunks not found: {path}")
            return

        data = json.loads(path.read_text(encoding="utf-8"))

        for entity_name, mapping in data.items():
            if entity_name in self._entities and isinstance(mapping, dict):
                self._entities[entity_name].chunk_ids = mapping.get("chunk_ids", [])

        logger.info(f"Entity-chunk mappings loaded")

    def get_chunk_content(self, chunk_id: str) -> str:
        """Get the text content for a chunk ID."""
        chunk = self._chunks.get(chunk_id)
        return chunk.content if chunk else ""

    def get_entity_descriptions(self, entity_name: str) -> List[str]:
        """Get all descriptions for an entity, deduplicated."""
        entity = self._entities.get(entity_name)
        if not entity:
            return []
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for d in entity.descriptions:
            normalized = d.strip().lower()
            if normalized not in seen:
                seen.add(normalized)
                unique.append(d.strip())
        return unique

    def get_entity_chunks_text(self, entity_name: str) -> List[str]:
        """Get all chunk texts associated with an entity."""
        entity = self._entities.get(entity_name)
        if not entity:
            return []
        return [
            self.get_chunk_content(cid)
            for cid in entity.chunk_ids
            if self.get_chunk_content(cid)
        ]

    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a specific type."""
        return [e for e in self._entities.values() if e.entity_type == entity_type]

    def get_neighbors(self, entity_name: str) -> List[str]:
        """Get neighboring entity names in the graph."""
        if self._graph and entity_name in self._graph:
            return list(self._graph.neighbors(entity_name))
        return []

    def get_paths(self, source: str, target: str, max_length: int = 3) -> List[List[str]]:
        """Get all simple paths between two entities up to max_length."""
        if not self._graph or source not in self._graph or target not in self._graph:
            return []
        try:
            return list(nx.all_simple_paths(self._graph, source, target, cutoff=max_length))
        except nx.NetworkXError:
            return []

    def get_edge_description(self, src: str, tgt: str) -> str:
        """Get the description for an edge between two nodes."""
        if self._graph and self._graph.has_edge(src, tgt):
            edge_data = self._graph.edges[src, tgt]
            desc = edge_data.get("description", "")
            # Take first description if <SEP>-separated
            if "<SEP>" in desc:
                desc = desc.split("<SEP>")[0].strip()
            return desc
        return ""
