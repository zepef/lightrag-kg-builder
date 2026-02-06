"""
Graph Merger - Merge multiple KG outputs with entity deduplication.

After parallel pipeline execution, this module merges the separate graph outputs
into a unified knowledge graph with deduplicated entities and merged relationships.
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents a knowledge graph entity."""
    name: str
    entity_type: str
    description: str = ""
    sources: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def merge_with(self, other: "Entity") -> "Entity":
        """Merge this entity with another, keeping the best attributes."""
        merged_desc = self.description if len(self.description) >= len(other.description) else other.description
        merged_sources = list(set(self.sources + other.sources))
        merged_attrs = {**other.attributes, **self.attributes}
        for key in merged_attrs:
            if key in self.attributes and key in other.attributes:
                if not self.attributes.get(key) and other.attributes.get(key):
                    merged_attrs[key] = other.attributes[key]

        return Entity(
            name=self.name,
            entity_type=self.entity_type or other.entity_type,
            description=merged_desc,
            sources=merged_sources,
            attributes=merged_attrs
        )


@dataclass
class Relationship:
    """Represents a knowledge graph relationship."""
    source: str
    target: str
    relation_type: str
    description: str = ""
    weight: float = 1.0
    sources: List[str] = field(default_factory=list)

    @property
    def key(self) -> tuple:
        """Unique key for this relationship."""
        return (self.source.lower().strip(), self.target.lower().strip(), self.relation_type.lower().strip())

    def merge_with(self, other: "Relationship") -> "Relationship":
        """Merge this relationship with another."""
        merged_desc = self.description if len(self.description) >= len(other.description) else other.description
        merged_sources = list(set(self.sources + other.sources))

        return Relationship(
            source=self.source,
            target=self.target,
            relation_type=self.relation_type,
            description=merged_desc,
            weight=max(self.weight, other.weight),
            sources=merged_sources
        )


class GraphMerger:
    """
    Merge multiple KG outputs with entity deduplication.

    Handles merging of graphml files from parallel pipeline runs,
    with name-based entity deduplication and relationship consolidation.
    """

    def __init__(self):
        if not HAS_NETWORKX:
            raise ImportError("networkx is required for graph merging: pip install networkx")

    def merge_graphs(
        self,
        input_dirs: List[Path],
        output_dir: Path,
        graph_filename: str = "graph_chunk_entity_relation.graphml"
    ) -> Path:
        """
        Merge multiple graph outputs into a unified graph.

        Args:
            input_dirs: List of pipeline output directories
            output_dir: Directory for merged output
            graph_filename: Name of graphml file in each input dir

        Returns:
            Path to merged graph file
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        all_entities: Dict[str, Entity] = {}
        all_relationships: Dict[tuple, Relationship] = {}

        for input_dir in input_dirs:
            graph_file = input_dir / graph_filename
            if not graph_file.exists():
                logger.warning(f"Graph file not found: {graph_file}")
                continue

            logger.info(f"Loading graph from {graph_file}")
            entities, relationships = self._load_graphml(graph_file)

            for entity in entities:
                key = entity.name.lower().strip()
                if key in all_entities:
                    all_entities[key] = all_entities[key].merge_with(entity)
                else:
                    all_entities[key] = entity

            for rel in relationships:
                if rel.key in all_relationships:
                    all_relationships[rel.key] = all_relationships[rel.key].merge_with(rel)
                else:
                    all_relationships[rel.key] = rel

        logger.info(f"Merged: {len(all_entities)} entities, {len(all_relationships)} relationships")

        merged_graph = self._build_graph(list(all_entities.values()), list(all_relationships.values()))

        output_file = output_dir / graph_filename
        nx.write_graphml(merged_graph, output_file)
        logger.info(f"Merged graph written to {output_file}")

        self._merge_kv_stores(input_dirs, output_dir)

        return output_file

    def _load_graphml(self, graph_file: Path) -> tuple[List[Entity], List[Relationship]]:
        """Load entities and relationships from a graphml file."""
        graph = nx.read_graphml(graph_file)
        entities = []
        relationships = []

        for node_id, node_data in graph.nodes(data=True):
            entity = Entity(
                name=node_data.get("entity_name", node_id),
                entity_type=node_data.get("entity_type", "unknown"),
                description=node_data.get("description", ""),
                sources=self._parse_sources(node_data.get("source_id", "")),
                attributes={k: v for k, v in node_data.items()
                           if k not in ["entity_name", "entity_type", "description", "source_id"]}
            )
            entities.append(entity)

        for source, target, edge_data in graph.edges(data=True):
            rel = Relationship(
                source=source,
                target=target,
                relation_type=edge_data.get("relation", "related_to"),
                description=edge_data.get("description", ""),
                weight=float(edge_data.get("weight", 1.0)),
                sources=self._parse_sources(edge_data.get("source_id", ""))
            )
            relationships.append(rel)

        return entities, relationships

    def _parse_sources(self, source_str: str) -> List[str]:
        """Parse source string into list."""
        if not source_str:
            return []
        if isinstance(source_str, list):
            return source_str
        return [s.strip() for s in source_str.split(",") if s.strip()]

    def _build_graph(self, entities: List[Entity], relationships: List[Relationship]) -> nx.Graph:
        """Build a networkx graph from entities and relationships."""
        graph = nx.Graph()

        for entity in entities:
            node_attrs = {
                "entity_name": entity.name,
                "entity_type": entity.entity_type,
                "description": entity.description,
                "source_id": ",".join(entity.sources),
                **entity.attributes
            }
            graph.add_node(entity.name.lower().strip(), **node_attrs)

        for rel in relationships:
            source_key = rel.source.lower().strip()
            target_key = rel.target.lower().strip()

            if source_key not in graph.nodes:
                graph.add_node(source_key, entity_name=rel.source, entity_type="unknown")
            if target_key not in graph.nodes:
                graph.add_node(target_key, entity_name=rel.target, entity_type="unknown")

            edge_attrs = {
                "relation": rel.relation_type,
                "description": rel.description,
                "weight": rel.weight,
                "source_id": ",".join(rel.sources)
            }
            graph.add_edge(source_key, target_key, **edge_attrs)

        return graph

    def _merge_kv_stores(self, input_dirs: List[Path], output_dir: Path):
        """Merge key-value stores from parallel pipelines."""
        kv_files = [
            "kv_store_full_docs.json",
            "kv_store_text_chunks.json",
            "kv_store_llm_response_cache.json"
        ]

        for kv_file in kv_files:
            merged_data = {}

            for input_dir in input_dirs:
                kv_path = input_dir / kv_file
                if kv_path.exists():
                    try:
                        with open(kv_path, 'r') as f:
                            data = json.load(f)
                            merged_data.update(data)
                    except Exception as e:
                        logger.warning(f"Failed to load {kv_path}: {e}")

            if merged_data:
                output_path = output_dir / kv_file
                with open(output_path, 'w') as f:
                    json.dump(merged_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Merged {kv_file}: {len(merged_data)} entries")

    def dedupe_entities(self, entities: List[Entity]) -> List[Entity]:
        """Simple name-based deduplication of entities."""
        merged: Dict[str, Entity] = {}

        for entity in entities:
            name = entity.name.lower().strip()
            if name in merged:
                merged[name] = merged[name].merge_with(entity)
            else:
                merged[name] = entity

        return list(merged.values())


def merge_pipeline_outputs(
    pipeline_dirs: List[str],
    output_dir: str = "data/kg/merged"
) -> Path:
    """
    Convenience function to merge pipeline outputs.

    Args:
        pipeline_dirs: List of pipeline output directory paths
        output_dir: Path for merged output

    Returns:
        Path to merged graph file
    """
    merger = GraphMerger()
    input_paths = [Path(d) for d in pipeline_dirs]
    output_path = Path(output_dir)
    return merger.merge_graphs(input_paths, output_path)


def get_graph_stats(graph_path: Path) -> Dict[str, Any]:
    """Get statistics about a graph file."""
    if not HAS_NETWORKX:
        return {"error": "networkx not installed"}

    try:
        graph = nx.read_graphml(graph_path)
        return {
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "density": nx.density(graph) if len(graph.nodes) > 0 else 0,
            "connected_components": nx.number_connected_components(graph) if len(graph.nodes) > 0 else 0,
        }
    except Exception as e:
        return {"error": str(e)}
