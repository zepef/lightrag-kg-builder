"""Tests for the graph merger."""

import json

import pytest
import networkx as nx
from pathlib import Path

from src.kg.merger import Entity, Relationship, GraphMerger, merge_pipeline_outputs, get_graph_stats


# ============================================================================
# Entity
# ============================================================================

class TestEntity:

    def test_merge_keeps_longer_description(self):
        e1 = Entity(name="Capital", entity_type="concept", description="Short desc")
        e2 = Entity(name="Capital", entity_type="concept", description="A much longer and more detailed description of capital")
        merged = e1.merge_with(e2)
        assert merged.description == e2.description

    def test_merge_shorter_first_picks_longer(self):
        e1 = Entity(name="X", entity_type="t", description="A much longer description here")
        e2 = Entity(name="X", entity_type="t", description="Short")
        merged = e1.merge_with(e2)
        assert merged.description == e1.description

    def test_merge_combines_sources(self):
        e1 = Entity(name="X", entity_type="t", sources=["s1", "s2"])
        e2 = Entity(name="X", entity_type="t", sources=["s2", "s3"])
        merged = e1.merge_with(e2)
        assert set(merged.sources) == {"s1", "s2", "s3"}

    def test_merge_preserves_entity_type(self):
        e1 = Entity(name="X", entity_type="concept")
        e2 = Entity(name="X", entity_type="")
        merged = e1.merge_with(e2)
        assert merged.entity_type == "concept"

    def test_merge_fills_empty_type(self):
        e1 = Entity(name="X", entity_type="")
        e2 = Entity(name="X", entity_type="data")
        merged = e1.merge_with(e2)
        assert merged.entity_type == "data"

    def test_merge_combines_attributes(self):
        e1 = Entity(name="X", entity_type="t", attributes={"a": 1})
        e2 = Entity(name="X", entity_type="t", attributes={"b": 2})
        merged = e1.merge_with(e2)
        assert merged.attributes["a"] == 1
        assert merged.attributes["b"] == 2

    def test_merge_self_attributes_win(self):
        e1 = Entity(name="X", entity_type="t", attributes={"key": "from_e1"})
        e2 = Entity(name="X", entity_type="t", attributes={"key": "from_e2"})
        merged = e1.merge_with(e2)
        assert merged.attributes["key"] == "from_e1"

    def test_merge_preserves_name(self):
        e1 = Entity(name="Capital", entity_type="concept")
        e2 = Entity(name="Capital", entity_type="concept")
        merged = e1.merge_with(e2)
        assert merged.name == "Capital"


# ============================================================================
# Relationship
# ============================================================================

class TestRelationship:

    def test_key_normalized(self):
        r = Relationship(source="  Capital  ", target=" Equity ", relation_type="IS_A")
        assert r.key == ("capital", "equity", "is_a")

    def test_key_consistency(self):
        r1 = Relationship(source="Capital", target="Equity", relation_type="is_a")
        r2 = Relationship(source="capital", target="equity", relation_type="IS_A")
        assert r1.key == r2.key

    def test_merge_keeps_longer_description(self):
        r1 = Relationship(source="A", target="B", relation_type="r", description="Short")
        r2 = Relationship(source="A", target="B", relation_type="r", description="A longer description of the relationship")
        merged = r1.merge_with(r2)
        assert merged.description == r2.description

    def test_merge_keeps_max_weight(self):
        r1 = Relationship(source="A", target="B", relation_type="r", weight=1.0)
        r2 = Relationship(source="A", target="B", relation_type="r", weight=3.5)
        merged = r1.merge_with(r2)
        assert merged.weight == 3.5

    def test_merge_max_weight_reversed(self):
        r1 = Relationship(source="A", target="B", relation_type="r", weight=5.0)
        r2 = Relationship(source="A", target="B", relation_type="r", weight=2.0)
        merged = r1.merge_with(r2)
        assert merged.weight == 5.0

    def test_merge_combines_sources(self):
        r1 = Relationship(source="A", target="B", relation_type="r", sources=["s1"])
        r2 = Relationship(source="A", target="B", relation_type="r", sources=["s2"])
        merged = r1.merge_with(r2)
        assert set(merged.sources) == {"s1", "s2"}


# ============================================================================
# GraphMerger
# ============================================================================

def _create_pipeline_dir(tmp_path: Path, name: str, nodes: list, edges: list) -> Path:
    """Create a pipeline directory with a GraphML file."""
    dir_path = tmp_path / name
    dir_path.mkdir()
    G = nx.Graph()
    for node_id, attrs in nodes:
        G.add_node(node_id, **attrs)
    for src, tgt, attrs in edges:
        G.add_edge(src, tgt, **attrs)
    nx.write_graphml(G, str(dir_path / "graph_chunk_entity_relation.graphml"))
    return dir_path


class TestGraphMerger:

    def test_merge_two_disjoint_graphs(self, tmp_path):
        d1 = _create_pipeline_dir(tmp_path, "p1",
            nodes=[("capital", {"entity_name": "Capital", "entity_type": "concept", "description": "Equity"})],
            edges=[],
        )
        d2 = _create_pipeline_dir(tmp_path, "p2",
            nodes=[("dette", {"entity_name": "Dette", "entity_type": "concept", "description": "Liability"})],
            edges=[],
        )

        merger = GraphMerger()
        output_dir = tmp_path / "merged"
        result = merger.merge_graphs([d1, d2], output_dir)

        assert result.exists()
        G = nx.read_graphml(str(result))
        assert len(G.nodes) == 2

    def test_deduplicate_same_entity(self, tmp_path):
        d1 = _create_pipeline_dir(tmp_path, "p1",
            nodes=[("capital", {"entity_name": "Capital", "entity_type": "concept", "description": "Short"})],
            edges=[],
        )
        d2 = _create_pipeline_dir(tmp_path, "p2",
            nodes=[("capital", {"entity_name": "Capital", "entity_type": "concept", "description": "A much longer description of capital"})],
            edges=[],
        )

        merger = GraphMerger()
        output_dir = tmp_path / "merged"
        result = merger.merge_graphs([d1, d2], output_dir)

        G = nx.read_graphml(str(result))
        assert len(G.nodes) == 1
        assert G.nodes["capital"]["description"] == "A much longer description of capital"

    def test_merge_edges_dedup(self, tmp_path):
        nodes = [
            ("a", {"entity_name": "A", "entity_type": "concept"}),
            ("b", {"entity_name": "B", "entity_type": "concept"}),
        ]
        d1 = _create_pipeline_dir(tmp_path, "p1",
            nodes=nodes,
            edges=[("a", "b", {"relation": "related_to", "description": "linked", "weight": 1.0})],
        )
        d2 = _create_pipeline_dir(tmp_path, "p2",
            nodes=nodes,
            edges=[("a", "b", {"relation": "related_to", "description": "strongly linked together", "weight": 2.0})],
        )

        merger = GraphMerger()
        output_dir = tmp_path / "merged"
        result = merger.merge_graphs([d1, d2], output_dir)

        G = nx.read_graphml(str(result))
        assert len(G.edges) == 1
        edge = G.edges["a", "b"]
        assert edge["description"] == "strongly linked together"
        assert float(edge["weight"]) == 2.0

    def test_missing_graph_file_skipped(self, tmp_path):
        d1 = tmp_path / "p1"
        d1.mkdir()
        # No GraphML file in this directory

        merger = GraphMerger()
        output_dir = tmp_path / "merged"
        result = merger.merge_graphs([d1], output_dir)

        G = nx.read_graphml(str(result))
        assert len(G.nodes) == 0

    def test_edges_create_missing_nodes(self, tmp_path):
        d1 = _create_pipeline_dir(tmp_path, "p1",
            nodes=[("a", {"entity_name": "A", "entity_type": "concept"})],
            edges=[("a", "b", {"relation": "links_to", "description": "test", "weight": 1.0})],
        )

        merger = GraphMerger()
        output_dir = tmp_path / "merged"
        result = merger.merge_graphs([d1], output_dir)

        G = nx.read_graphml(str(result))
        assert "a" in G.nodes
        assert "b" in G.nodes

    def test_merge_three_pipelines(self, tmp_path):
        for i, name in enumerate(["alpha", "beta", "gamma"]):
            _create_pipeline_dir(tmp_path, f"p{i+1}",
                nodes=[(name, {"entity_name": name.capitalize(), "entity_type": "concept", "description": f"Desc for {name}"})],
                edges=[],
            )
        dirs = [tmp_path / f"p{i+1}" for i in range(3)]

        merger = GraphMerger()
        output_dir = tmp_path / "merged"
        result = merger.merge_graphs(dirs, output_dir)

        G = nx.read_graphml(str(result))
        assert len(G.nodes) == 3


# ============================================================================
# KV store merging
# ============================================================================

class TestKVStoreMerge:

    def test_merge_full_docs(self, tmp_path):
        d1 = _create_pipeline_dir(tmp_path, "p1", nodes=[], edges=[])
        d2 = _create_pipeline_dir(tmp_path, "p2", nodes=[], edges=[])

        (d1 / "kv_store_full_docs.json").write_text(json.dumps({"doc1": {"content": "text1"}}))
        (d2 / "kv_store_full_docs.json").write_text(json.dumps({"doc2": {"content": "text2"}}))

        merger = GraphMerger()
        output_dir = tmp_path / "merged"
        merger.merge_graphs([d1, d2], output_dir)

        merged_kv = json.loads((output_dir / "kv_store_full_docs.json").read_text())
        assert "doc1" in merged_kv
        assert "doc2" in merged_kv

    def test_merge_text_chunks(self, tmp_path):
        d1 = _create_pipeline_dir(tmp_path, "p1", nodes=[], edges=[])
        (d1 / "kv_store_text_chunks.json").write_text(json.dumps({"c1": {"content": "chunk1"}}))

        merger = GraphMerger()
        output_dir = tmp_path / "merged"
        merger.merge_graphs([d1], output_dir)

        merged = json.loads((output_dir / "kv_store_text_chunks.json").read_text())
        assert "c1" in merged

    def test_no_kv_files_no_output(self, tmp_path):
        d1 = _create_pipeline_dir(tmp_path, "p1", nodes=[], edges=[])

        merger = GraphMerger()
        output_dir = tmp_path / "merged"
        merger.merge_graphs([d1], output_dir)

        assert not (output_dir / "kv_store_full_docs.json").exists()


# ============================================================================
# dedupe_entities method
# ============================================================================

class TestDedupeEntities:

    def test_dedup_by_name(self):
        merger = GraphMerger()
        entities = [
            Entity(name="Capital", entity_type="concept", description="Short"),
            Entity(name="capital", entity_type="concept", description="A longer description here"),
            Entity(name="CAPITAL", entity_type="concept", description="Medium length desc"),
        ]
        deduped = merger.dedupe_entities(entities)
        assert len(deduped) == 1
        assert deduped[0].description == "A longer description here"

    def test_dedup_preserves_distinct(self):
        merger = GraphMerger()
        entities = [
            Entity(name="Capital", entity_type="concept", description="Desc A"),
            Entity(name="Dette", entity_type="concept", description="Desc B"),
        ]
        deduped = merger.dedupe_entities(entities)
        assert len(deduped) == 2


# ============================================================================
# get_graph_stats
# ============================================================================

class TestGetGraphStats:

    def test_stats_from_file(self, tmp_path):
        G = nx.Graph()
        G.add_node("a")
        G.add_node("b")
        G.add_node("c")
        G.add_edge("a", "b")
        G.add_edge("b", "c")
        path = tmp_path / "test.graphml"
        nx.write_graphml(G, str(path))

        stats = get_graph_stats(path)
        assert stats["nodes"] == 3
        assert stats["edges"] == 2
        assert "density" in stats
        assert stats["connected_components"] == 1

    def test_stats_disconnected_graph(self, tmp_path):
        G = nx.Graph()
        G.add_node("a")
        G.add_node("b")
        # No edges — 2 components
        path = tmp_path / "test.graphml"
        nx.write_graphml(G, str(path))

        stats = get_graph_stats(path)
        assert stats["nodes"] == 2
        assert stats["edges"] == 0
        assert stats["connected_components"] == 2

    def test_stats_nonexistent_file(self, tmp_path):
        path = tmp_path / "nonexistent.graphml"
        stats = get_graph_stats(path)
        assert "error" in stats

    def test_stats_empty_graph(self, tmp_path):
        G = nx.Graph()
        path = tmp_path / "empty.graphml"
        nx.write_graphml(G, str(path))

        stats = get_graph_stats(path)
        assert stats["nodes"] == 0
        assert stats["edges"] == 0


# ============================================================================
# merge_pipeline_outputs convenience function
# ============================================================================

class TestMergePipelineOutputs:

    def test_convenience_function(self, tmp_path):
        d1 = _create_pipeline_dir(tmp_path, "p1",
            nodes=[("x", {"entity_name": "X", "entity_type": "concept"})],
            edges=[],
        )
        output_dir = tmp_path / "merged"
        result = merge_pipeline_outputs([str(d1)], str(output_dir))
        assert result.exists()
