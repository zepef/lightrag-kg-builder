#!/usr/bin/env python3
"""
Knowledge Graph Visualization

Generates an interactive HTML visualization of the KG using pyvis.
Dark theme, color-coded entity types, node size by degree centrality.

Usage:
    python scripts/visualize_kg.py --input data/kg --output data/kg/kg_visualization.html
    python scripts/visualize_kg.py --input data/kg  # defaults to {input}/kg_visualization.html
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import networkx as nx
    from pyvis.network import Network
except ImportError:
    print("Required: pip install networkx pyvis")
    sys.exit(1)


# Color palette for entity types (dark-theme friendly)
TYPE_COLORS = {
    "concept": "#7c3aed",    # violet
    "content": "#2563eb",    # blue
    "data": "#059669",       # emerald
    "method": "#d97706",     # amber
    "artifact": "#dc2626",   # red
    "organization": "#db2777",  # pink
    "event": "#0891b2",      # cyan
    "location": "#65a30d",   # lime
    "UNKNOWN": "#6b7280",    # gray
}


def build_visualization(input_dir: Path, output_path: Path, height: str = "900px"):
    """Build interactive KG visualization."""
    graphml_path = input_dir / "graph_chunk_entity_relation.graphml"
    if not graphml_path.exists():
        print(f"GraphML not found: {graphml_path}")
        sys.exit(1)

    G = nx.read_graphml(str(graphml_path))
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Compute degree centrality for node sizing
    centrality = nx.degree_centrality(G)
    max_centrality = max(centrality.values()) if centrality else 1

    # Create pyvis network
    net = Network(
        height=height,
        width="100%",
        bgcolor="#1a1a2e",
        font_color="#e2e8f0",
        directed=G.is_directed(),
    )

    # Physics settings for readable layout
    net.set_options(json.dumps({
        "physics": {
            "enabled": True,
            "barnesHut": {
                "gravitationalConstant": -8000,
                "centralGravity": 0.3,
                "springLength": 150,
                "springConstant": 0.04,
                "damping": 0.09,
            },
            "stabilization": {
                "iterations": 200,
                "updateInterval": 25,
            },
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 100,
            "navigationButtons": True,
            "keyboard": True,
        },
        "nodes": {
            "font": {"size": 12, "face": "Inter, sans-serif"},
            "borderWidth": 2,
            "borderWidthSelected": 4,
        },
        "edges": {
            "color": {"color": "#475569", "highlight": "#7c3aed"},
            "width": 1.5,
            "smooth": {"type": "continuous"},
        },
    }))

    # Add nodes
    for node, data in G.nodes(data=True):
        entity_type = data.get("entity_type", "UNKNOWN")
        color = TYPE_COLORS.get(entity_type, TYPE_COLORS["UNKNOWN"])

        # Size based on degree centrality (10-40 range)
        size = 10 + (centrality.get(node, 0) / max_centrality) * 30

        # Description for tooltip
        desc = data.get("description", "")
        if "<SEP>" in desc:
            desc = desc.split("<SEP>")[0]
        if len(desc) > 300:
            desc = desc[:300] + "..."

        title = (
            f"<b>{node}</b><br>"
            f"Type: {entity_type}<br>"
            f"Degree: {G.degree(node)}<br>"
            f"<hr>{desc}"
        )

        net.add_node(
            node,
            label=node if len(node) < 40 else node[:37] + "...",
            title=title,
            color=color,
            size=size,
        )

    # Add edges
    for src, tgt, data in G.edges(data=True):
        desc = data.get("description", "")
        if "<SEP>" in desc:
            desc = desc.split("<SEP>")[0]
        keywords = data.get("keywords", "")

        title = f"{src} → {tgt}<br>{keywords}<br>{desc[:200]}"

        weight = float(data.get("weight", 1))
        net.add_edge(src, tgt, title=title, width=min(weight * 1.5, 5))

    # Build legend HTML
    legend_items = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0">'
        f'<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:{color}"></span>'
        f'<span>{etype}</span></div>'
        for etype, color in TYPE_COLORS.items()
        if any(d.get("entity_type") == etype for _, d in G.nodes(data=True))
    )

    # Generate HTML and inject legend
    net.save_graph(str(output_path))

    # Inject legend overlay into the HTML
    html = output_path.read_text(encoding="utf-8")
    legend_div = (
        f'<div style="position:fixed;top:10px;right:10px;background:rgba(26,26,46,0.9);'
        f'border:1px solid #334155;border-radius:8px;padding:12px 16px;color:#e2e8f0;'
        f'font-family:Inter,sans-serif;font-size:13px;z-index:1000">'
        f'<div style="font-weight:bold;margin-bottom:8px;font-size:14px">Entity Types</div>'
        f'{legend_items}'
        f'<div style="margin-top:10px;font-size:11px;color:#94a3b8">'
        f'{G.number_of_nodes()} nodes · {G.number_of_edges()} edges</div>'
        f'</div>'
    )

    html = html.replace("</body>", f"{legend_div}</body>")
    output_path.write_text(html, encoding="utf-8")

    print(f"Visualization saved: {output_path}")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Entity types: {len(set(d.get('entity_type', 'UNKNOWN') for _, d in G.nodes(data=True)))}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Knowledge Graph")
    parser.add_argument("--input", required=True, help="KG directory with GraphML file")
    parser.add_argument("--output", default=None, help="Output HTML path (default: {input}/kg_visualization.html)")
    parser.add_argument("--height", default="900px", help="Canvas height (default: 900px)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output) if args.output else input_dir / "kg_visualization.html"

    build_visualization(input_dir, output_path, height=args.height)


if __name__ == "__main__":
    main()
