"""Mermaid rendering for graph-agent graphs."""

from __future__ import annotations

from graph_agent.core import Graph


def _escape_label(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace('"', "#quot;")
        .replace("\n", "<br/>")
    )


def to_mermaid(graph: Graph) -> str:
    """Render a graph-agent graph as a Mermaid flowchart."""
    node_names = set(graph.nodes)
    for edge in graph.edges:
        node_names.add(edge.source)
        node_names.add(edge.target)
    if graph.start_node is not None:
        node_names.add(graph.start_node)

    ordered_node_names = sorted(node_names)
    node_ids = {
        node_name: f"n{index}"
        for index, node_name in enumerate(ordered_node_names)
    }

    lines = ["flowchart TD"]
    for node_name in ordered_node_names:
        node = graph.nodes.get(node_name)
        label = node_name
        if node is not None:
            label = f"{node_name}\n{node.kind().value}"
        lines.append(f'    {node_ids[node_name]}["{_escape_label(label)}"]')

    if graph.start_node is not None:
        lines.append("    graph_start((start))")
        lines.append(f"    graph_start --> {node_ids[graph.start_node]}")

    for edge in graph.edges:
        source = node_ids[edge.source]
        target = node_ids[edge.target]
        label = _escape_label(edge.name)
        lines.append(f'    {source} -- "{label}" --> {target}')

    return "\n".join(lines)
