"""Graph builder for assembling graph runtime instances."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from graph_agent.core.edge import Edge, EdgePredicate
from graph_agent.core.graph import Graph, NodeState
from graph_agent.core.message import Message
from graph_agent.core.node import Node
from graph_agent.core.utils import _required_str


class GraphBuilder:
    def __init__(self, name: str = "graph") -> None:
        self.name = name
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.input_messages: list[Message] = []
        self.start_node: str | None = None
        self.serialized_state: Mapping[str, Any] | None = None

    def _prepare_node_states(self, graph: Graph) -> None:
        node_names = set(graph.nodes)
        if graph.start_node is not None:
            node_names.add(graph.start_node)
        for edge in graph.edges:
            node_names.add(edge.source)
            node_names.add(edge.target)

        graph.node_states = {node_name: NodeState() for node_name in node_names}
        in_edges_by_node: dict[str, list[Edge]] = {
            node_name: [] for node_name in node_names
        }
        out_edges_by_node: dict[str, list[Edge]] = {
            node_name: [] for node_name in node_names
        }
        for edge in graph.edges:
            out_edges_by_node[edge.source].append(edge)
            in_edges_by_node[edge.target].append(edge)

        for node_name, state in graph.node_states.items():
            node = graph.nodes.get(node_name)
            extra = (
                self._node_extra_from_edges(
                    node,
                    in_edges_by_node[node_name],
                    out_edges_by_node[node_name],
                    graph.nodes,
                )
                if node is not None
                else {}
            )
            state.init_from_edges(
                in_edges_by_node[node_name],
                out_edges_by_node[node_name],
                extra=extra,
            )

    def _node_extra_from_edges(
        self,
        node: Node,
        in_edges: list[Edge],
        out_edges: list[Edge],
        graph_nodes: Mapping[str, Node],
    ) -> Mapping[str, Any]:
        extra = node.init_from_edges(list(in_edges), list(out_edges), graph_nodes)
        if not isinstance(extra, Mapping):
            raise TypeError("node init_from_edges must return a mapping")
        return extra

    def input(self, input_messages: list[Message]) -> "GraphBuilder":
        self.input_messages = list(input_messages)
        return self

    def state(self, serialized_state: Mapping[str, Any]) -> "GraphBuilder":
        self.serialized_state = serialized_state
        return self

    def start(self, node_name: str) -> "GraphBuilder":
        if self.start_node not in {None, node_name}:
            raise ValueError(f"graph already has start node {self.start_node}")
        self.start_node = node_name
        return self

    def node(self, node: Node) -> "GraphBuilder":
        if node.name in self.nodes:
            raise KeyError(f"Node with name {node.name} already exists in graph")
        self.nodes[node.name] = node
        return self

    def edge(
        self,
        source: str,
        target: str,
        name: str,
        active: EdgePredicate | None = None,
    ) -> "GraphBuilder":
        if any(edge.name == name for edge in self.edges):
            raise KeyError(f"Edge with name {name} already exists in graph")
        self.edges.append(
            Edge(
                name=name,
                source=source,
                target=target,
                active=active,
            )
        )
        return self

    def build(self) -> Graph:
        start_node = self.start_node
        if start_node is None and self.serialized_state is not None:
            start_node = _required_str(
                self.serialized_state,
                "start_node",
                "graph_state.start_node",
            )
        if start_node is None:
            raise KeyError("graph must have a start node")
        if start_node not in self.nodes:
            raise KeyError(f"graph start node not found: {start_node}")

        graph = Graph(start_node, self.name)
        graph.nodes = dict(self.nodes)
        graph.edges = list(self.edges)
        graph.input_messages = list(self.input_messages)
        self._prepare_node_states(graph)
        if self.serialized_state is not None:
            graph.load_state(self.serialized_state)

        return graph
