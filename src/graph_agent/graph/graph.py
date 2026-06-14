"""Small asynchronous graph runtime for agent-style message flow."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from graph_agent.graph.edge import (
    Edge,
    EdgePredicate,
)
from graph_agent.graph.node import (
    Node,
    NodeResult,
    UpstreamOutputs,
)
from graph_agent.message import Message
from graph_agent.runtime import EventSink, RunContext, RuntimeEventName


@dataclass(frozen=True)
class NodeActivation:
    node: Node
    history: list[Message]
    upstream_outputs: UpstreamOutputs
    downstream_history: list[Message]
    activation_edges: list[Edge] = field(default_factory=list)


@dataclass(frozen=True)
class CompletedNode:
    activation: NodeActivation
    result: NodeResult

    @property
    def node(self) -> Node:
        return self.activation.node

    @property
    def history(self) -> list[Message]:
        return self.activation.history

    @property
    def upstream_outputs(self) -> UpstreamOutputs:
        return self.activation.upstream_outputs

    @property
    def downstream_history(self) -> list[Message]:
        return self.activation.downstream_history

    @property
    def source(self) -> str:
        return self.result.node.name

    @property
    def output(self) -> Message:
        return self.result.output


@dataclass
class NodeState:
    depends_on: int = 0
    finished_dependency: int = 0
    in_edges: list[Edge] = field(default_factory=list)
    out_edges: list[Edge] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    dependency_results: dict[str, CompletedNode] = field(default_factory=dict)
    dependency_outputs: UpstreamOutputs = field(default_factory=dict)

    def init_from_edges(
        self,
        in_edges: list[Edge],
        out_edges: list[Edge],
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        self.in_edges = list(in_edges)
        self.out_edges = list(out_edges)
        self.depends_on = len({edge.name for edge in self.in_edges})
        self.extra = dict(extra or {})
        self.extra.setdefault("tools", ())


@dataclass(frozen=True)
class GraphRunResult:
    output: list[Message]
    history: list[Message] = field(default_factory=list)


class Graph:
    def __init__(self, start_node: str, name: str = "graph") -> None:
        self.name = name
        self.nodes: dict[str, Node] = {}
        self.node_states: dict[str, NodeState] = {}
        self.edges: list[Edge] = []
        self.input_messages: list[Message] = []
        self.start_node = start_node
        self._has_run = False

    def _dependency_history(self, state: NodeState) -> list[Message]:
        history: list[Message] = []
        seen_sources: set[str] = set()
        for edge in state.in_edges:
            if edge.source in seen_sources:
                continue

            dependency = state.dependency_results[edge.name]
            seen_sources.add(edge.source)
            for message in dependency.downstream_history:
                if message not in history:
                    history.append(message)
        return history

    async def _invoke_activation(
        self,
        ctx: RunContext,
        activation: NodeActivation,
    ) -> NodeResult:
        extra = self.node_states[activation.node.name].extra
        return await activation.node.invoke(
            ctx,
            list(activation.history),
            dict(activation.upstream_outputs),
            **dict(extra),
        )

    async def _schedule_activation(
        self,
        ctx: RunContext,
        activation: NodeActivation,
    ) -> asyncio.Task[NodeResult]:
        if activation.activation_edges:
            await ctx.emit(
                RuntimeEventName.NODE_ACTIVATED,
                node=activation.node.name,
                target=activation.node.name,
                edges=[
                    {
                        "name": edge.name,
                        "source": edge.source,
                        "target": edge.target,
                    }
                    for edge in activation.activation_edges
                ],
                history=list(activation.history),
                upstream_outputs=dict(activation.upstream_outputs),
            )
        await ctx.emit(
            RuntimeEventName.NODE_STARTED,
            node=activation.node.name,
            history=list(activation.history),
        )
        return asyncio.create_task(
            self._invoke_activation(
                ctx.child_for_node(activation.node.name),
                activation,
            )
        )

    def _propagate_completed_node(self, completed_node: CompletedNode) -> list[str]:
        source_state = self.node_states[completed_node.source]
        affected_targets: list[str] = []
        for edge in source_state.out_edges:
            downstream_node = self.nodes[edge.target]
            if not edge.can_activate(completed_node.result, downstream_node):
                continue

            target_state = self.node_states[edge.target]
            # TODO: Will the edge active a node multiple times?
            if edge.name not in target_state.dependency_results:
                target_state.finished_dependency += 1
            target_state.dependency_results[edge.name] = completed_node
            target_state.dependency_outputs[edge.name] = completed_node.result.output

            if edge.target not in affected_targets:
                affected_targets.append(edge.target)
        return affected_targets

    def _activate_ready_targets(
        self,
        affected_targets: list[str],
    ) -> list[NodeActivation]:
        to_activate: list[NodeActivation] = []
        activated_targets: set[str] = set()
        for node_name in affected_targets:
            if node_name in activated_targets:
                continue
            activated_targets.add(node_name)

            state = self.node_states[node_name]
            if state.finished_dependency < state.depends_on:
                continue

            upstream_outputs = {
                edge.name: state.dependency_outputs[edge.name]
                for edge in state.in_edges
            }
            history = self._dependency_history(state)
            downstream_history = self.nodes[node_name].prepare_downstream_history(
                upstream_outputs,
                history,
            )

            to_activate.append(
                NodeActivation(
                    node=self.nodes[node_name],
                    history=history,
                    upstream_outputs=upstream_outputs,
                    downstream_history=downstream_history,
                    activation_edges=[
                        edge
                        for edge in state.in_edges
                        if edge.name in state.dependency_outputs
                    ],
                )
            )
        return to_activate

    def active_next_nodes(
        self,
        completed_node: CompletedNode,
    ) -> list[NodeActivation]:
        affected_targets = self._propagate_completed_node(completed_node)
        return self._activate_ready_targets(affected_targets)

    async def run(
        self,
        event_sink: EventSink | None = None,
    ) -> GraphRunResult:
        if self.start_node is None:
            raise KeyError("graph must have a start node")
        if self.start_node not in self.nodes:
            raise KeyError(f"graph start node not found: {self.start_node}")
        if not self.node_states:
            raise RuntimeError("graph must be built before it can run")
        if self._has_run:
            raise RuntimeError(f"graph {self.name} has already run")

        self._has_run = True
        running_tasks: dict[asyncio.Task[NodeResult], NodeActivation] = {}
        waits_for_activation_rounds = bool(
            getattr(event_sink, "waits_for_activation_rounds", False)
        )
        handles_activation_ready = waits_for_activation_rounds or bool(
            getattr(event_sink, "handles_activation_ready", False)
        )
        ctx = RunContext(event_sink=event_sink)
        await ctx.emit(RuntimeEventName.GRAPH_STARTED, graph=self.name)

        start_history = list(self.input_messages)
        node = self.nodes[self.start_node]
        upstream_outputs: UpstreamOutputs = {}
        downstream_history = node.prepare_downstream_history(
            dict(upstream_outputs),
            list(start_history),
        )
        activation = NodeActivation(
            node=node,
            history=list(start_history),
            upstream_outputs=upstream_outputs,
            downstream_history=list(downstream_history),
        )

        if handles_activation_ready:
            await self._emit_activation_ready(ctx, [activation])

        next_task = await self._schedule_activation(ctx, activation)
        running_tasks[next_task] = activation

        output: list[Message] | None = None
        history: list[Message] | None = None

        while running_tasks:
            return_when = (
                asyncio.ALL_COMPLETED
                if waits_for_activation_rounds
                else asyncio.FIRST_COMPLETED
            )
            done, _ = await asyncio.wait(
                running_tasks,
                return_when=return_when,
            )

            next_activations: list[NodeActivation] = []
            for task in done:
                activation = running_tasks.pop(task)
                result = task.result()
                completed_node = CompletedNode(
                    activation,
                    result,
                )
                output = [result.output]
                history = [*activation.downstream_history, result.output]
                await ctx.emit(
                    RuntimeEventName.NODE_FINISHED,
                    node=result.node.name,
                    output=result.output,
                )

                ready_activations = self.active_next_nodes(completed_node)
                if not waits_for_activation_rounds:
                    if ready_activations and handles_activation_ready:
                        await self._emit_activation_ready(ctx, ready_activations)
                    for activation in ready_activations:
                        next_task = await self._schedule_activation(ctx, activation)
                        running_tasks[next_task] = activation
                else:
                    next_activations.extend(ready_activations)

            if next_activations and handles_activation_ready:
                await self._emit_activation_ready(ctx, next_activations)

            for activation in next_activations:
                next_task = await self._schedule_activation(ctx, activation)
                running_tasks[next_task] = activation

        await ctx.emit(RuntimeEventName.GRAPH_FINISHED, graph=self.name)
        if output is None:
            raise RuntimeError(f"graph {self.name} did not produce output")
        if history is None:
            raise RuntimeError(f"graph {self.name} did not produce history")
        return GraphRunResult(output, history)

    async def _emit_activation_ready(
        self,
        ctx: RunContext,
        activations: list[NodeActivation],
    ) -> None:
        await ctx.emit(
            RuntimeEventName.ACTIVATION_READY,
            store=False,
            nodes=[activation.node.name for activation in activations],
            edges=[
                {
                    "name": edge.name,
                    "source": edge.source,
                    "target": edge.target,
                }
                for activation in activations
                for edge in activation.activation_edges
            ],
        )


class GraphBuilder:
    def __init__(self, name: str = "graph") -> None:
        self.name = name
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.input_messages: list[Message] = []
        self.start_node: str | None = None

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
        if self.start_node is None:
            raise KeyError("graph must have a start node")
        if self.start_node not in self.nodes:
            raise KeyError(f"graph start node not found: {self.start_node}")
        
        graph = Graph(self.start_node, self.name)
        graph.nodes = dict(self.nodes)
        graph.edges = list(self.edges)
        graph.input_messages = list(self.input_messages)
        self._prepare_node_states(graph)
        
        return graph
