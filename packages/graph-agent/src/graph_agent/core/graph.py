"""Small asynchronous graph runtime for agent-style message flow."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from graph_agent.core.edge import Edge
from graph_agent.core.node import (
    Node,
    NodeResult,
    UpstreamOutputs,
)
from graph_agent.core.message import Message, messages_from_dict, messages_to_dict
from graph_agent.core.utils import (
    _optional_str_list,
    _required_int,
    _required_list,
    _required_mapping,
    _required_str,
)
from graph_agent.runtime import EventSink, RunContext, RuntimeEventName

GRAPH_STATE_SCHEMA = "graph-agent.graph-state.v2"


@dataclass(frozen=True)
class NodeActivation:
    node: Node
    history: list[Message]
    upstream_outputs: UpstreamOutputs
    downstream_history: list[Message]
    activation_edges: list[Edge] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node.name,
            "history": messages_to_dict(self.history),
            "upstream_outputs": {
                name: message.to_dict()
                for name, message in self.upstream_outputs.items()
            },
            "downstream_history": messages_to_dict(self.downstream_history),
            "activation_edges": [edge.to_dict() for edge in self.activation_edges],
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        graph_nodes: Mapping[str, Node],
        edges_by_name: Mapping[str, Edge],
    ) -> "NodeActivation":
        if not isinstance(data, Mapping):
            raise TypeError("node activation data must be a JSON object")

        node_name = _required_str(data, "node", "node_activation.node")
        activation_edges: list[Edge] = []
        for edge_data in _required_list(
            data,
            "activation_edges",
            "node_activation.activation_edges",
        ):
            if not isinstance(edge_data, Mapping):
                raise TypeError("edge data must be a JSON object")
            serialized_edge = Edge.from_dict(edge_data)
            if serialized_edge.name not in edges_by_name:
                raise KeyError(
                    "serialized graph state references unknown edge: "
                    f"{serialized_edge.name}"
                )
            edge = edges_by_name[serialized_edge.name]
            if edge.source != serialized_edge.source or edge.target != serialized_edge.target:
                raise ValueError(
                    "serialized graph state edge does not match graph: "
                    f"{serialized_edge.name}"
                )
            if bool(edge_data.get("active", False)) != (edge.active is not None):
                raise ValueError(
                    "serialized graph state edge predicate shape does not match graph: "
                    f"{serialized_edge.name}"
                )
            activation_edges.append(edge)

        return cls(
            node=graph_nodes[node_name],
            history=messages_from_dict(
                _required_list(data, "history", "node_activation.history")
            ),
            upstream_outputs=messages_by_name_from_dict(
                _required_mapping(
                    data,
                    "upstream_outputs",
                    "node_activation.upstream_outputs",
                )
            ),
            downstream_history=messages_from_dict(
                _required_list(
                    data,
                    "downstream_history",
                    "node_activation.downstream_history",
                )
            ),
            activation_edges=activation_edges,
        )


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.source,
            "activation": self.activation.to_dict(),
            "output": self.output.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        graph_nodes: Mapping[str, Node],
        edges_by_name: Mapping[str, Edge],
    ) -> "CompletedNode":
        if not isinstance(data, Mapping):
            raise TypeError("completed node data must be a JSON object")

        node_name = _required_str(data, "node", "completed_node.node")
        activation = NodeActivation.from_dict(
            _required_mapping(data, "activation", "completed_node.activation"),
            graph_nodes,
            edges_by_name,
        )
        if activation.node.name != node_name:
            raise ValueError("completed node must match activation node")

        node = graph_nodes[node_name]
        output = Message.from_dict(
            _required_mapping(data, "output", "completed_node.output")
        )
        return cls(activation, NodeResult(node, output))


@dataclass
class NodeState:
    depends_on: int = 0
    finished_dependency: int = 0
    in_edges: list[Edge] = field(default_factory=list)
    out_edges: list[Edge] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    dependency_results: dict[str, CompletedNode] = field(default_factory=dict)
    completed: CompletedNode | None = None

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

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "depends_on": self.depends_on,
            "finished_dependency": self.finished_dependency,
            "in_edges": [edge.to_dict() for edge in self.in_edges],
            "out_edges": [edge.to_dict() for edge in self.out_edges],
            "dependency_results": {
                edge_name: result.to_dict()
                for edge_name, result in self.dependency_results.items()
            },
        }
        if self.completed is not None:
            data["completed"] = self.completed.to_dict()
        return data

    def load_dict(
        self,
        data: Mapping[str, Any],
        graph_nodes: Mapping[str, Node],
        edges_by_name: Mapping[str, Edge],
    ) -> None:
        if not isinstance(data, Mapping):
            raise TypeError("node state data must be a JSON object")

        self._validate_serialized_edges(
            "in_edges",
            self.in_edges,
            _required_list(data, "in_edges", "node_state.in_edges"),
        )
        self._validate_serialized_edges(
            "out_edges",
            self.out_edges,
            _required_list(data, "out_edges", "node_state.out_edges"),
        )
        depends_on = _required_int(data, "depends_on", "node_state.depends_on")
        if depends_on != self.depends_on:
            raise ValueError("serialized node state does not match graph dependencies")

        self.finished_dependency = _required_int(
            data,
            "finished_dependency",
            "node_state.finished_dependency",
        )
        self.dependency_results = {
            edge_name: CompletedNode.from_dict(result, graph_nodes, edges_by_name)
            for edge_name, result in _required_mapping(
                data,
                "dependency_results",
                "node_state.dependency_results",
            ).items()
        }
        self._validate_dependency_maps()
        completed_data = data.get("completed")
        self.completed = (
            CompletedNode.from_dict(completed_data, graph_nodes, edges_by_name)
            if completed_data is not None
            else None
        )

    def _validate_serialized_edges(
        self,
        field_name: str,
        edges: list[Edge],
        serialized_edges: list[Any],
    ) -> None:
        if len(serialized_edges) != len(edges):
            raise ValueError(f"serialized node state {field_name} do not match graph")
        for edge, serialized_edge in zip(edges, serialized_edges):
            if not isinstance(serialized_edge, Mapping):
                raise TypeError(f"node_state.{field_name} items must be JSON objects")
            name = _required_str(serialized_edge, "name", f"node_state.{field_name}.name")
            source = _required_str(
                serialized_edge,
                "source",
                f"node_state.{field_name}.source",
            )
            target = _required_str(
                serialized_edge,
                "target",
                f"node_state.{field_name}.target",
            )
            active = bool(serialized_edge.get("active", False))
            if (
                edge.name != name
                or edge.source != source
                or edge.target != target
                or (edge.active is not None) != active
            ):
                raise ValueError(
                    f"serialized node state {field_name} do not match graph"
                )

    def _validate_dependency_maps(self) -> None:
        inbound_edges = {edge.name: edge for edge in self.in_edges}
        result_edges = set(self.dependency_results)
        unknown_edges = result_edges.difference(inbound_edges)
        if unknown_edges:
            raise ValueError(
                "serialized node state dependency maps reference unknown edges: "
                f"{unknown_edges}"
            )
        if self.finished_dependency != len(result_edges):
            raise ValueError(
                "serialized node state finished dependency count does not match "
                "dependency results"
            )
        for edge_name, completed_node in self.dependency_results.items():
            edge = inbound_edges[edge_name]
            if completed_node.source != edge.source:
                raise ValueError(
                    "serialized node state dependency result source does not match "
                    "graph edge"
                )


class GraphRunStatus(StrEnum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class GraphRunResult:
    output: list[Message]
    history: list[Message] = field(default_factory=list)
    status: GraphRunStatus = GraphRunStatus.COMPLETED
    state: dict[str, Any] | None = None


class Graph:
    def __init__(self, start_node: str, name: str = "graph") -> None:
        self.name = name
        self.nodes: dict[str, Node] = {}
        self.node_states: dict[str, NodeState] = {}
        self.edges: list[Edge] = []
        self.input_messages: list[Message] = []
        self.start_node = start_node
        self.view_layout_direction = "horizontal"
        self.status: GraphRunStatus | None = None
        # Completed nodes whose outputs still need to be propagated on resume.
        self.pending_completed_nodes: list[str] = []
        self._loaded_from_state = False
        self._has_run = False

    # --- Runtime activation flow ---

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

    async def _start_activation(
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
        extra = self.node_states[activation.node.name].extra
        return asyncio.create_task(
            activation.node.invoke(
                ctx.child_for_node(activation.node.name),
                list(activation.history),
                dict(activation.upstream_outputs),
                **dict(extra),
            )
        )

    def active_next_nodes(
        self,
        completed_node: CompletedNode,
    ) -> list[NodeActivation]:
        source_state = self.node_states[completed_node.source]
        affected_targets: list[str] = []
        for edge in source_state.out_edges:
            downstream_node = self.nodes[edge.target]
            if not edge.can_activate(completed_node.result, downstream_node):
                continue

            target_state = self.node_states[edge.target]
            if edge.name not in target_state.dependency_results:
                target_state.finished_dependency += 1
            target_state.dependency_results[edge.name] = completed_node

            if edge.target not in affected_targets:
                affected_targets.append(edge.target)

        to_activate: list[NodeActivation] = []
        for node_name in affected_targets:
            state = self.node_states[node_name]
            if state.finished_dependency < state.depends_on:
                continue

            upstream_outputs = {
                edge.name: state.dependency_results[edge.name].output
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
                        if edge.name in state.dependency_results
                    ],
                )
            )
        return to_activate

    async def _complete_task(
        self,
        ctx: RunContext,
        task: asyncio.Task[NodeResult],
        activation: NodeActivation,
    ) -> CompletedNode:
        result = task.result()
        completed_node = CompletedNode(
            activation,
            result,
        )
        self.node_states[completed_node.source].completed = completed_node
        await ctx.emit(
            RuntimeEventName.NODE_FINISHED,
            node=result.node.name,
            output=result.output,
        )
        return completed_node

    async def _collect_completed_nodes(
        self,
        ctx: RunContext,
        running_tasks: dict[asyncio.Task[NodeResult], NodeActivation],
        tasks: Iterable[asyncio.Task[NodeResult]],
    ) -> list[CompletedNode]:
        completed_nodes: list[CompletedNode] = []
        for task in tasks:
            activation = running_tasks.pop(task)
            completed_nodes.append(await self._complete_task(ctx, task, activation))
        return completed_nodes

    async def _drain_running_tasks(
        self,
        ctx: RunContext,
        running_tasks: dict[asyncio.Task[NodeResult], NodeActivation],
    ) -> list[CompletedNode]:
        if not running_tasks:
            return []

        done, _ = await asyncio.wait(
            running_tasks,
            return_when=asyncio.ALL_COMPLETED,
        )
        return await self._collect_completed_nodes(ctx, running_tasks, done)

    async def _cancel_running_tasks(
        self,
        running_tasks: dict[asyncio.Task[NodeResult], NodeActivation],
    ) -> None:
        if not running_tasks:
            return

        tasks = list(running_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        running_tasks.clear()

    def _completed_node_result(
        self,
        completed_node: CompletedNode,
    ) -> tuple[list[Message], list[Message]]:
        return (
            [completed_node.output],
            [*completed_node.downstream_history, completed_node.output],
        )

    async def _run_activation_loop(
        self,
        ctx: RunContext,
        initial_activations: list[NodeActivation],
        output: list[Message] | None = None,
        history: list[Message] | None = None,
    ) -> GraphRunResult:
        running_tasks: dict[asyncio.Task[NodeResult], NodeActivation] = {}
        try:
            await self._start_ready_activations(
                ctx,
                running_tasks,
                initial_activations
            )

            while running_tasks:
                done, _ = await asyncio.wait(
                    running_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                completed_nodes = await self._collect_completed_nodes(
                    ctx,
                    running_tasks,
                    done,
                )

                output, history = self._completed_node_result(completed_nodes[-1])

                if ctx.cancelled:
                    drained_nodes = await self._drain_running_tasks(ctx, running_tasks)
                    for completed_node in drained_nodes:
                        output, history = self._completed_node_result(completed_node)
                    self.status = GraphRunStatus.CANCELLED
                    self.pending_completed_nodes = [
                        completed_node.source
                        for completed_node in [*completed_nodes, *drained_nodes]
                    ]
                    state = self.state_to_dict()
                    await ctx.emit(
                        RuntimeEventName.GRAPH_CANCELLED,
                        graph=self.name,
                        state=state,
                    )
                    return GraphRunResult(
                        output or [],
                        history or [],
                        status=GraphRunStatus.CANCELLED,
                        state=state,
                    )

                for completed_node in completed_nodes:
                    ready_activations = self.active_next_nodes(completed_node)
                    await self._start_ready_activations(
                        ctx,
                        running_tasks,
                        ready_activations,
                    )
        except Exception as exc:
            await self._cancel_running_tasks(running_tasks)
            self.status = GraphRunStatus.FAILED
            self.pending_completed_nodes = []
            await ctx.emit(
                RuntimeEventName.GRAPH_FAILED,
                graph=self.name,
                state=self.state_to_dict(),
                error={
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            )
            raise
        except BaseException:
            await self._cancel_running_tasks(running_tasks)
            raise

        await ctx.emit(RuntimeEventName.GRAPH_FINISHED, graph=self.name)
        
        if output is None:
            raise RuntimeError(f"graph {self.name} did not produce output")
        if history is None:
            raise RuntimeError(f"graph {self.name} did not produce history")
        
        self.status = GraphRunStatus.COMPLETED
        self.pending_completed_nodes = []
        
        return GraphRunResult(
            output,
            history,
            status=GraphRunStatus.COMPLETED,
            state=self.state_to_dict(),
        )

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
        if self._loaded_from_state:
            raise RuntimeError(
                f"graph {self.name} was loaded from state; use resume()"
            )

        self._has_run = True
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

        return await self._run_activation_loop(
            ctx, [activation],
        )

    async def resume(
        self,
        event_sink: EventSink | None = None,
    ) -> GraphRunResult:
        if not self.node_states:
            raise RuntimeError("graph must be built before it can resume")
        if self._has_run:
            raise RuntimeError(f"graph {self.name} has already run")
        if not self._loaded_from_state:
            raise RuntimeError(f"graph {self.name} was not loaded from state")
        if self.status == GraphRunStatus.COMPLETED:
            raise RuntimeError(f"graph {self.name} state is already completed")
        if self.status != GraphRunStatus.CANCELLED:
            raise RuntimeError(
                f"graph {self.name} state cannot resume without cancelled status"
            )
        if not self.pending_completed_nodes:
            raise RuntimeError(
                f"graph {self.name} state has no pending completed nodes"
            )

        self._has_run = True
        ctx = RunContext(event_sink=event_sink)
        await ctx.emit(RuntimeEventName.GRAPH_STARTED, graph=self.name)

        pending_nodes = [
            self.node_states[node_name].completed
            for node_name in self.pending_completed_nodes
        ]
        completed_nodes = [
            completed_node
            for completed_node in pending_nodes
            if completed_node is not None
        ]
        self.pending_completed_nodes = []

        initial_activations: list[NodeActivation] = []
        output: list[Message] | None = None
        history: list[Message] | None = None

        for completed_node in completed_nodes:
            output, history = self._completed_node_result(completed_node)
            initial_activations.extend(self.active_next_nodes(completed_node))

        return await self._run_activation_loop(
            ctx,
            initial_activations,
            output=output,
            history=history,
        )

    async def _start_ready_activations(
        self,
        ctx: RunContext,
        running_tasks: dict[asyncio.Task[NodeResult], NodeActivation],
        activations: list[NodeActivation]
    ) -> None:
        if not activations:
            return
        
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
        
        for activation in activations:
            next_task = await self._start_activation(ctx, activation)
            running_tasks[next_task] = activation

    def state_to_dict(self) -> dict[str, Any]:
        return {
            "schema": GRAPH_STATE_SCHEMA,
            "graph": self.name,
            "start_node": self.start_node,
            "status": self.status.value if self.status is not None else None,
            "pending_completed_nodes": list(self.pending_completed_nodes),
            "input_messages": messages_to_dict(self.input_messages),
            "node_states": {
                node_name: state.to_dict()
                for node_name, state in self.node_states.items()
            },
        }

    def load_state(self, data: Mapping[str, Any]) -> "Graph":
        if not isinstance(data, Mapping):
            raise TypeError("graph state data must be a JSON object")

        schema = _required_str(data, "schema", "graph_state.schema")
        if schema != GRAPH_STATE_SCHEMA:
            raise ValueError(f"unsupported graph state schema: {schema}")

        start_node = _required_str(data, "start_node", "graph_state.start_node")
        if start_node != self.start_node:
            raise ValueError("serialized graph state start node does not match graph")

        graph_name = _required_str(data, "graph", "graph_state.graph")
        self.name = graph_name
        self.input_messages = messages_from_dict(
            _required_list(data, "input_messages", "graph_state.input_messages")
        )
        self.status = _optional_graph_run_status(
            data,
            "status",
            "graph_state.status",
        )
        self.pending_completed_nodes = _optional_str_list(
            data,
            "pending_completed_nodes",
            "graph_state.pending_completed_nodes",
        )

        node_states_data = _required_mapping(
            data,
            "node_states",
            "graph_state.node_states",
        )
        unknown_nodes = set(node_states_data).difference(self.node_states)
        if unknown_nodes:
            raise KeyError(f"serialized graph state has unknown nodes: {unknown_nodes}")

        missing_nodes = set(self.node_states).difference(node_states_data)
        if missing_nodes:
            raise KeyError(f"serialized graph state is missing nodes: {missing_nodes}")

        edges_by_name = {edge.name: edge for edge in self.edges}
        for node_name, state_data in node_states_data.items():
            self.node_states[node_name].load_dict(
                state_data,
                self.nodes,
                edges_by_name,
            )
        self._validate_pending_completed_nodes()
        self._loaded_from_state = True
        return self

    def _validate_pending_completed_nodes(self) -> None:
        seen_nodes: set[str] = set()
        for node_name in self.pending_completed_nodes:
            if node_name in seen_nodes:
                raise ValueError(
                    "serialized graph state has duplicate pending completed nodes"
                )
            seen_nodes.add(node_name)
            if node_name not in self.node_states:
                raise KeyError(
                    "serialized graph state references unknown pending node: "
                    f"{node_name}"
                )
            if self.node_states[node_name].completed is None:
                raise ValueError(
                    "serialized graph state pending node is not completed: "
                    f"{node_name}"
                )
        if self.status == GraphRunStatus.COMPLETED and self.pending_completed_nodes:
            raise ValueError(
                "completed graph state cannot have pending completed nodes"
            )


def messages_by_name_from_dict(data: Mapping[str, Any]) -> UpstreamOutputs:
    messages: UpstreamOutputs = {}
    for name, message_data in data.items():
        if not isinstance(name, str):
            raise TypeError("message map keys must be strings")
        if not isinstance(message_data, Mapping):
            raise TypeError(f"messages.{name} must be a JSON object")
        messages[name] = Message.from_dict(message_data)
    return messages


def _optional_graph_run_status(
    data: Mapping[str, Any],
    key: str,
    path: str,
) -> GraphRunStatus | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a string")
    try:
        return GraphRunStatus(value)
    except ValueError as exc:
        raise ValueError(f"unsupported graph run status: {value}") from exc
