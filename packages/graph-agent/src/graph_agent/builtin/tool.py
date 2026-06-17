"""Tool registry and execution helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from graph_agent.core.edge import Edge, ToolSchemaProvider
from graph_agent.core.message import Message, MessageRole, ToolCallBlock, ToolResultBlock
from graph_agent.core.node import Node, NodeKind, NodeResult, UpstreamOutputs
from graph_agent.runtime import RunContext


@dataclass(frozen=True)
class ToolSchema:
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Tool(Protocol):
    def schema(self) -> ToolSchema:
        raise NotImplementedError

    async def invoke(self, arguments: dict[str, Any]) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class FunctionTool:
    name: str
    handler: Callable[[dict[str, Any]], str]
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=dict(self.parameters),
        )

    async def invoke(self, arguments: dict[str, Any]) -> str:
        return self.handler(arguments)


class ToolRegistry:
    def __init__(self) -> None:
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        name = tool.schema().name
        if name in self.tools:
            raise ValueError(f"tool already registered: {name}")
        self.tools[name] = tool

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def schemas(self) -> tuple[ToolSchema, ...]:
        return tuple(tool.schema() for tool in self.tools.values())


class ToolExecutor:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    async def execute_call(self, call: ToolCallBlock) -> Message:
        tool = self.registry.get(call.tool_name)
        if tool is None:
            return self._error(call, f"tool not found: {call.tool_name}")

        try:
            output = await tool.invoke(dict(call.arguments))
        except Exception as exc:
            return self._error(call, f"tool failed: {exc}")

        if not isinstance(output, str):
            return self._error(
                call,
                f"tool output must be a string, got {type(output).__name__}",
            )

        return Message.tool_result(
            ToolResultBlock(
                call_id=call.call_id,
                tool_name=call.tool_name,
                content=output,
            )
        )

    def _error(self, call: ToolCallBlock, content: str) -> Message:
        return Message.tool_result(
            ToolResultBlock(
                call_id=call.call_id,
                tool_name=call.tool_name,
                content=content,
                is_error=True,
            )
        )


ToolCallPredicate = Callable[[NodeResult, Edge, Node | None], bool]


def matches_tool_call(
    tool_schemas: Iterable[ToolSchema] | None = None,
) -> ToolCallPredicate:
    tool_names = (
        None
        if tool_schemas is None
        else frozenset(schema.name for schema in tool_schemas)
    )

    def predicate(result: NodeResult, edge: Edge, downstream_node: Node | None) -> bool:
        active_tool_names = tool_names
        if active_tool_names is None:
            if not isinstance(downstream_node, ToolSchemaProvider):
                return False
            active_tool_names = frozenset(
                schema.name for schema in downstream_node.available_tool_schemas()
            )
        return any(
            call.tool_name in active_tool_names
            for call in result.output.tool_calls()
        )

    return predicate


def matches_any_tool_call_for_downstream(
    result: NodeResult,
    edge: Edge,
    downstream_node: Node | None,
) -> bool:
    if not isinstance(downstream_node, ToolSchemaProvider):
        return False

    downstream_tool_names = frozenset(
        schema.name for schema in downstream_node.available_tool_schemas()
    )
    return any(
        call.tool_name in downstream_tool_names
        for call in result.output.tool_calls()
    )


class ToolCallNode(Node):
    """Graph node wrapper that executes tool calls from upstream messages."""

    def __init__(self, name: str = "tools", *tools: Tool) -> None:
        self.name = name
        self.registry = ToolRegistry()
        self.executor = ToolExecutor(self.registry)

        for tool in tools:
            self.registry.register(tool)

    def register_tool(self, tool: Tool) -> None:
        self.registry.register(tool)

    def available_tool_schemas(self) -> tuple[ToolSchema, ...]:
        return self.registry.schemas()

    async def invoke(
        self,
        ctx: RunContext,
        history: list[Message],
        upstream_outputs: UpstreamOutputs,
        **extra: Any,
    ) -> NodeResult:
        blocks = [
            result_block
            for message in upstream_outputs.values()
            for call in message.tool_calls()
            if self.registry.get(call.tool_name) is not None
            for result_block in (await self.executor.execute_call(call)).blocks
            if isinstance(result_block, ToolResultBlock)
        ]

        return NodeResult(self, Message(MessageRole.TOOL, tuple(blocks)))

    def kind(self) -> NodeKind:
        return NodeKind.LLM


__all__ = [
    "FunctionTool",
    "Tool",
    "ToolCallNode",
    "ToolExecutor",
    "ToolRegistry",
    "ToolSchema",
    "matches_any_tool_call_for_downstream",
    "matches_tool_call",
]
