"""Generic graph node wrapper for LLM providers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from graph_agent.core import Node, NodeKind, NodeResult, UpstreamOutputs
from graph_agent.core.edge import ToolSchemaProvider
from graph_agent.core.message import Message
from graph_agent.runtime import RunContext

from .base import LLMProvider, SystemPrompt

if TYPE_CHECKING:
    from graph_agent.builtin.tool import ToolSchema
    from graph_agent.core.edge import Edge


class LLMNode(Node):
    """Graph node wrapper that invokes an LLMProvider."""

    def __init__(
        self,
        name: str,
        provider: LLMProvider,
        *,
        system_prompt: SystemPrompt = None,
        tools: Iterable[ToolSchema] | None = None,
    ) -> None:
        self.name = name
        self.provider = provider
        self.tools = tuple(tools or ())
        self.system_messages: tuple[Message, ...]

        if system_prompt is None:
            self.system_messages = ()
        elif isinstance(system_prompt, str):
            self.system_messages = (Message.system_text(system_prompt),)
        else:
            raise TypeError("system_prompt must be a str")

    def init_from_edges(
        self,
        in_edges: list[Edge],
        out_edges: list[Edge],
        graph_nodes: Mapping[str, Node],
    ) -> dict[str, Any]:
        return {
            "tools": self._dedupe_tool_schemas(
                (*self.tools, *self._downstream_tool_schemas(out_edges, graph_nodes))
            )
        }

    @staticmethod
    def _downstream_tool_schemas(
        out_edges: list[Edge],
        graph_nodes: Mapping[str, Node],
    ) -> tuple[ToolSchema, ...]:
        schemas: list[ToolSchema] = []
        schemas_by_name: dict[str, ToolSchema] = {}

        for edge in out_edges:
            downstream_node = graph_nodes.get(edge.target)
            if not isinstance(downstream_node, ToolSchemaProvider):
                continue

            for schema in downstream_node.available_tool_schemas():
                existing_schema = schemas_by_name.get(schema.name)
                if existing_schema is None:
                    schemas_by_name[schema.name] = schema
                    schemas.append(schema)
                    continue
                if existing_schema != schema:
                    raise ValueError(
                        f"duplicate downstream tool schema name: {schema.name}"
                    )

        return tuple(schemas)

    @staticmethod
    def _dedupe_tool_schemas(
        schemas: Iterable[ToolSchema],
    ) -> tuple[ToolSchema, ...]:
        deduped: list[ToolSchema] = []
        schemas_by_name: dict[str, ToolSchema] = {}
        for schema in schemas:
            existing_schema = schemas_by_name.get(schema.name)
            if existing_schema is None:
                schemas_by_name[schema.name] = schema
                deduped.append(schema)
                continue
            if existing_schema != schema:
                raise ValueError(f"duplicate tool schema name: {schema.name}")
        return tuple(deduped)

    async def invoke(
        self,
        ctx: RunContext,
        history: list[Message],
        upstream_outputs: UpstreamOutputs,
        *,
        tools: Iterable[ToolSchema] | None = None,
        **response_options: Any,
    ) -> NodeResult:
        messages = [
            *self.system_messages,
            *history,
            *upstream_outputs.values(),
        ]
        active_tools = tuple(tools) if tools is not None else self.tools
        if active_tools:
            response_options = {**response_options, "tools": active_tools}

        output = await self.provider.generate(messages, **response_options)
        return NodeResult(self, output)

    def kind(self) -> NodeKind:
        return NodeKind.LLM
