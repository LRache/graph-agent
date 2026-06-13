"""Graph edge types and routing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .node import NodeResult

if TYPE_CHECKING:
    from graph_agent.graph.node import Node
    from graph_agent.tool import ToolSchema


class EdgePredicate(Protocol):
    def __call__(
        self,
        result: NodeResult,
        edge: "Edge",
        downstream_node: Node | None,
    ) -> bool:
        raise NotImplementedError


@runtime_checkable
class ToolSchemaProvider(Protocol):
    def available_tool_schemas(self) -> tuple[ToolSchema, ...]:
        raise NotImplementedError


@dataclass(frozen=True)
class Edge:
    name: str
    source: str
    target: str
    active: EdgePredicate | None = None

    def can_activate(
        self,
        result: NodeResult,
        downstream_node: Node | None = None,
    ) -> bool:
        if self.active is None:
            return True
        return self.active(result, self, downstream_node)
