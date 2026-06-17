"""Graph edge types and routing helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .node import NodeResult
from .utils import _required_str

if TYPE_CHECKING:
    from graph_agent.builtin.tool import ToolSchema
    from graph_agent.core.node import Node


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

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "source": self.source,
            "target": self.target,
        }
        if self.active is not None:
            data["active"] = True
        return data

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        active: EdgePredicate | None = None,
    ) -> "Edge":
        if not isinstance(data, Mapping):
            raise TypeError("edge data must be a JSON object")
        return cls(
            name=_required_str(data, "name", "edge.name"),
            source=_required_str(data, "source", "edge.source"),
            target=_required_str(data, "target", "edge.target"),
            active=active,
        )

    def can_activate(
        self,
        result: NodeResult,
        downstream_node: Node | None = None,
    ) -> bool:
        if self.active is None:
            return True
        return self.active(result, self, downstream_node)
