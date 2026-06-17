"""Runtime context and events emitted while a graph runs."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4


class RuntimeEventName(StrEnum):
    GRAPH_STARTED = "graph_started"
    GRAPH_FINISHED = "graph_finished"
    GRAPH_CANCELLED = "graph_cancelled"
    GRAPH_FAILED = "graph_failed"
    ACTIVATION_READY = "activation_ready"
    NODE_ACTIVATED = "node_activated"
    NODE_STARTED = "node_started"
    NODE_FINISHED = "node_finished"


@dataclass(frozen=True)
class RuntimeEvent:
    name: RuntimeEventName
    payload: dict[str, Any] = field(default_factory=dict)


EventSinkResult = Awaitable[None] | None
EventSink = Callable[[RuntimeEvent], EventSinkResult]


@dataclass
class _CancelState:
    cancelled: bool = False


class RunContext:
    def __init__(
        self,
        *,
        run_id: str | None = None,
        node_id: str | None = None,
        event_sink: EventSink | None = None,
        events: list[RuntimeEvent] | None = None,
        cancel_state: _CancelState | None = None,
    ) -> None:
        self.run_id = run_id or str(uuid4())
        self.node_id = node_id
        self.events = events if events is not None else []
        self._event_sink = event_sink
        self._cancel_state = cancel_state or _CancelState()

    @property
    def cancelled(self) -> bool:
        return self._cancel_state.cancelled

    def _emit_event(
        self,
        name: RuntimeEventName,
        *,
        store: bool,
        **payload: Any,
    ) -> EventSinkResult:
        event = RuntimeEvent(name, {"run_id": self.run_id, **payload})
        if store:
            self.events.append(event)
        if self._event_sink is not None:
            return self._event_sink(event)
        return None

    async def emit(
        self,
        name: RuntimeEventName,
        *,
        store: bool = True,
        **payload: Any,
    ) -> None:
        result = self._emit_event(name, store=store, **payload)
        if result is not None:
            await result

    def child_for_node(self, node_id: str) -> "RunContext":
        return RunContext(
            run_id=self.run_id,
            node_id=node_id,
            event_sink=self._event_sink,
            events=self.events,
            cancel_state=self._cancel_state,
        )

    def cancel(self) -> None:
        self._cancel_state.cancelled = True
