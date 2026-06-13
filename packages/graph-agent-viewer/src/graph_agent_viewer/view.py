"""React-backed local graph viewer."""

from __future__ import annotations

import asyncio
import importlib
from importlib import resources
import json
import webbrowser
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from graph_agent.graph import Graph, GraphRunResult
from graph_agent.message import (
    FileBlock,
    Message,
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from graph_agent.runtime import RuntimeEvent


JsonValue = Any
JsonObject = dict[str, Any]
_STATIC_CONTENT_TYPES = {
    "app.js": "application/javascript",
    "index.html": "text/html",
    "styles.css": "text/css",
}


def _load_aiohttp_web() -> Any:
    try:
        return importlib.import_module("aiohttp.web")
    except ImportError as exc:
        raise RuntimeError(
            "GraphView requires aiohttp. Install graph-agent-viewer with "
            "its viewer dependencies before calling GraphView.run()."
        ) from exc


def _read_static_text(name: str) -> str:
    if name not in _STATIC_CONTENT_TYPES:
        raise FileNotFoundError(name)
    return (
        resources.files("graph_agent_viewer")
        .joinpath("static", name)
        .read_text(encoding="utf-8")
    )


def _jsonable(value: Any) -> JsonValue:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return repr(value)


def _block_text(block: Any) -> str:
    if isinstance(block, TextBlock | ReasoningBlock):
        return block.text_value
    if isinstance(block, ToolCallBlock):
        return f"{block.tool_name}({_jsonable(block.arguments)!r})"
    if isinstance(block, ToolResultBlock):
        return block.content
    if isinstance(block, FileBlock):
        return block.name or block.path or block.file_id or ""
    try:
        text = block.text()
    except NotImplementedError:
        return repr(block)
    return text or ""


def _message_text(message: Message) -> str:
    return "\n".join(text for text in (_block_text(block) for block in message.blocks) if text)


def _message_to_dict(message: Message) -> JsonObject:
    return {
        "role": message.role.value,
        "text": _message_text(message),
        "blocks": [_block_to_dict(block) for block in message.blocks],
        "response_meta": _jsonable(message.response_meta),
        "extra": _jsonable(message.extra),
    }


def _block_to_dict(block: Any) -> JsonObject:
    if isinstance(block, TextBlock):
        return {"kind": block.kind.value, "text": block.text_value}
    if isinstance(block, ReasoningBlock):
        return {
            "kind": block.kind.value,
            "text": block.text_value,
            "signature": block.signature,
        }
    if isinstance(block, FileBlock):
        return {
            "kind": block.kind.value,
            "file_id": block.file_id,
            "path": block.path,
            "mime_type": block.mime_type,
            "name": block.name,
        }
    if isinstance(block, ToolCallBlock):
        return {
            "kind": block.kind.value,
            "call_id": block.call_id,
            "tool_name": block.tool_name,
            "arguments": _jsonable(block.arguments),
        }
    if isinstance(block, ToolResultBlock):
        return {
            "kind": block.kind.value,
            "call_id": block.call_id,
            "tool_name": block.tool_name,
            "content": block.content,
            "is_error": block.is_error,
        }
    return {"kind": str(block.kind), "text": _jsonable(block.text())}


def _payload_value_to_dict(value: Any) -> JsonValue:
    if isinstance(value, Message):
        return _message_to_dict(value)
    if isinstance(value, Mapping):
        return {str(key): _payload_value_to_dict(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_payload_value_to_dict(item) for item in value]
    return _jsonable(value)


def runtime_event_to_dict(event: RuntimeEvent) -> JsonObject:
    return {
        "name": event.name.value,
        "payload": {
            key: _payload_value_to_dict(value)
            for key, value in event.payload.items()
        },
    }


def graph_to_view_data(graph: Graph) -> JsonObject:
    node_names = set(graph.nodes)
    for edge in graph.edges:
        node_names.add(edge.source)
        node_names.add(edge.target)
    if graph.start_node is not None:
        node_names.add(graph.start_node)

    nodes: list[JsonObject] = []
    for node_name in sorted(node_names):
        node = graph.nodes.get(node_name)
        nodes.append(
            {
                "id": node_name,
                "label": node_name,
                "kind": node.kind().value if node is not None else "unknown",
                "is_start": node_name == graph.start_node,
            }
        )

    return {
        "name": graph.name,
        "start_node": graph.start_node,
        "input_messages": [
            _message_to_dict(message) for message in graph.input_messages
        ],
        "nodes": nodes,
        "edges": [
            {
                "id": edge.name,
                "name": edge.name,
                "source": edge.source,
                "target": edge.target,
                "conditional": edge.active is not None,
            }
            for edge in graph.edges
        ],
    }


class _ViewState:
    def __init__(self, graph: Graph) -> None:
        self.graph_data = graph_to_view_data(graph)
        self.events: list[JsonObject] = []
        self.clients: list[asyncio.Queue[JsonObject | None]] = []
        self.loop = asyncio.get_running_loop()

    def publish(self, event: RuntimeEvent) -> None:
        self.publish_data(runtime_event_to_dict(event))

    def publish_data(self, event_data: JsonObject) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is self.loop:
            self._publish_data(event_data)
            return
        self.loop.call_soon_threadsafe(self._publish_data, event_data)

    def _publish_data(self, event_data: JsonObject) -> None:
        self.events.append(event_data)
        for client in list(self.clients):
            client.put_nowait(event_data)

    def register(self) -> tuple[list[JsonObject], asyncio.Queue[JsonObject | None]]:
        client: asyncio.Queue[JsonObject | None] = asyncio.Queue()
        existing = list(self.events)
        self.clients.append(client)
        return existing, client

    def unregister(self, client: asyncio.Queue[JsonObject | None]) -> None:
        if client in self.clients:
            self.clients.remove(client)

    def close(self) -> None:
        clients = list(self.clients)
        self.clients.clear()
        for client in clients:
            client.put_nowait(None)


async def _write_sse(response: Any, event_data: JsonObject) -> None:
    body = json.dumps(event_data)
    await response.write(f"event: graph-event\ndata: {body}\n\n".encode("utf-8"))


def _build_app(state: _ViewState) -> Any:
    web = _load_aiohttp_web()
    app = web.Application()

    async def index(request: Any) -> Any:
        return web.Response(
            text=_read_static_text("index.html"),
            content_type=_STATIC_CONTENT_TYPES["index.html"],
        )

    async def static_asset(request: Any) -> Any:
        name = request.match_info["name"]
        if name not in _STATIC_CONTENT_TYPES:
            raise web.HTTPNotFound()
        return web.Response(
            text=_read_static_text(name),
            content_type=_STATIC_CONTENT_TYPES[name],
        )

    async def graph_data(request: Any) -> Any:
        return web.json_response(state.graph_data)

    async def events(request: Any) -> Any:
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await response.prepare(request)
        existing, client = state.register()
        try:
            await response.write(b": connected\n\n")
            for event_data in existing:
                await _write_sse(response, event_data)
            while True:
                queued_event = await client.get()
                if queued_event is None:
                    break
                await _write_sse(response, queued_event)
        except (BrokenPipeError, ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            state.unregister(client)
        return response

    app.router.add_get("/", index)
    app.router.add_get("/static/{name}", static_asset)
    app.router.add_get("/api/graph", graph_data)
    app.router.add_get("/api/events", events)
    return app


@dataclass(frozen=True)
class GraphView:
    host: str = "127.0.0.1"
    port: int = 0
    open_browser: bool = True
    keep_open: bool = True
    quiet: bool = False

    @classmethod
    def run(cls, graph: Graph, **kwargs: Any) -> GraphRunResult:
        return cls(**kwargs).serve(graph)

    @classmethod
    async def run_async(cls, graph: Graph, **kwargs: Any) -> GraphRunResult:
        return await cls(**kwargs).serve_async(graph)

    def serve(self, graph: Graph) -> GraphRunResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "GraphView.run() cannot be used inside a running event loop; "
                "use await GraphView.run_async(graph) instead."
            )
        return asyncio.run(self.serve_async(graph))

    async def serve_async(self, graph: Graph) -> GraphRunResult:
        state = _ViewState(graph)
        web = _load_aiohttp_web()
        runner = web.AppRunner(_build_app(state), access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        server = site._server
        if server is None or not server.sockets:
            raise RuntimeError("GraphView failed to start the aiohttp server.")
        actual_port = server.sockets[0].getsockname()[1]
        url = f"http://{self.host}:{actual_port}/"
        if not self.quiet:
            print(f"GraphView running at {url}")
        if self.open_browser:
            webbrowser.open(url)

        try:
            result = await graph.run(event_sink=state.publish)
            if self.keep_open:
                await self._wait_forever()
            return result
        except Exception as exc:
            state.publish_data(
                {
                    "name": "viewer_error",
                    "payload": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            )
            raise
        finally:
            state.close()
            await runner.cleanup()

    async def _wait_forever(self) -> None:
        try:
            while True:
                await asyncio.sleep(3600)
        except KeyboardInterrupt:
            if not self.quiet:
                print("GraphView stopped")
