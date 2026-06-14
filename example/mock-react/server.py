"""Dependency-free mock server for debugging the React GraphView stepper."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import threading
import time
from typing import Any
from urllib.parse import urlparse
import webbrowser


JsonObject = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = ROOT / "packages" / "graph-agent-viewer" / "src" / "graph_agent_viewer" / "static"
RUN_ID = "mock-react-run"
NODES = [
    ("prompt", "Read the user request and initialize state."),
    ("reason", "Decide which tool would be useful."),
    ("act", "Call the mocked tool with structured arguments."),
    ("observe", "Receive the mocked tool result."),
    ("answer", "Compose the final answer from the observation."),
]
EDGE_NAMES = ["plan", "tool", "result", "final"]
EDGES = [
    {
        "id": name,
        "name": name,
        "source": source,
        "target": target,
        "conditional": False,
    }
    for (source, _), (target, _), name in zip(NODES, NODES[1:], EDGE_NAMES)
]


def message(role: str, text: str) -> JsonObject:
    return {
        "role": role,
        "text": text,
        "blocks": [{"kind": "text", "text": text}],
        "response_meta": None,
        "extra": {},
    }


GRAPH: JsonObject = {
    "name": "mock_react_stepper",
    "start_node": NODES[0][0],
    "input_messages": [message("user", "mock request: plan, act, observe, answer")],
    "nodes": [
        {
            "id": name,
            "label": name,
            "kind": "llm",
            "is_start": index == 0,
        }
        for index, (name, _) in enumerate(NODES)
    ],
    "edges": EDGES,
}


class MockStepperState:
    def __init__(self, delay_seconds: float = 0.35) -> None:
        self.delay_seconds = delay_seconds
        self._lock = threading.Lock()
        self._clients: list[queue.Queue[JsonObject | None]] = []
        self._events: list[JsonObject] = []
        self._history: list[JsonObject] = list(GRAPH["input_messages"])
        self._last_output: JsonObject | None = None
        self._next_index = 0
        self._step = 0
        self._waiting = False
        self._current_edges: list[JsonObject] = []
        self._current_node: str | None = None
        self._append_event({"name": "graph_started", "payload": {"run_id": RUN_ID, "graph": GRAPH["name"]}})
        self._wait_for_next_node()

    def status(self) -> JsonObject:
        with self._lock:
            return self._status_locked()

    def register(self) -> tuple[list[JsonObject], queue.Queue[JsonObject | None]]:
        client: queue.Queue[JsonObject | None] = queue.Queue()
        with self._lock:
            existing = list(self._events)
            self._clients.append(client)
        return existing, client

    def unregister(self, client: queue.Queue[JsonObject | None]) -> None:
        with self._lock:
            if client in self._clients:
                self._clients.remove(client)

    def close(self) -> None:
        with self._lock:
            clients = list(self._clients)
            self._clients.clear()
        for client in clients:
            client.put_nowait(None)

    def advance(self) -> JsonObject:
        with self._lock:
            if not self._waiting or self._current_node is None:
                return {**self._status_locked(), "released": False}
            node = self._current_node
            edges = list(self._current_edges)
            step = self._step
            activation_history = list(self._history)
            upstream_output = self._last_output
            self._waiting = False
            self._append_event(
                {
                    "name": "viewer_step_released",
                    "payload": {"run_id": RUN_ID, "step": step},
                }
            )

        if edges:
            self._publish(
                {
                    "name": "node_activated",
                    "payload": {
                        "run_id": RUN_ID,
                        "node": node,
                        "target": node,
                        "edges": edges,
                        "history": activation_history,
                        "upstream_outputs": {
                            edges[0]["name"]: upstream_output,
                        },
                    },
                }
            )
        self._publish(
            {
                "name": "node_started",
                "payload": {
                    "run_id": RUN_ID,
                    "node": node,
                    "history": activation_history,
                },
            }
        )
        time.sleep(self.delay_seconds)

        output = message("assistant", dict(NODES)[node])
        with self._lock:
            self._last_output = output
            self._history.append(output)
            self._append_event(
                {
                    "name": "node_finished",
                    "payload": {
                        "run_id": RUN_ID,
                        "node": node,
                        "output": output,
                    },
                }
            )
            self._next_index += 1
            if self._next_index < len(NODES):
                self._wait_for_next_node()
            else:
                self._current_node = None
                self._current_edges = []
                self._append_event(
                    {
                        "name": "graph_finished",
                        "payload": {"run_id": RUN_ID, "graph": GRAPH["name"]},
                    }
                )
            return {**self._status_locked(), "released": True}

    def _status_locked(self) -> JsonObject:
        return {
            "enabled": True,
            "waiting": self._waiting,
            "step": self._step,
            "nodes": [self._current_node] if self._current_node else [],
            "edges": list(self._current_edges),
        }

    def _wait_for_next_node(self) -> None:
        node = NODES[self._next_index][0]
        self._step += 1
        self._waiting = True
        self._current_node = node
        self._current_edges = [] if self._next_index == 0 else [EDGES[self._next_index - 1]]
        self._append_event(
            {
                "name": "viewer_step_waiting",
                "payload": {
                    "run_id": RUN_ID,
                    "step": self._step,
                    "nodes": [node],
                    "edges": list(self._current_edges),
                },
            }
        )

    def _publish(self, event: JsonObject) -> None:
        with self._lock:
            self._append_event(event)

    def _append_event(self, event: JsonObject) -> None:
        self._events.append(event)
        for client in list(self._clients):
            client.put_nowait(event)


def make_handler(state: MockStepperState) -> type[BaseHTTPRequestHandler]:
    class MockViewerHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self._send_file(STATIC_DIR / "index.html", "text/html")
                return
            if path == "/api/graph":
                self._send_json(GRAPH)
                return
            if path == "/api/step":
                self._send_json(state.status())
                return
            if path == "/api/events":
                self._send_events()
                return
            if path.startswith("/static/"):
                name = path.rsplit("/", 1)[-1]
                content_type = {
                    "app.js": "application/javascript",
                    "styles.css": "text/css",
                }.get(name)
                if content_type is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self._send_file(STATIC_DIR / name, content_type)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path != "/api/step":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._send_json(state.advance())

        def _send_json(self, data: JsonObject) -> None:
            body = json.dumps(data).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path, content_type: str) -> None:
            if not path.exists():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_events(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            existing, client = state.register()
            try:
                self.wfile.write(b": connected\n\n")
                self.wfile.flush()
                for event in existing:
                    self._write_event(event)
                while True:
                    event = client.get()
                    if event is None:
                        break
                    self._write_event(event)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                state.unregister(client)

        def _write_event(self, event: JsonObject) -> None:
            body = json.dumps(event)
            self.wfile.write(f"event: graph-event\ndata: {body}\n\n".encode("utf-8"))
            self.wfile.flush()

    return MockViewerHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the mock React GraphView server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--delay", type=float, default=0.35)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    state = MockStepperState(delay_seconds=args.delay)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    host, port = server.server_address
    url = f"http://{host}:{port}/"
    print(f"Mock React GraphView running at {url}")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Mock React GraphView stopped")
    finally:
        state.close()
        server.server_close()


if __name__ == "__main__":
    main()
