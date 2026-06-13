import unittest

from graph_agent import (
    GraphBuilder,
    Message,
    Node,
    NodeKind,
    NodeResult,
    RuntimeEvent,
    RuntimeEventName,
    UpstreamOutputs,
)
from graph_agent.runtime import RunContext
from graph_agent_viewer import GraphView
from graph_agent_viewer.view import (
    _read_static_text,
    graph_to_view_data,
    runtime_event_to_dict,
)


class StaticNode(Node):
    def __init__(self, name: str) -> None:
        self.name = name

    def prepare_downstream_history(
        self,
        upstream_outputs: UpstreamOutputs,
        history: list[Message],
    ) -> list[Message]:
        return list(history)

    async def invoke(
        self,
        ctx: RunContext,
        history: list[Message],
        upstream_outputs: UpstreamOutputs,
        **extra,
    ) -> NodeResult:
        return NodeResult(self, Message.assistant_text(self.name))

    def kind(self) -> NodeKind:
        return NodeKind.LLM


class GraphViewTests(unittest.TestCase):
    def test_graph_view_is_exported(self) -> None:
        self.assertTrue(callable(GraphView.run))

    def test_static_frontend_assets_are_available(self) -> None:
        self.assertIn("/static/styles.css", _read_static_text("index.html"))
        self.assertIn("/static/app.js", _read_static_text("index.html"))
        self.assertIn("function App()", _read_static_text("app.js"))
        self.assertIn(".graph-canvas", _read_static_text("styles.css"))

    def test_graph_to_view_data_includes_static_structure(self) -> None:
        graph = (
            GraphBuilder("viewer_demo")
            .input([Message.user_text("hello")])
            .node(StaticNode("a"))
            .node(StaticNode("b"))
            .start("a")
            .edge("a", "b", name="to_b")
            .build()
        )

        data = graph_to_view_data(graph)

        self.assertEqual(data["name"], "viewer_demo")
        self.assertEqual(data["start_node"], "a")
        self.assertEqual(
            data["nodes"],
            [
                {"id": "a", "label": "a", "kind": "llm", "is_start": True},
                {"id": "b", "label": "b", "kind": "llm", "is_start": False},
            ],
        )
        self.assertEqual(
            data["edges"],
            [
                {
                    "id": "to_b",
                    "name": "to_b",
                    "source": "a",
                    "target": "b",
                    "conditional": False,
                }
            ],
        )
        self.assertEqual(data["input_messages"][0]["text"], "hello")

    def test_runtime_event_to_dict_serializes_messages_for_react(self) -> None:
        event = RuntimeEvent(
            RuntimeEventName.NODE_FINISHED,
            {
                "run_id": "run-1",
                "node": "a",
                "output": Message.assistant_text("done"),
            },
        )

        self.assertEqual(
            runtime_event_to_dict(event),
            {
                "name": "node_finished",
                "payload": {
                    "run_id": "run-1",
                    "node": "a",
                    "output": {
                        "role": "assistant",
                        "text": "done",
                        "blocks": [{"kind": "text", "text": "done"}],
                        "response_meta": None,
                        "extra": {},
                    },
                },
            },
        )

    def test_runtime_event_to_dict_serializes_tool_call_messages(self) -> None:
        event = RuntimeEvent(
            RuntimeEventName.NODE_FINISHED,
            {
                "run_id": "run-1",
                "node": "llm",
                "output": Message.tool_call("call-1", "add", {"a": 2, "b": 2}),
            },
        )

        data = runtime_event_to_dict(event)

        self.assertEqual(data["payload"]["output"]["role"], "assistant")
        self.assertEqual(data["payload"]["output"]["text"], "add({'a': 2, 'b': 2})")
        self.assertEqual(
            data["payload"]["output"]["blocks"],
            [
                {
                    "kind": "tool_call",
                    "call_id": "call-1",
                    "tool_name": "add",
                    "arguments": {"a": 2, "b": 2},
                }
            ],
        )

    def test_graph_view_run_starts_aiohttp_server_and_runs_graph(self) -> None:
        graph = (
            GraphBuilder("viewer_run")
            .node(StaticNode("a"))
            .start("a")
            .build()
        )

        result = GraphView.run(
            graph,
            open_browser=False,
            keep_open=False,
            quiet=True,
        )

        self.assertEqual([message.text() for message in result.output], ["a"])
