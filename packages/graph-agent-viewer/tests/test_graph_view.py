import asyncio
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
    _StepController,
    _ViewState,
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
        self.assertIn("function ResizeHandle", _read_static_text("app.js"))
        self.assertIn("runtimeEdges[runtimeEdges.length - 1]", _read_static_text("app.js"))
        self.assertIn("const titleMeta", _read_static_text("app.js"))
        self.assertIn(".graph-canvas", _read_static_text("styles.css"))
        self.assertIn(".splitter", _read_static_text("styles.css"))
        self.assertIn(".panel-title-meta", _read_static_text("styles.css"))

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
        self.assertEqual(data["layout_direction"], "horizontal")
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

    def test_graph_to_view_data_can_use_vertical_layout_direction(self) -> None:
        graph = (
            GraphBuilder("viewer_demo")
            .node(StaticNode("a"))
            .start("a")
            .build()
        )
        graph.view_layout_direction = "vertical"

        data = graph_to_view_data(graph)

        self.assertEqual(data["layout_direction"], "vertical")

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

    def test_step_controller_releases_one_activation_round_at_a_time(self) -> None:
        async def scenario() -> None:
            graph = (
                GraphBuilder("stepper")
                .node(StaticNode("a"))
                .node(StaticNode("b"))
                .start("a")
                .edge("a", "b", name="a_to_b")
                .build()
            )
            state = _ViewState(graph)
            controller = _StepController(state)

            first_step = asyncio.create_task(
                controller.wait_for_next_step(
                    RuntimeEvent(
                        RuntimeEventName.ACTIVATION_READY,
                        {"run_id": "run-1", "nodes": ["a"], "edges": []},
                    )
                )
            )
            await asyncio.sleep(0)

            self.assertEqual(controller.status()["step"], 1)
            self.assertTrue(controller.status()["waiting"])
            self.assertEqual(state.events[-1]["name"], "viewer_step_waiting")
            self.assertEqual(state.events[-1]["payload"]["nodes"], ["a"])

            self.assertTrue(controller.release())
            await asyncio.wait_for(first_step, timeout=1)
            self.assertFalse(controller.status()["waiting"])
            self.assertEqual(state.events[-1]["name"], "viewer_step_released")
            self.assertEqual(state.events[-1]["payload"]["step"], 1)

            second_step = asyncio.create_task(
                controller.wait_for_next_step(
                    RuntimeEvent(
                        RuntimeEventName.ACTIVATION_READY,
                        {
                            "run_id": "run-1",
                            "nodes": ["b"],
                            "edges": [
                                {
                                    "name": "a_to_b",
                                    "source": "a",
                                    "target": "b",
                                }
                            ],
                        },
                    )
                )
            )
            await asyncio.sleep(0)

            self.assertEqual(controller.status()["step"], 2)
            self.assertTrue(controller.status()["waiting"])
            self.assertEqual(state.events[-1]["payload"]["nodes"], ["b"])

            self.assertTrue(controller.release())
            await asyncio.wait_for(second_step, timeout=1)
            self.assertFalse(controller.release())

        asyncio.run(scenario())
