import asyncio
import json
import unittest

from graph_agent import (
    CompletedNode,
    ContentBlock,
    Edge,
    FunctionTool,
    GRAPH_STATE_SCHEMA,
    GraphBuilder,
    GraphRunStatus,
    LLMNode,
    matches_any_tool_call_for_downstream,
    matches_tool_call,
    Message,
    MessageRole,
    Node,
    NodeActivation,
    NodeKind,
    NodeResult,
    RuntimeEventName,
    ToolCallBlock,
    ToolCallNode,
    ToolSchema,
)
from graph_agent.runtime import RunContext

from tests.helpers import CallableNode, RecordingProvider


def _seed_node() -> CallableNode:
    return CallableNode(
        "seed",
        lambda ctx, history, upstream: Message.assistant_text("seed"),
    )


class _CancellingTestNode(Node):
    name = "left"

    def __init__(self, started: asyncio.Event):
        self.started = started

    def prepare_downstream_history(self, upstream_outputs, history):
        return list(history)

    async def invoke(self, ctx, history, upstream_outputs, **extra):
        self.started.set()
        ctx.cancel()
        return NodeResult(self, Message.assistant_text("left"))

    def kind(self):
        return NodeKind.LLM


class _BlockingTestNode(Node):
    name = "right"

    def __init__(
        self,
        started: asyncio.Event,
        release: asyncio.Event,
        finished: asyncio.Event | None = None,
    ):
        self.started = started
        self.release = release
        self.finished = finished

    def prepare_downstream_history(self, upstream_outputs, history):
        return list(history)

    async def invoke(self, ctx, history, upstream_outputs, **extra):
        self.started.set()
        await self.release.wait()
        if self.finished is not None:
            self.finished.set()
        return NodeResult(self, Message.assistant_text("right"))

    def kind(self):
        return NodeKind.LLM


class _FailingAfterPeerStartsNode(Node):
    def __init__(
        self,
        name: str,
        started: asyncio.Event,
        peer_started: asyncio.Event,
    ):
        self.name = name
        self.started = started
        self.peer_started = peer_started

    def prepare_downstream_history(self, upstream_outputs, history):
        return list(history)

    async def invoke(self, ctx, history, upstream_outputs, **extra):
        self.started.set()
        await self.peer_started.wait()
        raise RuntimeError(f"{self.name} failed")

    def kind(self):
        return NodeKind.LLM


class _CancellableBlockingTestNode(Node):
    def __init__(
        self,
        name: str,
        started: asyncio.Event,
        cancelled: asyncio.Event,
    ):
        self.name = name
        self.started = started
        self.cancelled = cancelled

    def prepare_downstream_history(self, upstream_outputs, history):
        return list(history)

    async def invoke(self, ctx, history, upstream_outputs, **extra):
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise

    def kind(self):
        return NodeKind.LLM


class _RecordingTestNode(Node):
    def __init__(self, name: str, invocations: list[str]):
        self.name = name
        self.invocations = invocations

    def prepare_downstream_history(self, upstream_outputs, history):
        return list(history)

    async def invoke(self, ctx, history, upstream_outputs, **extra):
        self.invocations.append(self.name)
        return NodeResult(self, Message.assistant_text(self.name))

    def kind(self):
        return NodeKind.LLM


class GraphRuntimeTests(unittest.TestCase):
    def test_default_node_prepare_downstream_history_deduplicates_uuid_in_order(self):
        class DefaultPrepareNode(Node):
            name = "default_prepare"

            async def invoke(self, ctx, history, upstream_outputs, **extra):
                return NodeResult(self, Message.assistant_text("output"))

            def kind(self):
                return NodeKind.LLM

        node = DefaultPrepareNode()
        history_message = Message.user_text(
            "history",
            uuid="00000000-0000-4000-8000-000000000001",
        )
        duplicate_uuid_output = Message.assistant_text(
            "duplicate uuid",
            uuid=history_message.uuid,
        )
        second_output = Message.assistant_text(
            "second",
            uuid="00000000-0000-4000-8000-000000000002",
        )

        downstream_history = node.prepare_downstream_history(
            {
                "duplicate": duplicate_uuid_output,
                "second": second_output,
            },
            [history_message],
        )

        self.assertEqual(downstream_history, [history_message, second_output])

    def test_edge_round_trips_serialized_shape(self):
        def active(result, edge, downstream_node):
            return True

        edge = Edge("to_target", "source", "target", active=active)

        self.assertEqual(
            edge.to_dict(),
            {
                "name": "to_target",
                "source": "source",
                "target": "target",
                "active": True,
            },
        )

        restored = Edge.from_dict(edge.to_dict(), active=active)

        self.assertEqual(restored.name, "to_target")
        self.assertEqual(restored.source, "source")
        self.assertEqual(restored.target, "target")
        self.assertIs(restored.active, active)

    def test_unbound_tool_call_predicate_does_not_match_without_tools(self):
        def source(ctx, history, upstream_outputs):
            return Message.assistant_text("source")

        edge = Edge(
            "to_tools",
            "llm",
            "tools",
            active=matches_tool_call(),
        )
        result = NodeResult(
            CallableNode("llm", source),
            Message.tool_call("call_1", "lookup"),
        )

        self.assertFalse(edge.can_activate(result))

    def test_matches_any_tool_call_for_downstream_can_be_edge_active(self):
        def source(ctx, history, upstream_outputs):
            return Message.assistant_text("source")

        lookup_node = ToolCallNode("lookup_tools")
        lookup_node.register_tool(FunctionTool("lookup", lambda args: "found"))
        edge = Edge(
            "to_lookup",
            "llm",
            "lookup_tools",
            active=matches_any_tool_call_for_downstream,
        )
        result = NodeResult(
            CallableNode("llm", source),
            Message.tool_call("call_1", "lookup"),
        )
        unmatched_result = NodeResult(
            CallableNode("llm", source),
            Message.tool_call("call_2", "echo"),
        )

        self.assertTrue(edge.can_activate(result, lookup_node))
        self.assertFalse(edge.can_activate(unmatched_result, lookup_node))
        self.assertFalse(edge.can_activate(result))

    def test_matches_any_tool_call_for_downstream_routes_matching_tool_node(self):
        def source(ctx, history, upstream_outputs):
            return Message.assistant_text("call tools")

        lookup_node = ToolCallNode("lookup_tools")
        lookup_node.register_tool(
            FunctionTool("lookup", lambda args: f"found {args['query']}")
        )
        echo_node = ToolCallNode("echo_tools")
        echo_node.register_tool(
            FunctionTool("echo", lambda args: args["value"])
        )
        graph = (
            GraphBuilder("tool_call_downstream_routing")
            .node(CallableNode("llm", source))
            .node(lookup_node)
            .node(echo_node)
            .start("llm")
            .edge(
                "llm",
                "lookup_tools",
                name="llm_to_lookup_tools",
                active=matches_any_tool_call_for_downstream,
            )
            .edge(
                "llm",
                "echo_tools",
                name="llm_to_echo_tools",
                active=matches_any_tool_call_for_downstream,
            )
            .build()
        )
        source_node = graph.nodes["llm"]
        output = Message.tool_call("call_lookup", "lookup", {"query": "weather"})

        activations = graph.active_next_nodes(
            CompletedNode(
                NodeActivation(source_node, [], {}, []),
                NodeResult(source_node, output),
            )
        )

        self.assertEqual(
            [activation.node.name for activation in activations],
            ["lookup_tools"],
        )
        self.assertEqual(graph.node_states["lookup_tools"].finished_dependency, 1)
        self.assertEqual(graph.node_states["echo_tools"].finished_dependency, 0)

    def test_graph_passes_downstream_tool_schemas_to_llm_node_provider(self):
        provider = RecordingProvider()
        lookup_tool = FunctionTool(
            "lookup",
            lambda args: args["query"],
            description="Look up a value.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        graph = (
            GraphBuilder("tool_schema_request")
            .input([Message.user_text("lookup weather")])
            .node(LLMNode("llm", provider))
            .node(ToolCallNode("lookup_tools", lookup_tool))
            .start("llm")
            .edge(
                "llm",
                "lookup_tools",
                name="llm_to_lookup_tools",
                active=matches_any_tool_call_for_downstream,
            )
            .build()
        )

        result = asyncio.run(graph.run())

        self.assertEqual([message.text() for message in result.output], ["recorded"])
        self.assertEqual(
            provider.calls[0][1]["tools"],
            (
                ToolSchema(
                    name="lookup",
                    description="Look up a value.",
                    parameters={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                ),
            ),
        )

    def test_graph_passes_node_state_extra_to_invoke(self):
        class InspectingNode(CallableNode):
            def init_from_edges(self, in_edges, out_edges, graph_nodes):
                return {
                    "edge_counts": (len(in_edges), len(out_edges)),
                    "known_nodes": tuple(sorted(graph_nodes)),
                }

        middle = InspectingNode(
            "middle",
            lambda ctx, history, upstream: Message.assistant_text("middle"),
        )
        graph = (
            GraphBuilder("node_extra_invoke")
            .node(
                CallableNode(
                    "source",
                    lambda ctx, history, upstream: Message.assistant_text("source"),
                )
            )
            .node(middle)
            .start("source")
            .edge("source", "middle", name="source_to_middle")
            .build()
        )
        graph.node_states["middle"].extra["from_state"] = "runtime"

        asyncio.run(graph.run())

        self.assertEqual(
            middle.extra,
            {
                "edge_counts": (1, 0),
                "known_nodes": ("middle", "source"),
                "from_state": "runtime",
                "tools": (),
            },
        )

    def test_tool_edges_activate_without_filtering_matching_tool_calls(self):
        def source(ctx, history, upstream_outputs):
            return Message.assistant_text("call tools")

        lookup_node = ToolCallNode("lookup_tools")
        lookup_node.register_tool(
            FunctionTool("lookup", lambda args: f"found {args['query']}")
        )
        echo_node = ToolCallNode("echo_tools")
        echo_node.register_tool(
            FunctionTool("echo", lambda args: args["value"])
        )
        graph = (
            GraphBuilder("tool_call_routing")
            .node(CallableNode("llm", source))
            .node(lookup_node)
            .node(echo_node)
            .start("llm")
            .edge(
                "llm",
                "lookup_tools",
                name="llm_to_lookup_tools",
                active=matches_tool_call(),
            )
            .edge(
                "llm",
                "echo_tools",
                name="llm_to_echo_tools",
                active=matches_tool_call(),
            )
            .build()
        )
        source_node = graph.nodes["llm"]
        output = Message.of(
            MessageRole.ASSISTANT,
            [
                ContentBlock.tool_call(
                    ToolCallBlock("call_lookup", "lookup", {"query": "weather"})
                ),
                ContentBlock.tool_call(
                    ToolCallBlock("call_echo", "echo", {"value": "hello"})
                ),
            ],
        )

        activations = graph.active_next_nodes(
            CompletedNode(
                NodeActivation(source_node, [], {}, []),
                NodeResult(source_node, output),
            )
        )

        self.assertEqual(
            [activation.node.name for activation in activations],
            ["lookup_tools", "echo_tools"],
        )
        lookup_output = activations[0].upstream_outputs["llm_to_lookup_tools"]
        echo_output = activations[1].upstream_outputs["llm_to_echo_tools"]
        self.assertEqual(
            [call.tool_name for call in lookup_output.tool_calls()],
            ["lookup", "echo"],
        )
        self.assertEqual(
            [call.tool_name for call in echo_output.tool_calls()],
            ["lookup", "echo"],
        )

        lookup_result = asyncio.run(
            activations[0].node.invoke(
                RunContext(),
                activations[0].history,
                activations[0].upstream_outputs,
            )
        )
        echo_result = asyncio.run(
            activations[1].node.invoke(
                RunContext(),
                activations[1].history,
                activations[1].upstream_outputs,
            )
        )

        self.assertEqual(lookup_result.output.text(), "found weather")
        self.assertEqual(echo_result.output.text(), "hello")

    def test_tool_edges_do_not_activate_without_matching_tool_calls(self):
        def source(ctx, history, upstream_outputs):
            return Message.assistant_text("call tools")

        lookup_node = ToolCallNode("lookup_tools")
        lookup_node.register_tool(
            FunctionTool("lookup", lambda args: f"found {args['query']}")
        )
        graph = (
            GraphBuilder("unmatched_tool_call_routing")
            .node(CallableNode("llm", source))
            .node(lookup_node)
            .start("llm")
            .edge(
                "llm",
                "lookup_tools",
                name="llm_to_lookup_tools",
                active=matches_tool_call(),
            )
            .build()
        )
        source_node = graph.nodes["llm"]
        output = Message.tool_call("call_echo", "echo", {"value": "hello"})

        activations = graph.active_next_nodes(
            CompletedNode(
                NodeActivation(source_node, [], {}, []),
                NodeResult(source_node, output),
            )
        )

        self.assertEqual(activations, [])
        self.assertEqual(graph.node_states["lookup_tools"].finished_dependency, 0)
        self.assertEqual(graph.node_states["lookup_tools"].dependency_results, {})

    def test_tool_edges_do_not_filter_without_explicit_active(self):
        def source(ctx, history, upstream_outputs):
            return Message.assistant_text("call tools")

        lookup_node = ToolCallNode("lookup_tools")
        lookup_node.register_tool(
            FunctionTool("lookup", lambda args: f"found {args['query']}")
        )
        graph = (
            GraphBuilder("unfiltered_tool_call_routing")
            .node(CallableNode("llm", source))
            .node(lookup_node)
            .start("llm")
            .edge("llm", "lookup_tools", name="llm_to_lookup_tools")
            .build()
        )
        source_node = graph.nodes["llm"]
        output = Message.tool_call("call_echo", "echo", {"value": "hello"})

        activations = graph.active_next_nodes(
            CompletedNode(
                NodeActivation(source_node, [], {}, []),
                NodeResult(source_node, output),
            )
        )

        self.assertEqual([activation.node.name for activation in activations], ["lookup_tools"])
        routed_output = activations[0].upstream_outputs["llm_to_lookup_tools"]
        self.assertEqual(
            [call.tool_name for call in routed_output.tool_calls()],
            ["echo"],
        )
        self.assertEqual(graph.node_states["lookup_tools"].finished_dependency, 1)

    def test_graph_start_outputs_configured_input(self):
        def echo(ctx, history, upstream_outputs):
            return Message.assistant_text(history[-1].text())

        graph = (
            GraphBuilder("input_echo")
            .input([Message.user_text("configured")])
            .node(CallableNode("echo", echo))
            .start("echo")
            .build()
        )

        result = asyncio.run(graph.run())

        self.assertEqual([message.text() for message in result.output], ["configured"])
        self.assertEqual(graph.start_node, "echo")

    def test_graph_run_returns_final_history(self):
        def first(ctx, history, upstream_outputs):
            return Message.assistant_text("first")

        def second(ctx, history, upstream_outputs):
            return Message.assistant_text("second")

        graph = (
            GraphBuilder("run_history")
            .input([Message.user_text("start")])
            .node(CallableNode("first", first))
            .node(CallableNode("second", second))
            .start("first")
            .edge("first", "second", name="first_to_second")
            .build()
        )

        result = asyncio.run(graph.run())

        self.assertEqual([message.text() for message in result.output], ["second"])
        self.assertEqual(
            [message.text() for message in result.history],
            ["start", "first", "second"],
        )
        self.assertIs(result.history[-1], result.output[0])

    def test_graph_run_returns_serialized_state(self):
        def first(ctx, history, upstream_outputs):
            return Message.assistant_text("first")

        def second(ctx, history, upstream_outputs):
            return Message.assistant_text("second")

        graph = (
            GraphBuilder("state_snapshot")
            .input([Message.user_text("start")])
            .node(CallableNode("first", first))
            .node(CallableNode("second", second))
            .start("first")
            .edge("first", "second", name="first_to_second")
            .build()
        )

        result = asyncio.run(graph.run())
        payload = json.loads(json.dumps(result.state))

        self.assertEqual(result.status, GraphRunStatus.COMPLETED)
        self.assertEqual(payload["schema"], GRAPH_STATE_SCHEMA)
        self.assertEqual(payload["graph"], "state_snapshot")
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["pending_completed_nodes"], [])
        self.assertEqual(
            payload["node_states"]["first"]["completed"]["output"]["blocks"],
            [{"kind": "text", "text": "first"}],
        )
        self.assertEqual(
            payload["node_states"]["second"]["dependency_results"][
                "first_to_second"
            ]["output"]["blocks"],
            [{"kind": "text", "text": "first"}],
        )
        self.assertEqual(
            payload["node_states"]["second"]["completed"]["output"]["blocks"],
            [{"kind": "text", "text": "second"}],
        )

    def test_graph_builder_restores_state_after_nodes_and_edges_are_registered(self):
        def first(ctx, history, upstream_outputs):
            return Message.assistant_text("first")

        def second(ctx, history, upstream_outputs):
            return Message.assistant_text("second")

        graph = (
            GraphBuilder("builder_restore")
            .input([Message.user_text("start")])
            .node(CallableNode("first", first))
            .node(CallableNode("second", second))
            .start("first")
            .edge("first", "second", name="first_to_second")
            .build()
        )
        state = json.loads(json.dumps(asyncio.run(graph.run()).state))

        restored = (
            GraphBuilder()
            .state(state)
            .node(CallableNode("first", first))
            .node(CallableNode("second", second))
            .edge("first", "second", name="first_to_second")
            .build()
        )

        self.assertEqual(restored.name, "builder_restore")
        self.assertEqual(restored.start_node, "first")
        self.assertEqual([message.text() for message in restored.input_messages], ["start"])
        self.assertEqual(
            restored.node_states["second"].finished_dependency,
            1,
        )
        first_dependency = restored.node_states["second"].dependency_results[
            "first_to_second"
        ]
        self.assertEqual(first_dependency.output.text(), "first")
        self.assertIs(first_dependency.node, restored.nodes["first"])
        self.assertIs(first_dependency.activation.node, restored.nodes["first"])
        self.assertEqual(
            restored.node_states["second"].completed.output.text(),
            "second",
        )

    def test_graph_builder_rejects_state_with_mismatched_edge_predicate_shape(self):
        def active(result, edge, downstream_node):
            return True

        def first(ctx, history, upstream_outputs):
            return Message.assistant_text("first")

        def second(ctx, history, upstream_outputs):
            return Message.assistant_text("second")

        graph = (
            GraphBuilder("predicate_state")
            .node(CallableNode("first", first))
            .node(CallableNode("second", second))
            .start("first")
            .edge("first", "second", name="first_to_second", active=active)
            .build()
        )
        state = json.loads(json.dumps(asyncio.run(graph.run()).state))

        with self.assertRaisesRegex(ValueError, "do not match graph"):
            (
                GraphBuilder()
                .state(state)
                .node(CallableNode("first", first))
                .node(CallableNode("second", second))
                .edge("first", "second", name="first_to_second")
                .build()
            )

    def test_graph_builder_rejects_state_with_mismatched_dependency_count(self):
        def first(ctx, history, upstream_outputs):
            return Message.assistant_text("first")

        def second(ctx, history, upstream_outputs):
            return Message.assistant_text("second")

        graph = (
            GraphBuilder("dependency_state")
            .node(CallableNode("first", first))
            .node(CallableNode("second", second))
            .start("first")
            .edge("first", "second", name="first_to_second")
            .build()
        )
        state = json.loads(json.dumps(asyncio.run(graph.run()).state))
        state["node_states"]["second"]["finished_dependency"] = 0

        with self.assertRaisesRegex(ValueError, "finished dependency count"):
            (
                GraphBuilder()
                .state(state)
                .node(CallableNode("first", first))
                .node(CallableNode("second", second))
                .edge("first", "second", name="first_to_second")
                .build()
            )

    def test_cancelled_graph_waits_for_running_nodes_and_serializes_state(self):
        async def scenario():
            left_started = asyncio.Event()
            right_started = asyncio.Event()
            right_release = asyncio.Event()
            right_finished = asyncio.Event()
            child_invocations = []

            graph = (
                GraphBuilder("cancel_state")
                .node(_seed_node())
                .node(_CancellingTestNode(left_started))
                .node(_BlockingTestNode(right_started, right_release, right_finished))
                .node(_RecordingTestNode("left_child", child_invocations))
                .node(_RecordingTestNode("right_child", child_invocations))
                .start("seed")
                .edge("seed", "left", name="seed_to_left")
                .edge("seed", "right", name="seed_to_right")
                .edge("left", "left_child", name="left_to_child")
                .edge("right", "right_child", name="right_to_child")
                .build()
            )

            run_task = asyncio.create_task(graph.run())
            await asyncio.wait_for(left_started.wait(), timeout=1)
            await asyncio.wait_for(right_started.wait(), timeout=1)
            await asyncio.sleep(0)

            self.assertFalse(right_finished.is_set())
            self.assertEqual(child_invocations, [])

            right_release.set()
            result = await asyncio.wait_for(run_task, timeout=1)
            payload = json.loads(json.dumps(result.state))

            self.assertEqual(result.status, GraphRunStatus.CANCELLED)
            self.assertTrue(right_finished.is_set())
            self.assertEqual(child_invocations, [])
            self.assertEqual([message.text() for message in result.output], ["right"])
            self.assertEqual(payload["schema"], GRAPH_STATE_SCHEMA)
            self.assertEqual(payload["status"], "cancelled")
            self.assertEqual(
                payload["pending_completed_nodes"],
                ["left", "right"],
            )
            self.assertEqual(
                payload["node_states"]["left"]["completed"]["output"]["blocks"],
                [{"kind": "text", "text": "left"}],
            )
            self.assertEqual(
                payload["node_states"]["right"]["completed"]["output"]["blocks"],
                [{"kind": "text", "text": "right"}],
            )
            self.assertNotIn("completed", payload["node_states"]["left_child"])
            self.assertEqual(
                payload["node_states"]["left_child"]["finished_dependency"],
                0,
            )
            self.assertEqual(
                payload["node_states"]["right_child"]["finished_dependency"],
                0,
            )

        asyncio.run(scenario())

    def test_graph_run_cancels_sibling_tasks_when_node_fails(self):
        async def scenario():
            left_started = asyncio.Event()
            right_started = asyncio.Event()
            right_cancelled = asyncio.Event()
            events = []
            graph = (
                GraphBuilder("failed_sibling_cleanup")
                .node(_seed_node())
                .node(
                    _FailingAfterPeerStartsNode(
                        "left",
                        left_started,
                        right_started,
                    )
                )
                .node(
                    _CancellableBlockingTestNode(
                        "right",
                        right_started,
                        right_cancelled,
                    )
                )
                .start("seed")
                .edge("seed", "left", name="seed_to_left")
                .edge("seed", "right", name="seed_to_right")
                .build()
            )

            run_task = asyncio.create_task(graph.run(event_sink=events.append))
            await asyncio.wait_for(left_started.wait(), timeout=1)
            await asyncio.wait_for(right_started.wait(), timeout=1)

            with self.assertRaisesRegex(RuntimeError, "left failed"):
                await asyncio.wait_for(run_task, timeout=1)

            self.assertTrue(right_cancelled.is_set())
            self.assertEqual(graph.status, GraphRunStatus.FAILED)
            self.assertEqual(graph.pending_completed_nodes, [])
            self.assertIsNone(graph.node_states["right"].completed)
            self.assertEqual(events[-1].name, RuntimeEventName.GRAPH_FAILED)
            self.assertEqual(events[-1].payload["graph"], "failed_sibling_cleanup")
            self.assertEqual(events[-1].payload["state"]["status"], "failed")
            self.assertEqual(events[-1].payload["error"]["type"], "RuntimeError")
            self.assertEqual(events[-1].payload["error"]["message"], "left failed")

        asyncio.run(scenario())

    def test_graph_resume_continues_from_cancelled_state(self):
        async def scenario():
            left_started = asyncio.Event()
            right_started = asyncio.Event()
            right_release = asyncio.Event()
            original_child_invocations = []

            graph = (
                GraphBuilder("resume_state")
                .node(_seed_node())
                .node(_CancellingTestNode(left_started))
                .node(_BlockingTestNode(right_started, right_release))
                .node(_RecordingTestNode("left_child", original_child_invocations))
                .node(_RecordingTestNode("right_child", original_child_invocations))
                .start("seed")
                .edge("seed", "left", name="seed_to_left")
                .edge("seed", "right", name="seed_to_right")
                .edge("left", "left_child", name="left_to_child")
                .edge("right", "right_child", name="right_to_child")
                .build()
            )

            run_task = asyncio.create_task(graph.run())
            await asyncio.wait_for(left_started.wait(), timeout=1)
            await asyncio.wait_for(right_started.wait(), timeout=1)
            right_release.set()
            cancelled = await asyncio.wait_for(run_task, timeout=1)
            state = json.loads(json.dumps(cancelled.state))

            self.assertEqual(original_child_invocations, [])
            resumed_invocations = []

            def restored_node(name):
                return _RecordingTestNode(name, resumed_invocations)

            restored_for_run = (
                GraphBuilder()
                .state(state)
                .node(restored_node("seed"))
                .node(restored_node("left"))
                .node(restored_node("right"))
                .node(restored_node("left_child"))
                .node(restored_node("right_child"))
                .edge("seed", "left", name="seed_to_left")
                .edge("seed", "right", name="seed_to_right")
                .edge("left", "left_child", name="left_to_child")
                .edge("right", "right_child", name="right_to_child")
                .build()
            )
            with self.assertRaisesRegex(RuntimeError, "use resume"):
                await restored_for_run.run()

            restored = (
                GraphBuilder()
                .state(state)
                .node(restored_node("seed"))
                .node(restored_node("left"))
                .node(restored_node("right"))
                .node(restored_node("left_child"))
                .node(restored_node("right_child"))
                .edge("seed", "left", name="seed_to_left")
                .edge("seed", "right", name="seed_to_right")
                .edge("left", "left_child", name="left_to_child")
                .edge("right", "right_child", name="right_to_child")
                .build()
            )

            result = await restored.resume()
            payload = json.loads(json.dumps(result.state))

            self.assertEqual(result.status, GraphRunStatus.COMPLETED)
            self.assertEqual(
                set(resumed_invocations),
                {"left_child", "right_child"},
            )
            self.assertNotIn("seed", resumed_invocations)
            self.assertNotIn("left", resumed_invocations)
            self.assertNotIn("right", resumed_invocations)
            self.assertIn(result.output[0].text(), {"left_child", "right_child"})
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["pending_completed_nodes"], [])
            self.assertEqual(
                payload["node_states"]["left_child"]["finished_dependency"],
                1,
            )
            self.assertEqual(
                payload["node_states"]["right_child"]["finished_dependency"],
                1,
            )
            self.assertEqual(
                payload["node_states"]["left_child"]["completed"]["output"][
                    "blocks"
                ],
                [{"kind": "text", "text": "left_child"}],
            )
            self.assertEqual(
                payload["node_states"]["right_child"]["completed"]["output"][
                    "blocks"
                ],
                [{"kind": "text", "text": "right_child"}],
            )

        asyncio.run(scenario())

    def test_graph_run_emits_graph_and_node_lifecycle_events(self):
        contexts = []
        input_message = Message.user_text("start")

        def first(ctx, history, upstream_outputs):
            contexts.append(ctx)
            return Message.assistant_text("first")

        def second(ctx, history, upstream_outputs):
            return Message.assistant_text("second")

        graph = (
            GraphBuilder("runtime_events")
            .input([input_message])
            .node(CallableNode("first", first))
            .node(CallableNode("second", second))
            .start("first")
            .edge("first", "second", name="first_to_second")
            .build()
        )

        asyncio.run(graph.run())

        events = contexts[0].events
        self.assertEqual(
            [event.name for event in events],
            [
                RuntimeEventName.GRAPH_STARTED,
                RuntimeEventName.NODE_STARTED,
                RuntimeEventName.NODE_FINISHED,
                RuntimeEventName.NODE_ACTIVATED,
                RuntimeEventName.NODE_STARTED,
                RuntimeEventName.NODE_FINISHED,
                RuntimeEventName.GRAPH_FINISHED,
            ],
        )
        self.assertEqual(events[0].payload["graph"], "runtime_events")
        self.assertEqual(events[-1].payload["graph"], "runtime_events")

        first_started = events[1].payload
        self.assertEqual(first_started["node"], "first")
        self.assertEqual(first_started["history"], [input_message])

        first_finished = events[2].payload
        self.assertEqual(first_finished["node"], "first")
        self.assertEqual(first_finished["output"].text(), "first")

        second_activated = events[3].payload
        self.assertEqual(second_activated["node"], "second")
        self.assertEqual(
            second_activated["edges"],
            [{"name": "first_to_second", "source": "first", "target": "second"}],
        )
        self.assertEqual(
            [message.text() for message in second_activated["history"]],
            ["start"],
        )
        self.assertEqual(
            second_activated["upstream_outputs"]["first_to_second"].text(),
            "first",
        )

        second_started = events[4].payload
        self.assertEqual(second_started["node"], "second")
        self.assertEqual(
            [message.text() for message in second_started["history"]],
            ["start"],
        )

        second_finished = events[5].payload
        self.assertEqual(second_finished["node"], "second")
        self.assertEqual(second_finished["output"].text(), "second")

    def test_graph_run_forwards_events_to_event_sink(self):
        events = []
        input_message = Message.user_text("start")

        def echo(ctx, history, upstream_outputs):
            return Message.assistant_text(history[-1].text())

        graph = (
            GraphBuilder("sink_events")
            .input([input_message])
            .node(CallableNode("echo", echo))
            .start("echo")
            .build()
        )

        result = asyncio.run(graph.run(event_sink=events.append))

        self.assertEqual([message.text() for message in result.output], ["start"])
        self.assertEqual(
            [event.name for event in events],
            [
                RuntimeEventName.GRAPH_STARTED,
                RuntimeEventName.ACTIVATION_READY,
                RuntimeEventName.NODE_STARTED,
                RuntimeEventName.NODE_FINISHED,
                RuntimeEventName.GRAPH_FINISHED,
            ],
        )
        self.assertEqual(events[2].payload["node"], "echo")
        self.assertEqual(events[2].payload["history"], [input_message])
        self.assertEqual(events[3].payload["output"].text(), "start")

    def test_graph_run_event_sink_waits_before_ready_activation(self):
        async def scenario():
            first_gate = asyncio.Event()
            second_gate = asyncio.Event()
            releases = [asyncio.Event(), asyncio.Event()]
            source_started = asyncio.Event()
            target_started = asyncio.Event()
            gated_nodes = []

            def source(ctx, history, upstream_outputs):
                source_started.set()
                return Message.assistant_text("source")

            def target(ctx, history, upstream_outputs):
                target_started.set()
                return Message.assistant_text("target")

            class BlockingSink:
                async def __call__(self, event):
                    if event.name != RuntimeEventName.ACTIVATION_READY:
                        return
                    gated_nodes.append(event.payload["nodes"])
                    if len(gated_nodes) == 1:
                        first_gate.set()
                    if len(gated_nodes) == 2:
                        second_gate.set()
                    await releases[len(gated_nodes) - 1].wait()

            graph = (
                GraphBuilder("gated_runtime")
                .node(CallableNode("source", source))
                .node(CallableNode("target", target))
                .start("source")
                .edge("source", "target", name="source_to_target")
                .build()
            )

            run_task = asyncio.create_task(graph.run(event_sink=BlockingSink()))
            await asyncio.wait_for(first_gate.wait(), timeout=1)
            self.assertEqual(gated_nodes, [["source"]])
            self.assertFalse(source_started.is_set())
            self.assertFalse(target_started.is_set())

            releases[0].set()
            await asyncio.wait_for(second_gate.wait(), timeout=1)
            self.assertEqual(gated_nodes, [["source"], ["target"]])
            self.assertTrue(source_started.is_set())
            self.assertFalse(target_started.is_set())

            releases[1].set()
            result = await asyncio.wait_for(run_task, timeout=1)

            self.assertTrue(target_started.is_set())
            self.assertEqual([message.text() for message in result.output], ["target"])

        asyncio.run(scenario())

    def test_graph_run_can_only_start_once(self):
        async def scenario():
            started = asyncio.Event()
            release = asyncio.Event()

            class BlockingNode(Node):
                name = "blocking"

                def prepare_downstream_history(self, upstream_outputs, history):
                    return list(history)

                async def invoke(self, ctx, history, upstream_outputs, **extra):
                    started.set()
                    await release.wait()
                    return NodeResult(self, Message.assistant_text("done"))

                def kind(self):
                    return NodeKind.LLM

            graph = (
                GraphBuilder("non_reentrant")
                .input([Message.user_text("start")])
                .node(BlockingNode())
                .start("blocking")
                .build()
            )

            first_run = asyncio.create_task(graph.run())
            await started.wait()

            with self.assertRaisesRegex(RuntimeError, "has already run"):
                await graph.run()

            release.set()
            result = await first_run
            self.assertEqual([message.text() for message in result.output], ["done"])

            with self.assertRaisesRegex(RuntimeError, "has already run"):
                await graph.run()

        asyncio.run(scenario())

    def test_graph_returns_start_node_output_without_end_node(self):
        def echo(ctx, history, upstream_outputs):
            return Message.assistant_text(history[-1].text())

        graph = (
            GraphBuilder("no_end")
            .input([Message.user_text("start")])
            .node(CallableNode("echo", echo))
            .start("echo")
            .build()
        )

        result = asyncio.run(graph.run())

        self.assertEqual([message.text() for message in result.output], ["start"])

    def test_edge_predicate_blocks_target_activation(self):
        target_inputs = []
        predicate_results = []
        predicate_edges = []
        predicate_targets = []

        def source(ctx, history, upstream_outputs):
            return Message.assistant_text("skip")

        def target(ctx, history, upstream_outputs):
            target_inputs.append(upstream_outputs)
            return Message.assistant_text("target")

        def active(result, edge, downstream_node):
            predicate_results.append(result)
            predicate_edges.append(edge)
            predicate_targets.append(downstream_node)
            return result.output.text() == "go"

        graph = (
            GraphBuilder("conditional_edge_blocked")
            .node(CallableNode("source", source))
            .node(CallableNode("target", target))
            .start("source")
            .edge("source", "target", name="source_to_target", active=active)
            .build()
        )

        result = asyncio.run(graph.run())

        self.assertEqual([message.text() for message in result.output], ["skip"])
        self.assertEqual(target_inputs, [])
        self.assertEqual(len(predicate_results), 1)
        self.assertIsInstance(predicate_results[0], NodeResult)
        self.assertEqual(predicate_results[0].output.text(), "skip")
        self.assertEqual([edge.name for edge in predicate_edges], ["source_to_target"])
        self.assertIs(predicate_edges[0], graph.node_states["source"].out_edges[0])
        self.assertIs(predicate_targets[0], graph.nodes["target"])
        self.assertEqual(graph.node_states["target"].finished_dependency, 0)
        self.assertEqual(graph.node_states["target"].dependency_results, {})

    def test_edge_predicate_allows_target_activation(self):
        target_inputs = []
        predicate_results = []
        predicate_edges = []
        predicate_targets = []

        def source(ctx, history, upstream_outputs):
            return Message.assistant_text("go")

        def target(ctx, history, upstream_outputs):
            target_inputs.append(
                {
                    edge_name: output.text()
                    for edge_name, output in upstream_outputs.items()
                }
            )
            return Message.assistant_text("target")

        def active(result, edge, downstream_node):
            predicate_results.append(result)
            predicate_edges.append(edge)
            predicate_targets.append(downstream_node)
            return result.output.text() == "go"

        graph = (
            GraphBuilder("conditional_edge_allowed")
            .node(CallableNode("source", source))
            .node(CallableNode("target", target))
            .start("source")
            .edge("source", "target", name="source_to_target", active=active)
            .build()
        )

        result = asyncio.run(graph.run())

        self.assertEqual([message.text() for message in result.output], ["target"])
        self.assertEqual(target_inputs, [{"source_to_target": "go"}])
        self.assertEqual(len(predicate_results), 1)
        self.assertIsInstance(predicate_results[0], NodeResult)
        self.assertEqual(predicate_results[0].output.text(), "go")
        self.assertEqual([edge.name for edge in predicate_edges], ["source_to_target"])
        self.assertIs(predicate_edges[0], graph.node_states["source"].out_edges[0])
        self.assertIs(predicate_targets[0], graph.nodes["target"])
        self.assertEqual(graph.node_states["target"].finished_dependency, 1)

    def test_node_waits_for_all_dependencies_before_activation(self):
        join_inputs = []

        def seed(ctx, history, upstream_outputs):
            return Message.assistant_text(history[-1].text())

        def source(name):
            return lambda ctx, history, upstream_outputs: Message.assistant_text(name)

        def join(ctx, history, upstream_outputs):
            history_texts = [message.text() for message in history]
            output_texts = {
                edge_name: message.text()
                for edge_name, message in upstream_outputs.items()
            }
            join_inputs.append((history_texts, output_texts))
            return Message.assistant_text("+".join([*history_texts, *output_texts.values()]))

        graph = (
            GraphBuilder("dependency_join")
            .input([Message.user_text("start")])
            .node(
                CallableNode(
                    "seed",
                    seed,
                    prepare_downstream_history=lambda upstream_outputs, history: [],
                )
            )
            .node(CallableNode("left", source("left")))
            .node(CallableNode("right", source("right")))
            .node(CallableNode("join", join))
            .start("seed")
            .edge("seed", "left", name="to_left")
            .edge("seed", "right", name="to_right")
            .edge("left", "join", name="left_result")
            .edge("right", "join", name="right_result")
            .build()
        )

        self.assertEqual(graph.node_states["join"].depends_on, 2)

        result = asyncio.run(graph.run())

        self.assertEqual(
            join_inputs,
            [(["start"], {"left_result": "left", "right_result": "right"})],
        )
        self.assertEqual(
            [message.text() for message in result.output],
            ["start+left+right"],
        )
        self.assertEqual(graph.node_states["join"].finished_dependency, 2)

    def test_completed_dependency_can_reactivate_node(self):
        prepare_downstream_history_calls = []

        def source(ctx, history, upstream_outputs):
            return Message.assistant_text(history[-1].text())

        def target(ctx, history, upstream_outputs):
            return Message.assistant_text(upstream_outputs["source_to_target"].text())

        def prepare_target(upstream_outputs, history):
            prepare_downstream_history_calls.append(
                upstream_outputs["source_to_target"].text()
            )
            return list(history)

        graph = (
            GraphBuilder("repeat_activation")
            .node(CallableNode("source", source))
            .node(
                CallableNode(
                    "target",
                    target,
                    prepare_downstream_history=prepare_target,
                )
            )
            .start("source")
            .edge("source", "target", name="source_to_target")
            .build()
        )
        source_node = graph.nodes["source"]

        first_activation = graph.active_next_nodes(
            CompletedNode(
                NodeActivation(
                    node=source_node,
                    history=[],
                    upstream_outputs={},
                    downstream_history=[],
                ),
                NodeResult(source_node, Message.assistant_text("first")),
            )
        )
        second_activation = graph.active_next_nodes(
            CompletedNode(
                NodeActivation(
                    node=source_node,
                    history=[],
                    upstream_outputs={},
                    downstream_history=[],
                ),
                NodeResult(source_node, Message.assistant_text("second")),
            )
        )

        self.assertEqual(
            [activation.node.name for activation in first_activation],
            ["target"],
        )
        self.assertEqual(
            [activation.node.name for activation in second_activation],
            ["target"],
        )
        self.assertEqual(prepare_downstream_history_calls, ["first", "second"])
        self.assertEqual(graph.node_states["target"].finished_dependency, 1)
        target_dependency = graph.node_states["target"].dependency_results[
            "source_to_target"
        ]
        self.assertEqual(target_dependency.output.text(), "second")

    def test_upstream_output_does_not_merge_completed_history(self):
        def source(ctx, history, upstream_outputs):
            return Message.assistant_text("source")

        def target(ctx, history, upstream_outputs):
            return Message.assistant_text(upstream_outputs["source_to_target"].text())

        graph = (
            GraphBuilder("unmerged_upstream_output")
            .node(CallableNode("source", source))
            .node(CallableNode("target", target))
            .start("source")
            .edge("source", "target", name="source_to_target")
            .build()
        )
        source_node = graph.nodes["source"]

        activations = graph.active_next_nodes(
            CompletedNode(
                NodeActivation(
                    node=source_node,
                    history=[Message.user_text("history")],
                    upstream_outputs={},
                    downstream_history=[Message.user_text("history")],
                ),
                NodeResult(source_node, Message.assistant_text("source")),
            )
        )

        self.assertEqual([activation.node.name for activation in activations], ["target"])
        target_dependency = graph.node_states["target"].dependency_results[
            "source_to_target"
        ]
        self.assertEqual(target_dependency.output.text(), "source")
        self.assertEqual(
            activations[0].upstream_outputs["source_to_target"].text(),
            "source",
        )

    def test_single_completed_node_activates_each_target_once(self):
        prepare_downstream_history_calls = []

        def source(ctx, history, upstream_outputs):
            return Message.assistant_text("source")

        def target(ctx, history, upstream_outputs):
            return Message.assistant_text(
                "+".join(output.text() for output in upstream_outputs.values())
            )

        def prepare_target(upstream_outputs, history):
            prepare_downstream_history_calls.append(
                {
                    edge_name: output.text()
                    for edge_name, output in upstream_outputs.items()
                }
            )
            return list(history)

        graph = (
            GraphBuilder("single_source_multiple_edges")
            .node(CallableNode("source", source))
            .node(
                CallableNode(
                    "target",
                    target,
                    prepare_downstream_history=prepare_target,
                )
            )
            .start("source")
            .edge("source", "target", name="first_edge")
            .edge("source", "target", name="second_edge")
            .build()
        )
        source_node = graph.nodes["source"]

        activations = graph.active_next_nodes(
            CompletedNode(
                NodeActivation(
                    node=source_node,
                    history=[],
                    upstream_outputs={},
                    downstream_history=[],
                ),
                NodeResult(source_node, Message.assistant_text("source")),
            )
        )

        self.assertEqual([activation.node.name for activation in activations], ["target"])
        self.assertEqual(
            {
                edge_name: output.text()
                for edge_name, output in activations[0].upstream_outputs.items()
            },
            {"first_edge": "source", "second_edge": "source"},
        )
        self.assertEqual([message.text() for message in activations[0].history], [])
        self.assertEqual(
            [message.text() for message in activations[0].downstream_history],
            [],
        )
        self.assertEqual(
            prepare_downstream_history_calls,
            [{"first_edge": "source", "second_edge": "source"}],
        )
        self.assertEqual(graph.node_states["target"].finished_dependency, 2)

    def test_prepare_downstream_history_receives_upstream_outputs_and_accumulated_history(self):
        prepare_downstream_history_calls = []
        invoke_inputs = []

        def first(ctx, history, upstream_outputs):
            return Message.assistant_text("first")

        def prepare_downstream_history_second(upstream_outputs, history):
            prepare_downstream_history_calls.append(
                (
                    {
                        edge_name: output.text()
                        for edge_name, output in upstream_outputs.items()
                    },
                    [message.text() for message in history],
                )
            )
            messages = list(history)
            for output in upstream_outputs.values():
                messages.append(output)
            return messages

        def second(ctx, history, upstream_outputs):
            invoke_inputs.append(
                (
                    [message.text() for message in history],
                    {
                        edge_name: output.text()
                        for edge_name, output in upstream_outputs.items()
                    },
                )
            )
            return Message.assistant_text("second")

        graph = (
            GraphBuilder("prepare_downstream_history")
            .input([Message.user_text("start")])
            .node(CallableNode("first", first))
            .node(
                CallableNode(
                    "second",
                    second,
                    prepare_downstream_history=prepare_downstream_history_second,
                )
            )
            .start("first")
            .edge("first", "second", name="first_to_second")
            .build()
        )

        result = asyncio.run(graph.run())

        self.assertEqual(
            prepare_downstream_history_calls,
            [({"first_to_second": "first"}, ["start"])],
        )
        self.assertEqual(invoke_inputs, [(["start"], {"first_to_second": "first"})])
        self.assertEqual([message.text() for message in result.output], ["second"])
        first_dependency = graph.node_states["second"].dependency_results["first_to_second"]
        self.assertIsInstance(first_dependency, CompletedNode)
        self.assertEqual(first_dependency.output.text(), "first")
        self.assertEqual([message.text() for message in first_dependency.history], ["start"])
        self.assertEqual(
            [message.text() for message in first_dependency.downstream_history],
            ["start"],
        )

    def test_node_does_not_activate_until_all_dependencies_finish(self):
        blocked_inputs = []

        def ready(ctx, history, upstream_outputs):
            return Message.assistant_text("ready")

        def blocked(ctx, history, upstream_outputs):
            blocked_inputs.append((history, upstream_outputs))
            return Message.assistant_text("blocked")

        graph = (
            GraphBuilder("blocked_dependency")
            .input([Message.user_text("start")])
            .node(CallableNode("ready", ready))
            .node(CallableNode("blocked", blocked))
            .start("ready")
            .edge("ready", "blocked", name="ready_to_blocked")
            .edge("missing", "blocked", name="missing_to_blocked")
            .build()
        )

        result = asyncio.run(graph.run())

        self.assertEqual(blocked_inputs, [])
        self.assertEqual([message.text() for message in result.output], ["ready"])
        self.assertEqual(graph.node_states["blocked"].depends_on, 2)
        self.assertEqual(graph.node_states["blocked"].finished_dependency, 1)

    def test_graph_output_comes_from_last_node(self):
        def seed(ctx, history, upstream_outputs):
            return Message.assistant_text(history[-1].text())

        def source(name):
            return lambda ctx, history, upstream_outputs: Message.assistant_text(name)

        def join(ctx, history, upstream_outputs):
            return Message.assistant_text(
                "+".join(output.text() for output in upstream_outputs.values())
            )

        graph = (
            GraphBuilder("last_node_output")
            .input([Message.user_text("start")])
            .node(
                CallableNode(
                    "seed",
                    seed,
                    prepare_downstream_history=lambda upstream_outputs, history: [],
                )
            )
            .node(CallableNode("left", source("left")))
            .node(CallableNode("right", source("right")))
            .node(CallableNode("join", join))
            .start("seed")
            .edge("seed", "left", name="to_left")
            .edge("seed", "right", name="to_right")
            .edge("left", "join", name="left_result")
            .edge("right", "join", name="right_result")
            .build()
        )

        result = asyncio.run(graph.run())

        self.assertIsInstance(result.output, list)
        self.assertEqual([message.text() for message in result.output], ["left+right"])
        self.assertEqual(graph.node_states["join"].depends_on, 2)
