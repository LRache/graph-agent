import asyncio
import unittest

from graph_agent import (
    CompletedNode,
    ContentBlock,
    Edge,
    FunctionTool,
    GraphBuilder,
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



class GraphRuntimeTests(unittest.TestCase):
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
        self.assertEqual(graph.node_states["lookup_tools"].dependency_outputs, {})

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
                RuntimeEventName.NODE_STARTED,
                RuntimeEventName.NODE_FINISHED,
                RuntimeEventName.GRAPH_FINISHED,
            ],
        )
        self.assertEqual(events[1].payload["node"], "echo")
        self.assertEqual(events[1].payload["history"], [input_message])
        self.assertEqual(events[2].payload["output"].text(), "start")

    def test_graph_run_event_sink_waits_before_next_round(self):
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
                waits_for_activation_rounds = True

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

    def test_graph_run_event_sink_steps_full_rounds(self):
        async def scenario():
            gate_calls = []
            first_gate = asyncio.Event()
            second_gate = asyncio.Event()
            third_gate = asyncio.Event()
            releases = [asyncio.Event(), asyncio.Event(), asyncio.Event()]
            right_release = asyncio.Event()

            def immediate(name):
                return lambda ctx, history, upstream_outputs: Message.assistant_text(name)

            class BlockingSink:
                waits_for_activation_rounds = True

                async def __call__(self, event):
                    if event.name != RuntimeEventName.ACTIVATION_READY:
                        return
                    gate_calls.append(event.payload["nodes"])
                    if len(gate_calls) == 1:
                        first_gate.set()
                    if len(gate_calls) == 2:
                        second_gate.set()
                    if len(gate_calls) == 3:
                        third_gate.set()
                    await releases[len(gate_calls) - 1].wait()

            class BlockingRight(CallableNode):
                async def invoke(self, ctx, history, upstream_outputs, **extra):
                    await right_release.wait()
                    return await super().invoke(ctx, history, upstream_outputs, **extra)

            graph = (
                GraphBuilder("gated_rounds")
                .node(CallableNode("seed", immediate("seed")))
                .node(CallableNode("left", immediate("left")))
                .node(BlockingRight("right", immediate("right")))
                .node(CallableNode("left_child", immediate("left_child")))
                .node(CallableNode("right_child", immediate("right_child")))
                .start("seed")
                .edge("seed", "left", name="seed_to_left")
                .edge("seed", "right", name="seed_to_right")
                .edge("left", "left_child", name="left_to_child")
                .edge("right", "right_child", name="right_to_child")
                .build()
            )

            run_task = asyncio.create_task(graph.run(event_sink=BlockingSink()))
            await asyncio.wait_for(first_gate.wait(), timeout=1)
            self.assertEqual(gate_calls[0], ["seed"])

            releases[0].set()
            await asyncio.wait_for(second_gate.wait(), timeout=1)
            self.assertEqual(set(gate_calls[1]), {"left", "right"})

            releases[1].set()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            self.assertEqual(len(gate_calls), 2)
            right_release.set()
            await asyncio.wait_for(third_gate.wait(), timeout=1)
            self.assertEqual(set(gate_calls[2]), {"left_child", "right_child"})

            releases[2].set()
            result = await asyncio.wait_for(run_task, timeout=1)
            self.assertEqual(len(result.output), 1)

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
        self.assertEqual(graph.node_states["target"].dependency_outputs, {})

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
        self.assertEqual(
            graph.node_states["target"].dependency_outputs["source_to_target"].text(),
            "second",
        )

    def test_dependency_output_does_not_merge_completed_history(self):
        def source(ctx, history, upstream_outputs):
            return Message.assistant_text("source")

        def target(ctx, history, upstream_outputs):
            return Message.assistant_text(upstream_outputs["source_to_target"].text())

        graph = (
            GraphBuilder("unmerged_dependency_output")
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
        self.assertEqual(
            graph.node_states["target"].dependency_outputs["source_to_target"].text(),
            "source",
        )
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
        self.assertEqual(
            graph.node_states["second"].dependency_outputs["first_to_second"].text(),
            "first",
        )
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
