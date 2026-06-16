import asyncio
import unittest

from graph_agent import (
    Edge,
    FunctionTool,
    GraphBuilder,
    LLMNode,
    Message,
    NodeState,
    ToolCallNode,
    ToolSchema,
)

from tests.helpers import CallableNode, StaticProvider



class GraphBuilderTests(unittest.TestCase):
    def test_edge_has_name_source_and_target(self):
        edge = Edge("to_join", "left", "join")

        self.assertEqual(edge.name, "to_join")
        self.assertEqual(edge.source, "left")
        self.assertEqual(edge.target, "join")
        self.assertNotIn("tool_schemas", Edge.__dataclass_fields__)
        self.assertNotIn("signal", Edge.__dataclass_fields__)

    def test_build_collects_tool_schemas_from_downstream_tool_nodes(self):
        llm_node = LLMNode("llm", StaticProvider())
        lookup_node = ToolCallNode("lookup_tools")
        echo_node = ToolCallNode("echo_tools")
        builder = (
            GraphBuilder("tool_schema_edges")
            .node(llm_node)
            .edge("llm", "lookup_tools", name="llm_to_lookup_tools")
            .edge("llm", "echo_tools", name="llm_to_echo_tools")
            .node(lookup_node)
            .node(echo_node)
            .start("llm")
        )
        lookup_node.register_tool(
            FunctionTool(
                "lookup",
                lambda args: args["query"],
                description="Look up a value.",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            )
        )
        echo_node.register_tool(
            FunctionTool(
                "echo",
                lambda args: args["value"],
                description="Echo a value.",
                parameters={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            )
        )

        graph = builder.build()

        lookup_schema = ToolSchema(
            name="lookup",
            description="Look up a value.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        echo_schema = ToolSchema(
            name="echo",
            description="Echo a value.",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        )
        self.assertNotIn("tool_schemas", NodeState.__dataclass_fields__)
        self.assertNotIn("tool_schemas", Edge.__dataclass_fields__)
        self.assertEqual(
            graph.node_states["llm"].extra["tools"],
            (lookup_schema, echo_schema),
        )
        self.assertFalse(hasattr(llm_node, "tool_schemas"))
        self.assertEqual(graph.node_states["lookup_tools"].extra["tools"], ())
        self.assertEqual(graph.node_states["echo_tools"].extra["tools"], ())

    def test_build_stores_node_init_extra_in_node_state(self):
        class InspectingNode(CallableNode):
            def init_from_edges(self, in_edges, out_edges, graph_nodes):
                return {
                    "edge_counts": (len(in_edges), len(out_edges)),
                    "known_nodes": tuple(sorted(graph_nodes)),
                }

        graph = (
            GraphBuilder("node_init_extra")
            .node(
                CallableNode(
                    "source",
                    lambda ctx, history, upstream: Message.assistant_text("source"),
                )
            )
            .node(
                InspectingNode(
                    "middle",
                    lambda ctx, history, upstream: Message.assistant_text("middle"),
                )
            )
            .node(
                CallableNode(
                    "target",
                    lambda ctx, history, upstream: Message.assistant_text("target"),
                )
            )
            .start("source")
            .edge("source", "middle", name="source_to_middle")
            .edge("middle", "target", name="middle_to_target")
            .build()
        )

        self.assertEqual(graph.node_states["middle"].extra["edge_counts"], (1, 1))
        self.assertEqual(
            graph.node_states["middle"].extra["known_nodes"],
            ("middle", "source", "target"),
        )
        self.assertEqual(graph.node_states["middle"].extra["tools"], ())

    def test_build_rejects_non_mapping_node_init_extra(self):
        class InvalidInitNode(CallableNode):
            def init_from_edges(self, in_edges, out_edges, graph_nodes):
                return None

        with self.assertRaisesRegex(TypeError, "must return a mapping"):
            (
                GraphBuilder("invalid_node_extra")
                .node(
                    InvalidInitNode(
                        "invalid",
                        lambda ctx, history, upstream: Message.assistant_text(
                            "invalid"
                        ),
                    )
                )
                .start("invalid")
                .build()
            )

    def test_builder_build_returns_distinct_equivalent_graphs(self):
        def echo(ctx, history, upstream_outputs):
            return Message.assistant_text(history[-1].text())

        builder = (
            GraphBuilder("repeatable_build")
            .input([Message.user_text("configured")])
            .node(CallableNode("echo", echo))
            .start("echo")
        )

        first = builder.build()
        second = builder.build()

        self.assertIsNot(first, second)
        self.assertFalse(hasattr(builder, "graph"))
        self.assertIsNot(first.nodes, second.nodes)
        self.assertIsNot(first.edges, second.edges)
        self.assertIsNot(first.node_states, second.node_states)
        self.assertEqual(first.name, second.name)
        self.assertEqual(first.start_node, second.start_node)
        self.assertEqual(first.input_messages, second.input_messages)
        self.assertEqual(first.edges, second.edges)
        self.assertEqual(set(first.nodes), set(second.nodes))

        result = asyncio.run(first.run())

        self.assertEqual([message.text() for message in result.output], ["configured"])

    def test_graph_requires_start_node(self):
        def echo(ctx, history, upstream_outputs):
            return Message.assistant_text(history[-1].text())

        with self.assertRaisesRegex(KeyError, "start node"):
            (
                GraphBuilder("missing_start")
                .input([Message.user_text("start")])
                .node(CallableNode("echo", echo))
                .build()
            )

    def test_graph_allows_only_one_start_node(self):
        def first(ctx, history, upstream_outputs):
            return Message.assistant_text("first")

        def second(ctx, history, upstream_outputs):
            return Message.assistant_text("second")

        builder = (
            GraphBuilder("duplicate_start")
            .node(CallableNode("first", first))
            .node(CallableNode("second", second))
            .start("first")
        )

        builder.start("first")
        with self.assertRaisesRegex(ValueError, "already has start node first"):
            builder.start("second")

    def test_graph_requires_unique_edge_names(self):
        def source(ctx, history, upstream_outputs):
            return Message.assistant_text("source")

        builder = (
            GraphBuilder("duplicate_edge")
            .node(CallableNode("source", source))
            .node(CallableNode("left", source))
            .node(CallableNode("right", source))
            .start("source")
            .edge("source", "left", name="duplicate")
        )

        with self.assertRaisesRegex(KeyError, "Edge with name duplicate"):
            builder.edge("source", "right", name="duplicate")
