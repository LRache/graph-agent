import asyncio
import unittest

from graph_agent import (
    ContentBlock,
    FunctionTool,
    Message,
    MessageRole,
    ToolCallBlock,
    ToolCallNode,
    ToolExecutor,
    ToolRegistry,
    ToolResultBlock,
)
from graph_agent.runtime import RunContext



class ToolTests(unittest.TestCase):
    def test_tool_output_rejects_unstructured_values(self):
        registry = ToolRegistry()
        registry.register(FunctionTool("lookup", lambda _: {"raw": "value"}))
        executor = ToolExecutor(registry)

        result = asyncio.run(executor.execute_call(ToolCallBlock("call_1", "lookup")))

        self.assertEqual(result.role, MessageRole.TOOL)
        tool_result = result.blocks[0]
        assert isinstance(tool_result, ToolResultBlock)
        self.assertTrue(tool_result.is_error)
        self.assertIn("tool output must be", tool_result.content)

    def test_tool_call_node_invokes_registered_tools_from_upstream_messages(self):
        node = ToolCallNode()
        node.register_tool(
            FunctionTool(
                "lookup",
                lambda args: f"found {args['query']}",
            )
        )
        node.register_tool(
            FunctionTool(
                "echo",
                lambda args: args["value"],
            )
        )
        upstream = Message.of(
            MessageRole.ASSISTANT,
            [
                ContentBlock.text_block("checking tools"),
                ContentBlock.tool_call(
                    ToolCallBlock("call_1", "lookup", {"query": "weather"})
                ),
                ContentBlock.tool_call(
                    ToolCallBlock("call_2", "echo", {"value": "hello"})
                ),
            ],
        )

        result = asyncio.run(node.invoke(RunContext(), [], {"llm_to_tools": upstream}))

        self.assertIs(result.node, node)
        self.assertEqual(result.output.role, MessageRole.TOOL)
        self.assertEqual(
            result.output.blocks,
            (
                ToolResultBlock(
                    call_id="call_1",
                    tool_name="lookup",
                    content="found weather",
                ),
                ToolResultBlock(
                    call_id="call_2",
                    tool_name="echo",
                    content="hello",
                ),
            ),
        )

    def test_tool_executor_returns_error_result_for_unavailable_tool(self):
        executor = ToolExecutor(ToolRegistry())
        result = asyncio.run(
            executor.execute_call(
                ToolCallBlock("call_1", "missing", {"query": "weather"})
            )
        )

        self.assertEqual(result.role, MessageRole.TOOL)
        self.assertEqual(len(result.blocks), 1)
        block = result.blocks[0]
        assert isinstance(block, ToolResultBlock)
        self.assertEqual(block.call_id, "call_1")
        self.assertEqual(block.tool_name, "missing")
        self.assertTrue(block.is_error)
        self.assertIn("tool not found", block.content)

    def test_tool_call_node_ignores_unavailable_tools(self):
        node = ToolCallNode()
        upstream = Message.tool_call("call_1", "missing", {"query": "weather"})

        result = asyncio.run(node.invoke(RunContext(), [], {"llm_to_tools": upstream}))

        self.assertEqual(result.output.role, MessageRole.TOOL)
        self.assertEqual(result.output.blocks, ())

    def test_tool_call_node_uses_default_empty_init_from_edges(self):
        node = ToolCallNode()

        self.assertNotIn("init_from_edges", ToolCallNode.__dict__)
        self.assertEqual(node.init_from_edges([], [], {}), {})
