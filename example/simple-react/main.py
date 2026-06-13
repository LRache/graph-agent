from graph_agent.graph import *
from graph_agent.message import Message
from graph_agent.provider import OpenAIProvider, LLMNode
from graph_agent.tool import FunctionTool, ToolCallNode
from openai import AsyncOpenAI

import asyncio
import os

SYSTEM_PROMPT = """你是一个 llm，现在需要你帮助我测试我的 agent 框架中的工具调用。"""

OPENAI_KEY = os.getenv("OPENAI_KEY", None)
OPENAI_BASEURL = os.getenv("OPENAI_BASEURL", None)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")

if OPENAI_KEY is None or OPENAI_BASEURL is None:
    raise ValueError("OPENAI_KEY and OPENAI_BASEURL environment variables must be set")

print("Using OpenAI provider with model:", OPENAI_MODEL, "at base URL:", OPENAI_BASEURL, ", key =", OPENAI_KEY[:4] + "...")

if __name__ == "__main__":
    tool_node = ToolCallNode(
        "tool_node", 
        FunctionTool(
            "add", 
            lambda args: str(args["a"] + args["b"]),
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
            description="Add a and b"
        )
    )
    
    builder = GraphBuilder()
    builder.node(
        LLMNode(
            "llm", 
            OpenAIProvider(
                OPENAI_MODEL, 
                AsyncOpenAI(
                    api_key=OPENAI_KEY, 
                    base_url=OPENAI_BASEURL
                ),
                api="chat_completions"
            ),
            system_prompt=SYSTEM_PROMPT
        )
    )
    builder.node(tool_node)
    builder.edge(
        source="llm",
        target="tool_node",
        name="llm->toolcall",
        active=matches_any_tool_call_for_downstream,
    )
    builder.edge(
        source="tool_node",
        target="llm",
        name="toolcall->llm",
    )
    builder.start("llm")

    builder.input([Message.user_text("使用工具计算 2 + 2")])

    graph = builder.build()
    result = asyncio.run(graph.run())

    print("Graph execution complete.")
    print("history:", result.history)
    print("Final result:", result.output)
