"""Mock ReAct-style graph for debugging the React GraphView stepper."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

from graph_agent import (
    GraphBuilder,
    Message,
    NodeKind,
    NodeResult,
    RunContext,
    UpstreamOutputs,
)
from graph_agent_viewer import GraphView


@dataclass(frozen=True)
class MockReactNode:
    name: str
    output_text: str
    delay_seconds: float = 0.35

    def prepare_downstream_history(
        self,
        upstream_outputs: UpstreamOutputs,
        history: list[Message],
    ) -> list[Message]:
        return [*history, *upstream_outputs.values()]

    async def invoke(
        self,
        ctx: RunContext,
        history: list[Message],
        upstream_outputs: UpstreamOutputs,
        **extra: object,
    ) -> NodeResult:
        await asyncio.sleep(self.delay_seconds)
        return NodeResult(self, Message.assistant_text(self.output_text))

    def kind(self) -> NodeKind:
        return NodeKind.LLM


def build_graph():
    builder = GraphBuilder("mock_react_stepper")
    builder.input([Message.user_text("mock request: plan, act, observe, answer")])

    nodes = [
        MockReactNode("prompt", "Read the user request and initialize state."),
        MockReactNode("reason", "Decide which tool would be useful."),
        MockReactNode("act", "Call the mocked tool with structured arguments."),
        MockReactNode("observe", "Receive the mocked tool result."),
        MockReactNode("answer", "Compose the final answer from the observation."),
    ]
    edge_names = ["plan", "tool", "result", "final"]
    for node in nodes:
        builder.node(node)

    builder.start(nodes[0].name)
    for source, target, edge_name in zip(nodes, nodes[1:], edge_names):
        builder.edge(
            source.name,
            target.name,
            name=edge_name,
        )

    return builder.build()


if __name__ == "__main__":
    graph = build_graph()
    result = GraphView.run(
        graph,
        host=os.getenv("GRAPH_VIEW_HOST", "127.0.0.1"),
        port=int(os.getenv("GRAPH_VIEW_PORT", "0")),
        open_browser=os.getenv("GRAPH_VIEW_OPEN_BROWSER", "1") != "0",
        keep_open=os.getenv("GRAPH_VIEW_KEEP_OPEN", "1") != "0",
        step_mode=True,
    )
    print("Graph execution complete.")
    print("history:", result.history)
    print("Final result:", result.output)
