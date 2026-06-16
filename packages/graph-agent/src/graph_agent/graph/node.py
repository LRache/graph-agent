"""Graph node runtime abstractions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from graph_agent.message import ContentBlock, Message, MessageRole
from graph_agent.runtime import RunContext

if TYPE_CHECKING:
    from graph_agent.graph.edge import Edge


class NodeKind(StrEnum):
    LLM = "llm"


Active = Callable[[list[bool]], bool]
UpstreamOutputs = dict[str, Message]
NodeCallable = Callable[[RunContext, list[Message], UpstreamOutputs], Message]


@dataclass(frozen=True)
class NodeResult:
    node: Node
    output: Message


def merge_messages(
    messages: list[Message],
    *,
    default_role: MessageRole = MessageRole.ASSISTANT,
) -> Message:
    if len(messages) == 1:
        return messages[0]

    blocks: list[ContentBlock] = []
    for message in messages:
        blocks.extend(message.blocks)
    return Message.of(default_role, blocks)


@runtime_checkable
class Node(Protocol):
    name: str

    def init_from_edges(
        self,
        in_edges: list[Edge],
        out_edges: list[Edge],
        graph_nodes: Mapping[str, Node],
    ) -> dict[str, Any]:
        return {}

    def prepare_downstream_history(
        self,
        upstream_outputs: UpstreamOutputs,
        history: list[Message],
    ) -> list[Message]:
        """根据直属上游输出和当前历史上下文，准备传给下游节点的历史上下文。

        约定：
        - invoke(history, upstream_outputs) 中的 history 是当前节点继承到的
          上游历史上下文。
        - upstream_outputs 只包含直属上游节点本次产生的 output message，
          按入边名称索引，不隐式携带任何历史上下文。
        - invoke 返回值只表示当前节点本次新产生的 message，不负责合并上下文。
        - prepare_downstream_history 的返回值只会作为下游节点看到的 history，
          不会再传回当前节点的 invoke。

        例：llm_a 激活 tool_a 和 tool_b，tool_a/tool_b 都完成后再激活 llm_b。
        llm_b.invoke 收到的 history 不包含 tool_a/tool_b 的输出；tool_a/tool_b
        的输出会单独放在 upstream_outputs 中，key 是入边名称，形如
        {"tool_a_result": message, "tool_b_result": message}。
        llm_b.prepare_downstream_history 的返回值会作为 llm_b 下游节点看到的 history。
        """
        raise NotImplementedError

    async def invoke(
        self,
        ctx: RunContext,
        history: list[Message],
        upstream_outputs: UpstreamOutputs,
        **extra: Any,
    ) -> NodeResult:
        """执行当前节点并返回本次新产生的 message。

        history 是当前节点继承到的上游历史上下文；upstream_outputs 只包含
        直属上游节点本次产生的 output message，按入边名称索引，不隐式携带
        任何历史上下文。返回的 NodeResult.output 只表示当前节点本次新产生的
        message，不应该把 history 合并进去。

        extra 是 build 阶段由 init_from_edges 返回并保存在 NodeState.extra
        中的节点初始化数据，会在每次 invoke 时原样展开传入。

        如果当前节点希望调整下游节点看到的 history，应通过
        prepare_downstream_history 实现；该方法的返回值不会传回当前 invoke。
        """
        raise NotImplementedError

    def kind(self) -> NodeKind:
        raise NotImplementedError
