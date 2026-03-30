"""Generic factory for specialist ReAct agents.

Creates a 3-node subgraph:
    agent_node → (tool_node →)* agent_node → finalize_node → END

Each tool call is gated by an interrupt() approval request.
The finalize_node writes the agent's last message content to `output_key`.
"""

from __future__ import annotations

from typing import Callable, Type

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from llm_config import get_llm
from gmail.client import get_gmail_tools, get_readonly_gmail_tools


def create_specialist_agent(
    state_class: Type,
    system_prompt: str,
    output_key: str,
    output_status: str,
    agent_name: str,
    seed_message_fn: Callable,
    readonly: bool = True,
    use_tools: bool = True,
    temperature: float = 0,
) -> CompiledStateGraph:
    """Build and compile a specialist ReAct subgraph.

    Args:
        state_class:      The TypedDict subgraph state class.
        system_prompt:    System prompt for the LLM.
        output_key:       State key to write the final LLM output to.
        output_status:    Value to write to `status` when done.
        agent_name:       Shown in interrupt approval payloads.
        seed_message_fn:  callable(state) -> str — the initial HumanMessage content.
        readonly:         If True, use get_readonly_gmail_tools (no send_message).
        use_tools:        If False, LLM gets no tools (composer, reviewer).
        temperature:      LLM temperature.
    """
    _llm: list = [None]  # mutable container for lazy init

    def _get_llm():
        if _llm[0] is None:
            _llm[0] = get_llm(agent_name, temperature)
        return _llm[0]

    async def agent_node(state) -> dict:
        if use_tools:
            tools = (
                await get_readonly_gmail_tools()
                if readonly
                else await get_gmail_tools()
            )
            llm = _get_llm().bind_tools(list(tools.values()))
        else:
            tools = {}
            llm = _get_llm()

        messages = state.get("messages") or []

        if not messages:
            messages = [
                SystemMessage(system_prompt),
                HumanMessage(seed_message_fn(state)),
            ]

        response = await llm.ainvoke(messages)
        return {"messages": [response]}

    async def tool_node(state) -> dict:  # only reachable when use_tools=True
        last = state["messages"][-1]
        tool_calls = last.tool_calls

        approval = interrupt({
            "type": "tool_approval",
            "agent": agent_name,
            "task": agent_name,
            "tool_name": ", ".join(tc["name"] for tc in tool_calls),
            "tool_args": tool_calls[0]["args"] if len(tool_calls) == 1 else {
                tc["name"]: tc["args"] for tc in tool_calls
            },
            "message": f"[{agent_name}] wants to call: {', '.join(tc['name'] for tc in tool_calls)}",
        })

        tool_messages = []
        tools = (
            await get_readonly_gmail_tools()
            if readonly
            else await get_gmail_tools()
        )

        for tc in tool_calls:
            if approval.get("approved", True):
                try:
                    result = await tools[tc["name"]].ainvoke(tc["args"])
                    content = str(result)
                except Exception as e:
                    content = f"Error calling {tc['name']}: {e}"
            else:
                content = f"Tool call denied by user: {approval.get('reason', '')}"

            tool_messages.append(
                ToolMessage(content=content, tool_call_id=tc["id"])
            )

        return {"messages": tool_messages}

    async def finalize_node(state) -> dict:
        output = state["messages"][-1].content.strip()
        return {output_key: output, "status": output_status}

    def _route(state) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tool_node"
        return "finalize_node"

    subgraph = StateGraph(state_class)
    subgraph.add_node("agent_node", agent_node)
    subgraph.add_node("tool_node", tool_node)
    subgraph.add_node("finalize_node", finalize_node)

    subgraph.set_entry_point("agent_node")
    subgraph.add_conditional_edges("agent_node", _route, {
        "tool_node": "tool_node",
        "finalize_node": "finalize_node",
    })
    subgraph.add_edge("tool_node", "agent_node")
    subgraph.add_edge("finalize_node", END)

    return subgraph.compile()
