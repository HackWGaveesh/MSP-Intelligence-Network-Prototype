"""LangGraph graph composition for MSP agents.

This module compiles a routing graph that behaves like a mixture-of-experts:
all requests enter through a router node that selects the appropriate agent
handler based on ``agent_type``.
"""
from __future__ import annotations

from typing import Any, Dict, TypedDict

from langgraph.graph import END, StateGraph

from .handlers import HANDLERS, initialize_models


class AgentState(TypedDict, total=False):
    agent_type: str
    payload: Dict[str, Any]
    result: Dict[str, Any]


def _router(state: AgentState) -> str:
    return state["agent_type"]


def _make_node(agent_key: str):
    def node(state: AgentState) -> AgentState:
        handler = HANDLERS[agent_key]
        state["result"] = handler(state.get("payload", {}))
        return state

    return node


def build_graph() -> Any:
    initialize_models()
    graph = StateGraph(AgentState)
    graph.add_node("input", lambda s: s)
    graph.set_entry_point("input")

    agent_nodes = {}
    for key in HANDLERS:
        node_name = key
        graph.add_node(node_name, _make_node(key))
        graph.add_edge(node_name, END)
        agent_nodes[key] = node_name

    graph.add_conditional_edges("input", _router, agent_nodes)
    return graph.compile()


_AGENT_GRAPH = build_graph()


def run_graph(agent_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    state = {"agent_type": agent_type, "payload": payload}
    final_state = _AGENT_GRAPH.invoke(state)
    return final_state.get("result", {})



