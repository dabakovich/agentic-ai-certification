from typing import Any, Dict

from constants import NODE
from langgraph.graph import END, START, StateGraph
from nodes.a3_nodes import (
    make_manager_node,
    make_reviewer_node,
    make_title_generator_node,
    make_tldr_generator_node,
    route_from_reviewer,
)
from states.a3_state import A3State


def build_a3_graph(a3_config: Dict[str, Any]) -> StateGraph:
    graph = StateGraph(A3State)

    # Add nodes
    manager_node = make_manager_node(a3_config["agents"][NODE.MANAGER.value]["llm"])
    graph.add_node(NODE.MANAGER.value, manager_node)

    tldr_generator_node = make_tldr_generator_node(
        a3_config["agents"][NODE.TLDR_GENERATOR.value]["llm"]
    )
    graph.add_node(NODE.TLDR_GENERATOR.value, tldr_generator_node)
    title_generator_node = make_title_generator_node(
        a3_config["agents"][NODE.TITLE_GENERATOR.value]["llm"]
    )
    graph.add_node(NODE.TITLE_GENERATOR.value, title_generator_node)

    reviewer_node = make_reviewer_node(a3_config["agents"][NODE.REVIEWER.value]["llm"])
    graph.add_node(NODE.REVIEWER.value, reviewer_node)

    # Add edges and flows
    graph.add_edge(START, NODE.MANAGER.value)

    graph.add_edge(NODE.MANAGER.value, NODE.TLDR_GENERATOR.value)
    graph.add_edge(NODE.MANAGER.value, NODE.TITLE_GENERATOR.value)

    graph.add_edge(
        [NODE.TLDR_GENERATOR.value, NODE.TITLE_GENERATOR.value], NODE.REVIEWER.value
    )

    graph.add_conditional_edges(
        NODE.REVIEWER.value,
        route_from_reviewer,
        {
            NODE.TLDR_GENERATOR.value: NODE.TLDR_GENERATOR.value,
            NODE.TITLE_GENERATOR.value: NODE.TITLE_GENERATOR.value,
            "end": END,
        },
    )

    return graph.compile()
