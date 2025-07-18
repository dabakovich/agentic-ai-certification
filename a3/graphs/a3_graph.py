from typing import Any, Dict

from constants import MANAGER, REVIEWER, TITLE_GENERATOR, TLDR_GENERATOR
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
    manager_node = make_manager_node(a3_config["agents"][MANAGER]["llm"])
    graph.add_node(MANAGER, manager_node)

    tldr_generator_node = make_tldr_generator_node(
        a3_config["agents"][TLDR_GENERATOR]["llm"]
    )
    graph.add_node(TLDR_GENERATOR, tldr_generator_node)
    title_generator_node = make_title_generator_node(
        a3_config["agents"][TITLE_GENERATOR]["llm"]
    )
    graph.add_node(TITLE_GENERATOR, title_generator_node)

    reviewer_node = make_reviewer_node(a3_config["agents"][REVIEWER]["llm"])
    graph.add_node(REVIEWER, reviewer_node)

    # Add edges and flows
    graph.add_edge(START, MANAGER)

    graph.add_edge(MANAGER, TLDR_GENERATOR)
    graph.add_edge(MANAGER, TITLE_GENERATOR)

    graph.add_edge([TLDR_GENERATOR, TITLE_GENERATOR], REVIEWER)

    graph.add_conditional_edges(
        REVIEWER,
        route_from_reviewer,
        {
            TLDR_GENERATOR: TLDR_GENERATOR,
            TITLE_GENERATOR: TITLE_GENERATOR,
            "end": END,
        },
    )

    return graph.compile()
