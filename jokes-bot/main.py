from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from state import JokeState
from nodes import all_nodes, show_menu


def build_joke_graph() -> CompiledStateGraph:
    workflow = StateGraph(JokeState)

    workflow.add_node("show_menu", show_menu)

    for node in all_nodes:
        workflow.add_node(node["name"], node["function"])

    workflow.set_entry_point("show_menu")

    workflow.add_conditional_edges(
        "show_menu",
        lambda state: state.user_choice,
    )

    workflow.add_edge("next_joke", "show_menu")
    workflow.add_edge("change_category", "show_menu")
    workflow.add_edge("change_language", "show_menu")
    workflow.add_edge("reset_jokes", "show_menu")
    workflow.add_edge("exit_bot", END)

    return workflow.compile()


def main():
    graph = build_joke_graph()
    # Save png image of the graph
    # graph.get_graph().draw_mermaid_png(output_file_path="joke_graph.png")
    graph.invoke(JokeState(), {"recursion_limit": 100})


if __name__ == "__main__":
    main()
