from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from nodes import (
    critic_joke,
    main_menu_nodes,
    retries_end,
    show_approved_joke,
    show_menu,
)
from state import JokeState


def critic_router(state: JokeState):
    if state.approved:
        return "show_approved_joke"
    elif len(state.rejected_jokes) > 4:
        return "retries_end"
    else:
        return "generate_joke"


def build_joke_graph() -> CompiledStateGraph:
    workflow = StateGraph(JokeState)

    workflow.add_node("show_menu", show_menu)

    for node in main_menu_nodes:
        workflow.add_node(node["name"], node["function"])

    workflow.add_node("critic_joke", critic_joke)
    workflow.add_node("show_approved_joke", show_approved_joke)
    workflow.add_node("retries_end", retries_end)

    workflow.set_entry_point("show_menu")

    workflow.add_conditional_edges(
        "show_menu",
        lambda state: state.user_choice,
    )

    workflow.add_conditional_edges("critic_joke", critic_router)

    workflow.add_edge("generate_joke", "critic_joke")
    workflow.add_edge("show_approved_joke", "show_menu")
    workflow.add_edge("retries_end", "show_menu")
    workflow.add_edge("change_category", "show_menu")
    workflow.add_edge("change_language", "show_menu")
    workflow.add_edge("show_saved_jokes", "show_menu")
    workflow.add_edge("reset_jokes", "show_menu")
    workflow.add_edge("exit_bot", END)

    return workflow.compile()


def main():
    load_dotenv()

    graph = build_joke_graph()
    # Save png image of the graph
    # graph.get_graph().draw_mermaid_png(output_file_path="joke_graph.png")
    graph.invoke(JokeState(), {"recursion_limit": 100})


if __name__ == "__main__":
    main()
