from operator import add
from typing import Annotated, List, Literal
from pydantic import BaseModel
from pyjokes import get_joke, CATEGORY_VALUES
import inquirer
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, END


class Joke(BaseModel):
    text: str
    category: str


class JokeState(BaseModel):
    jokes: Annotated[List[Joke], add] = []
    user_choice: Literal["n", "c", "q"] = "n"  # next, category, quit
    category: str = "neutral"
    language: str = "en"
    quit: bool = False


def show_menu(state: JokeState) -> dict:
    print("-" * 80)
    print(f"You fetched {len(state.jokes)} jokes.")
    print("Select menu option:")

    choices = {"Next joke": "n", "Change category": "c", "Exit": "q"}

    choice = inquirer.prompt(
        [
            inquirer.List(
                "choice",
                message="What category do you want to use?",
                choices=choices.keys(),
            )
        ]
    )
    print("-" * 80)

    return {"user_choice": choices[choice["choice"]]}


def fetch_joke(state: JokeState) -> dict:
    joke_text = get_joke(language=state.language, category=state.category)
    new_joke = Joke(text=joke_text, category=state.category)
    print(joke_text)
    return {"jokes": [new_joke]}


def update_category(state: JokeState) -> dict:
    new_category = inquirer.prompt(
        [
            inquirer.List(
                "category",
                message="What category do you want to use?",
                choices=CATEGORY_VALUES,
            )
        ]
    )

    return {"category": new_category["category"]}


def exit_bot(state: JokeState) -> dict:
    return {"quit": True}


def route_choice(state: JokeState) -> str:
    if state.user_choice == "n":
        return "fetch_joke"
    elif state.user_choice == "c":
        return "update_category"
    elif state.user_choice == "q":
        return "exit_bot"

    return "exit_bot"


def build_joke_graph() -> CompiledStateGraph:
    workflow = StateGraph(JokeState)

    workflow.add_node("show_menu", show_menu)
    workflow.add_node("fetch_joke", fetch_joke)
    workflow.add_node("update_category", update_category)
    workflow.add_node("exit_bot", exit_bot)

    workflow.set_entry_point("show_menu")

    workflow.add_conditional_edges(
        "show_menu",
        route_choice,
        {
            "fetch_joke": "fetch_joke",
            "update_category": "update_category",
            "exit_bot": "exit_bot",
        },
    )

    workflow.add_edge("fetch_joke", "show_menu")
    workflow.add_edge("update_category", "show_menu")
    workflow.add_edge("exit_bot", END)

    return workflow.compile()


def main():
    graph = build_joke_graph()
    graph.invoke(JokeState(), {"recursion_limit": 100})


if __name__ == "__main__":
    main()
