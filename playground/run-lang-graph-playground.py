from operator import add
from typing import Annotated, List, Literal
from pydantic import BaseModel
from pyjokes import get_joke, CATEGORY_VALUES, LANGUAGE_VALUES
import inquirer
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, END


class Joke(BaseModel):
    text: str
    category: str


class JokeState(BaseModel):
    jokes: Annotated[List[Joke], add] = []
    user_choice: str = "next_joke"
    category: str = "neutral"
    language: str = "en"
    quit: bool = False


def show_menu(state: JokeState) -> dict:
    print("-" * 80)
    print(
        f"| MENU | Category: {state.category} | Language: {state.language} | Jokes: {len(state.jokes)} |"
    )
    print("Select an option:")

    result = inquirer.prompt(
        [
            inquirer.List(
                "description",
                message="What option do you want to use?",
                choices=[node["description"] for node in nodes],
            )
        ]
    )
    print("-" * 80)

    user_choice = [
        node["name"] for node in nodes if node["description"] == result["description"]
    ][0]

    return {"user_choice": user_choice}


def next_joke(state: JokeState) -> dict:
    joke_text = get_joke(language=state.language, category=state.category)
    new_joke = Joke(text=joke_text, category=state.category)
    print(joke_text)
    return {"jokes": [new_joke]}


def change_category(state: JokeState) -> dict:
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


def change_language(state: JokeState) -> dict:
    print(f"Current language: {state.language}")

    new_language = ""

    while not new_language:
        user_input = input(
            "Enter your new language code ('en', 'de', 'es'), press q for exit: "
        )

        if user_input.strip().lower() == "q":
            return

        if user_input not in LANGUAGE_VALUES:
            print(f"The {new_language} is not available")
            continue

        new_language = user_input

    return {"language": new_language}


def exit_bot(state: JokeState) -> dict:
    return {"quit": True}


nodes = [
    {
        "name": "next_joke",
        "short": "n",
        "description": "Next joke",
        "function": next_joke,
    },
    {
        "name": "change_category",
        "short": "c",
        "description": "Change category",
        "function": change_category,
    },
    {
        "name": "change_language",
        "short": "l",
        "description": "Change language",
        "function": change_language,
    },
    {
        "name": "exit_bot",
        "short": "q",
        "description": "Exit",
        "function": exit_bot,
    },
]


def build_joke_graph() -> CompiledStateGraph:
    workflow = StateGraph(JokeState)

    workflow.add_node("show_menu", show_menu)

    for node in nodes:
        workflow.add_node(node["name"], node["function"])

    workflow.set_entry_point("show_menu")

    workflow.add_conditional_edges(
        "show_menu",
        lambda state: state.user_choice,
    )

    workflow.add_edge("next_joke", "show_menu")
    workflow.add_edge("change_category", "show_menu")
    workflow.add_edge("change_language", "show_menu")
    workflow.add_edge("exit_bot", END)

    return workflow.compile()


def main():
    graph = build_joke_graph()
    graph.invoke(JokeState(), {"recursion_limit": 100})


if __name__ == "__main__":
    main()
