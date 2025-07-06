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
    user_choice: Literal["n", "c", "l", "q"] = "n"  # next, category, language, quit
    category: str = "neutral"
    language: str = "en"
    quit: bool = False


def show_menu(state: JokeState) -> dict:
    print("-" * 80)
    print(
        f"| MENU | Category: {state.category} | Language: {state.language} | Jokes: {len(state.jokes)} |"
    )
    print("Select an option:")

    choices = {
        "Next joke": "n",
        "Change category": "c",
        "Change language": "l",
        "Exit": "q",
    }

    choice = inquirer.prompt(
        [
            inquirer.List(
                "choice",
                message="What option do you want to use?",
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


def route_choice(state: JokeState) -> str:
    if state.user_choice == "n":
        return "fetch_joke"
    elif state.user_choice == "c":
        return "update_category"
    elif state.user_choice == "l":
        return "change_language"
    elif state.user_choice == "q":
        return "exit_bot"

    return "exit_bot"


def build_joke_graph() -> CompiledStateGraph:
    workflow = StateGraph(JokeState)

    workflow.add_node("show_menu", show_menu)
    workflow.add_node("fetch_joke", fetch_joke)
    workflow.add_node("change_category", change_category)
    workflow.add_node("change_language", change_language)
    workflow.add_node("exit_bot", exit_bot)

    workflow.set_entry_point("show_menu")

    workflow.add_conditional_edges(
        "show_menu",
        route_choice,
        {
            "fetch_joke": "fetch_joke",
            "change_category": "change_category",
            "change_language": "change_language",
            "exit_bot": "exit_bot",
        },
    )

    workflow.add_edge("fetch_joke", "show_menu")
    workflow.add_edge("change_category", "show_menu")
    workflow.add_edge("change_language", "show_menu")
    workflow.add_edge("exit_bot", END)

    return workflow.compile()


def main():
    graph = build_joke_graph()
    graph.invoke(JokeState(), {"recursion_limit": 100})


if __name__ == "__main__":
    main()
