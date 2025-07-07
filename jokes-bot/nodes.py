import inquirer
from constants import categories, languages
from langchain.schema import HumanMessage
from prompts import joke_prompt, system_message
from state import Joke, JokeState

from llm import llm


def show_menu(state: JokeState) -> dict:
    print("-" * 80)
    print(
        f"| MENU | Category: {state.category} | Language: {state.language} | Jokes: {len(state.jokes)} |"
    )
    print("Select an option:")

    result = inquirer.prompt(
        [
            inquirer.List(
                "name",
                message="What option do you want to use?",
                choices=[(node["description"], node["name"]) for node in all_nodes],
            )
        ]
    )
    print("-" * 80)

    return {"user_choice": result["name"]}


def next_joke(state: JokeState) -> dict:
    prompt = joke_prompt.format(
        category=state.category,
        language=state.language,
        prev_jokes="\n\n".join([joke.text for joke in state.jokes]),
    )
    llm_answer = llm.invoke([system_message, HumanMessage(prompt)])
    joke_text = llm_answer.content
    print(joke_text)
    new_joke = Joke(text=joke_text, category=state.category)
    return {"jokes": [new_joke]}


def change_category(state: JokeState) -> dict:
    new_category = inquirer.prompt(
        [
            inquirer.List(
                "category",
                message="What category do you want to use?",
                choices=categories,
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

        if user_input not in languages:
            print(f"The {new_language} is not available")
            continue

        new_language = user_input

    return {"language": new_language}


def reset_jokes(state: JokeState) -> dict:
    print("Clearing jokes...")
    return {"jokes": []}


def exit_bot(state: JokeState) -> dict:
    return {"quit": True}


all_nodes = [
    {
        "name": "next_joke",
        "description": "Next joke",
        "function": next_joke,
    },
    {
        "name": "change_category",
        "description": "Change category",
        "function": change_category,
    },
    {
        "name": "change_language",
        "description": "Change language",
        "function": change_language,
    },
    {
        "name": "reset_jokes",
        "description": "Reset jokes",
        "function": reset_jokes,
    },
    {
        "name": "exit_bot",
        "description": "Exit",
        "function": exit_bot,
    },
]
