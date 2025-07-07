import inquirer
from constants import (
    categories,
    fun_treshold,
    languages,
    is_provide_prev_jokes_in_context,
)
from langchain.schema import HumanMessage
from prompts import (
    critic_prompt,
    critic_system_message,
    writer_prompt,
    writer_system_message,
)
from state import Joke, JokeState

from llm import llm
from paths import VECTOR_DB_DIR
from vector_store import ChromaDB, VectorStore

store = VectorStore("jokes", chromadb=ChromaDB(VECTOR_DB_DIR))


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
                choices=[
                    (node["description"], node["name"]) for node in main_menu_nodes
                ],
                carousel=True,
            )
        ]
    )
    print("-" * 80)

    return {"user_choice": result["name"]}


def generate_joke(state: JokeState) -> dict:
    print("Generating new joke...")
    prompt = writer_prompt.format(
        category=state.category,
        language=state.language,
        prev_jokes="\n\n".join([joke.text for joke in state.jokes])
        if is_provide_prev_jokes_in_context
        else None,
        rejected_jokes="\n\n".join([joke.text for joke in state.rejected_jokes]),
    )
    llm_answer = llm.invoke([writer_system_message, HumanMessage(prompt)])
    joke_text = llm_answer.content

    print(
        f"Generated new joke: {joke_text}, rejected count: {len(state.rejected_jokes)}"
    )

    joke_for_review = Joke(text=joke_text, category=state.category)
    return {"joke_for_review": joke_for_review}


def critic_joke(state: JokeState) -> dict:
    if not is_provide_prev_jokes_in_context:
        print("Analazing if the joke was used...")
        is_joke_exists = store.is_joke_exists(state.joke_for_review)

        if is_joke_exists:
            print("Joke already exists, rejecting...")

            return {
                "approved": False,
                "rejected_jokes": [state.joke_for_review],
            }

    print("Analyzing how funny joke is...")
    prompt = critic_prompt.format(joke=state.joke_for_review.text)

    llm_answer = llm.invoke([critic_system_message, HumanMessage(prompt)])
    print(llm_answer.content)

    approved = True if float(llm_answer.content.strip()) > fun_treshold else False

    print(f"The joke was {'approved' if approved else 'rejected, trying again'}")

    return {
        "approved": approved,
        "rejected_jokes": [] if approved else [state.joke_for_review],
    }


def show_approved_joke(state: JokeState) -> dict:
    print(state.joke_for_review.text)

    store.insert_joke(state.joke_for_review)

    return {
        "jokes": [state.joke_for_review],
        "joke_for_review": None,
        "approved": False,
    }


def retries_end(state: JokeState) -> dict:
    print("No one joke was approved")
    return {"joke_for_review": None, "rejected_jokes": []}


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


def show_saved_jokes(state: JokeState) -> dict:
    saved_jokes = store.retrieve_jokes()

    print("\n\n".join([joke.text for joke in saved_jokes]))
    return {}


def reset_jokes(state: JokeState) -> dict:
    print("Clearing jokes...")
    return {"jokes": []}


def exit_bot(state: JokeState) -> dict:
    return {"quit": True}


main_menu_nodes = [
    {
        "name": "generate_joke",
        "description": "Next joke",
        "function": generate_joke,
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
        "name": "show_saved_jokes",
        "description": "Show saved jokes",
        "function": show_saved_jokes,
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
