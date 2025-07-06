from typing import Annotated, List
from pydantic import BaseModel
from pyjokes import LANGUAGE_VALUES
import inquirer
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama


system_message = SystemMessage(
    """You are a basic joke generator.
You will be providen category and language.
Answer only with joke text, without any formatting and additional text.
Don't provide translations if you generated joke in different language than English.
You will be providen your previous jokes, don't repeat them."""
)

joke_prompt = PromptTemplate(
    input_variables=["category", "language"],
    template="""
Category: {category}

Language: {language}

Previous jokes:
{prev_jokes}
""",
)

llm = ChatOllama(model="gemma3:4b", temperature=0.0)

categories = ["programmer", "chuck", "dad"]
languages = LANGUAGE_VALUES.copy()
languages.add("ua")


class Joke(BaseModel):
    text: str
    category: str


def joke_reducer(current: List[Joke], new: List[Joke]) -> List[Joke]:
    # Clear jokes if we're providing empty list
    if not new:
        return []

    # Just concatenate jokes if we're providing new joke
    return current + new


class JokeState(BaseModel):
    jokes: Annotated[List[Joke], joke_reducer] = []
    user_choice: str = "next_joke"
    category: str = categories[0]
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
                "name",
                message="What option do you want to use?",
                choices=[(node["description"], node["name"]) for node in nodes],
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


nodes = [
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
