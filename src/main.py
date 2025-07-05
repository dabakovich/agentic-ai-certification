import inquirer

from common.utils import load_env
from container import Container
from core.ui import launch_conversation
from vector_store.ui import show_vector_store_options


def main():
    load_env()

    choices = {
        "Launch a conversation": launch_conversation,
        "Show vector store options": show_vector_store_options,
    }

    # Ask user if he wants to insert new publications or launch a conversation
    answer = inquirer.prompt(
        [
            inquirer.List(
                "action",
                message="What do you want to do?",
                choices=choices.keys(),
            )
        ]
    )

    choices[answer["action"]]()


if __name__ == "__main__":
    container = Container()
    container.wire(modules=[__name__, "core.conversation"])

    main()  # <-- dependency is injected automatically
