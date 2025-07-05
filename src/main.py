import os
from core.conversation import Conversation
from common.utils import (
    load_env,
    load_publications,
    get_llm_choices,
    get_files_in_directory,
)
from paths import DATA_DIR
from vector_store import VectorStore
import inquirer


def insert_publications():
    list_of_files = get_files_in_directory(DATA_DIR)

    if not list_of_files:
        print(f"No files found in the {DATA_DIR} directory")
        return

    # Ask user for the path to the publications
    answer = inquirer.prompt(
        [
            inquirer.List(
                "file_path",
                message="What publication do you want to insert?",
                choices=list_of_files,
            )
        ]
    )

    path = os.path.join(DATA_DIR, answer["file_path"])

    publications = load_publications(path)
    vector_store = VectorStore("publications")
    vector_store.insert_publications(publications)
    print("Publications inserted successfully")


def launch_conversation():
    llm_choices = get_llm_choices()

    # Ask which LLM to use
    answer = inquirer.prompt(
        [
            inquirer.List(
                "llm_name",
                message="Which LLM do you want to use?",
                choices=[choice["name"] for choice in llm_choices],
            )
        ]
    )

    conversation = Conversation(llm_name=answer["llm_name"])
    conversation.run()


def main():
    load_env()

    # Ask user if he wants to insert new publications or launch a conversation
    answer = inquirer.prompt(
        [
            inquirer.List(
                "action",
                message="What do you want to do?",
                choices=["Insert new publications", "Launch a conversation"],
            )
        ]
    )

    if answer["action"] == "Insert new publications":
        insert_publications()

    elif answer["action"] == "Launch a conversation":
        launch_conversation()


if __name__ == "__main__":
    main()
