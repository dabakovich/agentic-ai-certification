import os

import inquirer
from dependency_injector.wiring import Provide, inject

from common.utils import (
    get_files_in_directory,
    load_publications,
)
from container import Container
from paths import DATA_DIR
from vector_store import VectorStore


@inject
def show_vector_store_options(
    vector_store: VectorStore = Provide[Container.vector_store],
):
    choices = [
        "Insert new publications",
        "Show count of vector store publications",
        "Clear vector store",
    ]

    answer = inquirer.prompt(
        [
            inquirer.List(
                "action",
                message="What do you want to do?",
                choices=choices,
            )
        ]
    )

    if answer["action"] == choices[0]:
        insert_publications(vector_store)
    elif answer["action"] == choices[1]:
        show_count_of_publications(vector_store)
    elif answer["action"] == choices[2]:
        vector_store.clear_collection()
        print("Vector store cleared successfully")


@inject
def show_count_of_publications(vector_store: VectorStore):
    count_of_publications = vector_store.collection.count()
    print(f"Vector store contains {count_of_publications} publications")


@inject
def insert_publications(vector_store: VectorStore):
    print("Retrieving list of files in the /data directory")

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
    vector_store.insert_publications(publications)
    print("Publications inserted successfully")
