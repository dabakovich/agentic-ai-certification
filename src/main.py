from database import get_db_collection, initialize_db
from embedding import insert_publications
from utils import load_env, load_publications
from paths import PUBLICATIONS_PATH
from assistant import run_conversation
from rag import retrieve_relevant_documents


def main():
    load_env()
  
    # collection = initialize_db(collection_name="publications_test")
    collection = get_db_collection(collection_name="publications_test")
    # publications = load_publications(PUBLICATIONS_PATH)
    # insert_publications(collection, publications)

    # relevant_documents = retrieve_relevant_documents(collection, "How docstrings could be useful for ML?")

    run_conversation(collection)

if __name__ == "__main__":
    main()