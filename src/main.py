from core.conversation import Conversation
from common.utils import load_env, load_publications
from paths import PUBLICATIONS_PATH
from vector_store import VectorStore


def main():
    load_env()
    conversation = Conversation(llm_name="gemma")
    conversation.run()

    # _publications = load_publications(PUBLICATIONS_PATH)

    # vector_store = VectorStore("publications")

    # vector_store.insert_publications(publications)
    # vector_store.retrieve_publications("Types of memory")
    # vector_store.retrieve_publications("Types of memory")
    # vector_store.retrieve_publications("Types of memory")
    # results = vector_store.retrieve_publications("Hey, can I use docstrings in ML?")
    # print(results["distances"])


if __name__ == "__main__":
    main()
