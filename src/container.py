from vector_store.chromadb import ChromaDB
from dependency_injector import containers, providers
from paths import VECTOR_DB_DIR
from vector_store import VectorStore


class Container(containers.DeclarativeContainer):
    config = providers.Configuration()

    chromadb = providers.Singleton(ChromaDB, VECTOR_DB_DIR)
    vector_store = providers.Singleton(VectorStore, "publications_test", chromadb)
