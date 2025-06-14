import chromadb
import os
import shutil
from paths import VECTOR_DB_DIR

def initialize_db(
    persist_directory: str = VECTOR_DB_DIR,
    collection_name: str = "publications",
    delete_existing: bool = False,
) -> chromadb.Collection:
    """
    Initialize a ChromaDB instance and persist it to disk.

    Args:
        persist_directory (str): The directory where ChromaDB will persist data. Defaults to "./vector_db"
        collection_name (str): The name of the collection to create/get. Defaults to "publications"
        delete_existing (bool): Whether to delete the existing database if it exists. Defaults to False
    Returns:
        chromadb.Collection: The ChromaDB collection instance
    """
    if os.path.exists(persist_directory) and delete_existing:
        shutil.rmtree(persist_directory)

    os.makedirs(persist_directory, exist_ok=True)

    # Initialize ChromaDB client with persistent storage
    client = chromadb.PersistentClient(path=persist_directory)

    # Create or get a collection
    try:
        # Try to get existing collection first
        collection = client.get_collection(name=collection_name)
        print(f"Retrieved existing collection: {collection_name}")
    except Exception:
        # If collection doesn't exist, create it
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"Created new collection: {collection_name}")

    print(f"ChromaDB initialized with persistent storage at: {persist_directory}")

    return collection

def get_db_collection(
    persist_directory: str = VECTOR_DB_DIR,
    collection_name: str = "publications",
) -> chromadb.Collection:
    """
    Get a ChromaDB client instance.

    Args:
        persist_directory (str): The directory where ChromaDB persists data
        collection_name (str): The name of the collection to get

    Returns:
        chromadb.PersistentClient: The ChromaDB client instance
    """
    return chromadb.PersistentClient(path=persist_directory).get_collection(
        name=collection_name
    )