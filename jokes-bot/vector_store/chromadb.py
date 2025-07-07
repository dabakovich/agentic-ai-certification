import chromadb
import os


class ChromaDB:
    def __init__(self, persist_dir: str):
        os.makedirs(persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_dir)

        print(f"ChromaDB initialized with persistent storage at: {persist_dir}")

    def clear_collection(self, name: str):
        self.client.delete_collection(name=name)

    def get_collection(self, name: str) -> chromadb.Collection:
        # Create or get a collection
        try:
            # Try to get existing collection first
            collection = self.client.get_collection(name=name)
            print(f"Retrieved existing collection: {name}")
        except Exception:
            # If collection doesn't exist, create it
            collection = self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
            print(f"Created new collection: {name}")

        return collection
