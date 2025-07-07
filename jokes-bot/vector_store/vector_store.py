from langchain_huggingface import HuggingFaceEmbeddings
import torch
from vector_store.chromadb import ChromaDB
from classes import Joke
from constants import joke_similarity_treshold


class VectorStore:
    def __init__(
        self,
        collection_name: str,
        chromadb: ChromaDB,
    ):
        self.client = chromadb
        self.collection = self.client.get_collection(collection_name)
        self.collection_name = collection_name

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "mps"},
        )

    def insert_joke(
        self,
        joke: Joke,
    ):
        """
        Insert new joke into a vector store.
        """
        index = self.collection.count() + 1

        embeddings = self.embeddings_model.embed_query(joke.text)

        documents = [joke.text]
        ids = [str(index)]
        metadatas = [{"category": joke.category}]

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )

    def retrieve_jokes(
        self,
        joke: Joke,
        threshold: float = joke_similarity_treshold,
    ) -> list[dict]:
        """
        Retrieve similar jokes from the vector store.
        """
        embeddings = self.embeddings_model.embed_query(joke.text)

        results = self.collection.query(
            query_embeddings=embeddings,
            n_results=5,
        )

        # Filtering results
        filtered_result_indexes = [
            i for i, result in enumerate(results["distances"][0]) if result < threshold
        ]

        relevant_results = {
            "ids": [results["ids"][0][i] for i in filtered_result_indexes],
            "jokes": [results["documents"][0][i] for i in filtered_result_indexes],
            "distances": [results["distances"][0][i] for i in filtered_result_indexes],
        }

        return relevant_results

    def is_joke_exists(self, joke: Joke):
        similar_jokes = self.retrieve_jokes(joke)

        return len(similar_jokes["jokes"]) > 0

    def clear_collection(self):
        self.client.clear_collection(self.collection_name)
