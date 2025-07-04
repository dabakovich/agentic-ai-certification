from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from vector_store.chromadb import ChromaDB
from paths import VECTOR_DB_DIR


class VectorStore:
    def __init__(self, collection_name: str):
        self.client = ChromaDB(VECTOR_DB_DIR)
        self.collection = self.client.get_collection(collection_name)

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

        # print number of documents in the collection
        print(self.collection.count())

    def chunk_text(self, text: str, size: int = 1000, overlap: int = 200) -> list[str]:
        """
        Chunk text into smaller pieces.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        return text_splitter.split_text(text)

    def embed_chunks(self, chunks: list[str]) -> list[list[float]]:
        """
        Embed text chunks using a model.
        """
        embeddings = self.embeddings_model.embed_documents(chunks)

        return embeddings

    # publication: {id: str, title: str, description: str}
    def insert_publications(
        self,
        publications: list[dict],
    ):
        """
        Ingest publications into a vector store.
        """
        for publication in publications:
            chunks = self.chunk_text(publication["description"])

            embeddings = self.embed_chunks(chunks)

            ids = [f"{publication['id']}-{index}" for index in range(len(chunks))]
            metadatas = [{"title": publication["title"]} for _ in chunks]

            self.collection.add(
                documents=chunks,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas,
            )

    def retrieve_publications(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.5,
    ) -> list[dict]:
        """
        Retrieve publications from the vector store.
        """
        embeddings = self.embeddings_model.embed_query(query)

        results = self.collection.query(
            query_embeddings=embeddings,
            n_results=k,
        )

        # Filtering results
        filtered_result_indexes = [
            i for i, result in enumerate(results["distances"][0]) if result < threshold
        ]

        relevant_results = {
            "ids": [results["ids"][0][i] for i in filtered_result_indexes],
            "documents": [results["documents"][0][i] for i in filtered_result_indexes],
            "distances": [results["distances"][0][i] for i in filtered_result_indexes],
        }

        return relevant_results
