from database import get_db_collection
from embedding import embed_documents
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

# Set up our embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def retrieve_relevant_documents(
    collection: chromadb.Collection,
    query: str,
    n_results: int = 5,
    threshold: float = 0.4,
) -> list[str]:
    """
    Query the ChromaDB database with a string query.

    Args:
        collection (chromadb.Collection): The collection to query
        query (str): The search query string
        n_results (int): Number of results to return (default: 5)
        threshold (float): Threshold for the cosine similarity score (default: 0.3)

    Returns:
        dict: Query results containing ids, documents, distances, and metadata
    """
    print(f"Retrieving relevant documents for query: {query}")
    relevant_results = {
        "ids": [],
        "documents": [],
        "distances": [],
    }
    # Embed the query using the same model used for documents
    print("Embedding query...")
    query_embedding = embeddings.embed_query(query)

    print("Querying collection...")
    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    print(results["distances"])

    print("Filtering results...")
    keep_item = [False] * len(results["ids"][0])
    for i, distance in enumerate(results["distances"][0]):
        if distance < threshold:
            keep_item[i] = True

    for i, keep in enumerate(keep_item):
        if keep:
            relevant_results["ids"].append(results["ids"][0][i])
            relevant_results["documents"].append(results["documents"][0][i])
            relevant_results["distances"].append(results["distances"][0][i])

    return relevant_results["documents"]