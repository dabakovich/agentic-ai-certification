from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import chromadb

def chunk_publication(
    publication_content: str, title: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[str]:
    """
    Chunk the publication into smaller documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_text(publication_content)
    chunks_with_metadata = []
    for i, chunk in enumerate(chunks):
        chunks_with_metadata.append({
            "content": chunk,
            "title": title,
            "chunk_id": f"{title}_{i}",
        })
    return chunks_with_metadata

def embed_documents(documents: list[str]) -> list[list[float]]:
    """
    Embed documents using a model.
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
    )

    embeddings = model.embed_documents(documents)
    return embeddings


def insert_publications(collection: chromadb.Collection, publications: list):
    """
    Insert documents into a ChromaDB collection.

    Args:
        collection (chromadb.Collection): The collection to insert documents into
        documents (list[str]): The documents to insert

    Returns:
        None
    """
    next_id = collection.count()

    for publication in publications:
        chunked_publication = chunk_publication(publication["description"], publication["title"])
        chunked_publication_content = [chunk["content"] for chunk in chunked_publication]
        embeddings = embed_documents(chunked_publication_content)
        ids = list(range(next_id, next_id + len(chunked_publication)))
        ids = [f"document_{id}" for id in ids]
        collection.add(
            embeddings=embeddings,
            ids=ids,
            documents=chunked_publication_content,
            metadatas=[{"title": chunk["title"], "chunk_id": chunk["chunk_id"]} for chunk in chunked_publication],
        )
        next_id += len(chunked_publication)