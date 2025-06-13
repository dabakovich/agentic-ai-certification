import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from utils import load_publications, get_llm, load_env
import torch
from langchain.prompts import PromptTemplate

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./publications_db")
collection = client.get_or_create_collection(
    name="publications",
    metadata={"hnsw:space": "cosine"}
)

# Set up our embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

publications = load_publications(json_path="./data/project_1_publications.json")

# print(publications[0])

def chunk_publication(publication_content, title):
    """Break a research paper into searchable chunks"""
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,          # ~200 words per chunk
        chunk_overlap=200,        # Overlap to preserve context
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_text(publication_content)
    
    # Add metadata to each chunk
    chunk_data = []
    for i, chunk in enumerate(chunks):
        chunk_data.append({
            "content": chunk,
            "title": title,
            "chunk_id": f"{title}_{i}",
        })
    
    return chunk_data


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
        publications (list[str]): The documents to insert

    Returns:
        None
    """
    next_id = collection.count()

    for publication in publications:
        print(f"Chunking publication: {publication['title']}")
        chunked_publication = chunk_publication(publication["description"], publication["title"])

        chunked_publication_content = [chunk["content"] for chunk in chunked_publication]
        # Embed list of strings from chunked_publication
        embeddings = embed_documents(chunked_publication_content)
        ids = list(range(next_id, next_id + len(chunked_publication)))
        ids = [f"document_{id}" for id in ids]
        # Create metadata for each chunk
        metadatas = [{"title": chunk["title"], "chunk_id": chunk["chunk_id"]} for chunk in chunked_publication]
        
        collection.add(
            embeddings=embeddings,
            ids=ids,
            documents=chunked_publication_content,
            metadatas=metadatas,
        )
        next_id += len(chunked_publication)

def search_research_db(query, collection, embeddings, top_k=5):
    """Find the most relevant research chunks for a query"""
    
    # Convert question to vector
    query_vector = embeddings.embed_query(query)
    
    # Search for similar content
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # print(results)
    
    # Format results
    relevant_chunks = []
    for i, doc in enumerate(results["documents"][0]):
        # Handle case where metadata might be None
        metadata = results["metadatas"][0][i]
        title = metadata["title"] if metadata and "title" in metadata else "Unknown Publication"
        
        relevant_chunks.append({
            "content": doc,
            "title": title,  # Use safe title extraction
            "similarity": 1 - results["distances"][0][i]  # Convert distance to similarity
        })
    
    return relevant_chunks


def answer_research_question(query, collection, embeddings, llm):
    """Generate an answer based on retrieved research"""
    
    # Get relevant research chunks
    relevant_chunks = search_research_db(query, collection, embeddings, top_k=3)

    # print(relevant_chunks)
    
    # Build context from research
    context = "\n\n".join([
        f"From {chunk['title']}:\n{chunk['content']}" 
        for chunk in relevant_chunks
    ])
    
    # Create research-focused prompt
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Based on the following research findings, answer the researcher's question:

Research Context:
{context}

Researcher's Question: {question}

Answer: Provide a comprehensive answer based on the research findings above.
"""
    )
    
    # Generate answer
    prompt = prompt_template.format(context=context, question=query)
    response = llm.invoke(prompt)
    
    return response.content, relevant_chunks

# insert_publications(collection, publications)

def main():
    load_env()
    llm = get_llm(llm="ollama")

    # insert_publications(collection, publications)

    query = "How can I use docstrings in ML?"
    answer, relevant_chunks = answer_research_question(query, collection, embeddings, llm)
    print(answer)
    # print(relevant_chunks)

if __name__ == "__main__":
    main()