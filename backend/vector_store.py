import chromadb
from sentence_transformers import SentenceTransformer
import os

# Initialize the sentence transformer model for embedding generation
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize a persistent ChromaDB client that stores data in ./chroma_db
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create or get the collection named banking_docs
collection = chroma_client.get_or_create_collection(name="banking_docs")


def add_documents(chunks: list, ids: list, metadatas: list):
    """Add document chunks with embeddings and metadata to the Chroma collection."""
    try:
        # Generate embeddings for all chunks using the sentence transformer model
        embeddings = model.encode(chunks).tolist()

        # Add the documents, embeddings, ids, and metadatas to the collection
        collection.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)

        # Print the number of chunks added
        print(f"Added {len(chunks)} chunks to the collection.")
    except Exception as e:
        # Print any error encountered during addition
        print(f"Error adding documents to collection: {e}")


def search(query: str, n_results: int = 5):
    """Search the Chroma collection for documents matching the query."""
    try:
        # Encode the query into an embedding vector
        query_embedding = model.encode([query]).tolist()

        # Perform a similarity search in the collection
        results = collection.query(query_embeddings=query_embedding, n_results=n_results)

        # Return the first result set of documents or an empty list if none found
        return results["documents"][0] if results["documents"] else []
    except Exception as e:
        # Print the error and return an empty list on failure
        print(f"Error searching collection: {e}")
        return []


def get_collection_count():
    """Return the number of items stored in the Chroma collection."""
    try:
        return collection.count()
    except Exception as e:
        # Print the error and return zero if the count cannot be retrieved
        print(f"Error retrieving collection count: {e}")
        return 0
