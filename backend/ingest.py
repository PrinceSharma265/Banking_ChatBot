from vector_store import add_documents, get_collection_count
import os
import uuid


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100):
    """Split text into overlapping chunks for vector indexing."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def ingest_file(filepath: str, filename: str):
    """Read a file, chunk its text, and index the chunks into the vector store."""
    try:
        print(f"Ingesting {filename}...")

        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()

        if not text.strip():
            print(f"Skipping {filename}: file contains no readable text.")
            return

        chunks = chunk_text(text)
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": filename} for _ in chunks]

        add_documents(chunks, ids, metadatas)
        print(f"Successfully indexed {len(chunks)} chunks from {filename}")
    except Exception as e:
        print(f"Error ingesting {filename}: {e}")


def main():
    """Ingest all supported files from the data folder into ChromaDB."""
    data_folder = "./data"
    print("Starting document ingestion...")

    try:
        for filename in os.listdir(data_folder):
            filepath = os.path.join(data_folder, filename)
            if not os.path.isfile(filepath):
                continue

            if filename.lower().endswith(".txt") or filename.lower().endswith(".pdf"):
                ingest_file(filepath, filename)

        print("Ingestion complete!")
        print(f"Total documents in collection: {get_collection_count()}")
    except Exception as e:
        print(f"Error during ingestion: {e}")


if __name__ == "__main__":
    main()
