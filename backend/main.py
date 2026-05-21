from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_pipeline import get_response
from vector_store import add_documents, get_collection_count
import pypdf
import uuid
import io
import os

# Initialize the FastAPI application with a title
app = FastAPI(title="Banking Chatbot API")

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# In-memory sessions dictionary to store chat history per session
sessions = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


@app.get("/health")
async def health():
    """Health endpoint to verify the API is running and count indexed documents."""
    try:
        return {"status": "ok", "documents_indexed": get_collection_count()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")


@app.post("/chat")
async def chat(request: ChatRequest):
    """Process a chat request using the RAG pipeline and maintain session history."""
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        # Retrieve existing history for the session or start a new list
        history = sessions.get(request.session_id, [])

        # Generate the assistant response using the RAG pipeline
        reply = get_response(request.message, history)

        # Append the user message and assistant reply to the session history
        history.append({"role": "user", "content": request.message})
        history.append({"role": "assistant", "content": reply})

        # Keep only the last 10 messages in the history for context
        sessions[request.session_id] = history[-10:]

        return {"reply": reply, "session_id": request.session_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {e}")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100):
    """Split text into overlapping chunks for document indexing."""
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


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Accept PDF or text files, extract text, chunk it, and index into the vector store."""
    filename = file.filename
    _, ext = os.path.splitext(filename.lower())

    if ext not in {".pdf", ".txt"}:
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported")

    try:
        content = await file.read()

        if ext == ".pdf":
            # Read PDF bytes and extract text from each page
            reader = pypdf.PdfReader(io.BytesIO(content))
            text = ""
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        else:
            # Decode text file bytes as UTF-8
            text = content.decode("utf-8")

        if not text.strip():
            raise HTTPException(status_code=400, detail="Uploaded file contains no readable text")

        # Chunk the extracted text into overlapping segments
        chunks = chunk_text(text, chunk_size=500, overlap=100)

        # Generate unique IDs and metadata for each chunk
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"filename": filename} for _ in chunks]

        # Add the chunks to the vector store
        add_documents(chunks, ids, metadatas)

        return {"status": "success", "filename": filename, "chunks_indexed": len(chunks)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {e}")
