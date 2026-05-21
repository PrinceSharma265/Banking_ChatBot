# Banking Chatbot Architecture

## System Components

1. **Frontend** - index.html (HTML, CSS, JavaScript)
   - Chat interface
   - File upload
   - Connects to Railway backend

2. **Backend** - FastAPI (Python)
   - POST /chat
   - POST /upload  
   - GET /health

3. **RAG Pipeline** - rag_pipeline.py
   - Retrieves context from ChromaDB
   - Sends context + query to Groq LLM
   - Returns response to user

4. **Vector Database** - ChromaDB
   - Stores 106 document chunks
   - Performs similarity search
   - Returns top 5 relevant chunks

5. **Embeddings** - sentence-transformers
   - Model: all-MiniLM-L6-v2
   - Converts text to vectors locally

6. **LLM** - Groq API
   - Model: Llama 3.1 8B Instant
   - Generates context-aware responses

7. **Banking Documents** - 4 text files
   - loan_policy.txt
   - credit_card_guide.txt
   - banking_faq.txt
   - rbi_guidelines.txt

## Flow
User → Frontend → FastAPI → RAG Pipeline → ChromaDB → Groq LLM → Response
