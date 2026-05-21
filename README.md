# Banking Support Chatbot

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111.1-brightgreen)
![Railway](https://img.shields.io/badge/Deployment-Railway-purple)

## Overview

Banking Support Chatbot is an AI-powered banking support assistant built with a Retrieval-Augmented Generation (RAG) pipeline. It helps users answer banking queries using indexed banking documents, delivers accurate responses via Groq LLM, and supports PDF/TXT upload for expanding knowledge coverage.

## Tech Stack

- **Backend:** FastAPI (Python)
- **LLM:** Groq API (Llama 3.1)
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector Database:** ChromaDB
- **Frontend:** HTML, CSS, JavaScript
- **Deployment:** Railway

## Features

- Conversational chat interface
- RAG pipeline for accurate, context-aware answers
- PDF and TXT document upload support
- Session-based context retention
- Banking data coverage including loans, credit cards, FAQs, RBI guidelines

## API Endpoints

- `GET /health` - Check server status
- `POST /chat` - Send a message and receive a response
- `POST /upload` - Upload PDF or TXT documents for indexing

## Architecture

1. User sends a chat message from the frontend.
2. Backend searches ChromaDB for relevant document chunks.
3. Retrieved chunks are sent as context to the Groq LLM.
4. Groq generates a response using the provided context.
5. Response is returned to the frontend and displayed in the chat.

## Setup Instructions

```bash
git clone https://github.com/PrinceSharma265/Banking_ChatBot.git
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

1. Create a `.env` file in the `backend` folder.
2. Add your Railway/Groq API key:

```text
GROQ_API_KEY=your_key
```

3. Ingest data documents:

```bash
python ingest.py
```

4. Start the backend server:

```bash
uvicorn main:app --reload --port 8000
```

5. Open `frontend/index.html` in your browser.

## Deployment

- Backend deployed on Railway
- Production URL: `https://bankingchatbot-production.up.railway.app`

## Screenshots

> Screenshots coming soon. Replace this section with actual UI images once available.

## Future Improvements

- Add user authentication and secure session handling
- Support more document formats such as DOCX and CSV
- Improve prompt engineering for even better banking answer quality
- Add analytics dashboard for usage and document performance
- Enable browser-based conversation history persistence

## License

This project is licensed under the MIT License.
