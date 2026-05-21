from groq import Groq
from vector_store import search
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Read the GROQ API key from environment variables
api_key = os.getenv("GROQ_API_KEY")

# Initialize the Groq client with the API key
client = Groq(api_key=api_key)

# Set the model name to use for chat completions
model_name = "llama-3.1-8b-instant"


def get_response(query: str, history: list) -> str:
    """Retrieve relevant context from the vector store and generate a response using Groq."""
    try:
        # Search the vector store for relevant document chunks
        chunks = search(query, n_results=5)

        # Join the chunks into a single context string
        context = "\n\n".join(chunks)

        # Build the system prompt with instructions and context
        system_prompt = (
            "You are a helpful banking assistant for an Indian bank.\n"
            "Answer customer queries based ONLY on the context provided below.\n"
            "If the answer is not in the context, say I don't have information about that. Please contact our customer support.\n"
            "Always be polite, professional and helpful.\n"
            "Context:\n"
            f"{context}"
        )

        # Build the chat messages list for Groq
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        for item in history:
            messages.append({"role": item["role"], "content": item["content"]})

        messages.append({"role": "user", "content": query})

        # Call the Groq chat completions API
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
        )

        # Return the generated assistant content
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, I am unable to process your request right now. Please try again."
