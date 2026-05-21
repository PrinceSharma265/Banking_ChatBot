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
        system_prompt = f"""You are SecureBank AI Assistant, 
an expert banking support chatbot for an Indian bank.

Your role is to help customers with:
- Loan queries (personal, home, vehicle, education loans)
- Credit card information (types, fees, rewards, limits)
- Banking FAQs (accounts, KYC, NEFT, RTGS, UPI, ATM)
- RBI guidelines and banking policies

STRICT RULES:
1. Answer ONLY from the context provided below
2. If answer is not in context, say exactly: 
   "I don't have specific information about that. 
   Please visit your nearest branch or call our 
   customer care at 1800-XXX-XXXX for assistance."
3. Always be polite, professional and empathetic
4. Give specific numbers, rates and figures when available
5. Keep answers concise but complete
6. Never make up information not in the context
7. If customer seems distressed, acknowledge their concern first
8. End responses with a helpful follow-up offer

CONTEXT FROM KNOWLEDGE BASE:
{context}

Remember: You represent SecureBank. 
Every response reflects our brand values of 
trust, transparency and customer-first approach."""

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


def get_streaming_response(query: str, history: list):
    try:
        chunks = search(query, n_results=5)
        context = "\n\n".join(chunks) if chunks else "No context found."
        
        system_prompt = f"""You are SecureBank AI Assistant, 
an expert banking support chatbot for an Indian bank.

Your role is to help customers with:
- Loan queries (personal, home, vehicle, education loans)
- Credit card information (types, fees, rewards, limits)
- Banking FAQs (accounts, KYC, NEFT, RTGS, UPI, ATM)
- RBI guidelines and banking policies

STRICT RULES:
1. Answer ONLY from the context provided below
2. If answer is not in context, say exactly: 
   "I don't have specific information about that. 
   Please visit your nearest branch or call our 
   customer care at 1800-XXX-XXXX for assistance."
3. Always be polite, professional and empathetic
4. Give specific numbers, rates and figures when available
5. Keep answers concise but complete
6. Never make up information not in the context

CONTEXT FROM KNOWLEDGE BASE:
{context}"""

        messages = [{"role": "system", "content": system_prompt}]
        for item in history:
            messages.append({"role": item["role"], "content": item["content"]})
        messages.append({"role": "user", "content": query})
        
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        print(f"Streaming error: {e}")
        yield "Sorry, I am unable to process your request right now. Please try again."
