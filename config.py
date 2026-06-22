import requests
from langchain_ollama import ChatOllama
from langchain_core.messages import AnyMessage, SystemMessage

OLLAMA_BASE_URL  = "http://localhost:11434"
CHAT_MODEL       = "llama3.1:latest"       
EMBED_MODEL      = "nomic-embed-text:latest"  
WINDOW_SIZE      = 10
CHROMA_PATH      = "./chroma_bookstore_ollama"
COLLECTION_NAME  = "user_memories_ollama"

SYSTEM_PROMPT_BASE = """You are MannKiBot, a warm and knowledgeable customer support agent for PageTurner Books — an online bookstore.

Your capabilities:
- Help customers track and place orders
- Recommend books based on their preferences
- Handle returns, complaints, and general queries
- Remember details about returning customers and personalise your responses

Tone: Friendly, concise, and helpful. Use the customer's name when you know it.

{long_term_context}"""

def check_ollama(required_models: list[str]) -> bool:
    """Verify Ollama is running and the required models are pulled."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Cannot reach Ollama at {OLLAMA_BASE_URL}: {e}")
        print("    Start Ollama with:  ollama serve")
        return False
    
    try:
        pulled = {m["name"].split(":")[0] for m in r.json().get("models", [])}
    except ValueError as e:
        print(f"Failed to parse Ollama response: {e}")
        return False

    missing = [m for m in required_models if m.split(":")[0] not in pulled]
    if missing:
        print(f"❌  Missing Ollama models: {missing}")
        for m in missing:
            print(f"    Pull with:  ollama pull {m}")
        return False

    print(f"Ollama running | models available: {required_models}\n")
    return True

try:
    _llm = ChatOllama(
        model=CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.7,
    )
except Exception as e:
    print(f"Failed to initialize ChatOllama: {e}")
    _llm = None

def _ollama_complete(system: str, history: list[AnyMessage]) -> str:
    """
    Build the message list for ChatOllama and return the reply text.
    ChatOllama.invoke([SystemMessage, HumanMessage, AIMessage, ...]) → AIMessage
    """
    try:
        if _llm is None:
            raise RuntimeError("ChatOllama model was not properly initialized.")
            
        msgs = [SystemMessage(content=system)] + list(history)
        response = _llm.invoke(msgs)
        return response.content.strip()
    except Exception as e:
        print(f"Error during LLM completion: {e}")
        return "I am experiencing technical difficulties. Please try again later."
