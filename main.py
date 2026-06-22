import sys
from config import CHAT_MODEL, EMBED_MODEL, check_ollama
from memory import LongTermMemory
from graph import build_graph
from session import BookstoreSession

def run_interactive():
    print("\n" + "=" * 62)
    print("  Interactive PageTurner Books Support Agent")
    print("  Type 'quit' or 'exit' to end the session.")
    print("=" * 62 + "\n")

    # Verify Ollama is up and models are available
    if not check_ollama([CHAT_MODEL, EMBED_MODEL]):
        return

    try:
        ltm = LongTermMemory()
        app = build_graph(ltm)
        
        # Ask for a username so ChromaDB knows who it's talking to
        user_id = input("Enter your username (e.g., alice_123): ").strip()
        if not user_id:
            user_id = "guest_user"
            
        session = BookstoreSession(app, ltm, user_id=user_id)
        
        print("\n🤖 MannKiBot: Hello! I'm MannKiBot from PageTurner Books. How can I help you today?\n")
        
        # The interactive Chat Loop
        while True:
            try:
                user_input = input("👤 You: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
                
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
                
            if not user_input:
                continue
                
            reply = session.chat(user_input)
            print(f"\n🤖 MannKiBot: {reply}\n")
            
            # Optional: Uncomment this if you want to monitor the memory state live!
            # print(f"  [Memory Status: {session.memory_status()}]\n")
            
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
    finally:
        print("\nEnding session and saving memories...")
        try:
            if 'session' in locals():
                session.end()
        except Exception:
            pass
        print("Goodbye!")

if __name__ == "__main__":
    run_interactive()
