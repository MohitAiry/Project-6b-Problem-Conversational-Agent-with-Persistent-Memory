import uuid
from langchain_core.messages import HumanMessage, AIMessage

from config import CHAT_MODEL, OLLAMA_BASE_URL
from graph import extract_facts, AgentState
from memory import LongTermMemory

def make_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}

class BookstoreSession:
    """
    Thin wrapper around the compiled LangGraph app.
    Each session → unique thread_id → isolated MemorySaver state.
    ChromaDB (LTM) persists across sessions.
    """
    def __init__(self, app, ltm: LongTermMemory, user_id: str):
        try:
            self.app = app
            self.ltm = ltm
            self.user_id = user_id
            self.thread_id = f"{user_id}_{str(uuid.uuid4())[:8]}"
            self.session_id = self.thread_id.split("_", 1)[1]
            self._first_turn = True

            print(f"\n{'═'*60}")
            print(f"  SESSION {self.session_id}  |  User: {user_id}")
            print(f"  Model: {CHAT_MODEL} via Ollama ({OLLAMA_BASE_URL})")
            print(f"  LangGraph thread_id: {self.thread_id}")
            print(f"{'═'*60}\n")

            self._initial_state: AgentState = {
                "messages": [],
                "summary": "",
                "user_id": user_id,
                "session_id": self.session_id,
                "lt_context": "",
                "turn_count": 0,
            }
        except Exception as e:
            print(f"Error initializing BookstoreSession: {e}")
            raise

    def chat(self, user_input: str) -> str:
        try:
            config = make_config(self.thread_id)

            if self._first_turn:
                inp = {**self._initial_state,
                       "messages": [HumanMessage(content=user_input)]}
                self._first_turn = False
            else:
                inp = {"messages": [HumanMessage(content=user_input)]}

            result = self.app.invoke(inp, config=config)

            ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
            return ai_msgs[-1].content if ai_msgs else "(no reply)"
        except Exception as e:
            print(f"Error during chat execution: {e}")
            return "An error occurred during communication."

    def end(self):
        print(f"\n{'─'*60}")
        print("  Session ending — extracting facts for long-term memory...")
        print(f"{'─'*60}")

        try:
            config = make_config(self.thread_id)
            snapshot = self.app.get_state(config)
            if not snapshot or not snapshot.values:
                print("  (No state to persist)\n")
                return

            state = snapshot.values
            messages = state.get("messages", [])
            summary = state.get("summary", "")

            if not messages:
                print("  (Empty conversation — nothing to store)\n")
                return

            facts = extract_facts(messages, summary)
            if facts:
                self.ltm.store(self.user_id, facts, self.session_id)
                print(f"Session {self.session_id} complete.\n")
            else:
                print("Could not extract facts from conversation.\n")
        except Exception as e:
            print(f"Error ending session: {e}")

    def memory_status(self) -> str:
        try:
            config = make_config(self.thread_id)
            snapshot = self.app.get_state(config)
            if not snapshot or not snapshot.values:
                return "buffer=0 | summary=no"
            state = snapshot.values
            n_msgs = len(state.get("messages", []))
            has_sum = bool(state.get("summary", ""))
            n_ltm = self.ltm.col.count()
            return (
                f"buffer={n_msgs} msgs | "
                f"summary={'yes' if has_sum else 'no'} | "
                f"ltm_docs={n_ltm}"
            )
        except Exception as e:
            print(f"Error retrieving memory status: {e}")
            return "Status unavailable"
