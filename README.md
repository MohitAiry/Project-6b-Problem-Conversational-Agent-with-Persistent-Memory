# Problem 6b: Stateful Conversational Agent with Persistent Memory

This agent is a fully local, modular conversational AI agent designed for an online bookstore (PageTurner Books). It demonstrates how to handle context window limits and provide a personalized experience by utilizing a **Three-Layered Memory Architecture**. The agent successfully remembers facts both within a single conversation and across completely separate sessions using LangGraph, Ollama, and ChromaDB.

---

## Table of Contents
1. [What You're Building](#1-what-youre-building)
2. [Core Concepts](#2-core-concepts)
3. [Project Files Overview](#3-project-files-overview)
4. [The Modular Implementation](#4-the-modular-implementation)
5. [The Interactive Demo Explained](#5-the-interactive-demo-explained)
6. [Key Design Decisions](#6-key-design-decisions)
7. [Extending the Project](#7-extending-the-project)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. What You're Building

A **stateful customer support chatbot** for an online bookstore (PageTurner Books) named MannKiBot. Unlike a standard chatbot that forgets everything after a conversation ends, MannKiBot:

- Remembers what happened earlier in the current conversation (sliding window + summary)
- Remembers users across completely separate sessions (ChromaDB vector store)
- Can correctly answer "What did I order last time?" in a new session without being told

The scenario that proves it works:
**Session 1** — You log in as a new customer (e.g. `mohit_47`), order *Dark Matter*, say you love sci-fi, and then type `quit`.
**Session 2 (new process, fresh memory)** — You run the script again as `mohit_47` and say "Hey there!". MannKiBot correctly greets you, recalls the books you ordered, and remembers your genre preferences — all from ChromaDB.

---

## 2. Core Concepts

### 2.1 Why Memory Matters
Every LLM call is stateless. The model has no idea what was said in a previous message unless you include it in the current request. All "memory" in LLM applications is context management — deciding what text to include in the prompt.

The challenge is the **context window limit**. You can't keep appending every message forever. A 100-turn conversation would eventually be too long and too expensive to send with every new message. This project implements three strategies to handle this.

### 2.2 The Three Memory Layers
```
┌───────────────────────────────────────────────────────────────┐
│                       MEMORY ARCHITECTURE                     │
├─────────────────┬──────────────────┬──────────────────────────┤
│  Layer 1        │  Layer 2         │  Layer 3                 │
│  Sliding Window │  Summary Memory  │  Long-Term (ChromaDB)    │
├─────────────────┼──────────────────┼──────────────────────────┤
│ Last 10 msgs    │ LLM-compressed   │ Embedded facts stored    │
│ Lives in RAM    │ older messages   │ on disk as vectors       │
│ Dies on exit    │ Lives in RAM     │ Persists forever         │
│                 │ Dies on exit     │                          │
├─────────────────┼──────────────────┼──────────────────────────┤
│ In-session      │ In-session       │ Cross-session            │
│ Short-term      │ Short-term       │ Long-term                │
└─────────────────┴──────────────────┴──────────────────────────┘
```

**Layer 1 — Sliding Window Buffer**
Always expose only the most recent `N` messages (default: 10) to the LLM. When message 11 arrives, message 1 is dropped. Simple and cheap — no extra LLM calls needed.

**Layer 2 — Summary Memory**
Instead of silently dropping old messages, call the LLM to compress them into 2-3 sentences before they fall out of the window. That summary is prepended to future prompts as context. This preserves semantic continuity at the cost of one extra LLM call per overflow event.

**Layer 3 — ChromaDB Long-Term Memory**
After a session ends, extract key facts (name, orders, preferences) using an LLM call and store them as vector embeddings in ChromaDB. At the start of the next session, retrieve semantically relevant facts and inject them into the system prompt. This is the only layer that survives application restarts.

### 2.3 Memory Strategy Comparison
| Strategy | Cost | What's Preserved | When It Fails |
|---|---|---|---|
| Buffer memory (store everything) | High tokens | Everything | Context window overflow |
| Sliding window | Low tokens | Recent context only | Loses early facts |
| Summary memory | Medium tokens | Semantic gist | Nuance lost in compression |
| Long-term (ChromaDB) | Disk + embedding | Key facts across sessions | Only as good as extraction |

---

## 3. Project Files Overview

The codebase is highly modular, splitting responsibilities cleanly:
```
Problem6b/
├── config.py             # Constants, system prompts, Ollama HTTP checks
├── memory.py             # ChromaDB vector storage and extraction
├── graph.py              # LangGraph nodes, state machine, and router
├── session.py            # BookstoreSession wrapper over the graph
├── main.py               # The interactive CLI loop
├── chroma_bookstore_ollama/ # Created at runtime (database folder)
```

---

## 4. The Modular Implementation

Here is an in-depth breakdown of the major functions within each file:

### 4.1 `config.py` - Configuration and Ollama Interface
* **`check_ollama(required_models)`**: Runs an HTTP request to `localhost:11434/api/tags` to verify the local daemon is running and has the required `llama3.1` and `nomic-embed-text` models downloaded.
```python
def check_ollama(required_models: list[str]) -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        r.raise_for_status()
        pulled = {m["name"].split(":")[0] for m in r.json().get("models", [])}
        # Checks if base model names match required ones
        ...
```
* **`_ollama_complete(system, history)`**: A wrapper for `ChatOllama.invoke`. It takes a system prompt string and a list of LangChain message objects (`HumanMessage`, `AIMessage`), prepends the system prompt as a `SystemMessage`, and directly queries the local LLM.
```python
def _ollama_complete(system: str, history: list[AnyMessage]) -> str:
    llm = ChatOllama(model=CHAT_MODEL, base_url=OLLAMA_BASE_URL)
    messages = [SystemMessage(content=system)] + history
    return llm.invoke(messages).content
```

### 4.2 `memory.py` - Persistent Vector Storage
* **`OllamaChromaEF`**: A custom embedding function class wrapping `langchain_ollama.OllamaEmbeddings` so ChromaDB can use local Nomic embeddings.
```python
class OllamaChromaEF(EmbeddingFunction):
    def __init__(self, model=EMBED_MODEL, base_url=OLLAMA_BASE_URL):
        self._emb = OllamaEmbeddings(model=model, base_url=base_url)

    def __call__(self, input: Documents) -> Embeddings:
        return self._emb.embed_documents(list(input))
```
* **`LongTermMemory`**:
  * **`retrieve(user_id, query)`**: Performs a semantic search against the database using `where={"user_id": user_id}` as a metadata filter.
  ```python
  results = self.col.query(
      query_texts=[query],
      n_results=min(n, self.col.count()),
      where={"user_id": user_id} # Ensures isolation
  )
  ```
  * **`store(user_id, facts, session_id)`**: Uses ChromaDB's `upsert` mechanism with a deterministic `doc_id` to prevent duplicate fact entries.
  ```python
  doc_id = f"{user_id}_{session_id}"
  self.col.upsert(
      ids=[doc_id],
      documents=[facts],
      metadatas=[{"user_id": user_id, "session_id": session_id, ...}]
  )
  ```

### 4.3 `graph.py` - LangGraph Flow and Logic

LangGraph makes the conversation structure explicit and inspectable. Here is the topology of our state machine:

```text
          ┌──────────┐
          │ load_ltm │  ← Reads ChromaDB on first turn of session
          └────┬─────┘
               │
          ┌────▼─────┐
          │   chat   │  ← Core inference (Layer 1: sliding window)
          └────┬─────┘
               │
        ┌──────▼──────┐
        │   router    │  ← len(messages) > WINDOW_SIZE?
        └──────┬──────┘
       yes     │     no
       ┌───────▼───┐  └────────────────► END
       │ summarise │  (Layer 2: Compresses older messages)
       └───────────┘
             │
            END
```

* **`AgentState`**: A `TypedDict` defining the graph's memory structure using LangGraph's `add_messages` reducer.
```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str
    user_id: str
    session_id: str
    lt_context: str
    turn_count: int
```
* **`node_load_ltm`**: The entry point. Checks `state["turn_count"]` to pull from ChromaDB only on the first turn.
```python
def _load_ltm(state: AgentState) -> dict:
    if state.get("turn_count", 0) > 0:
        return {} # Skips database query on subsequent turns
    return node_load_ltm(state, ltm)
```
* **`node_chat`**: Implements the Sliding Window (Layer 1). It slices `messages[-WINDOW_SIZE:]` to strictly enforce the context limit.
```python
def node_chat(state: AgentState) -> dict:
    ...
    window_msgs.extend(messages[-WINDOW_SIZE:])
    reply = _ollama_complete(system, window_msgs)
    return {
        "messages": [AIMessage(content=reply)],
        "turn_count": state.get("turn_count", 0) + 1,
    }
```
* **`node_summarise`**: Implements Summary Memory (Layer 2). It isolates the older overflow messages, prompts the LLM to compress them, and returns `RemoveMessage` objects to physically delete them.
```python
def node_summarise(state: AgentState) -> dict:
    overflow_count = len(messages) - WINDOW_SIZE
    to_compress = messages[:overflow_count]
    ...
    removals = [RemoveMessage(id=m.id) for m in to_compress]
    return {"summary": accumulated, "messages": removals}
```
* **`router_should_summarise`**: A conditional edge. Evaluates if `len(messages) > WINDOW_SIZE` to trigger compression.
```python
def router_should_summarise(state: AgentState) -> Literal["summarise", "__end__"]:
    if len(state["messages"]) > WINDOW_SIZE:
        return "summarise"
    return "__end__"
```

### 4.4 `session.py` - Interaction Abstraction
* **`BookstoreSession`**: Abstracts LangGraph's complex threading mechanisms. 
  * **`__init__`**: Generates a unique `thread_id` to isolate the in-memory `MemorySaver` checkpointer.
  ```python
  self.session_id = str(uuid.uuid4())[:8]
  self.config = {"configurable": {"thread_id": f"{user_id}_{self.session_id}"}}
  ```
  * **`end()`**: Wraps up the conversation by retrieving the final graph state, extracting facts, and saving them.
  ```python
  final_state = self.app.get_state(self.config).values
  facts = extract_facts(final_state["messages"], final_state.get("summary", ""))
  self.ltm.store(self.user_id, facts, self.session_id)
  ```

### 4.5 `main.py` - The User Interface
* **`run_interactive()`**: An infinite loop utilizing python's `input()` with a `finally` block to guarantee `session.end()` runs on crash/exit.
```python
try:
    while True:
        user_input = input("👤 You: ").strip()
        reply = session.chat(user_input)
        print(f"\n🤖 MannKiBot: {reply}\n")
finally:
    print("\nEnding session and saving memories...")
    if 'session' in locals():
        session.end()
```

---

## 5. The Interactive Demo Explained

When you run `python main.py`, the CLI asks for your username.
1. Entering your username initializes `BookstoreSession`.
2. The agent queries ChromaDB. If you are new, it proceeds normally. If you return, it invisibly injects your previous history into `lt_context`.
3. You converse. If your conversation gets long (over `WINDOW_SIZE=10`), the agent automatically compresses the oldest messages, preserving tokens.
4. When you type `quit`, the loop breaks, triggering the `finally` block. The agent reads the graph state, extracts facts into prose, and saves them permanently.

---

## 6. Key Design Decisions

### Why prose for fact storage, not JSON?
Facts are stored as plain English sentences rather than a JSON object. Prose embeddings carry richer semantic meaning. When you query ChromaDB with "what did this user order", the cosine similarity between the query embedding and a prose embedding of facts is more reliable than the similarity to a JSON string. JSON structure can interfere with embedding quality.

### Why upsert instead of add for ChromaDB?
```python
self.collection.upsert(ids=[doc_id], ...)
```
Using `upsert` means re-running the same session won't create duplicate entries. The `doc_id` format (`user_id_session_id`) is deterministic, so the same session re-run just overwrites the previous entry. `add` would throw an error on the second run because the ID already exists.

### Why a synthetic Human/AI pair for the summary injection?
```python
api_messages.append({"role": "user",      "content": "[System note: earlier summary]"})
api_messages.append({"role": "assistant", "content": f"[Summary]: {summary}"})
```
Most LLMs expect messages to alternate between human and assistant roles. Injecting the summary as a single system-like message or a bare assistant message can cause issues. Framing it as a completed exchange (question + answer) is reliably interpreted as shared context from earlier in the conversation.

### Why `RemoveMessage` instead of returning a shorter list?
In LangGraph, the `add_messages` reducer is append-only by design. If you return `{"messages": short_list}`, the reducer tries to append `short_list` to the existing list, not replace it. `RemoveMessage` objects are the proper way to signal deletions — the reducer recognises them and deletes the matching message by ID before merging.

### Why per-session `thread_id` instead of per-user?
```python
self.thread_id = f"{user_id}_{self.session_id}"
```
Using the user ID directly as `thread_id` would cause Session 2 to continue loading the in-session state from Session 1 (via MemorySaver). A new UUID suffix means each session starts with a clean in-memory slate. Cross-session memory comes from ChromaDB, not from the MemorySaver — this separation is intentional.

---

## 7. Extending the Project

### Swap the embedding model
To improve retrieval quality, swap to a larger embedding model:
```bash
ollama pull mxbai-embed-large   # 1024-dim, higher quality
```
In `config.py`:
```python
EMBED_MODEL = "mxbai-embed-large"
```
Then delete the old ChromaDB directory (embedding dimensions must match) and re-run.

### Add more LTM collections
Instead of one collection for all user facts, use separate collections for different memory types:
```python
orders_col      = chroma.get_or_create_collection("orders")
preferences_col = chroma.get_or_create_collection("preferences")
```

---

## 8. Troubleshooting

### `Cannot reach Ollama at http://localhost:11434`
Ollama isn't running. Start it:
```bash
ollama serve          # Linux/macOS foreground
```

### `Missing Ollama models`
```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

### ChromaDB embedding dimension mismatch
```
InvalidDimensionException: Embedding dimension X does not match collection dimensionality Y
```
You changed the embedding model but the existing collection was built with the old one. Delete it:
```bash
rm -rf ./chroma_bookstore_ollama/    
python3 main.py       # re-run to rebuild
```

### Session 2 doesn't recall Session 1 facts
Check in order:
1. Did Session 1 complete without errors? The `quit` command must be run properly.
2. Is `user_id` the exactly same in both sessions? (e.g. `"mohit_47"`)
3. Does the ChromaDB directory exist?
4. Add a debug print before querying: `print(ltm.col.count())` — should be ≥ 1 after Session 1.
