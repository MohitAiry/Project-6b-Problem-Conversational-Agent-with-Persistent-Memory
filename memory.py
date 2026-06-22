import datetime
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from langchain_ollama import OllamaEmbeddings

from config import EMBED_MODEL, OLLAMA_BASE_URL, CHROMA_PATH, COLLECTION_NAME

class OllamaChromaEF(EmbeddingFunction):
    """
    Wraps langchain_ollama.OllamaEmbeddings into the chromadb
    EmbeddingFunction interface so it can be passed to get_or_create_collection.
    """
    def __init__(self, model: str = EMBED_MODEL, base_url: str = OLLAMA_BASE_URL):
        try:
            self._emb = OllamaEmbeddings(model=model, base_url=base_url)
        except Exception as e:
            print(f"Failed to initialize OllamaEmbeddings: {e}")
            raise

    def __call__(self, input: Documents) -> Embeddings:
        try:
            return self._emb.embed_documents(list(input))
        except Exception as e:
            print(f"Embedding generation failed: {e}")
            raise

class LongTermMemory:
    """
    Handles Layer 3 Long-Term Memory via ChromaDB.
    """
    def __init__(self, path: str = CHROMA_PATH):
        try:
            self.chroma = chromadb.PersistentClient(path=path)
            self.ef = OllamaChromaEF()
            self.col = self.chroma.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=self.ef,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"ChromaDB loaded — {self.col.count()} memory entries found")
            print(f"       Embedding model: {EMBED_MODEL} via Ollama\n")
        except Exception as e:
            print(f"Failed to initialize ChromaDB LongTermMemory: {e}")
            raise

    def user_exists(self, user_id: str) -> bool:
        try:
            return len(self.col.get(where={"user_id": user_id})["ids"]) > 0
        except Exception as e:
            print(f"Error checking user existence: {e}")
            return False

    def retrieve(
        self,
        user_id: str,
        query: str = "user name past orders reading preferences genres",
        n: int = 3
    ) -> str:
        try:
            if self.col.count() == 0:
                return ""
            results = self.col.query(
                query_texts=[query],
                n_results=min(n, self.col.count()),
                where={"user_id": user_id}
            )
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            
            if not docs:
                return ""
                
            lines = []
            for doc, meta in zip(docs, metas):
                ts = meta.get("timestamp", "")[:10]
                lines.append(f"[Memory from {ts}]: {doc}")
            return "\n".join(lines)
        except Exception as e:
            print(f"Error retrieving long-term memory: {e}")
            return ""

    def store(self, user_id: str, facts: str, session_id: str):
        try:
            doc_id = f"{user_id}_{session_id}"
            self.col.upsert(
                ids=[doc_id],
                documents=[facts],
                metadatas=[{
                    "user_id": user_id,
                    "session_id": session_id,
                    "timestamp": datetime.datetime.now().isoformat()
                }]
            )
            print(f"\n[LTM] Stored facts for '{user_id}' → doc '{doc_id}'")
            preview = facts[:180] + "..." if len(facts) > 180 else facts
            print(f"     {preview}\n")
        except Exception as e:
            print(f"Error storing facts to long-term memory: {e}")
