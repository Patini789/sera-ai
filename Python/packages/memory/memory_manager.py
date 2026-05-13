import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from ...core.logger import get_logger

logger = get_logger(__name__)

class MemoryManager:
    def __init__(self, json_path, api_client, embedding_dim=None):
        """
        - json_path: file path for storing memories
        - api_client: client used for generating embeddings
        - embedding_dim: optional; if None, inferred from existing memories or API default
        """
        self.api_client = api_client
        self.json_path = json_path
        self.memories = []  # List of dicts: id, text, tags, timestamp, embedding
        self._load_memories()
        self.last_text = None  # Cache for last retrieval query to avoid redundant embeddings
        self.query_emb = None  # Cache for last query embedding

        # Infer embedding dimension if not provided
        if embedding_dim is None:
            if self.memories:
                self.embedding_dim = len(self.memories[0]["embedding"])
            else:
                self.embedding_dim = getattr(api_client, 'embedding_dim', 768)
        else:
            self.embedding_dim = embedding_dim

        # Validate and adjust loaded embeddings if needed
        for mem in self.memories:
            emb = mem["embedding"]
            dim = emb.shape[0]
            if dim != self.embedding_dim:
                logger.warning(f"Invalid embedding in {mem['id']}: dim {dim}, adjusting to {self.embedding_dim}")
                if dim > self.embedding_dim:
                    mem["embedding"] = emb[:self.embedding_dim]
                else:
                    pad = np.zeros(self.embedding_dim - dim, dtype=np.float32)
                    mem["embedding"] = np.concatenate([emb, pad])

    def _load_memories(self):
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r', encoding="utf-8") as f:
                    data = json.load(f)
                for mem in data:
                    arr = np.array(mem.get("embedding", []), dtype=np.float32)
                    self.memories.append({
                        "id": mem.get("id"),
                        "text": mem.get("text"),
                        "tags": mem.get("tags", []),
                        "timestamp": mem.get("timestamp"),
                        "embedding": arr
                    })
            except Exception as e:
                logger.error(f"Error loading memories: {e}")
                self.memories = []
        else:
            os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
            with open(self.json_path, 'w', encoding="utf-8") as f:
                json.dump([], f)

    def _save_memories(self):
        serializable = []
        for mem in self.memories:
            serializable.append({
                "id": mem["id"],
                "text": mem["text"],
                "tags": mem["tags"],
                "timestamp": mem["timestamp"],
                "embedding": mem["embedding"].tolist()
            })
        with open(self.json_path, 'w', encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=4)

    def add_memory(self, text, tags=None, timestamp=None, embedding=None):
        """
        Add a new memory.
        - text: content to store.
        - tags: list of strings for filtering.
        - timestamp: ISO 8601 string; if None, current UTC time is used.
        - embedding: np. array or list; if None, generate via API client.
        """
        if tags is None:
            tags = []
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()
        if embedding is None:
            emb = self.api_client.embed(text)
        else:
            emb = np.array(embedding, dtype=np.float32)
        if emb.shape[0] != self.embedding_dim:
            logger.warning(f"Adjusting new memory embedding from {emb.shape[0]} to {self.embedding_dim}")
            if emb.shape[0] > self.embedding_dim:
                emb = emb[:self.embedding_dim]
            else:
                pad = np.zeros(self.embedding_dim - emb.shape[0], dtype=np.float32)
                emb = np.concatenate([emb, pad])

        mem_id = f"mem_{len(self.memories) + 1:04d}"
        self.memories.append({
            "id": mem_id,
            "text": text,
            "tags": tags,
            "timestamp": timestamp,
            "embedding": emb
        })
        self._save_memories()
        return mem_id

    def search_memories(self, query_embedding, top_k=5, tag_filter=None):
        """
        Search memories by cosine similarity to the query_embedding.
        Returns list of (memory_dict, similarity_score).
        """
        if not self.memories:
            return []
        
        if tag_filter:
            tag_set = set(tag_filter)
            candidates = [mem for mem in self.memories if tag_set.intersection(mem["tags"])]
            if not candidates:
                return []
        else:
            candidates = self.memories

        emb_matrix = np.stack([mem["embedding"] for mem in candidates], axis=0)
        q_emb = np.array(query_embedding, dtype=np.float32)

        if q_emb.shape[0] != self.embedding_dim:
            logger.warning(f"Adjusting query embedding from {q_emb.shape[0]} to {self.embedding_dim}")
            if q_emb.shape[0] > self.embedding_dim:
                q_emb = q_emb[:self.embedding_dim]
            else:
                pad = np.zeros(self.embedding_dim - q_emb.shape[0], dtype=np.float32)
                q_emb = np.concatenate([q_emb, pad])

        # Normalize for cosine similarity
        emb_norm = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-9)
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        sims = emb_norm.dot(q_norm)

        # Optimization: argpartition is O(N) to find top K, argsort is O(N log N)
        k = min(top_k, len(sims))
        if len(sims) > k:
            # Find the indices of the top K similarities without fully sorting the entire array
            top_indices_unsorted = np.argpartition(sims, -k)[-k:]
            # Order just the top K indices by their similarity scores
            top_indices = top_indices_unsorted[np.argsort(sims[top_indices_unsorted])[::-1]]
        else:
            top_indices = np.argsort(sims)[::-1]

        return [(candidates[i], float(sims[i])) for i in top_indices]

    def to_dataframe(self):
        data = []
        for mem in self.memories:
            data.append({
                "id": mem["id"],
                "text": mem["text"],
                "tags": mem["tags"],
                "timestamp": mem["timestamp"],
                "embedding_dim": len(mem["embedding"]),
            })
        return pd.DataFrame(data)

    def retrieve(self, text: str, top_k: int = 5, min_similarity: float = 0.8, tags: list[str] = None) -> list[dict]:
        try:
            if text != self.last_text:
                # Only embed if the query text has changed since the last retrieval to save resources
                self.last_text = text
                self.query_emb = self.api_client.embed(text)
            results = self.search_memories(self.query_emb, top_k=top_k, tag_filter=tags)

            enriched = []
            for mem, score in results:
                if score >= min_similarity:
                    days = self._days_since(mem["timestamp"])
                    enriched.append({
                        "text": mem["text"],
                        "days_ago": days,
                        "score": score,
                        "id": mem["id"]
                    })
            return enriched
        except Exception as e:
            logger.error(f"Memory retrieval error: {e}")
            return []

    def deduplicate_memories(self, memories: list[dict]) -> list[dict]:
        seen_ids = set()
        deduped = []
        for mem in memories:
            if mem["id"] not in seen_ids:
                deduped.append(mem)
                seen_ids.add(mem["id"])
        return deduped

    def _days_since(self, timestamp: str) -> int:
        memory_time = datetime.fromisoformat(timestamp)
        now = datetime.now(memory_time.tzinfo)
        return (now - memory_time).days

    def name(self):
        return "MemoryManager"
