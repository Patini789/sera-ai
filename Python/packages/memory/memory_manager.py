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
        self._last_mtime: float = 0.0  # For hot-reload detection
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
                self.memories = []
                for mem in data:
                    arr = np.array(mem.get("embedding", []), dtype=np.float32)
                    self.memories.append({
                        "id": mem.get("id"),
                        "text": mem.get("text"),
                        "tags": mem.get("tags", []),
                        "timestamp": mem.get("timestamp"),
                        "embedding": arr
                    })
                self._last_mtime = os.path.getmtime(self.json_path)
            except Exception as e:
                logger.error(f"Error loading memories: {e}")
                self.memories = []
        else:
            os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
            with open(self.json_path, 'w', encoding="utf-8") as f:
                json.dump([], f)
            self._last_mtime = os.path.getmtime(self.json_path)

    def _reload_if_changed(self):
        """Hot-reload memories from disk if the JSON file was modified externally.

        This allows the MCP tool process to write new memories and have them
        reflected immediately in the main process without a restart.
        """
        try:
            current_mtime = os.path.getmtime(self.json_path)
            if current_mtime != self._last_mtime:
                logger.debug("Memory file changed externally — hot-reloading...")
                self._load_memories()
                # Invalidate query cache after reload
                self.last_text = None
                self.query_emb = None
        except FileNotFoundError:
            pass

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
        # Update mtime cache so we don't reload our own write
        self._last_mtime = os.path.getmtime(self.json_path)

    def _next_id(self) -> str:
        """Generate a collision-safe memory ID using the max existing numeric index."""
        if not self.memories:
            return "mem_0001"
        existing_nums = []
        for mem in self.memories:
            mem_id = mem.get("id", "")
            if mem_id.startswith("mem_"):
                try:
                    existing_nums.append(int(mem_id[4:]))
                except ValueError:
                    pass
        next_num = (max(existing_nums) + 1) if existing_nums else 1
        return f"mem_{next_num:04d}"

    def add_memory(self, text, tags=None, timestamp=None, embedding=None):
        """Add a new memory with automatic semantic deduplication.

        Before saving, checks if a memory with >= 92% cosine similarity already
        exists. If a near-duplicate is found, merges tags into the existing memory
        and returns its ID without creating a duplicate. Otherwise, creates a new
        memory entry.

        Args:
            text: Content to store.
            tags: List of strings for filtering/boosting (soft-tags).
            timestamp: ISO 8601 string; if None, current UTC time is used.
            embedding: np.array or list; if None, generated via API client.

        Returns:
            The memory ID (existing or newly created).
        """
        if tags is None:
            tags = []
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        # Generate embedding for the new text
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

        # ── Semantic deduplication (threshold: 0.92) ──────────────────
        if self.memories:
            emb_matrix = np.stack([m["embedding"] for m in self.memories], axis=0)
            emb_norm = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-9)
            q_norm = emb / (np.linalg.norm(emb) + 1e-9)
            sims = emb_norm.dot(q_norm)
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])

            if best_score >= 0.92:
                existing = self.memories[best_idx]
                existing_id = existing["id"]
                logger.info(
                    f"Near-duplicate found (score={best_score:.3f}): '{existing_id}'. "
                    f"Merging tags instead of creating duplicate."
                )
                # Merge new tags into existing, avoiding duplicates
                merged_tags = list(set(existing["tags"]) | set(tags))
                if merged_tags != existing["tags"]:
                    existing["tags"] = merged_tags
                    existing["timestamp"] = timestamp  # Freshen timestamp
                    self._save_memories()
                    logger.debug(f"  Updated tags for {existing_id}: {merged_tags}")
                return existing_id

        # ── No duplicate found — create new memory ────────────────────
        mem_id = self._next_id()
        self.memories.append({
            "id": mem_id,
            "text": text,
            "tags": tags,
            "timestamp": timestamp,
            "embedding": emb
        })
        self._save_memories()
        logger.info(f"New memory saved: {mem_id}")
        return mem_id

    def search_memories(self, query_embedding, top_k=5, tag_filter=None):
        """Search memories by cosine similarity with optional soft-tag boosting.

        Unlike strict keyword filtering, tag_filter acts as a BOOST: memories
        that match the requested tags receive a +0.15 score bonus but are NOT
        excluded if they lack the tag. This ensures relevant memories are never
        missed due to missing or incomplete tags.

        Args:
            query_embedding: Embedding vector to search against.
            top_k: Maximum number of results to return.
            tag_filter: Optional list of tags. Matching memories get +0.15 boost.

        Returns:
            List of (memory_dict, similarity_score) sorted by score descending.
        """
        # Hot-reload before every search to pick up MCP-written memories
        self._reload_if_changed()

        if not self.memories:
            return []

        # Always search ALL memories — no hard exclusion by tag
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

        # ── Soft-tag boost (+0.15 for tag matches, clamped to 1.0) ────
        if tag_filter:
            tag_set = set(tag_filter)
            for i, mem in enumerate(candidates):
                if tag_set.intersection(mem["tags"]):
                    sims[i] = min(1.0, sims[i] + 0.15)

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

    def buscar_palabra_clave(self, keyword: str) -> list[tuple[dict, float]]:
        """Exact keyword search with a point-based scoring system.

        Does NOT use embeddings. Iterates over all memory texts looking for
        the keyword as a literal string match (case-insensitive).

        Scoring:
            - 1.0 base point if the keyword is found anywhere in the text.
            - +0.2 for each additional occurrence beyond the first.
            - +0.5 if the keyword matches any of the memory's tags exactly
              (case-insensitive comparison).

        Args:
            keyword: The exact word or phrase to search for.

        Returns:
            List of (memory_dict, score) sorted by score descending.
            Empty list if nothing is found.
        """
        self._reload_if_changed()

        keyword_lower = keyword.lower()
        results = []

        for mem in self.memories:
            text_lower = mem["text"].lower()
            count = text_lower.count(keyword_lower)

            if count == 0:
                continue  # Keyword not present in this memory

            # Base score for finding the keyword at all
            score = 1.0
            # Bonus for each additional occurrence beyond the first
            score += 0.2 * (count - 1)
            # Tag exact-match bonus
            if any(keyword_lower == tag.lower() for tag in mem["tags"]):
                score += 0.5

            results.append((mem, round(score, 2)))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results

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
