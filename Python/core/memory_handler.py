"""Memory retrieval, deduplication, and long-term storage.

This module encapsulates all memory-related logic that was previously
mixed into ``app.py``.  It manages a **temporary memory cache** with
per-turn lifetimes so that relevant memories stay in the system prompt
for several turns before fading, and it handles the summarise-and-store
pipeline that decides whether a conversation fragment is worth
persisting.
"""

import json
import re
from .logger import get_logger

logger = get_logger(__name__)


class MemoryHandler:
    """Manages short-term memory retrieval and long-term memory storage.

    Args:
        memory_manager: A :class:`MemoryManager` instance used for
            vector-based retrieval and persistence.
        use_memories: Whether to retrieve memories on each turn.
        create_memories: Whether to attempt storing new memories after
            a response is generated.
    """

    def __init__(self, memory_manager, use_memories: bool = True, create_memories: bool = False):
        self.memory = memory_manager
        self.use_memories = use_memories
        self.create_memories = create_memories

        # Temporary cache: each entry is {"memory": <dict>, "turns_left": int}
        self.temporary_memories: list[dict] = []
        self.memory_lifetime: int = 4

    # ─── Public API ───────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all temporary memory state."""
        self.temporary_memories = []
        self.memory_lifetime = 4

    def retrieve_relevant_memories(self, message: str, author: str) -> str:
        """Retrieve relevant memories and return formatted text for the system prompt.

        Memories are fetched from three pools (user-specific, global, and
        Sera's own memories), deduplicated, and kept alive in a
        ``temporary_memories`` cache that decrements a *turns_left*
        counter on each call.

        Args:
            message: The current user message to search against.
            author: The message author, used as a memory tag filter.

        Returns:
            A newline-separated string of formatted memories, or an
            empty string if none are found.
        """
        if self.use_memories:
            user_memories = self.memory.retrieve(
                message, top_k=3, min_similarity=0.80, tags=[author]
            )
            global_memories = self.memory.retrieve(
                message, top_k=2, min_similarity=0.70, tags=["global"]
            )
            sera_memories = self.memory.retrieve(
                message, top_k=3, min_similarity=0.80, tags=["Sera"]
            )

        try:
            new_memories = user_memories + global_memories + sera_memories
        except UnboundLocalError:
            new_memories = []

        # Add genuinely new memories to the temporary cache
        for mem in new_memories:
            if mem not in [m["memory"] for m in self.temporary_memories]:
                self.temporary_memories.append(
                    {"memory": mem, "turns_left": self.memory_lifetime}
                )

        # Decrement lifetime and prune expired entries
        self.temporary_memories = [
            {"memory": m["memory"], "turns_left": m["turns_left"] - 1}
            for m in self.temporary_memories
            if m["turns_left"] > 1
        ]
        self.temporary_memories = [
            m for m in self.temporary_memories if m["turns_left"] > 0
        ]

        combined_memories = self.memory.deduplicate_memories(
            [m["memory"] for m in self.temporary_memories]
        )

        # Debug logging
        logger.debug(f"🧠 Relevant memories with lifetime ({len(self.temporary_memories)}):")
        for m in self.temporary_memories:
            logger.debug(
                f"  ({m['turns_left']} turns left) "
                f"[{m['memory']['days_ago']} days_ago] "
                f"{m['memory']['text']}"
            )

        if not combined_memories:
            logger.debug("  No memories found.")

        memory_lines = [
            f"[{mem['days_ago']} days_ago] {mem['text']}"
            for mem in combined_memories
        ]
        return "\n".join(memory_lines) if memory_lines else ""

    def summarize_and_store(self, raw_log_list: list, api_client) -> None:
        """Summarize the conversation and store a memory if relevant.

        Runs the raw conversation log through the LLM to determine
        whether the exchange contains information worth remembering.
        On success the ``raw_log_list`` is **cleared** so that
        subsequent calls do not re-process the same conversation.

        Args:
            raw_log_list: *Mutable* reference to the engine's raw
                conversation log (a ``list[str]``).  Cleared on
                successful save.
            api_client: API client used for the summarisation call.
        """
        try:
            text = "\n".join(raw_log_list)
            # TODO: replace with a proper summarisation prompt.
            summary_prompt = "Made a resumen"
            raw_response = api_client.complete(
                summary_prompt, temperature=0.3, max_tokens=256
            )

            logger.debug(f"🎙️ Raw summary response: {raw_response}")

            # Strip markdown code fences that the model sometimes adds.
            cleaned = re.sub(r"```json\s*|```", "", raw_response).strip()

            if not cleaned:
                logger.warning("AI returned empty summary.")
                return

            summary_data = json.loads(cleaned)

            if not summary_data.get("recuerdo", False):
                logger.info("Nothing worthy of memory according to AI.")
                return

            summary_text = summary_data["text"]
            tags = summary_data.get("tags", [])

            # Check for near-duplicate memories before saving
            similar_memories = self.memory.retrieve(
                summary_text, top_k=3, min_similarity=0.87, tags=tags
            )

            if similar_memories:
                logger.info("⛔ Similar memory already exists. Skipping save.")
                return

            self.memory.add_memory(summary_text, tags=tags)
            raw_log_list.clear()
            logger.info("💾 Memory saved successfully.")

        except json.JSONDecodeError as e:
            logger.error(f"Memory summary JSON decode error: {e}")
            logger.debug(f"Received content: {raw_response}")

        except Exception as e:
            logger.error(f"Unexpected error in memory summarization: {e}")
