"""Memory retrieval and long-term storage handler.

This module encapsulates all memory-related logic that was previously
mixed into ``app.py``.  It manages a **temporary memory cache** with
per-turn lifetimes so that relevant memories stay in the system prompt
for several turns before fading.

Memory *creation* is now handled agentically via MCP Tools: Sera decides
when to save a memory by calling ``guardar_recuerdo`` directly.  The
``summarize_and_store`` batch pipeline has been removed.
"""

from .logger import get_logger

logger = get_logger(__name__)


class MemoryHandler:
    """Manages short-term memory retrieval and injection into prompts.

    Args:
        memory_manager: A :class:`MemoryManager` instance used for
            vector-based retrieval and persistence.
        use_memories: Whether to retrieve memories on each turn and
            inject them into the system prompt.
    """

    def __init__(self, memory_manager, use_memories: bool = True):
        self.memory = memory_manager
        self.use_memories = use_memories

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
            author: The message author, used as a soft-tag boost filter.

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
        logger.debug(f"Relevant memories with lifetime ({len(self.temporary_memories)}):")
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
