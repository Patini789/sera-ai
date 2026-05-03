"""Conversation engine — manages conversation state and LLM interactions.

This is the central conversation processor for Sera AI.  It:

1. Builds system prompts enriched with memory context and user metadata.
2. Routes requests through Gemini or a local LLM backend.
3. Manages the full message history for multi-turn conversations.
4. Fires a callback after each completed response so the orchestrator
   can trigger side-effects (TTS, memory creation, etc.).
"""

import json
import re
import threading
from datetime import datetime
from typing import Callable, Optional

from .training_data import save_training_data


class ConversationEngine:
    """Handles conversation flow, history management, and LLM API calls.

    Args:
        api_client: API client providing ``gemini_complete``,
            ``local_complete``, and ``local_complete_stream`` methods.
        memory_handler: A :class:`MemoryHandler` used to inject
            relevant memories into the system prompt.
        max_tokens: Maximum tokens for LLM responses.
        gemini_enable: If *True*, try the Gemini API before falling
            back to the local model.
        user_context_json: ``dict`` mapping author names to context
            strings that are injected into the system prompt.
        instructions: Base system-prompt instructions.
        on_response_complete: Optional callback invoked after every
            completed response with signature
            ``(clean_response: str, raw_conversation_log: list[str])``.
    """

    def __init__(
        self,
        api_client,
        memory_handler,
        max_tokens: int,
        gemini_enable: bool,
        opencode_enable: bool,
        user_context_json: dict,
        instructions: str,
        on_response_complete: Optional[Callable] = None,
    ):
        self.api_client = api_client
        self.memory_handler = memory_handler
        self.max_tokens = max_tokens
        self.gemini_enable = gemini_enable
        self.opencode_enable = opencode_enable
        self.user_context_json = user_context_json
        self.instructions = instructions
        self._on_response_complete = on_response_complete
        self.current_game_state = "" # New attribute to track game state, can be updated by tools and injected into prompts.

        # Conversational state
        self.current_authors: list[str] = []
        self.conversation_history: list[dict] = []
        self.raw_conversation_log: list[str] = []
        self.user_context: list = []

    # ─── State management ─────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all conversational state for a fresh session."""
        self.current_authors = []
        self.conversation_history = []
        self.raw_conversation_log = []
        self.user_context = []
        self.memory_handler.reset()

    # ─── Internal helpers ─────────────────────────────────────────────

    def _prepare_messages(self, author: str, message: str):
        """Build the API message list and the system prompt.

        Side-effects:
            * Appends the user message to ``conversation_history`` and
              ``raw_conversation_log``.
            * Registers new authors in ``current_authors``.

        Returns:
            ``(messages_for_api, system_prompt_text)`` tuple.
        """
        self.raw_conversation_log.append(f"author: {author}\ncontent: {message}")
        self.conversation_history.append(
            {"role": "user", "content": f"{author}: {message}"}
        )

        # Retrieve memories relevant to this turn
        memory_text = self.memory_handler.retrieve_relevant_memories(message, author)

        # Track author for per-user context injection
        if author in self.user_context_json and author not in self.current_authors:
            print(f"Adding context for {author}")
            self.current_authors.append(author)

        combined_user_contexts = []
        for a in self.current_authors:
            ctx = self.user_context_json.get(a, "").strip()
            if ctx:
                combined_user_contexts.append(f"Context of {a}: {ctx}")

        date_hour = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        game_context = ""
        if self.current_game_state:
            game_context = f"\n\n--- ESTADO ACTUAL DEL JUEGO ---\n{self.current_game_state}\n-------------------------------"

        system_prompt_text = (
            f"{self.instructions}\n"
            f"Day and hour: {date_hour}.\n"
            f"{memory_text}\n"
            + "\n".join(combined_user_contexts)
            + game_context
        )

        messages_for_api = [
            {"role": "system", "content": system_prompt_text}
        ] + self.conversation_history

        print("🏷️ PROMPT Sent (Clean JSON):\n" + "-" * 50)
        print(json.dumps(messages_for_api, indent=2, ensure_ascii=False))
        print("-" * 50)

        return messages_for_api, system_prompt_text

    def _finalize_response(
        self,
        author: str,
        message: str,
        full_response: str,
        messages_for_api: list,
        system_prompt_text: str,
    ) -> None:
        """Post-processing after a complete LLM response.

        1. Strips internal model tags from the response.
        2. Adds the CLEAN response to history to prevent prompt poisoning.
        3. Saves the RAW response for fine-tuning.
        4. Invokes the callback.
        """
        
        # 1. Clean the response
        clean = re.sub(
            r"<think>.*?</think>|<lmstudio_tools>.*?</lmstudio_tools>",
            "",
            full_response,
            flags=re.DOTALL,
        ).strip()

        # Console log for debugging
        self.raw_conversation_log.append(
            f"author: Sera\ncontent: {full_response}\n"
        )
        
        self.conversation_history.append(
            {"role": "assistant", "content": clean}
        )

        print("🪧 Response (Internal with thoughts):\n" + "-" * 50)
        print(full_response)
        print("-" * 50)
        print("🗣️ Response (Clean for users):\n" + "-" * 50)
        print(clean)
        print("-" * 50)

        # 3. Fine-tuning data gets the raw response
        threading.Thread(
            target=save_training_data,
            args=(
                system_prompt_text,
                self.conversation_history,
                f"{author}: {message}",
                full_response,
            ),
            daemon=True,
        ).start()

        if self._on_response_complete:
            self._on_response_complete(clean, self.raw_conversation_log)

    # ─── Public conversation API ──────────────────────────────────────

    def handle_new_message(self, author: str, message: str) -> str:
        """Handle a new user message and return the model's response.

        Uses the Gemini API when enabled, falling back to the local
        model on failure or when Gemini is disabled.

        Args:
            author: Identifier of the message author.
            message: The message text.  Send ``"cls"`` to reset the
                conversation.

        Returns:
            The model's response string, or an error message on failure.
        """
        if message == "cls":
            self.reset()
            return "Clean memory and context."

        messages_for_api, system_prompt_text = self._prepare_messages(author, message)
        response = None

        try:
            if self.gemini_enable:
                try:
                    clean_history_text = "\n\n".join(
                        [
                            f"{msg['role'].upper()}:\n{msg['content']}"
                            for msg in self.conversation_history
                        ]
                    )
                    response = self.api_client.gemini_complete(
                        prompt=clean_history_text,
                        system_instr=system_prompt_text,
                    )
                except Exception as e:
                    print(f"Gemini failed ({e})... Changing to LOCAL.")
                    response = None

            if response is None:
                print("Using local model...")
                response = self.api_client.local_complete(
                    messages=messages_for_api,
                    max_tokens=self.max_tokens,
                    temperature=0.6,
                )
            if not response:
                raise ValueError("La API no devolvió ninguna respuesta (vacía).")
        except Exception as e:
            error_msg = f"Internal system error: {e}"
            print(f"❌ Error CRÍTICO en handle_new_message: {e}")
            return error_msg

        self._finalize_response(
            author, message, response, messages_for_api, system_prompt_text
        )
        return response

    def handle_new_message_stream(self, author: str, message: str):
        """Generator that yields response tokens.

        When Gemini is enabled, calls ``gemini_complete`` (non-streaming)
        and yields the entire response in one chunk.  Falls back to the
        local streaming endpoint on failure or when Gemini is disabled.

        Args:
            author: Identifier of the message author.
            message: The message text.  Send ``"cls"`` to reset the
                conversation.

        Yields:
            Individual response tokens as strings.
        """
        if message == "cls":
            self.reset()
            yield "Clean memory and context."
            return

        messages_for_api, system_prompt_text = self._prepare_messages(author, message)

        full_response = ""
        used_api_model = False

        if self.opencode_enable:
            try:
                clean_history_text = "\n\n".join(
                    [
                        f"{msg['role'].upper()}:\n{msg['content']}"
                        for msg in self.conversation_history
                    ]
                )
                response = self.api_client.opencode_complete(
                    prompt=clean_history_text,
                    system_instr=system_prompt_text,

                )
                if response:
                    full_response = response
                    used_api_model = True
                    yield full_response  # Single chunk (OpenCode is not streamed)
            except Exception as e:
                used_api_model = False
                print(f"OpenCode failed ({e})... Changing to LOCAL stream.")
        # Fin test desechable

        # ── Try Gemini first if enabled ──────────────────────────
        elif self.gemini_enable:
            try:
                clean_history_text = "\n\n".join(
                    [
                        f"{msg['role'].upper()}:\n{msg['content']}"
                        for msg in self.conversation_history
                    ]
                )
                gemini_response = self.api_client.gemini_complete(
                    prompt=clean_history_text,
                    system_instr=system_prompt_text,
                )
                if gemini_response:
                    full_response = gemini_response
                    used_api_model = True
                    yield full_response  # Single chunk (Gemini is not streamed)
            except Exception as e:
                print(f"Gemini failed ({e})... Changing to LOCAL stream.")

        # ── Fallback to local streaming ──────────────────────────
        if not used_api_model:
            for token in self.api_client.local_complete_stream(
                messages=messages_for_api, temperature=0.6
            ):
                full_response += token
                yield token

        self._finalize_response(
            author, message, full_response, messages_for_api, system_prompt_text
        )

    def handle_new_message_wrapper(self, author: str, text: str) -> None:
        """Simple wrapper for threading compatibility."""
        self.handle_new_message(author, text)
