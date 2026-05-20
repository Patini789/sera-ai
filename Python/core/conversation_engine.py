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
from .logger import get_logger
from ..packages.screenshot_utils import capture_screenshot_b64

logger = get_logger(__name__)


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
        opencode_enable: If *True*, try the OpenCode API before falling
            back to the local model.
        openrouter_enable: If *True*, try the OpenRouter API before falling
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
        openrouter_enable: bool,
        user_context_json: dict,
        instructions: str,
        on_response_complete: Optional[Callable] = None,
        screenshot_max_dimension: int = 768,
        screenshot_jpeg_quality: int = 70,
    ):
        self.api_client = api_client
        self.memory_handler = memory_handler
        self.max_tokens = max_tokens
        self.gemini_enable = gemini_enable
        self.opencode_enable = opencode_enable
        self.openrouter_enable = openrouter_enable
        self.user_context_json = user_context_json
        self.instructions = instructions
        self._on_response_complete = on_response_complete
        self.current_game_state = "" # New attribute to track game state, can be updated by tools and injected into prompts.

        # Gaming Mode
        self.gaming_mode = False
        self.IMAGE_EXPIRY_TURNS = 3
        self.screenshot_max_dimension = screenshot_max_dimension
        self.screenshot_jpeg_quality = screenshot_jpeg_quality

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

    @staticmethod
    def _extract_text_from_content(content) -> str:
        """Extract plain text from a message content (handles text or multipart list)."""
        if isinstance(content, list):
            return " ".join(
                part.get("text", "") for part in content if part.get("type") == "text"
            )
        return str(content)

    def _strip_images_for_local(self, messages: list) -> list:
        """Return a copy of messages with image attachments removed for text-only models."""
        cleaned = []
        for msg in messages:
            if isinstance(msg.get("content"), list):
                text_parts = [
                    part["text"] for part in msg["content"]
                    if part.get("type") == "text"
                ]
                cleaned.append({**msg, "content": "\n".join(text_parts)})
            else:
                cleaned.append(msg)
        return cleaned

    def _prepare_messages(self, author: str, message: str, image_b64: str | None = None):
        """Build the API message list and the system prompt.

        Side-effects:
            * Appends the user message to ``conversation_history`` and
              ``raw_conversation_log``.
            * Registers new authors in ``current_authors``.

        Returns:
            ``(messages_for_api, system_prompt_text)`` tuple.
        """
        self.raw_conversation_log.append(f"author: {author}\ncontent: {message}")

        user_content = f"{author}: {message}"

        if self.gaming_mode and image_b64:
            # Inject screenshot as OpenAI-compatible vision content
            self.conversation_history.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_content},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                            "detail": "low",
                        },
                    },
                ],
            })
        else:
            self.conversation_history.append(
                {"role": "user", "content": user_content}
            )

        # ── Rolling window: strip images from older user messages ───
        user_indices = [
            i for i, m in enumerate(self.conversation_history)
            if m["role"] == "user"
        ]
        if len(user_indices) > self.IMAGE_EXPIRY_TURNS:
            for idx in user_indices[:-self.IMAGE_EXPIRY_TURNS]:
                msg = self.conversation_history[idx]
                if isinstance(msg.get("content"), list):
                    text = next(
                        (
                            p["text"] for p in msg["content"]
                            if p.get("type") == "text"
                        ),
                        "",
                    )
                    self.conversation_history[idx] = {
                        "role": "user", "content": text,
                    }

        # Retrieve memories relevant to this turn
        memory_text = self.memory_handler.retrieve_relevant_memories(message, author)

        # Track author for per-user context injection
        if author in self.user_context_json and author not in self.current_authors:
            logger.info(f"Adding context for {author}")
            self.current_authors.append(author)

        combined_user_contexts = []
        for a in self.current_authors:
            ctx = self.user_context_json.get(a, "").strip()
            if ctx:
                combined_user_contexts.append(f"Context of {a}: {ctx}")

        system_prompt_text = (
            f"{self.instructions}\n"
            f"{memory_text}\n"
            + "\n".join(combined_user_contexts)
        )

        now = datetime.now()
        day_name = now.strftime("%A")
        date_hour = now.strftime("%Y-%m-%d %H:%M")

        game_context = ""
        if self.current_game_state:
            game_context = f"\n\n--- ESTADO ACTUAL DEL JUEGO ---\n{self.current_game_state}\n-------------------------------"

        # Cache
        messages_for_api = [
            {"role": "system", "content": system_prompt_text}
        ] + self.conversation_history

        if messages_for_api and messages_for_api[-1]["role"] == "user":
            invisible_context = (
                f"\n\n[System Data Invisble: Hoy es {day_name}, "
                f"Hora: {date_hour}{game_context}]"
            )
            last_msg = messages_for_api[-1]
            if isinstance(last_msg["content"], list):
                for part in last_msg["content"]:
                    if part.get("type") == "text":
                        part["text"] += invisible_context
                        break
            else:
                last_msg["content"] += invisible_context

        logger.debug(
            "🏷️ PROMPT Sent (Clean JSON):\n"
            + json.dumps(messages_for_api, indent=2, ensure_ascii=False)
        )

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

        logger.info(f"Response (Clean for users): {clean}")
        logger.debug(f"Response (Internal with thoughts): {full_response}")

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

        image_b64 = None
        if self.gaming_mode:
            image_b64 = capture_screenshot_b64(
                max_dimension=self.screenshot_max_dimension,
                jpeg_quality=self.screenshot_jpeg_quality,
            )

        messages_for_api, system_prompt_text = self._prepare_messages(
            author, message, image_b64=image_b64
        )
        response = None

        # When gaming mode is active we pass the full messages_for_api (which contains
        # the vision payload) directly.  Otherwise we fall back to the text-only
        # clean_history_text prompt for backwards-compatible behaviour.
        vision_messages = messages_for_api if (self.gaming_mode and image_b64) else None
        has_image = vision_messages is not None
        logger.info(f"[GamingMode] Screenshot captured: {has_image}, gaming_mode={self.gaming_mode}")

        try:
            if self.openrouter_enable:
                logger.info("[Routing] Trying OpenRouter (non-streaming)...")
                try:
                    clean_history_text = "\n\n".join(
                        f"{msg['role'].upper()}:\n{self._extract_text_from_content(msg['content'])}"
                        for msg in self.conversation_history
                    )
                    response = self.api_client.openrouter_complete(
                        prompt=clean_history_text,
                        system_instr=system_prompt_text,
                        messages=vision_messages,
                    )
                    logger.info("[Routing] OpenRouter succeeded.")
                except Exception as e:
                    logger.warning(f"[Routing] OpenRouter failed ({e})... Changing to next fallback.")
                    response = None

            if response is None and self.opencode_enable:
                logger.info("[Routing] Trying OpenCode (non-streaming)...")
                try:
                    clean_history_text = "\n\n".join(
                        f"{msg['role'].upper()}:\n{self._extract_text_from_content(msg['content'])}"
                        for msg in self.conversation_history
                    )
                    response = self.api_client.opencode_complete(
                        prompt=clean_history_text,
                        system_instr=system_prompt_text,
                        messages=vision_messages,
                    )
                    logger.info("[Routing] OpenCode succeeded.")
                except Exception as e:
                    logger.warning(f"[Routing] OpenCode failed ({e})... Changing to next fallback.")
                    response = None

            if response is None and self.gemini_enable:
                try:
                    clean_history_text = "\n\n".join(
                        f"{msg['role'].upper()}:\n{self._extract_text_from_content(msg['content'])}"
                        for msg in self.conversation_history
                    )
                    response = self.api_client.gemini_complete(
                        prompt=clean_history_text,
                        system_instr=system_prompt_text,
                        messages=vision_messages,
                    )
                except Exception as e:
                    logger.warning(f"Gemini failed ({e})... Changing to LOCAL.")
                    response = None

            if response is None:
                logger.info("Using local model...")
                response = self.api_client.local_complete(
                    messages=self._strip_images_for_local(messages_for_api),
                    max_tokens=self.max_tokens,
                    temperature=0.6,
                )
            if not response:
                raise ValueError("La API no devolvió ninguna respuesta (vacía).")
        except Exception as e:
            error_msg = f"Internal system error: {e}"
            logger.error(f"Error CRÍTICO en handle_new_message: {e}")
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

        image_b64 = None
        if self.gaming_mode:
            image_b64 = capture_screenshot_b64(
                max_dimension=self.screenshot_max_dimension,
                jpeg_quality=self.screenshot_jpeg_quality,
            )

        messages_for_api, system_prompt_text = self._prepare_messages(
            author, message, image_b64=image_b64
        )

        # When gaming mode is active we pass the full messages_for_api (which contains
        # the vision payload) directly.  Otherwise we fall back to the text-only
        # clean_history_text prompt for backwards-compatible behaviour.
        vision_messages = messages_for_api if (self.gaming_mode and image_b64) else None
        has_image = vision_messages is not None
        logger.info(f"[GamingMode] Screenshot captured: {has_image}, gaming_mode={self.gaming_mode}")

        full_response = ""
        used_api_model = False
        if self.openrouter_enable:
            logger.info("[Routing] Trying OpenRouter (streaming)...")
            try:
                clean_history_text = "\n\n".join(
                    [
                        f"{msg['role'].upper()}:\n{self._extract_text_from_content(msg['content'])}"
                        for msg in self.conversation_history
                    ]
                )
                for token in self.api_client.openrouter_complete_stream(
                    prompt=clean_history_text,
                    system_instr=system_prompt_text,
                    messages=vision_messages,
                ):
                    full_response += token
                    used_api_model = True
                    yield token
                logger.info("[Routing] OpenRouter stream succeeded.")
            except Exception as e:
                used_api_model = False
                logger.warning(f"[Routing] OpenRouter stream failed ({e})... Changing to next fallback.")

        elif self.opencode_enable:
            try:
                clean_history_text = "\n\n".join(
                    [
                        f"{msg['role'].upper()}:\n{self._extract_text_from_content(msg['content'])}"
                        for msg in self.conversation_history
                    ]
                )
                for token in self.api_client.opencode_complete_stream(
                    prompt=clean_history_text,
                    system_instr=system_prompt_text,
                    messages=vision_messages,
                ):
                    full_response += token
                    used_api_model = True
                    yield token
            except Exception as e:
                used_api_model = False
                logger.warning(f"OpenCode failed ({e})... Changing to LOCAL stream.")

        # ── Try Gemini first if enabled ──────────────────────────
        elif self.gemini_enable:
            try:
                clean_history_text = "\n\n".join(
                    [
                        f"{msg['role'].upper()}:\n{self._extract_text_from_content(msg['content'])}"
                        for msg in self.conversation_history
                    ]
                )
                for token in self.api_client.gemini_complete_stream(
                    prompt=clean_history_text,
                    system_instr=system_prompt_text,
                    messages=vision_messages,
                ):
                    full_response += token
                    used_api_model = True
                    yield token
            except Exception as e:
                logger.warning(f"Gemini failed ({e})... Changing to LOCAL stream.")

        # ── Fallback to local streaming ──────────────────────────
        if not used_api_model:
            for token in self.api_client.local_complete_stream(
                messages=self._strip_images_for_local(messages_for_api),
                temperature=0.6,
            ):
                full_response += token
                yield token

        self._finalize_response(
            author, message, full_response, messages_for_api, system_prompt_text
        )

    def handle_new_message_wrapper(self, author: str, text: str) -> None:
        """Simple wrapper for threading compatibility."""
        self.handle_new_message(author, text)
