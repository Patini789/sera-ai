"""Application orchestrator — composes core modules and manages services.

This module provides the :class:`App` class, which is the single entry
point consumed by ``__main__.py``, ``ui.py``, and ``discord_bot.py``.
It delegates all conversation logic to :class:`ConversationEngine` and
all memory logic to :class:`MemoryHandler`, keeping only service
lifecycle management and cross-cutting concerns (TTS playback) here.
"""

import time
import threading
import asyncio

from .packages.godot_bridge import godot_bridge_loop
from .packages.game_bridge import game_bridge_loop
from .core import ConversationEngine, MemoryHandler
from .io.voice import Voice



class App:
    """Main application class that wires together all subsystems.

    Acts as a thin orchestrator: creates the conversation engine,
    memory handler, and manages service lifecycle (TTS, Discord, Godot).

    The public interface is intentionally kept identical to the
    pre-refactor version so that **no downstream modules need changes**.

    Args:
        api_client: The API client for LLM completions and embeddings.
        memory_manager: The :class:`MemoryManager` instance for vector
            memory storage.
        tts_service: Optional TTS service with an async ``speak()``
            method.  Pass ``None`` to disable voice output.
        settings: Application :class:`Settings` object.
    """

    def __init__(self, api_client, memory_manager, tts_service, settings):
        # Injected services
        self.api_client = api_client
        self.tts = tts_service
        self.voice_active = settings.voice_active


        # Memory handler
        self.memory_handler = MemoryHandler(
            memory_manager=memory_manager,
            use_memories=settings.use_memories,
            create_memories=settings.create_memories,
        )

        # Conversation engine
        self.engine = ConversationEngine(
            api_client=api_client,
            memory_handler=self.memory_handler,
            max_tokens=settings.max_tokens,
            gemini_enable=settings.gemini_enabled,
            opencode_enable=settings.opencode_enabled,
            user_context_json=settings.user_context,
            instructions=settings.instructions,
            on_response_complete=self._on_response_complete,
        )

        # Service registry
        self.background_services = []
        self.blocking_service = None
        
        self.stt = None
        self.audio_player_hooks = []

        if settings.speech_to_text:
            self.stt = Voice(on_speech_detected_fn=self.handle_new_message_wrapper)
            self.background_services.append(self.stt)
            
    def register_audio_player(self, func_player):
        self.audio_player_hooks.append(func_player)

    # ─── Properties (read/write access for ui.py) ─────────────────────

    @property
    def gemini_enable(self):
        """Whether the Gemini API is used as the primary backend."""
        return self.engine.gemini_enable

    @gemini_enable.setter
    def gemini_enable(self, value: bool):
        self.engine.gemini_enable = value

    @property
    def create_memories(self):
        """Whether new memories are created after responses."""
        return self.memory_handler.create_memories

    @create_memories.setter
    def create_memories(self, value: bool):
        self.memory_handler.create_memories = value

    @property
    def use_memories(self):
        """Whether memories are retrieved and injected into prompts."""
        return self.memory_handler.use_memories

    @use_memories.setter
    def use_memories(self, value: bool):
        self.memory_handler.use_memories = value

    @property
    def instructions(self):
        """The base system-prompt instructions."""
        return self.engine.instructions

    @instructions.setter
    def instructions(self, value: str):
        self.engine.instructions = value

    @property
    def disable_thinking(self):
        """Whether the model's thinking/reasoning phase is suppressed."""
        return self.api_client.disable_thinking

    @disable_thinking.setter
    def disable_thinking(self, value: bool):
        self.api_client.disable_thinking = value

    # ─── Public conversation API ──────────────────────────────────────

    def handle_new_message(self, author: str, message: str) -> str:
        """Handle a new user message synchronously.

        Delegates to :meth:`ConversationEngine.handle_new_message`.
        """
        return self.engine.handle_new_message(author, message)

    def handle_new_message_stream(self, author: str, message: str):
        """Generator that yields response tokens via streaming.

        Delegates to :meth:`ConversationEngine.handle_new_message_stream`.
        """
        yield from self.engine.handle_new_message_stream(author, message)

    def handle_new_message_wrapper(self, author: str, text: str) -> None:
        """Thread-safe wrapper for ``handle_new_message``."""
        self.engine.handle_new_message_wrapper(author, text)

    def reset_lists(self) -> None:
        """Reset all conversational and memory state."""
        self.engine.reset()

    # ─── Post-response callback ───────────────────────────────────────

    def _on_response_complete(self, clean_response: str, raw_conversation_log: list) -> None:
        """Invoked by the engine after each completed LLM response.

        Handles cross-cutting concerns:
        * **TTS playback** — speaks the clean response asynchronously.
        * **Memory creation** — summarises the conversation log and
          persists it if deemed relevant.
        """
        # TTS playback
        if self.voice_active and self.tts:
            def run_tts():
                try:
                    audio_file_path = asyncio.run(self.tts.speak(clean_response, voice_instance=self.stt))
                    if self.audio_player_hooks and audio_file_path:
                        for hook in self.audio_player_hooks:
                            hook(audio_file_path)
                except RuntimeError as e:
                    print(f"⚠️ TTS thread error: {e}")

            threading.Thread(target=run_tts, daemon=True).start()

        # Memory creation
        if self.memory_handler.create_memories:
            print("🪧 Conversation passed to memory summarizer:\n" + "-" * 50)
            print(raw_conversation_log)
            print("-" * 50)

            threading.Thread(
                target=self.memory_handler.summarize_and_store,
                args=(raw_conversation_log, self.api_client),
                daemon=True,
            ).start()

    # ─── Service management ───────────────────────────────────────────

    def add_service(self, service, is_blocking: bool = False) -> None:
        """Register a service to be started when :meth:`run` is called.

        Args:
            service: Any object with a ``start()`` method.
            is_blocking: If ``True`` the service will run on the main
                thread (e.g. the Discord event loop).
        """
        if is_blocking:
            self.blocking_service = service
        else:
            self.background_services.append(service)

    def run(self) -> None:
        """Start all registered services and the Godot bridge.

        1. Background services are launched in daemon threads.
        2. The Godot WebSocket bridge starts in its own thread.
        3. If a blocking service is registered, it takes over the
           main thread; otherwise, a simple keep-alive loop runs
           until interrupted.
        """
        # 1. Background services (e.g. TTS)
        for service in self.background_services:
            if service and hasattr(service, "start"):
                threading.Thread(target=service.start, daemon=True).start()

        # 2. Godot bridge
        godot_thread = threading.Thread(
            target=lambda: asyncio.run(godot_bridge_loop()),
            daemon=True,
        )
        godot_thread.start()

        # 2.5 Game bridge (new)
        game_thread = threading.Thread(
            target=lambda: asyncio.run(game_bridge_loop(self)),
            daemon=True,
        )
        game_thread.start()

        # 3. Blocking service or keep-alive
        if self.blocking_service:
            print("Starting main blocking service...")
            self.blocking_service.start()
        else:
            print("Running in non-discord mode. Press Ctrl+C to exit.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("Exiting.")
