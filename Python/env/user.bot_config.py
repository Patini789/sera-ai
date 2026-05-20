# bot_config.py — Central configuration file for Sera AI
# Copy this file as 'bot_config.py' in the same directory and customize.

bot_settings = {
    "MAX_TOKENS": 8192,

    # System prompt — defines Sera's personality and behavior.
    # Write your own character here or leave empty for a generic assistant.
    "INSTRUCTIONS": """
You are Sera, a helpful AI assistant.
""",

    # ── Feature Toggles ──────────────────────────────────────
    "VOICE_ACTIVE": False,      # Enable Text-to-Speech output
    "VOICE_API": False,         # Use external TTS API (Kokoro) instead of Edge TTS
    "SPEECH_TO_TEXT": False,    # Enable Whisper STT (requires VOICE_ACTIVE)
    "CREATE_MEMORIES": False,   # Auto-create memories from conversations
    "USE_MEMORIES": True,       # Retrieve and inject relevant memories into prompts
    "DISABLE_THINKING": False,  # Skip model's reasoning/thinking phase (useful for TTS)
    "JUST_DISCORD_TTS": False,  # Route TTS only to Discord (no local playback)

    # ── Gaming Mode (Vision & PTT) ───────────────────────────
    "GAMING_MODE": False,               # Auto-capture screenshot on every input as vision payload
    "STT_GAMING_MODE": False,           # Push-to-talk (Ctrl+Alt+V) for STT instead of silence detection
    "SCREENSHOT_MAX_DIMENSION": 768,    # Max dimension (width/height) for screenshot resizing
    "SCREENSHOT_JPEG_QUALITY": 70,      # JPEG quality (1-100) for compression
    # ── Scheduler / Agenda ────────────────────────────────
    # Enable the background scheduler that fires daily digests and proximity alerts.
    "SCHEDULER_ENABLED": False,
    "SCHEDULER_CRON_HOUR": 4,             # Hour (0-23) for the daily digest
    "SCHEDULER_CRON_MINUTE": 0,           # Minute (0-59) for the daily digest
    "SCHEDULER_PROXIMITY_CHECK_MINUTES": 10,   # How often (minutes) to check for upcoming events
    "SCHEDULER_PROXIMITY_WINDOW_MINUTES": 15,  # Alert when an event is within this many minutes
    "SCHEDULER_PROMPT_TEMPLATE": "[System] It is {hora}. Here are today's events:\n{eventos}\nRemind the user naturally.",
    "SCHEDULER_PROXIMITY_PROMPT": "[System] Heads up! In {minutos} minutes you have: {evento}. Let the user know.",

    # ── Cloud API: Gemini / Gemma ────────────────────────────
    # If True, tries the Gemini API before falling back to the local LLM.
    "GEMINI_ENABLE": False,

    # Ordered fallback list — if the active model's quota is exhausted (HTTP 429),
    # SeraAI will automatically try the next one.
    #
    # Supported models (as of 2026):
    #   gemma-4-31b-it           — Gemma 4 31B  (function calling, inline thinking)
    #   gemma-4-26b-a4b-it       — Gemma 4 26B  (function calling, inline thinking)
    #   gemini-3-flash-preview   — Gemini 3     (function calling, no thinking API)
    #   gemini-2.5-flash         — Gemini 2.5   (function calling, thinking API)
    #   gemini-2.5-flash-lite    — Gemini 2.5   (function calling, thinking API)
    #   gemini-3.1-flash-lite    — Gemini 3.1   (function calling, no thinking API)
    #   gemma-3-27b-it           — Gemma 3 27B  (basic, no native tool support)
    "GEMINI_MODELS": [
        "gemma-4-31b-it",
        "gemma-4-26b-a4b-it",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ],

    # ── Cloud API: OpenCode ──────────────────────────────────
    # If True, tries OpenCode API after Gemini (if enabled) and before local LLM.
    "OPENCODE_ENABLED": False,
    "OPENCODE_MODELS": ["big-pickle"],

    # ── Cloud API: OpenRouter ────────────────────────────────
    # If True, tries OpenRouter API before falling back to the local LLM.
    "OPENROUTER_ENABLED": False,
    "OPENROUTER_MODELS": ["google/gemini-3.1-flash-lite"],

    # ── Voice Settings (Edge TTS) ────────────────────────────
    # See available voices: https://speech.platform.bing.com/consumer/speech/synthesize/readaloud/voices/list
    "VOICE": "es-HN-KarlaNeural",
    "RATE": "+10%",
    "PITCH": "-10Hz",

    # ── Extra Data Tags ──────────────────────────────────────
    # Tags for memory database filtering
    "EXTRA_DATA": [],
}

discord_config = {
    "ENABLE_DISCORD_BOT": False,

    # Help message for the $help command
    "HELP": """
**Available Commands:**
1. **$chat [message]**: Send a message to Sera.
2. **$help**: Show this help message.
3. **$cls**: Clear Sera's conversation memory.
4. **$context [info]**: Set your personal context for more tailored responses.
5. **$check [user|list]**: View saved user contexts.
""",

    # Map Discord usernames to display names
    "DISCORD_NAMES": {
        # "discord_username": "DisplayName",
    },

    # Discord notification settings
    # Add user/channel IDs to "active" to receive online notifications
    "notify": {
        "users": {
            "active": {},
        },
        "channels": {
            "active": {},
        }
    }
}