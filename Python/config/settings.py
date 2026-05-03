"""Load and validate environment variables."""
# settings.py
import os
import json
from dotenv import load_dotenv
from pathlib import Path
from ..env.bot_config import bot_settings, discord_config


dotenv_path = Path(__file__).resolve().parent.parent / "env" / ".env"
load_dotenv(dotenv_path)

class Settings:
    def __init__(self):
        """Just a simple settings class to load and store configurations."""
        base = Path(__file__).resolve().parent.parent
        self.instructions = bot_settings["INSTRUCTIONS"]

        self.memory_path = base / "packages" / "memory" / "memories.json"
        self.file_path = base / ".." / "data" / "current_conversation.json"
        self.emotion_path = base / ".." / "data" / "current_emotion.json"
        self.extra_data = bot_settings["EXTRA_DATA"]

        # AI
        self.max_tokens = int(bot_settings["MAX_TOKENS"])
        self.prompter_route = base / "packages" / "models"
        
        # Model API
        get_api_path = base / "env" / "modelAPI.json"
        with open(get_api_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.get_api = data

            model_data = data.get("model", {})
            embed_data = data.get("embedding", {})
            
            self.api_key = os.getenv("API_KEY")

            self.completion_model = model_data.get("model")
            self.completion_url = model_data.get("url")

            self.embedding_model = embed_data.get("model")
            self.embedding_url = embed_data.get("url")
            self.local_token = os.getenv("LOCAL_TOKEN")

            
            self.opencode_key = os.getenv("OPENCODE_KEY")

        # Boolean Toggles
        self.voice_active = bot_settings.get("VOICE_ACTIVE", False)
        self.voice_api_active = bot_settings.get("VOICE_API", False)
        self.enable_discord_bot = discord_config.get("ENABLE_DISCORD_BOT", False)
        self.school = bot_settings.get("SCHOOL", False)
        self.speech_to_text = bot_settings.get("SPEECH_TO_TEXT", False)
        self.image_vision = bot_settings.get("IMAGE_VISION", False)
        self.extra_data = bot_settings.get("EXTRA_DATA", False) 
        self.create_memories = bot_settings.get("CREATE_MEMORIES", False)
        self.use_memories = bot_settings.get("USE_MEMORIES", True)

        self.opencode_enabled = bot_settings.get("OPENCODE_ENABLED", True)
        self.opencode_models = bot_settings.get("OPENCODE_MODELS", ["big-pickle"])        

        self.gemini_enabled = bot_settings.get("GEMINI_ENABLE", False)
        self.gemini_models = bot_settings.get("GEMINI_MODELS", ["gemma-4-26b"])
        self.disable_thinking = bot_settings.get("DISABLE_THINKING", False)
        self.just_discord = bot_settings.get("JUST_DISCORD_TTS", False)

        # Voice
        if self.voice_api_active and self.voice_active:
            voice_data = data.get("voice", {})
            self.voice_api = voice_data.get("api")
            self.voice_model = voice_data.get("model")
            self.voice_url = voice_data.get("url")
        elif self.voice_active:
            self.voice = bot_settings.get("VOICE")
            self.rate = bot_settings.get("RATE")
            self.pitch = bot_settings.get("PITCH")
        else:
            self.voice = None
            self.rate = None
            self.pitch = None

        # Discord
        if self.enable_discord_bot:
            self.discord_token = os.getenv("DISCORD_TOKEN")
            self.discord_names = discord_config.get("DISCORD_NAMES")
            self.discord_help = discord_config.get("HELP")

            # Discord Notify
            notify_dicc = discord_config.get("notify", {})
            self.notify_user_ids = list(notify_dicc.get("users", {}).get("active", {}).values())
            self.notify_channel_ids = list(notify_dicc.get("channels", {}).get("active", {}).values())
            self.user_ids = notify_dicc.get("users", {})
        else:
            self.discord_token = None
            self.discord_names = None
            self.discord_help = None
            self.notify_user_ids = []
            self.notify_channel_ids = []
            self.user_ids = {}
        
        # User Context
        get_api_path = base / "env" / "user_context.json"
        if os.path.exists(get_api_path):
            with open(get_api_path, "r", encoding="utf-8") as f:
                self.user_context = json.load(f)
            self.user_context_path = get_api_path
        else:
            self.user_context = {}
            self.user_context_path = None

        # Schedules
        if self.school:
            self.schedules_path = base / "env" / "schedules.json"
        else:
            self.schedules_path = None

settings = Settings()
