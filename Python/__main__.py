import time
import sys
import os

if sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Python.config.settings import settings
from Python.io.APIClient import APIClient
from Python.packages.memory.memory_manager import MemoryManager
from Python.app import App
from Python.packages.ui import start_ui_in_background

running = True

def main():
    print("Building dependencies...")

    api_client = APIClient(
        embedding_url=settings.embedding_url,
        embedding_model=settings.embedding_model,
        completion_url=settings.completion_url,
        completion_model=settings.completion_model,
        gemini_key=settings.api_key,
        gemini_models=settings.gemini_models,
        local_token=settings.local_token,
        disable_thinking=settings.disable_thinking,
        opencode_key=settings.opencode_key,
        opencode_models=settings.opencode_models
    )

    memory_manager = MemoryManager(
        json_path=settings.memory_path, 
        api_client=api_client
    )

    tts_service = None
    if settings.voice_active:
        if getattr(settings, 'voice_api_active', False):
            from Python.packages.tts_api import TTS
            tts_service = TTS(settings.voice_api, settings.voice_model, settings.voice_url, just_discord=settings.just_discord)
        else:
            from Python.packages.tts import TTS
            tts_service = TTS(settings.rate, settings.pitch, settings.voice, just_discord=settings.just_discord)
    

    app = App(
        api_client=api_client,
        memory_manager=memory_manager,
        tts_service=tts_service,
        settings=settings
    )

    if settings.enable_discord_bot:
        from Python.packages.discord_bot.discord_bot import DiscordBot
        
        shared_state = {
            "memory": memory_manager,
            "max_tokens": settings.max_tokens,
            "notify_user_ids": settings.notify_user_ids,
            "notify_channel_ids": settings.notify_channel_ids,
            "user_ids": settings.user_ids,
            "user_context_path": settings.user_context_path,
            "discord_names": settings.discord_names,
            "discord_help": settings.discord_help,
            "app_instance": app,
        }
        
        discord_bot = DiscordBot(
            shared_state, 
            on_new_message_fn=app.handle_new_message, 
            reset_lists=app.reset_lists
        )
        app.add_service(discord_bot, is_blocking=True)
    
    print("Starting Web UI Control Panel...")
    start_ui_in_background(app)

    app.run()

if __name__ == '__main__':
    main()
    while running:
        time.sleep(1)