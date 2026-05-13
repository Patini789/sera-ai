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
from Python.core.logger import setup_logging, get_logger

# Initialize logging
setup_logging(level=getattr(settings, 'log_level', 'INFO'))
logger = get_logger(__name__)

running = True

def main():
    logger.debug("Building dependencies...")

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

        import re
        import asyncio as _asyncio

        def discord_notification_hook(message: str):
            """Send scheduler output as DMs to configured notify users."""
            async def _send():
                clean_message = re.sub(r'<think>.*?</think>|<lmstudio_tools>.*?</lmstudio_tools>', '', message, flags=re.DOTALL).strip()
                if not clean_message:
                    return

                for user_id in settings.notify_user_ids:
                    try:
                        user = await discord_bot.bot.fetch_user(user_id)
                        if user:
                            # Split long messages to respect Discord's 2000 char limit
                            for i in range(0, len(clean_message), 2000):
                                await user.send(clean_message[i:i + 2000])
                    except Exception as e:
                        logger.warning(f"Failed to send scheduler DM to {user_id}: {e}")

            if discord_bot.bot.loop and discord_bot.bot.loop.is_running():
                _asyncio.run_coroutine_threadsafe(_send(), discord_bot.bot.loop)

        app.register_notification_hook(discord_notification_hook)

    # ── Scheduler Service ─────────────────────────────────────────
    if settings.scheduler_enabled:
        from Python.packages.scheduler_service import SchedulerService
        scheduler = SchedulerService(
            app=app,
            cron_hour=settings.scheduler_cron_hour,
            cron_minute=settings.scheduler_cron_minute,
            prompt_template=settings.scheduler_prompt_template,
            proximity_check_interval=settings.scheduler_proximity_check_interval,
            proximity_window=settings.scheduler_proximity_window,
            proximity_prompt=settings.scheduler_proximity_prompt,
        )
        app.add_service(scheduler)
        logger.info("Scheduler service registered")

    logger.debug("Starting Web UI Control Panel...")
    start_ui_in_background(app)

    app.run()

if __name__ == '__main__':
    main()
    while running:
        time.sleep(1)