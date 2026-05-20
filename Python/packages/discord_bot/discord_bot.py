# Python\packages\discord_bot\discord_bot.py
"""Discord bot module for the Sera AI application."""

import asyncio
import re
import time
import os
import socket
import threading
import requests
import numpy as np
import discord
from discord.ext import commands
from Python.config.settings import settings
from typing import Callable, Any
from Python.core.logger import get_logger

logger = get_logger(__name__)


class UDPWhisperServer:
    """Receives Discord voice audio from Node.js via UDP and feeds it to Whisper STT."""
    def __init__(self, app_instance, discord_names_dict):
        self.app_instance = app_instance
        self.discord_names = {k.lower(): v for k, v in discord_names_dict.items()}
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('127.0.0.1', 5005))
        self.is_running = True
        self.user_buffers = {}
        self.user_last_time = {}
        
        threading.Thread(target=self._listen, daemon=True).start()
        threading.Thread(target=self._vad_monitor, daemon=True).start()

    def _listen(self):
        while self.is_running:
            try:
                data, _ = self.sock.recvfrom(12288)
                name_len = data[0]
                raw_name = data[1:1+name_len].decode('utf-8').lower()
                pcm_data = data[1+name_len:]
                
                if raw_name not in self.user_buffers:
                    self.user_buffers[raw_name] = bytearray()
                
                self.user_buffers[raw_name].extend(pcm_data)
                self.user_last_time[raw_name] = time.time()
            except: pass

    def _vad_monitor(self):
        while self.is_running:
            now = time.time()
            for username in list(self.user_last_time.keys()):
                if now - self.user_last_time[username] > 0.8 and len(self.user_buffers[username]) > 0:
                    audio_data = bytes(self.user_buffers[username])
                    self.user_buffers[username] = bytearray()

                    display_name = self.discord_names.get(username, username)
                    
                    # Process audio
                    raw_np = np.frombuffer(audio_data, np.int16)
                    if np.max(np.abs(raw_np)) > 500 and len(raw_np) > 16000 * 0.4:
                        audio_float32 = raw_np.astype(np.float32) / 32768.0
                        if self.app_instance and self.app_instance.stt:
                            threading.Thread(
                                target=self.app_instance.stt.process_discord_audio, 
                                args=(display_name, audio_float32),
                                daemon=True
                            ).start()
            time.sleep(0.1)

# ==========================================================
# EL BOT DE DISCORD PRINCIPAL
# ==========================================================
intents = discord.Intents.default()
intents.message_content = True

class DiscordBot:
    def __init__(self, shared_state: dict[str, Any], on_new_message_fn: Callable[[str, str], str], reset_lists: Callable[[], None] = None):

        self.shared_state = shared_state
        self.bot = commands.Bot(command_prefix='$', intents=intents)
        self.on_new_message_fn = on_new_message_fn
        self.reset_lists = reset_lists

        self.ids_to_notify = shared_state.get("notify_user_ids", [])
        self.user_ids = shared_state.get("user_ids", [])
        self.channels_to_notify = shared_state.get("notify_channel_ids", [])
        self.user_context_path = shared_state.get("user_context_path", None)
        self.discord_names = shared_state.get("discord_names", {})
        self.discord_help = shared_state.get("discord_help", "")
        self._app_instance = shared_state.get("app_instance", None)

        # Start UDP listener for voice audio from Node.js
        self.udp_server = UDPWhisperServer(self._app_instance, self.discord_names)

        if self._app_instance:
            self._app_instance.register_audio_player(self.play_discord_audio)

        @self.bot.event
        async def on_ready():
            logger.info(f'Bot connected as {self.bot.user}')
            for user_id in self.ids_to_notify:
                try:
                    user = await self.bot.fetch_user(user_id)
                    if user:
                        #await user.send("Bot Active.")
                        pass
                except Exception:
                    pass

        @self.bot.command(name='join')
        async def join(ctx):
            if not ctx.author.voice:
                return await ctx.send("You must be in a voice channel first!")
            
            channel = ctx.author.voice.channel
            msg = await ctx.send("⏳ Connecting to voice via Node.js...")
            
            try:
                requests.post('http://127.0.0.1:3000/join', json={
                    "channelId": str(channel.id),
                    "guildId": str(ctx.guild.id)
                })
                await msg.edit(content=f"✅ Connected to **{channel.name}** via Node.js microservice.")
            except Exception as e:
                await msg.edit(content="❌ Error: Node.js microservice is not running. Start it with `node index.js` in the `voice_service` folder.")

        @self.bot.command(name='leave')
        async def leave(ctx):
            try:
                requests.post('http://127.0.0.1:3000/leave')
                await ctx.send("👋 Disconnected from voice channel.")
            except:
                await ctx.send("Could not contact Node.js to disconnect.")
        @self.bot.command(name='chat')
        async def chat(ctx, *, content):
            async with ctx.typing():
                author_name = str(ctx.author.name)
                author = self.discord_names.get(author_name, author_name)
                
                app = self._app_instance
                if app and hasattr(app, 'handle_new_message_stream'):
                    await self._stream_to_discord(ctx.message, app, author, content)
                else:
                    answer = await asyncio.to_thread(self.on_new_message_fn, author, content)
                    clean = re.sub(r'<think>.*?</think>|<lmstudio_tools>.*?</lmstudio_tools>', '', answer, flags=re.DOTALL).strip()
                    await ctx.send(clean[:2000])

        @self.bot.command(name='cls')
        async def cls(ctx):
            if self.reset_lists:
                self.reset_lists()
                await ctx.send("Memory cleaned.")

        @self.bot.event
        async def on_message(message):
            if message.author == self.bot.user:
                return
            await self.bot.process_commands(message)

    def set_app_instance(self, app):
        self._app_instance = app
        self._app_instance.register_audio_player(self.play_discord_audio)
        if hasattr(self, 'udp_server'):
            self.udp_server.app_instance = app
    
    def play_discord_audio(self, file_path):
        """Send playback command to Node.js voice microservice."""
        try:
            abs_path = os.path.abspath(file_path)
            requests.post('http://127.0.0.1:3000/play', json={"filepath": abs_path})
        except Exception as e:
            logger.error(f"Error playing audio in Discord (Node.js not responding): {e}")

    async def _stream_to_discord(self, message, app, author, content):
        EDIT_INTERVAL = 1.5
        MAX_DISPLAY = 1950
        active_bot_msg = await message.channel.send("⏳ Generando...")
        accumulated = ""
        last_edit_time = time.monotonic()
        stream_done = False

        def collect_stream():
            nonlocal accumulated, stream_done
            try:
                for delta in app.handle_new_message_stream(author, content):
                    accumulated += delta
            except Exception as e:
                accumulated += f"\n[Error: {e}]"
            finally:
                stream_done = True

        thread = asyncio.get_event_loop().run_in_executor(None, collect_stream)
        prev_snapshot = ""
        current_msg_index = 0

        while not stream_done:
            await asyncio.sleep(0.3)
            now = time.monotonic()
            if now - last_edit_time < EDIT_INTERVAL: continue
            
            snapshot = accumulated
            if snapshot == prev_snapshot: continue

            in_think = "<think>" in snapshot and "</think>" not in snapshot
            display_text = re.sub(r'<think>.*?</think>|<lmstudio_tools>.*?</lmstudio_tools>', '', snapshot, flags=re.DOTALL).strip()
            
            if in_think:
                display_text = re.sub(r'<think>[\s\S]*$', '', snapshot).strip()
                display_text = "💭 *Pensando...*\n" + display_text if display_text else "💭 *Pensando...*"
            if not display_text:
                display_text = "💭 *Pensando...*"

            parts = [display_text[i:i + MAX_DISPLAY] for i in range(0, len(display_text), MAX_DISPLAY)]
            
            if len(parts) - 1 > current_msg_index:
                try: await active_bot_msg.edit(content=parts[current_msg_index])
                except discord.HTTPException: pass
                
                current_msg_index = len(parts) - 1
                active_bot_msg = await message.channel.send(parts[current_msg_index] + " ▌")
                last_edit_time = time.monotonic()
                prev_snapshot = snapshot
                continue

            display = parts[current_msg_index] + " ▌"
            try:
                await active_bot_msg.edit(content=display)
                last_edit_time = time.monotonic()
                EDIT_INTERVAL = 1.5 
            except discord.RateLimited as e:
                await asyncio.sleep(e.retry_after)
            except discord.HTTPException:
                EDIT_INTERVAL = min(EDIT_INTERVAL + 1.0, 4.0)

            prev_snapshot = snapshot

        await thread
        clean = re.sub(r'<think>.*?</think>|<lmstudio_tools>.*?</lmstudio_tools>', '', accumulated, flags=re.DOTALL).strip()
        if not clean: clean = "*(respuesta vacía)*"
        final_parts = [clean[i:i + MAX_DISPLAY] for i in range(0, len(clean), MAX_DISPLAY)]

        try:
            final_content = final_parts[current_msg_index] if len(final_parts) > current_msg_index else final_parts[-1]
            await active_bot_msg.edit(content=final_content)
        except discord.HTTPException:
            await asyncio.sleep(1)
            await active_bot_msg.edit(content=final_content)

        for i in range(current_msg_index + 1, len(final_parts)):
            await message.channel.send(final_parts[i])

    def start(self):
        self.bot.run(settings.discord_token)