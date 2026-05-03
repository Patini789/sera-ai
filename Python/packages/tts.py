import os
import edge_tts
import socket
import io
import time
import asyncio
import numpy as np
import pyaudio
import requests
from pydub import AudioSegment

class TTS:
    def __init__(self, rate="+10%", pitch="-10Hz", voice="es-HN-KarlaNeural", just_discord=False): 
        self.rate = rate
        self.pitch = pitch
        self.voice = voice
        self.just_discord = just_discord
                
        # UDP Setup
        self.udp_ip = "127.0.0.1"
        self.udp_port = 4242
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    async def speak(self, text, app_instance=None, voice_instance=None):
        """Convert text to speech via Edge TTS with local playback and UDP lip-sync."""
        print(f"🗣️ EdgeTTS: {text}")
        
        try:
            communicate = edge_tts.Communicate(
                text,
                voice=self.voice,
                rate=self.rate,
                pitch=self.pitch
            )
            
            mp3_fp = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_fp.write(chunk["data"])
            
            mp3_fp.seek(0)
            audio_seg = AudioSegment.from_file(mp3_fp, format="mp3")
            
            if audio_seg.channels > 1:
                audio_seg = audio_seg.set_channels(1)

            discord_file = os.path.abspath("temp_tts_discord.wav")
            audio_seg.export(discord_file, format="wav")
            
            # Forward audio to Discord voice via Node.js microservice
            try:
                requests.post('http://127.0.0.1:3000/play', json={"filepath": discord_file})
            except Exception:
                pass
            
            if app_instance and hasattr(app_instance, 'play_audio'):
                app_instance.play_audio(discord_file)

            # Playback with ghost-clock timing for Discord-only mode
            raw_data = audio_seg.raw_data
            chunk_size = 1024
            
            p = None
            stream = None
            
            if not self.just_discord:
                p = pyaudio.PyAudio()
                stream = p.open(format=p.get_format_from_width(audio_seg.sample_width),
                                channels=audio_seg.channels,
                                rate=audio_seg.frame_rate,
                                output=True)

            bytes_por_frame = audio_seg.frame_width
            frames_por_segundo = audio_seg.frame_rate

            for i in range(0, len(raw_data), chunk_size * 2):
                chunk = raw_data[i : i + chunk_size * 2]
                
                if not self.just_discord:
                    stream.write(chunk)
                else:
                    duracion_chunk = len(chunk) / (bytes_por_frame * frames_por_segundo)
                    await asyncio.sleep(duracion_chunk) 
                
                # --- UDP LIP SYNC LOGIC ---
                if len(chunk) > 0:
                    try:
                        data_int16 = np.frombuffer(chunk, dtype=np.int16)
                        data_float = data_int16.astype(np.float32)
                        
                        DIVISOR = 3500.0
                        EXPONENT = 1.5
                        
                        rms = np.sqrt(np.mean(data_float**2))
                        if np.isnan(rms): rms = 0.0
                        
                        normalized = min(rms / DIVISOR, 1.0)
                        mouth_open = normalized ** EXPONENT
                        
                        self.sock.sendto(str(mouth_open).encode(), (self.udp_ip, self.udp_port))
                    except Exception as e:
                        pass

            # Cleanup
            if not self.just_discord:
                stream.stop_stream()
                stream.close()
                p.terminate()

        except Exception as e:
            print(f"⚠️ Error in EdgeTTS Speak: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.sock.sendto(b"0.0", (self.udp_ip, self.udp_port))
            if voice_instance:
                voice_instance.unlock()

    def start(self):
        pass

def make_chunks(audio_segment, chunk_length):
    number_of_chunks = len(audio_segment) // chunk_length
    chunks = [audio_segment[i * chunk_length:(i + 1) * chunk_length]
              for i in range(number_of_chunks)]
    return chunks