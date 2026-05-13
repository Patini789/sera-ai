import asyncio
import os
import socket
import io
import time
import numpy as np
import pyaudio
import requests
import traceback
from pydub import AudioSegment
from ..core.logger import get_logger

logger = get_logger(__name__)

class TTS:
    def __init__(self, voice_api="8de335b3-4e1d-47d1-af39-9d9929b4841b", voice_model="es", voice_url="http://127.0.0.1:17493/generate", just_discord=False): 
        self.profile_id = voice_api     
        self.language = voice_model     
        self.api_url = "http://127.0.0.1:17493/generate"
        self.base_url = "http://127.0.0.1:17493"

        self.just_discord = just_discord
        
        # UDP Setup for Lip-Sync
        self.udp_ip = "127.0.0.1"
        self.udp_port = 4242
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    async def speak(self, text, app_instance=None, voice_instance=None):
        """Convert text to speech via local Kokoro API with playback and UDP lip-sync."""
        import re
        text = re.sub(r'[^\w\s.,!?:;\'"¡¿()[\]{}\-+$%€/=&@]', '', text)
        logger.info(f"🗣️ Kokoro API TTS: {text}")
        
        try:
            payload = {
                "profile_id": self.profile_id,
                "text": text,
                "language": self.language,
                "engine": "kokoro" 
            }

            response = requests.post(self.api_url, json=payload)
            response.raise_for_status() 

            content_type = response.headers.get('Content-Type', '')
            audio_bytes = None

            if 'application/json' in content_type:
                data = response.json()
                job_id = data.get("id")
                logger.debug(f"Generating audio. Ticket ID: {job_id}")
                
                audio_url = f"{self.base_url}/audio/{job_id}"
                
                for _ in range(50):
                    time.sleep(0.5)
                    audio_res = requests.get(audio_url)
                    
                    if audio_res.status_code == 200:
                        audio_bytes = audio_res.content
                        logger.debug("Audio generated successfully!")
                        break

                if not audio_bytes:
                    logger.error("Audio generation timed out.")
                    return
            else:
                audio_bytes = response.content

            audio_fp = io.BytesIO(audio_bytes)
            audio_seg = AudioSegment.from_file(audio_fp)
            
            if audio_seg.channels > 1:
                audio_seg = audio_seg.set_channels(1)

            discord_file = os.path.abspath("temp_tts_discord.wav")
            audio_seg.export(discord_file, format="wav")
            
            try:
                requests.post('http://127.0.0.1:3000/play', json={"filepath": discord_file})
            except Exception:
                pass 
            
            if app_instance and hasattr(app_instance, 'play_audio'): 
                app_instance.play_audio(discord_file)

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
            
            # UDP lip-sync loop
            for i in range(0, len(raw_data), chunk_size * 2):
                chunk = raw_data[i : i + chunk_size * 2]
                
                if not self.just_discord:
                    stream.write(chunk)
                else:
                    duracion_chunk = len(chunk) / (bytes_por_frame * frames_por_segundo)
                    await asyncio.sleep(duracion_chunk)
                
                # UDP amplitude data for lip-sync
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
            logger.error(f"Error in API TTS: {e}")
            logger.debug(f"API TTS Traceback: {traceback.format_exc()}")
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