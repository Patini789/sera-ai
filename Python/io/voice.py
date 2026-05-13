
import threading
import time
import numpy as np
from faster_whisper import WhisperModel
from ..core.logger import get_logger

logger = get_logger(__name__)

class Voice:
    def __init__(self, on_speech_detected_fn, model_size="small", language="es"):
        self.on_speech_detected_fn = on_speech_detected_fn
        self.language = language
        
        # Lock state: False = Open (Listen and Respond), True = Closed (Listen only and save)
        self.is_locked = False

        self.conversation_buffer = []
        self.last_speech_time = time.time()
        self.buffer_lock = threading.Lock()

        try:
            logger.info("Loading Whisper (CUDA)...")
            self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
        except:
            logger.info("Loading Whisper (CPU)...")
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

        threading.Thread(target=self._aggregator_loop, daemon=True).start()

    def lock(self):
        """Lock buffer while Sera is speaking (TTS synchronization)."""
        logger.info("Buffer locked (Sera is speaking).")
        self.is_locked = True

    def unlock(self):
        """Unlock buffer when Sera finishes speaking."""
        logger.info("Buffer unlocked (Sera finished speaking).")
        self.is_locked = False

    def process_discord_audio(self, user_name, audio_np):
        """Receive Discord audio, transcribe with Whisper, and buffer results."""
        if len(audio_np) < 16000 * 0.5: 
            return
        
        logger.debug(f"Transcribing {user_name}...")
        
        segments, _ = self.model.transcribe(
            audio_np, 
            language=self.language,
            beam_size=5,  
            initial_prompt="Technical conversation with AI assistant Sera.",
            vad_filter=True,
            condition_on_previous_text=True
        )
        text = " ".join([s.text for s in segments]).strip()
        
        if text:
            logger.info(f"[{user_name}]: {text}")
            
            with self.buffer_lock:
                self.conversation_buffer.append(f"{user_name}: {text}")
                self.last_speech_time = time.time()

    def _aggregator_loop(self):
        """Monitor for 2s silence gap, then send buffered text to the AI."""
        while True:
            time.sleep(0.5)
            
            with self.buffer_lock:
                if not self.conversation_buffer:
                    continue
                
                if time.time() - self.last_speech_time > 2.0 and not self.is_locked:
                    
                    # Separate authors from their messages
                    users_in_buffer = []
                    clean_messages = []
                    
                    for item in self.conversation_buffer:
                        # Split "Name: Message" strings
                        parts = item.split(": ", 1) 
                        if len(parts) == 2:
                            users_in_buffer.append(parts[0])
                            clean_messages.append(parts[1])
                    
                    # Get unique users preserving order
                    unique_users = list(dict.fromkeys(users_in_buffer))

                    # Format output based on speaker count
                    if len(unique_users) == 1:
                        # Single speaker: author is that person, text is clean
                        author_to_send = unique_users[0]
                        text_to_send = " ".join(clean_messages)
                    else:
                        # Multiple speakers: send raw transcript
                        author_to_send = "" 
                        text_to_send = "\n".join(self.conversation_buffer)
                    
                    # Clear buffer and lock
                    self.conversation_buffer = []
                    self.lock()

                    # Send to AI
                    threading.Thread(
                        target=self.on_speech_detected_fn, 
                        args=(author_to_send, text_to_send) 
                    ).start()
  

    def start(self): pass
    def stop(self): pass