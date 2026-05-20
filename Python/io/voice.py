import threading
import time
import numpy as np
import keyboard  # <-- Nueva librería
from faster_whisper import WhisperModel
from ..core.logger import get_logger

logger = get_logger(__name__)

class Voice:
    # Cambiamos el default a "medium". 
    # TIP: Con tu RTX 5060 Ti, puedes poner model_size="large-v3" para una transcripción perfecta en español.
    def __init__(self, on_speech_detected_fn, model_size="large-v3", language="es", stt_gaming_mode=False):
        self.on_speech_detected_fn = on_speech_detected_fn
        self.language = language
        
        # Estado de bloqueo original
        self.is_locked = False
        
        # NUEVO: Bandera para retener mensajes si el usuario habló mientras Sera hablaba
        self.waiting_for_new_speech = False

        # --- Gaming Mode (Push-to-Talk) ---
        self.gaming_mode = stt_gaming_mode
        self.ptt_hotkey = "ctrl+alt+v" 
        self.ptt_active = False
        self.force_send = False
        
        # Tiempo de gracia (en segundos) tras soltar la tecla.
        # Evita que se corte la frase por micro-pausas y da tiempo 
        # a que lleguen los últimos lotes de audio de Discord.
        self.ptt_grace_period = 1.8 
        # --------------------------------------------------------

        self.conversation_buffer = []
        self.last_speech_time = time.time()
        self.buffer_lock = threading.Lock()

        try:
            logger.info(f"Loading Whisper {model_size} (CUDA)...")
            self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
        except:
            logger.info(f"Loading Whisper {model_size} (CPU)...")
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

        # Iniciar hilos
        threading.Thread(target=self._aggregator_loop, daemon=True).start()
        threading.Thread(target=self._ptt_monitor_loop, daemon=True).start()

    def set_gaming_mode(self, enabled: bool, hotkey: str = "ctrl+alt+v"):
        """Activa o desactiva el modo gaming y configura el shortcut."""
        self.gaming_mode = enabled
        self.ptt_hotkey = hotkey
        self.force_send = False
        logger.info(f"Modo Gaming: {'ACTIVADO (Atajo: ' + hotkey + ')' if enabled else 'DESACTIVADO'}")

    def _ptt_monitor_loop(self):
        """Monitorea el teclado con un 'Grace Period' (tiempo de gracia) para no cortar lotes."""
        was_pressed = False
        release_time = 0

        while True:
            time.sleep(0.05) 
            if not self.gaming_mode:
                continue
            
            try:
                is_pressed = keyboard.is_pressed(self.ptt_hotkey)
            except Exception:
                is_pressed = False

            if is_pressed:
                if not self.ptt_active:
                    self.ptt_active = True
                    logger.info("PTT: Escuchando...")
                
                was_pressed = True
                release_time = 0  # Si vuelve a presionar, cancelamos la cuenta regresiva de cierre
            else:
                if was_pressed:
                    # Se acaba de soltar la tecla físicamente
                    was_pressed = False
                    release_time = time.time()
                
                # Si está suelta pero ptt_active sigue activo, validamos el tiempo de gracia
                if self.ptt_active and release_time > 0:
                    if (time.time() - release_time) > self.ptt_grace_period:
                        self.ptt_active = False
                        self.force_send = True 
                        release_time = 0
                        logger.info("PTT: Tecla soltada definitivamente (procesando lotes finales)...")

    def lock(self):
        """Lock buffer while Sera is speaking (TTS synchronization)."""
        logger.info("Buffer locked (Sera is speaking).")
        with self.buffer_lock:
            self.is_locked = True

    def unlock(self):
        """Unlock buffer when Sera finishes speaking."""
        logger.info("Buffer unlocked (Sera finished speaking).")
        with self.buffer_lock:
            # Si el usuario habló y se guardó en el buffer MIENTRAS Sera hablaba, 
            # no enviamos inmediatamente. Esperamos a que vuelva a hablar.
            if len(self.conversation_buffer) > 0:
                self.waiting_for_new_speech = True
                logger.info("Mensaje cacheado durante el lock. Esperando a que el usuario hable de nuevo...")
            
            self.is_locked = False

    def process_discord_audio(self, user_name, audio_np):
        """Receive Discord audio, transcribe with Whisper, and buffer results."""
        
        if self.gaming_mode and not self.ptt_active:
            return 
            
        # Reducido de 0.5 a 0.3 segundos. Si descartamos medio segundo de audio, 
        # nos podemos comer sílabas iniciales o monosílabos ("sí", "no").
        if len(audio_np) < 16000 * 0.3: 
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
                
                # Si el lock está abierto, significa que el usuario habló después
                # de que Sera terminara. Cancelamos la espera forzada.
                if not self.is_locked:
                    self.waiting_for_new_speech = False

    def _aggregator_loop(self):
        """Monitor for silence gap OR Push-to-Talk release, then send to AI."""
        while True:
            time.sleep(0.1) 
            
            with self.buffer_lock:
                if not self.conversation_buffer:
                    continue
                
                if self.gaming_mode:
                    if not self.force_send:
                        continue
                else:
                    # En modo normal, validamos:
                    # 1. Silencio > 2.0s
                    # 2. Que no esté bloqueado por voz de Sera.
                    # 3. Que NO estemos en estado de retención.
                    if time.time() - self.last_speech_time <= 2.0 or self.is_locked or self.waiting_for_new_speech:
                        continue
                
                # --- Preparar envío ---
                if self.gaming_mode:
                    self.force_send = False # Resetear la bandera
                
                users_in_buffer = []
                clean_messages = []
                
                for item in self.conversation_buffer:
                    parts = item.split(": ", 1) 
                    if len(parts) == 2:
                        users_in_buffer.append(parts[0])
                        clean_messages.append(parts[1])
                
                unique_users = list(dict.fromkeys(users_in_buffer))

                if len(unique_users) == 1:
                    author_to_send = unique_users[0]
                    text_to_send = " ".join(clean_messages)
                else:
                    author_to_send = "" 
                    text_to_send = "\n".join(self.conversation_buffer)
                
                # Clear buffer and lock
                self.conversation_buffer = []
                
                if not self.gaming_mode:
                    self.is_locked = True # Aplicar el bloqueo directamente dentro de buffer_lock
                    logger.info("Buffer locked (Sera is speaking).")

                # Send to AI
                threading.Thread(
                    target=self.on_speech_detected_fn, 
                    args=(author_to_send, text_to_send) 
                ).start()

    def start(self): pass
    def stop(self): pass