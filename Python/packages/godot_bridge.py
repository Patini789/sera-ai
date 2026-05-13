import asyncio
import json
import psutil
import win32gui
import websockets
from websockets.server import serve
from ..core.logger import get_logger

logger = get_logger(__name__)


HOST = "127.0.0.1"
PORT = 8080
POLL_RATE = 0.5

active_connection = None

ultima_imagen_b64 = None

def get_active_window_title():
    try:
        window = win32gui.GetForegroundWindow()
        return win32gui.GetWindowText(window)
    except Exception:
        return ""

def classify_app(title):
    title = title.lower()
    if "music" in title and "youtube" in title: return "YouTube Music"
    if "youtube" in title: return "YouTube"
    if "vlc" in title: return "Video Player"
    if "visual studio" in title or "godot" in title or ".py" in title: return "Code"
    if "game" in title or "juego" in title: return "Juego"
    if "brave" in title: return "Navegando"
    return "Other"

async def enviar_comando(comando_dict):
    """
    Public function for other parts of the app (like the LLM) to send messages to Godot.
    """
    global active_connection
    if active_connection:
        try:
            await active_connection.send(json.dumps(comando_dict))
        except Exception as e:
            logger.error(f"Error sending command to Godot: {e}")

async def send_telemetry_loop(websocket):
    """IA telemetry loop — sends system metrics to Godot."""
    try:
        while True:
            cpu_usage = psutil.cpu_percent(interval=None)
            ram_usage = psutil.virtual_memory().percent
            window_title = get_active_window_title()
            
            payload = {
                "comando": "sensor_update",
                "cpu": cpu_usage,
                "ram": ram_usage,
                "app": classify_app(window_title),
                "raw_title": window_title
            }
            await websocket.send(json.dumps(payload))
            await asyncio.sleep(POLL_RATE)
    except websockets.ConnectionClosed:
        pass # Se maneja en el handler principal

async def receive_from_godot_loop(websocket):
    """Receive messages from Godot and react accordingly."""
    try:
        async for message in websocket:
            data = json.loads(message)
            evento = data.get("evento")
            
            if evento == "usuario_me_hizo_click":
                pass
                # future use
                
            elif evento == "me_estan_acariciando":
                pass
                # future use
                
            elif evento == "vision_update":
                global ultima_imagen_b64
                ultima_imagen_b64 = data.get("image_base64")
                logger.debug("Image received from Godot.")
                
    except websockets.ConnectionClosed:
        pass
async def pedir_vision_a_godot():
    """Ask Godot for a vision update."""
    global ultima_imagen_b64
    ultima_imagen_b64 = None
    
    await enviar_comando({"comando": "pedir_vision"})
    logger.info("Asking Godot for a vision update...")
    
    tiempo_espera = 0
    while ultima_imagen_b64 is None and tiempo_espera < 5.0:
        await asyncio.sleep(0.1)
        tiempo_espera += 0.1
        
    if ultima_imagen_b64:
        return ultima_imagen_b64
    else:
        logger.warning("Godot failed to provide vision.")
        return None
    
async def godot_handler(websocket):
    """Main WebSocket handler."""
    global active_connection
    active_connection = websocket
    logger.info(f"Godot connected from {websocket.remote_address}!")
    
    tarea_enviar = asyncio.create_task(send_telemetry_loop(websocket))
    tarea_escuchar = asyncio.create_task(receive_from_godot_loop(websocket))
    
    done, pending = await asyncio.wait(
        [tarea_enviar, tarea_escuchar], 
        return_when=asyncio.FIRST_COMPLETED
    )
    
    for task in pending:
        task.cancel()
        
    active_connection = None
    logger.info("Godot lost connection. Waiting for new connection...")

async def godot_bridge_loop():
    logger.info(f"Bridge running on ws://{HOST}:{PORT}...")
    async with serve(godot_handler, HOST, PORT):
        await asyncio.get_running_loop().create_future()

if __name__ == "__main__":
    try:
        asyncio.run(godot_bridge_loop())
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")