import asyncio
import json
import websockets
from websockets.server import serve
import functools

HOST = "127.0.0.1"
PORT = 8081

active_godot_ws = None

async def game_handler(websocket, app_instance):
    global active_godot_ws
    
    try:
        async for message in websocket:
            data = json.loads(message)
            msg_type = data.get("type")

            # Route: MCP -> Godot
            if msg_type == "mcp_command":
                if active_godot_ws is not None and not active_godot_ws.closed:
                    try:
                        await active_godot_ws.send(json.dumps(data["payload"]))
                        await websocket.send(json.dumps({"status": "ok"}))
                    except Exception as e:
                        await websocket.send(json.dumps({"status": "error", "msg": f"Error: {e}"}))
                else:
                    await websocket.send(json.dumps({"status": "error", "msg": "Godot no conectado"}))
                continue

            # Route: Godot -> LLM
            active_godot_ws = websocket
            
            game_state = data.get("state", "")
            if game_state:
                app_instance.engine.current_game_state = json.dumps(game_state, indent=2)
            
            if msg_type == "input":
                user_prompt = data.get("prompt", "Analiza el estado...")
                print(f"🎮 [Godot Input]: {user_prompt}")
                
                # Forward the prompt to the AI as a new conversation turn
                loop = asyncio.get_running_loop()
                loop.run_in_executor(
                    None,
                    app_instance.handle_new_message_wrapper,
                    "SISTEMA_JUEGO", 
                    user_prompt
                )

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        if active_godot_ws == websocket:
            active_godot_ws = None
            print("🎮 [Game Bridge] Godot se ha desconectado.")

async def game_bridge_loop(app_instance):
    print(f"🎮 Game Bridge corriendo en ws://{HOST}:{PORT}...")
    bound_handler = functools.partial(game_handler, app_instance=app_instance)
    async with serve(bound_handler, HOST, PORT):
        await asyncio.get_running_loop().create_future()