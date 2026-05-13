# Python/packages/ui.py
import os
import json
import threading
import logging
from flask import Flask, request, jsonify, render_template, Response
from ..core.logger import get_logger

logger = get_logger(__name__)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Resolve paths relative to this file's location
_this_dir = os.path.dirname(os.path.abspath(__file__)) # .../Python/packages
_python_dir = os.path.dirname(_this_dir) # .../Python
_template_dir = os.path.join(_python_dir, 'templates') # .../Python/templates
_static_dir = _this_dir # .../Python/packages  (biblioteca.jpg lives here)

import queue

app_flask = Flask(
    __name__,
    template_folder=_template_dir,
    static_folder=_static_dir,
    static_url_path='/static'
)
bot_app = None  

# --- Global Web UI State ---
clients = []
chat_lock = threading.Lock()
ui_history = []
is_generating = False
current_generation = ""

def broadcast(event, data=None):
    """Sends an SSE event to all connected clients."""
    for client in clients.copy():
        q = client['q'] if isinstance(client, dict) else client
        try:
            q.put_nowait({"event": event, "data": data or {}})
        except queue.Full:
            pass

def broadcast_active_users():
    active_users = []
    seen = set()
    for obj in clients.copy():
        if isinstance(obj, dict):
            name = obj.get('name', 'Lector')
            if name and name not in seen:
                seen.add(name)
                active_users.append(name)
    broadcast('active_users', active_users)

# ─── Routes ───────────────────────────────────────────────────────────────────

@app_flask.route('/')
def index():
    return render_template('chat.html')


@app_flask.after_request
def add_no_cache(response):
    """Prevent browser from caching HTML/SSE so code changes take effect immediately."""
    if 'text/html' in response.content_type or 'text/event-stream' in response.content_type:
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
    return response


@app_flask.route('/api/state', methods=['GET'])
def get_state():
    return jsonify({
        "gemini_enable": bot_app.gemini_enable,
        "voice_active": bot_app.voice_active,
        "create_memories": bot_app.create_memories,
        "use_memories": bot_app.use_memories,
        "disable_thinking": bot_app.disable_thinking,
        "instructions": bot_app.instructions
    })


@app_flask.route('/api/state', methods=['POST'])
def update_state():
    data = request.json
    if "gemini_enable" in data: bot_app.gemini_enable = data["gemini_enable"]
    if "voice_active" in data: bot_app.voice_active = data["voice_active"]
    if "create_memories" in data: bot_app.create_memories = data["create_memories"]
    if "use_memories" in data: bot_app.use_memories = data["use_memories"]
    if "disable_thinking" in data: bot_app.disable_thinking = data["disable_thinking"]
    if "instructions" in data: bot_app.instructions = data["instructions"]
    return jsonify({"status": "success"})

@app_flask.route('/api/presence', methods=['POST'])
def update_presence():
    data = request.json
    client_id = data.get('id')
    name = data.get('name')
    if not client_id or not name:
        return jsonify({"status": "error"}), 400
    for obj in clients:
        if isinstance(obj, dict) and obj.get('id') == client_id:
            obj['name'] = name
    broadcast_active_users()
    return jsonify({"status": "success"})


@app_flask.route('/api/stream')
def stream():
    client_id = request.args.get('id', 'unknown')
    name = request.args.get('name', 'Lector')

    def generate():
        q = queue.Queue()
        client_obj = {'q': q, 'id': client_id, 'name': name}
        clients.append(client_obj)
        broadcast_active_users()
        
        # Determine if there is already a generation in progress
        with chat_lock:
            current_is_generating = is_generating
            current_history = list(ui_history)
            current_gen_text = current_generation

        # On connect, flush history and initial status
        try:
            yield f"event: history\ndata: {json.dumps({'history': current_history, 'is_generating': current_is_generating, 'current_gen': current_gen_text}, ensure_ascii=False)}\n\n"

            while True:
                msg = q.get()
                payload = json.dumps(msg["data"], ensure_ascii=False)
                yield f"event: {msg['event']}\ndata: {payload}\n\n"
        except GeneratorExit:
            if client_obj in clients:
                clients.remove(client_obj)
            elif q in clients:
                clients.remove(q)
            broadcast_active_users()

    return Response(generate(), mimetype='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app_flask.route('/api/chat', methods=['POST'])
def handle_chat_broadcast():
    global is_generating, ui_history

    data = request.json
    message = data.get("message", "").strip()
    author = data.get("author", "WebUser").strip()

    if not message or not bot_app:
        return jsonify({"status": "error"}), 400

    if message.lower() == "cls":
        with chat_lock:
            ui_history.clear()
            is_generating = False
        bot_app.reset_lists()
        broadcast("clear")
        return jsonify({"status": "success"})

    with chat_lock:
        if is_generating:
            return jsonify({"status": "rejected", "reason": "Sera is already generating"})
        is_generating = True
        
        user_msg_entry = {"type": "user", "author": author, "text": message}
        ui_history.append(user_msg_entry)
        
    broadcast("user_msg", user_msg_entry)
    broadcast("start_gen")

    def generator_thread():
        global is_generating, current_generation
        full_response = ""
        current_generation = ""
        try:
            for token in bot_app.handle_new_message_stream(author, message):
                full_response += token
                with chat_lock:
                    current_generation = full_response
                broadcast("token", {"token": token})
            
            with chat_lock:
                ui_history.append({"type": "bot", "text": full_response})
                is_generating = False
                current_generation = ""
                
            broadcast("end_gen")
        except Exception as e:
            with chat_lock:
                is_generating = False
            broadcast("end_gen")
            logger.error(f"Flask background generator error: {e}")

    threading.Thread(target=generator_thread, daemon=True).start()
    return jsonify({"status": "success"})


# ─── Background server ───────────────────────────────────────────────────────

def start_ui_in_background(app_instance):
    global bot_app
    bot_app = app_instance
    
    def run_server():
        app_flask.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    logger.info("🌐 Dashboard Web UI corriendo en: http://127.0.0.1:5000")