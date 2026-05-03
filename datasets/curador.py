import os
import json
import re
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Configuración de rutas
INPUT_FILE = "datasets/dataset_raw.jsonl"
OUTPUT_DIR = "datasets/curated"
OUTPUT_RAW = os.path.join(OUTPUT_DIR, "curated_raw.jsonl")
OUTPUT_CLEAN = os.path.join(OUTPUT_DIR, "curated_clean.jsonl")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progreso.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_current_index():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return int(f.read().strip() or 0)
    return 0

def save_current_index(index):
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(index))

def load_line(index, filepath=INPUT_FILE):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == index:
                    return json.loads(line)
        return None # Fin del archivo
    except FileNotFoundError:
        return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/get_next", methods=["GET"])
def get_next():
    current_idx = get_current_index()
    data = load_line(current_idx)
    
    if not data:
        return jsonify({"status": "completed", "message": "¡Felicidades! Has terminado el dataset."})

    messages = data.get("messages", [])
    if not messages:
        return jsonify({"status": "error", "message": "Formato de dataset inválido."})

    # 1. Extraer componentes del array 'messages'
    system_text = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    assistant_msg = messages[-1] if messages and messages[-1]["role"] == "assistant" else {}
    user_msg = messages[-2] if len(messages) >= 2 and messages[-2]["role"] == "user" else {}
    
    # El historial es todo lo que está entre el system y el último user
    history_msgs = messages[1:-2] if len(messages) > 2 else []
    history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history_msgs])

    raw_response = assistant_msg.get("content", "")
    
    # 2. Extraer el bloque think y el texto limpio
    think_match = re.search(r'(<think>.*?</think>)', raw_response, flags=re.DOTALL)
    think_block = think_match.group(1).strip() if think_match else ""
    clean_response = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()

    # Variables de edición (por si cargamos progreso anterior)
    edited_clean = clean_response
    edited_think = think_block
    edited_system = system_text
    edited_message = user_msg.get("content", "")
    
    # 3. Intentar cargar datos previamente curados si volvemos atrás
    curated_clean_data = load_line(current_idx, OUTPUT_CLEAN)
    if curated_clean_data and "messages" in curated_clean_data:
        c_msgs = curated_clean_data["messages"]
        if c_msgs:
            edited_clean = c_msgs[-1].get("content", edited_clean)
            edited_system = c_msgs[0].get("content", edited_system) if c_msgs[0]["role"] == "system" else edited_system
            edited_message = c_msgs[-2].get("content", edited_message) if len(c_msgs) >= 2 else edited_message

    curated_raw_data = load_line(current_idx, OUTPUT_RAW)
    if curated_raw_data and "messages" in curated_raw_data:
        r_msgs = curated_raw_data["messages"]
        if r_msgs:
            r_resp = r_msgs[-1].get("content", "")
            t_match = re.search(r'(<think>.*?</think>)', r_resp, flags=re.DOTALL)
            if t_match:
                edited_think = t_match.group(1).strip()

    # Enviamos a la UI los datos
    return jsonify({
        "status": "success",
        "index": current_idx,
        "system": edited_system,
        "history": history_text,
        "message": edited_message,
        "clean_response": edited_clean,
        "hidden_think_block": edited_think
    })

@app.route("/api/set_index", methods=["POST"])
def set_index():
    payload = request.json
    new_index = payload.get("index")
    if new_index is not None and new_index >= 0:
        save_current_index(new_index)
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 400

@app.route("/api/save", methods=["POST"])
def save_data():
    payload = request.json
    idx = payload.get("index", get_current_index())
    
    if payload.get("deleted"):
        raw_data = {}
        clean_data = {}
    else:
        edited_system = payload.get("system", "")
        edited_message = payload.get("message", "")
        edited_clean_response = payload.get("clean_response", "")
        hidden_think_block = payload.get("hidden_think_block", "")
        history_text = payload.get("history", "")
        
        # 1. Recuperar los tool_calls originales (La UI no los edita, solo los pasa)
        original_data = load_line(idx)
        original_tools = []
        if original_data and "messages" in original_data:
            original_tools = original_data["messages"][-1].get("tool_calls", [])

        # 2. Reconstruir el historial intermedio (muy básico, asumiendo formato "ROLE: texto")
        history_msgs = []
        if history_text:
            for line in history_text.split('\n'):
                if ':' in line:
                    role, content = line.split(':', 1)
                    history_msgs.append({"role": role.strip().lower(), "content": content.strip()})

        # 3. Base Messages compartida
        base_messages = [{"role": "system", "content": edited_system}]
        base_messages.extend(history_msgs)
        base_messages.append({"role": "user", "content": edited_message})

        # 4. DATASET RAW: Pensamiento + Respuesta
        raw_content = f"{hidden_think_block}\n\n{edited_clean_response}".strip() if hidden_think_block else edited_clean_response
        assistant_raw = {"role": "assistant", "content": raw_content}
        if original_tools:
            assistant_raw["tool_calls"] = original_tools
            
        raw_data = {"messages": base_messages + [assistant_raw]}

        # 5. DATASET CLEAN: Solo respuesta
        assistant_clean = {"role": "assistant", "content": edited_clean_response}
        if original_tools:
            assistant_clean["tool_calls"] = original_tools

        clean_data = {
            "messages": base_messages + [assistant_clean],
            "metadata": {
                "think_log": hidden_think_block
            }
        }
    
    def update_file(filepath, new_data, data_idx):
        lines = []
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
        
        new_line = json.dumps(new_data, ensure_ascii=False) + "\n"
        
        if data_idx < len(lines):
            lines[data_idx] = new_line
        else:
            while len(lines) < data_idx:
                lines.append("{}\n")
            lines.append(new_line)
            
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(lines)
            
    update_file(OUTPUT_RAW, raw_data, idx)
    update_file(OUTPUT_CLEAN, clean_data, idx)
    
    save_current_index(idx + 1)
    
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(debug=True, port=7000)