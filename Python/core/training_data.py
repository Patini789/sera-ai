"""Training data collection for fine-tuning datasets.

Saves each LLM interaction in two complementary JSONL files:

* ``dataset_raw.jsonl`` — preserves ``<think>`` blocks and tool
  calls for full reproducibility.
* ``dataset_clean.jsonl`` — user-facing response with separated
  ``tools_used`` and ``think_log`` fields for easier curation.

.. note::

   Data collection is **temporarily disabled** because the streaming
   mode can produce malformed ``<think>`` tags.  Remove the early
   ``return`` in :func:`save_training_data` once the format is
   confirmed stable.
"""

import json
import os
import re
from .logger import get_logger

logger = get_logger(__name__)


def save_training_data(
        system_text: str,
        conversation_history: list[dict],
        user_message: str,
        raw_response: str,
    ) -> None:
    """Persist a single interaction to the raw and clean dataset files.

    Args:
        system_text: The system prompt used for this interaction.
        conversation_history: The full conversation history list
            (each entry has ``role`` and ``content`` keys).
        user_message: The formatted user message string
            (e.g. ``"Author: message"``).
        raw_response: The raw model response, potentially including
            ``<think>`` and ``<lmstudio_tools>`` tags.
    """
    # ── Everything below is unreachable until the guard above is removed ──

    os.makedirs("datasets", exist_ok=True)

    # 1. Extract <think> blocks
    think_match = re.search(r"<think>(.*?)</think>", raw_response, flags=re.DOTALL)
    think_content = think_match.group(1).strip() if think_match else ""

    # 2. Extract tool calls
    extracted_tools: list = []
    tools_match = re.search(
        r"<lmstudio_tools>(.*?)</lmstudio_tools>", raw_response, flags=re.DOTALL
    )
    if tools_match:
        try:
            extracted_tools = json.loads(tools_match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Build clean and raw response variants
    clean_text = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL)
    clean_text = re.sub(
        r"<lmstudio_tools>.*?</lmstudio_tools>", "", clean_text, flags=re.DOTALL
    ).strip()

    raw_text = re.sub(
        r"<lmstudio_tools>.*?</lmstudio_tools>", "", raw_response, flags=re.DOTALL
    ).strip()

    # 4. Reconstruct history in standard format (OpenAI / ChatML)
    # Note: skip the last message in conversation_history because
    # ConversationEngine already appended the clean assistant response.
    # We add our own structured version instead.
    base_messages = [{"role": "system", "content": system_text}]
    
    # Assume the second-to-last message is the current user_message
    # and the last is the assistant response. Take up to the user message.
    for msg in conversation_history[:-1]:
        base_messages.append({"role": msg["role"], "content": msg["content"]})

    # --- RAW DATASET (ideal for training with reasoning + tools) ---
    assistant_raw_msg = {
        "role": "assistant",
        "content": raw_text  # Mantiene las etiquetas <think> intactas
    }
    if extracted_tools:
        assistant_raw_msg["tool_calls"] = extracted_tools

    raw_data = {
        "messages": base_messages + [assistant_raw_msg]
    }

    # --- CLEAN DATASET (ideal for curation or training without reasoning) ---
    assistant_clean_msg = {
        "role": "assistant",
        "content": clean_text
    }
    if extracted_tools:
        assistant_clean_msg["tool_calls"] = extracted_tools

    clean_data = {
        "messages": base_messages + [assistant_clean_msg],
        "metadata": {
            "think_log": think_content  # Lo guardamos fuera del content por si quieres leerlo
        }
    }

    # 5. Write to JSONL files
    try:
        with open("datasets/dataset_raw.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(raw_data, ensure_ascii=False) + "\n")

        with open("datasets/dataset_clean.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(clean_data, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"Error saving training data: {e}")
