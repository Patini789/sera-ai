"""Core modules for the Sera AI application.

This package contains the decomposed business logic previously
housed in a monolithic ``app.py``:

* :class:`ConversationEngine` — conversation state and LLM interaction.
* :class:`MemoryHandler` — semantic memory retrieval and storage.
* :func:`save_training_data` — dataset persistence for fine-tuning.
"""

from .conversation_engine import ConversationEngine
from .memory_handler import MemoryHandler
from .training_data import save_training_data

__all__ = ["ConversationEngine", "MemoryHandler", "save_training_data"]
