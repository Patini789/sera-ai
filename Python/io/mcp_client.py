"""
MCP Client — Launches MCP servers as subprocesses and communicates
via the Model Context Protocol (JSON-RPC over stdin/stdout).

Provides:
* Automatic subprocess lifecycle management.
* Tool discovery via ``tools/list``.
* Tool execution via ``tools/call``.
* Conversion of MCP tool schemas to Gemini ``functionDeclarations``.

All connections are non-blocking: servers are initialized in
background threads so the main application never freezes. Read
operations use a dedicated reader thread with configurable timeouts
to avoid hanging on unresponsive servers.
"""

import json
import os
import subprocess
import threading
import queue
import sys
from pathlib import Path
from typing import Any
from ..core.logger import get_logger

logger = get_logger(__name__)


def _load_mcp_servers() -> dict[str, dict]:
    """Load MCP server definitions from ``Python/env/mcp_servers.json``.

    Falls back to an empty dict if the file does not exist, allowing
    SeraAI to run without any MCP tools configured.
    """
    config_path = Path(__file__).resolve().parent.parent / "env" / "mcp_servers.json"
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load MCP config ({config_path}): {e}")
        return {}


MCP_SERVERS: dict[str, dict] = _load_mcp_servers()

# Timeout (seconds) for MCP init/handshake.
_INIT_TIMEOUT = 15
# Timeout for regular tool-list / tool-call operations.
_CALL_TIMEOUT = 60


class MCPServerConnection:
    """Manages a single MCP server subprocess and JSON-RPC communication."""

    def __init__(self, name: str, command: str, args: list[str],
                 cwd: str | None = None, timeout: int = 300):
        self.name = name
        self.command = command
        self.args = args
        self.cwd = cwd  # Working directory for the subprocess
        self.timeout = timeout
        self.process: subprocess.Popen | None = None
        self._request_id = 0
        self._lock = threading.Lock()
        self.tools: list[dict] = []
        self.ready = False  # True after successful init + tool list

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def start(self) -> bool:
        """Launch the MCP server subprocess."""
        try:
            # Resolve cwd: relative paths are resolved against the project root
            # (SeraAI/), not os.getcwd() which may differ at runtime.
            project_root = Path(__file__).resolve().parent.parent.parent
            cwd = self.cwd
            if cwd:
                cwd_path = Path(cwd)
                if not cwd_path.is_absolute():
                    cwd_path = (project_root / cwd_path).resolve()
                cwd = str(cwd_path)
            elif self.args:
                cwd = os.path.dirname(os.path.abspath(self.args[0]))

            # Resolve command: if it's a relative path (e.g. ".env/Scripts/python.exe"),
            # resolve it against the resolved cwd.
            command = self.command
            if cwd and not Path(command).is_absolute() and os.sep in command or "/" in command:
                candidate = Path(cwd) / command
                if candidate.exists():
                    command = str(candidate.resolve())

            self.process = subprocess.Popen(
                [command] + self.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                bufsize=0,
            )
            logger.debug(f"🔌 MCP [{self.name}] subprocess started (PID: {self.process.pid}, cwd: {cwd})")
            return True
        except Exception as e:
            logger.error(f"❌ MCP [{self.name}] failed to start: {e}")
            return False

    def stop(self):
        """Terminate the MCP server subprocess."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            logger.info(f"🔌 MCP [{self.name}] subprocess stopped.")

    # ── Low-level I/O with timeouts ────────────────────────────────
    #
    # FastMCP's stdio transport uses **newline-delimited JSON**:
    #   • Send: one JSON object per line, terminated by \n
    #   • Receive: one JSON object per line on stdout
    # NO Content-Length headers.

    def _send_jsonrpc(
        self, method: str, params: dict | None = None, timeout: float = _CALL_TIMEOUT
    ) -> dict | None:
        """Send a JSON-RPC request and read the matching response."""
        if not self.process or self.process.poll() is not None:
            logger.warning(f"⚠️ MCP [{self.name}] process not running, restarting...")
            if not self.start():
                return None

        with self._lock:
            request_id = self._next_id()
            request = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
            }
            if params is not None:
                request["params"] = params

            # Newline-delimited JSON: one object + \n
            line = json.dumps(request).encode("utf-8") + b"\n"

            try:
                self.process.stdin.write(line)
                self.process.stdin.flush()

                # Read response with timeout
                response = self._read_response_with_timeout(request_id, timeout)
                return response
            except Exception as e:
                logger.error(f"❌ MCP [{self.name}] JSON-RPC error ({method}): {e}")
                return None

    def _send_notification(self, method: str, params: dict | None = None) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        if not self.process or self.process.poll() is not None:
            return
        notif: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            notif["params"] = params
        line = json.dumps(notif).encode("utf-8") + b"\n"
        try:
            self.process.stdin.write(line)
            self.process.stdin.flush()
        except Exception:
            pass

    def _read_response_with_timeout(
        self, request_id: int, timeout: float
    ) -> dict | None:
        """Read JSON lines from stdout until we get the response matching request_id."""
        result_queue: queue.Queue[dict | None] = queue.Queue()

        def _reader():
            try:
                result_queue.put(self._read_until_response(request_id))
            except Exception as e:
                logger.error(f"❌ MCP [{self.name}] reader thread error: {e}")
                result_queue.put(None)

        t = threading.Thread(target=_reader, daemon=True)
        t.start()

        try:
            return result_queue.get(timeout=timeout)
        except queue.Empty:
            logger.warning(f"⏰ MCP [{self.name}] read timed out after {timeout}s")
            return None

    def _read_until_response(self, request_id: int) -> dict | None:
        """Read JSON lines from stdout, skip notifications, return matching response."""
        stdout = self.process.stdout
        while True:
            raw_line = stdout.readline()
            if not raw_line:
                return None  # EOF

            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue  # Skip empty lines

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                # Skip non-JSON output (e.g. debug prints)
                continue

            # Notifications have no "id" — skip them
            if "id" not in msg:
                continue

            # This is a response — check if it matches our request
            if msg.get("id") == request_id:
                return msg

            # Mismatched id — skip (stale response)

    # ── MCP Protocol methods ──────────────────────────────────────

    def initialize(self) -> bool:
        """Perform the MCP initialize handshake."""
        response = self._send_jsonrpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "SeraAI",
                "version": "1.0.0"
            }
        }, timeout=_INIT_TIMEOUT)

        if response and "result" in response:
            # Send initialized notification (no response expected)
            self._send_notification("notifications/initialized")
            logger.info(f"✅ MCP [{self.name}] initialized successfully.")
            return True

        logger.error(f"❌ MCP [{self.name}] initialization failed (timeout or bad response)")
        return False

    def list_tools(self) -> list[dict]:
        """Fetch available tools from the MCP server."""
        response = self._send_jsonrpc("tools/list", {}, timeout=_INIT_TIMEOUT)
        if response and "result" in response:
            self.tools = response["result"].get("tools", [])
            tool_names = [t.get("name", "?") for t in self.tools]
            logger.debug(f"🛠️ MCP [{self.name}] tools: {tool_names}")
            return self.tools
        logger.warning(f"⚠️ MCP [{self.name}] tools/list failed")
        return []

    def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """Execute a tool on the MCP server and return the result."""
        response = self._send_jsonrpc("tools/call", {
            "name": tool_name,
            "arguments": arguments
        }, timeout=self.timeout)

        if response and "result" in response:
            result = response["result"]
            # MCP returns content as a list of content blocks
            content_blocks = result.get("content", [])
            text_parts = []
            for block in content_blocks:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            return "\n".join(text_parts) if text_parts else str(result)

        if response and "error" in response:
            error = response["error"]
            return f"Error MCP: {error.get('message', str(error))}"

        return f"Error: No response from MCP [{self.name}]"


class MCPClient:
    """Manages multiple MCP server connections and provides a unified
    interface for tool discovery and execution.

    Connections are established in **background threads** so the main
    application can start immediately without waiting for all servers.
    """

    def __init__(self, server_configs: dict[str, dict] | None = None):
        self.servers: dict[str, MCPServerConnection] = {}
        self._tool_to_server: dict[str, str] = {}  # tool_name → server_name
        self._gemini_declarations: list[dict] = []
        self._ready_event = threading.Event()

        configs = server_configs or MCP_SERVERS
        for name, cfg in configs.items():
            conn = MCPServerConnection(
                name=name,
                command=cfg["command"],
                args=cfg["args"],
                cwd=cfg.get("cwd"),
                timeout=cfg.get("timeout", 300),
            )
            self.servers[name] = conn

    def connect_all(self) -> None:
        """Start all MCP servers in background threads.

        Returns immediately — tools become available as each server
        finishes its handshake.  Use :meth:`wait_ready` if you need
        to block until connections are settled.
        """
        threads = []
        for name, conn in self.servers.items():
            t = threading.Thread(
                target=self._connect_one, args=(name, conn), daemon=True
            )
            t.start()
            threads.append(t)

        # Wait for all connection threads (with a generous timeout)
        def _waiter():
            for t in threads:
                t.join(timeout=_INIT_TIMEOUT + 5)
            # Rebuild declarations after all servers are settled
            self._gemini_declarations = self._build_gemini_declarations()
            count = len(self._gemini_declarations)
            logger.debug(f"📋 Total MCP tools available for Gemini: {count}")
            self._ready_event.set()

        threading.Thread(target=_waiter, daemon=True).start()

    def _connect_one(self, name: str, conn: MCPServerConnection) -> None:
        """Connect a single MCP server (runs in its own thread)."""
        try:
            if conn.start():
                if conn.initialize():
                    tools = conn.list_tools()
                    for tool in tools:
                        tool_name = tool.get("name", "")
                        self._tool_to_server[tool_name] = name
                    conn.ready = True
                else:
                    logger.warning(f"⚠️ MCP [{name}] init failed, tools unavailable.")
            else:
                logger.warning(f"⚠️ MCP [{name}] couldn't start.")
        except Exception as e:
            logger.error(f"❌ MCP [{name}] connection error: {e}")

    def wait_ready(self, timeout: float = 30) -> bool:
        """Block until all server connections are settled."""
        return self._ready_event.wait(timeout=timeout)

    def disconnect_all(self) -> None:
        """Stop all MCP server subprocesses."""
        for conn in self.servers.values():
            conn.stop()

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Route a tool call to the correct MCP server."""
        # If tools aren't loaded yet, wait briefly
        if not self._ready_event.is_set():
            logger.info("⏳ Waiting for MCP servers to finish connecting...")
            self._ready_event.wait(timeout=20)

        server_name = self._tool_to_server.get(tool_name)
        if not server_name:
            return f"Error: Unknown tool '{tool_name}'"

        conn = self.servers.get(server_name)
        if not conn:
            return f"Error: Server '{server_name}' not found"

        logger.info(f"🔧 Calling MCP tool: {tool_name} on [{server_name}]")
        result = conn.call_tool(tool_name, arguments)
        logger.debug(f"🔧 MCP tool result ({tool_name}): {str(result)[:200]}...")
        return str(result)

    def get_gemini_function_declarations(self) -> list[dict]:
        """Return tool schemas formatted as Gemini API functionDeclarations.

        If servers are still connecting, waits briefly for them.
        """
        if not self._ready_event.is_set():
            self._ready_event.wait(timeout=20)
        return self._gemini_declarations

    def _build_gemini_declarations(self) -> list[dict]:
        """Convert all MCP tool schemas into Gemini-compatible format."""
        declarations = []
        for name, conn in self.servers.items():
            for tool in conn.tools:
                decl = self._mcp_tool_to_gemini(tool)
                if decl:
                    declarations.append(decl)
        return declarations

    @staticmethod
    def _mcp_tool_to_gemini(mcp_tool: dict) -> dict | None:
        """Convert a single MCP tool definition into a Gemini functionDeclaration.

        MCP tool format:
            {"name": "...", "description": "...", "inputSchema": {JSON Schema}}

        Gemini format:
            {"name": "...", "description": "...", "parameters": {OpenAPI subset}}
        """
        name = mcp_tool.get("name")
        description = mcp_tool.get("description", "")
        input_schema = mcp_tool.get("inputSchema", {})

        if not name:
            return None

        # Gemini expects "parameters" in a subset of OpenAPI schema.
        # MCP's inputSchema is already JSON Schema which is very close.
        parameters = {}
        if input_schema:
            parameters = {
                "type": input_schema.get("type", "object"),
                "properties": input_schema.get("properties", {}),
            }
            if "required" in input_schema:
                parameters["required"] = input_schema["required"]

        declaration = {
            "name": name,
            "description": description,
        }
        # Only add parameters if the tool actually takes arguments
        if parameters.get("properties"):
            declaration["parameters"] = parameters

        return declaration
