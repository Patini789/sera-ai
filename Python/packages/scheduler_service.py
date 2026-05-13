"""scheduler_service.py — Background scheduler service for Sera AI.

Implements a hybrid scheduling system:
  1. **Daily Digest**: Fires once at a configured time (e.g. 4:00 AM) with all
     today's events, sending the formatted prompt to Sera.
  2. **Proximity Alerts**: Checks every N minutes for events happening within
     the next M minutes and fires a prompt for each upcoming event.

Both behaviors are fully configurable via ``bot_config.py`` settings.
The service follows the ``background_services`` pattern — it has a blocking
``start()`` method designed to run in a daemon thread.
"""

import os
import time
import subprocess
import sys
import json
from datetime import datetime
from pathlib import Path
from ..core.logger import get_logger

logger = get_logger(__name__)

# Path to the MCP agenda script (relative to project root)
MCP_AGENDA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "MCPs" / "agenda.py"
MCP_PYTHON = Path(__file__).resolve().parent.parent.parent.parent / "MCPs" / ".env" / "Scripts" / "python.exe"


class SchedulerService:
    """Background service that fires daily digests and proximity alerts.

    Args:
        app: The :class:`App` instance to dispatch prompts through.
        cron_hour: Hour (0-23) for the daily digest.
        cron_minute: Minute (0-59) for the daily digest.
        prompt_template: Template for the daily digest prompt.
            Placeholders: ``{hora}`` (current time), ``{eventos}`` (event list).
        proximity_check_interval: Minutes between proximity checks.
        proximity_window: Minutes ahead to look for upcoming events.
        proximity_prompt: Template for proximity alert prompts.
            Placeholders: ``{minutos}`` (minutes until event), ``{evento}`` (event details).
    """

    def __init__(
        self,
        app,
        cron_hour: int = 4,
        cron_minute: int = 0,
        prompt_template: str = "",
        proximity_check_interval: int = 10,
        proximity_window: int = 15,
        proximity_prompt: str = "",
    ):
        self.app = app
        self.cron_hour = cron_hour
        self.cron_minute = cron_minute
        self.prompt_template = prompt_template
        self.proximity_check_interval = proximity_check_interval
        self.proximity_window = proximity_window
        self.proximity_prompt = proximity_prompt

        # State tracking
        self._digest_fired_today = False
        self._last_digest_date = None
        self._last_proximity_check = 0
        self._alerted_events = set()  # Track already-alerted events to avoid duplicates

    def start(self):
        """Blocking loop — runs in a daemon thread via ``background_services``.

        Checks every 30 seconds:
          - Whether it's time for the daily digest
          - Whether it's time for a proximity check
        """
        logger.info(
            f"Scheduler started — Daily digest at {self.cron_hour:02d}:{self.cron_minute:02d}, "
            f"proximity check every {self.proximity_check_interval}min "
            f"(window: {self.proximity_window}min)"
        )

        while True:
            try:
                now = datetime.now()

                # ── Daily digest ──────────────────────────────────
                today_str = now.strftime("%Y-%m-%d")
                if self._last_digest_date != today_str:
                    self._digest_fired_today = False
                    self._alerted_events.clear()
                    self._last_digest_date = today_str

                if (
                    now.hour == self.cron_hour
                    and now.minute == self.cron_minute
                    and not self._digest_fired_today
                ):
                    self._fire_daily_digest()
                    self._digest_fired_today = True

                # ── Proximity alerts ──────────────────────────────
                elapsed = time.time() - self._last_proximity_check
                if elapsed >= self.proximity_check_interval * 60:
                    self._check_proximity()
                    self._last_proximity_check = time.time()

            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")

            time.sleep(30)

    # ── Internal: Daily Digest ────────────────────────────────────────

    def _fire_daily_digest(self):
        """Fetch today's events and send the digest prompt to Sera."""
        logger.info("Scheduler: Firing daily digest...")

        events_text = self._call_mcp_tool("obtener_eventos_hoy")
        if not events_text:
            events_text = "No hay eventos agendados para hoy."

        prompt = self.prompt_template.format(
            hora=datetime.now().strftime("%H:%M"),
            eventos=events_text,
        )

        logger.debug(f"Scheduler digest prompt: {prompt}")
        response = self.app.handle_new_message("Sistema", prompt)

        # Dispatch through notification hooks
        if response:
            for hook in self.app.notification_hooks:
                try:
                    hook(response)
                except Exception as e:
                    logger.error(f"Notification hook error: {e}")

    # ── Internal: Proximity Alerts ────────────────────────────────────

    def _check_proximity(self):
        """Check for events within the proximity window and alert if found."""
        upcoming = self._call_mcp_tool(
            "eventos_proximos",
            {"minutos": self.proximity_window}
        )

        if not upcoming or not upcoming.strip():
            return

        # Deduplicate: only alert for events we haven't alerted about yet
        alert_key = f"{datetime.now().strftime('%Y-%m-%d')}_{hash(upcoming)}"
        if alert_key in self._alerted_events:
            return
        self._alerted_events.add(alert_key)

        logger.info(f"Scheduler: Proximity alert — upcoming events detected")

        prompt = self.proximity_prompt.format(
            minutos=self.proximity_window,
            evento=upcoming,
        )

        logger.debug(f"Scheduler proximity prompt: {prompt}")
        response = self.app.handle_new_message("Sistema", prompt)

        if response:
            for hook in self.app.notification_hooks:
                try:
                    hook(response)
                except Exception as e:
                    logger.error(f"Notification hook error: {e}")

    # ── Internal: MCP Tool Caller ─────────────────────────────────────

    def _call_mcp_tool(self, tool_name: str, args: dict = None) -> str:
        """Call an agenda MCP tool by invoking the script directly.

        Since the scheduler runs in a background thread and the MCP server
        communicates via stdio JSON-RPC, we call the agenda module's functions
        directly by importing it as a subprocess with a small helper script.

        Args:
            tool_name: Name of the tool function to call.
            args: Optional dictionary of keyword arguments.

        Returns:
            The string result from the tool, or empty string on failure.
        """
        try:
            # Build a small inline script that imports and calls the function
            args_json = json.dumps(args or {})
            script = (
                f"import asyncio, sys; "
                f"sys.path.insert(0, r'{MCP_AGENDA_PATH.parent}'); "
                f"from agenda import {tool_name}; "
                f"result = asyncio.run({tool_name}(**{args_json})); "
                f"print(result)"
            )

            python_exe = str(MCP_PYTHON) if MCP_PYTHON.exists() else sys.executable

            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            result = subprocess.run(
                [python_exe, "-c", script],
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=env,
                timeout=30,
                cwd=str(MCP_AGENDA_PATH.parent),
            )

            if result.returncode == 0:
                return result.stdout.strip()
            else:
                logger.error(f"MCP tool '{tool_name}' failed: {result.stderr.strip()}")
                return ""

        except subprocess.TimeoutExpired:
            logger.error(f"MCP tool '{tool_name}' timed out")
            return ""
        except Exception as e:
            logger.error(f"Error calling MCP tool '{tool_name}': {e}")
            return ""
