"""
Background shell tasks (mirrors learn-claude-code s08).

The agent fires off long-running shell work (`pytest`, `npm install`, …) in a
daemon thread and immediately returns to thinking. Results are queued and
drained into the next tool turn as a synthetic ``role=user`` message so the
model sees them before its next decision.

State lives entirely in-process — restarts wipe the queue (intentional; this
is for *during a session*, not cross-session orchestration).

Two tools:

- ``bg_run(command, description?, timeout_sec?)``
   spawns and returns ``{task_id, status: "running", command}``.
- ``bg_check(task_id?)``
   if ``task_id`` is omitted, lists all current tasks; otherwise returns just that one.

The orchestrator should call ``drain_notifications()`` at the top of each
agent turn — finished tasks get injected as a structured user message so the
model can reason about them without explicitly polling.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

BG_MAX_OUTPUT_CHARS = int(os.getenv("AGENT_BG_MAX_OUTPUT_CHARS", "20000"))
BG_NOTIF_PREVIEW_CHARS = int(os.getenv("AGENT_BG_NOTIF_PREVIEW_CHARS", "1500"))
BG_DEFAULT_TIMEOUT_SEC = float(os.getenv("AGENT_BG_DEFAULT_TIMEOUT_SEC", "300"))
BG_HARD_TIMEOUT_SEC = float(os.getenv("AGENT_BG_HARD_TIMEOUT_SEC", "1800"))


@dataclass
class BgTask:
    task_id: str
    command: str
    description: str = ""
    status: str = "running"  # running | done | failed | timeout
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    exit_code: int | None = None
    output: str = ""

    def to_dict(self, *, include_output: bool = True) -> dict[str, Any]:
        d = asdict(self)
        if not include_output:
            d.pop("output", None)
        return d


class BackgroundManager:
    """Thread-safe registry of background subprocess tasks + a notification queue."""

    def __init__(self) -> None:
        self._tasks: dict[str, BgTask] = {}
        self._notif: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    # ---- spawn / inspect ----------------------------------------------------

    def run(
        self,
        command: str,
        *,
        description: str = "",
        timeout_sec: float | None = None,
        cwd: str | Path | None = None,
    ) -> BgTask:
        if not command or not command.strip():
            raise ValueError("bg_run: command is required")
        timeout = float(timeout_sec) if timeout_sec else BG_DEFAULT_TIMEOUT_SEC
        timeout = max(1.0, min(timeout, BG_HARD_TIMEOUT_SEC))

        task_id = uuid.uuid4().hex[:8]
        task = BgTask(task_id=task_id, command=command, description=description.strip())
        with self._lock:
            self._tasks[task_id] = task

        thread = threading.Thread(
            target=self._execute,
            args=(task_id, command, timeout, str(cwd) if cwd else None),
            daemon=True,
            name=f"bg-{task_id}",
        )
        thread.start()
        return task

    def get(self, task_id: str) -> BgTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def list_all(self) -> list[BgTask]:
        with self._lock:
            return sorted(self._tasks.values(), key=lambda t: t.started_at)

    def drain_notifications(self) -> list[dict[str, Any]]:
        """Return queued completion records and clear the queue. Thread-safe."""
        with self._lock:
            out, self._notif = self._notif, []
        return out

    # ---- internals ----------------------------------------------------------

    def _execute(self, task_id: str, command: str, timeout: float, cwd: str | None) -> None:
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            output = (proc.stdout or "") + (
                ("\n--- stderr ---\n" + proc.stderr) if (proc.stderr or "").strip() else ""
            )
            output = output[:BG_MAX_OUTPUT_CHARS]
            new_status = "done" if proc.returncode == 0 else "failed"
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            output = f"[bg] timeout after {timeout}s"
            new_status = "timeout"
            exit_code = None
        except OSError as exc:
            output = f"[bg] failed to start: {exc}"
            new_status = "failed"
            exit_code = None

        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.status = new_status
            task.finished_at = time.time()
            task.exit_code = exit_code
            task.output = output
            self._notif.append(
                {
                    "task_id": task_id,
                    "command": command,
                    "status": new_status,
                    "exit_code": exit_code,
                    "duration_sec": round(task.finished_at - task.started_at, 2),
                    "output_preview": output[:BG_NOTIF_PREVIEW_CHARS],
                    "output_truncated": len(output) > BG_NOTIF_PREVIEW_CHARS,
                }
            )


# Process-wide singleton so the orchestrator and the tool dispatcher share state.
_MANAGER = BackgroundManager()


def manager() -> BackgroundManager:
    return _MANAGER


# ---- tool entry points ----------------------------------------------------


def handle_bg_run(payload: dict[str, Any]) -> str:
    if os.getenv("AGENT_ALLOW_BASH", "").strip() != "1":
        return (
            "bg_run is disabled. Set AGENT_ALLOW_BASH=1 in the environment to enable "
            "background shell execution (high risk: runs shell on the host)."
        )
    try:
        task = _MANAGER.run(
            command=str(payload.get("command", "")),
            description=str(payload.get("description", "")),
            timeout_sec=payload.get("timeout_sec"),
            cwd=payload.get("cwd"),
        )
    except ValueError as exc:
        return f"bg_run error: {exc}"
    return json.dumps(task.to_dict(include_output=False), ensure_ascii=False, indent=2)


def handle_bg_check(payload: dict[str, Any]) -> str:
    task_id = payload.get("task_id")
    if task_id:
        task = _MANAGER.get(str(task_id))
        if task is None:
            return f"bg_check: task {task_id!r} not found"
        return json.dumps(task.to_dict(), ensure_ascii=False, indent=2)
    tasks = _MANAGER.list_all()
    if not tasks:
        return "(no background tasks)"
    return json.dumps(
        [t.to_dict(include_output=False) for t in tasks],
        ensure_ascii=False,
        indent=2,
    )


def render_drain_message(notifs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Format drained notifications as a synthetic ``role=user`` chat message."""
    if not notifs:
        return None
    blocks = []
    for n in notifs:
        head = (
            f"[bg:{n['task_id']}] status={n['status']} exit={n.get('exit_code')} "
            f"dur={n.get('duration_sec')}s cmd={n['command']!r}"
        )
        body = n.get("output_preview") or ""
        suffix = "  (output truncated; call bg_check task_id=… for full body)" if n.get("output_truncated") else ""
        blocks.append(f"{head}\n{body}{suffix}".strip())
    body = "\n\n".join(blocks)
    return {
        "role": "user",
        "content": f"<background-results count={len(notifs)}>\n{body}\n</background-results>",
    }
