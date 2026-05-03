"""
Persistent task graph (mirrors learn-claude-code s07).

Each task is one JSON file under ``.claude/tasks/task_<id>.json`` with fields:

    id          int          (auto-assigned, monotonically increasing)
    subject     str          one-line title
    description str          longer body (optional)
    status      str          pending | in_progress | completed | cancelled
    blockedBy   [int, ...]   task ids this one waits for
    owner       str          who's working on it (free-form, optional)
    created_at  float        unix ts
    updated_at  float        unix ts

Why a graph (and not just the in-session ``todo_write``):

- ``todo_write`` is per-conversation. It survives compression but it's a flat
  list with no dependency edges and gets cleared with ``action="clear"``.
- The task graph survives context compression *and* full restart, supports
  ``blockedBy`` so the agent can answer "what is unblocked right now?", and
  is the right substrate for sub-agents / worktree binding (Phase 7).

This module is intentionally stdlib-only.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

VALID_STATUSES = ("pending", "in_progress", "completed", "cancelled")


def tasks_dir(root: Path) -> Path:
    return root / ".claude" / "tasks"


def _ensure_dir(root: Path) -> Path:
    d = tasks_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _task_path(root: Path, task_id: int) -> Path:
    return tasks_dir(root) / f"task_{task_id:04d}.json"


def _load_one(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _save(root: Path, task: dict[str, Any]) -> None:
    _ensure_dir(root)
    task["updated_at"] = time.time()
    path = _task_path(root, int(task["id"]))
    path.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")


def _all_tasks(root: Path) -> list[dict[str, Any]]:
    d = tasks_dir(root)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("task_*.json")):
        t = _load_one(p)
        if t is not None:
            out.append(t)
    return sorted(out, key=lambda t: int(t.get("id", 0)))


def _next_id(root: Path) -> int:
    return (max((int(t.get("id", 0)) for t in _all_tasks(root)), default=0) + 1)


def _normalize_status(s: Any) -> str:
    s = (str(s or "pending")).strip().lower()
    return s if s in VALID_STATUSES else "pending"


def _strip_completed_dep(root: Path, completed_id: int) -> int:
    """Remove ``completed_id`` from every other task's ``blockedBy``. Returns count touched."""
    n = 0
    for t in _all_tasks(root):
        deps = t.get("blockedBy") or []
        if completed_id in deps:
            t["blockedBy"] = [d for d in deps if d != completed_id]
            _save(root, t)
            n += 1
    return n


# ---------- mutations ------------------------------------------------------


def task_create(
    root: Path,
    *,
    subject: str,
    description: str = "",
    blocked_by: Iterable[int] | None = None,
    owner: str = "",
) -> dict[str, Any]:
    subject = subject.strip()
    if not subject:
        raise ValueError("task_create: subject is required")
    deps = sorted({int(x) for x in (blocked_by or [])})
    now = time.time()
    task = {
        "id": _next_id(root),
        "subject": subject,
        "description": description.strip(),
        "status": "pending",
        "blockedBy": deps,
        "owner": owner.strip(),
        "created_at": now,
        "updated_at": now,
    }
    _save(root, task)
    return task


def task_update(
    root: Path,
    task_id: int,
    *,
    status: str | None = None,
    subject: str | None = None,
    description: str | None = None,
    owner: str | None = None,
    add_blocked_by: Iterable[int] | None = None,
    remove_blocked_by: Iterable[int] | None = None,
) -> dict[str, Any]:
    p = _task_path(root, task_id)
    task = _load_one(p)
    if task is None:
        raise KeyError(f"task_update: task {task_id} not found")
    if status is not None:
        new_status = _normalize_status(status)
        was = task.get("status")
        task["status"] = new_status
        if new_status == "completed" and was != "completed":
            _save(root, task)
            _strip_completed_dep(root, task_id)
            return task
    if subject is not None and subject.strip():
        task["subject"] = subject.strip()
    if description is not None:
        task["description"] = description.strip()
    if owner is not None:
        task["owner"] = owner.strip()
    if add_blocked_by:
        task["blockedBy"] = sorted(set(task.get("blockedBy") or []) | {int(x) for x in add_blocked_by})
    if remove_blocked_by:
        rm = {int(x) for x in remove_blocked_by}
        task["blockedBy"] = [d for d in (task.get("blockedBy") or []) if d not in rm]
    _save(root, task)
    return task


def task_get(root: Path, task_id: int) -> dict[str, Any]:
    t = _load_one(_task_path(root, task_id))
    if t is None:
        raise KeyError(f"task_get: task {task_id} not found")
    return t


def task_list(
    root: Path,
    *,
    status: str | None = None,
    only_unblocked: bool = False,
) -> list[dict[str, Any]]:
    items = _all_tasks(root)
    if status:
        s = _normalize_status(status)
        items = [t for t in items if t.get("status") == s]
    if only_unblocked:
        items = [t for t in items if not t.get("blockedBy")]
    return items


# ---------- rendering & tool dispatch --------------------------------------


def render_task_summary(tasks: list[dict[str, Any]]) -> str:
    if not tasks:
        return "(no tasks)"
    lines: list[str] = []
    counts = {"pending": 0, "in_progress": 0, "completed": 0, "cancelled": 0}
    for t in tasks:
        s = t.get("status", "pending")
        counts[s] = counts.get(s, 0) + 1
        box = {"completed": "[x]", "in_progress": "[>]", "cancelled": "[-]"}.get(s, "[ ]")
        deps = t.get("blockedBy") or []
        dep_s = f" (blocked by {','.join(str(d) for d in deps)})" if deps else ""
        owner = t.get("owner") or ""
        owner_s = f" @{owner}" if owner else ""
        lines.append(f"- {box} #{t['id']:>3} {t.get('subject','').strip()}{owner_s}{dep_s}")
    lines.append(
        f"\n_summary: {counts.get('completed',0)} done, "
        f"{counts.get('in_progress',0)} in progress, "
        f"{counts.get('pending',0)} pending, "
        f"{counts.get('cancelled',0)} cancelled_"
    )
    return "\n".join(lines)


def handle_task_tool(action: str, payload: dict[str, Any], *, root: Path) -> str:
    """Dispatch a single ``task_graph`` tool call. Returns markdown or JSON text."""
    act = (action or "").strip().lower()
    try:
        if act == "create":
            t = task_create(
                root,
                subject=str(payload.get("subject", "")),
                description=str(payload.get("description", "")),
                blocked_by=payload.get("blocked_by") or [],
                owner=str(payload.get("owner", "")),
            )
            return f"Created task #{t['id']}: {t['subject']}"
        if act == "update":
            tid = int(payload["id"])
            t = task_update(
                root,
                tid,
                status=payload.get("status"),
                subject=payload.get("subject"),
                description=payload.get("description"),
                owner=payload.get("owner"),
                add_blocked_by=payload.get("add_blocked_by"),
                remove_blocked_by=payload.get("remove_blocked_by"),
            )
            return f"Updated task #{t['id']} (status={t['status']})"
        if act == "complete":
            tid = int(payload["id"])
            t = task_update(root, tid, status="completed")
            return f"Completed task #{t['id']}; downstream deps unblocked."
        if act == "get":
            tid = int(payload["id"])
            return json.dumps(task_get(root, tid), ensure_ascii=False, indent=2)
        if act == "list":
            return render_task_summary(
                task_list(
                    root,
                    status=payload.get("status"),
                    only_unblocked=bool(payload.get("only_unblocked")),
                )
            )
        return f"task: unknown action {action!r}; use create, update, complete, get, list"
    except (KeyError, ValueError) as exc:
        return f"task error: {exc}"


# ---------- system-prompt helper -------------------------------------------


def render_task_prompt_section(root: Path, *, max_items: int | None = None) -> str:
    items = _all_tasks(root)
    if not items:
        return (
            "(no persistent tasks yet — use the `task` tool to create a graph "
            "for any work that needs to survive context compression or restart; "
            f"storage: `{tasks_dir(root)}`)"
        )
    cap = max_items or int(os.getenv("AGENT_TASK_PROMPT_MAX_ITEMS", "20"))
    if len(items) > cap:
        active = [t for t in items if t.get("status") in ("pending", "in_progress")][:cap]
        return render_task_summary(active) + f"\n_(showing {len(active)} of {len(items)} tasks; call `task` action=list for all)_"
    return render_task_summary(items)
