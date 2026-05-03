"""
Git worktree task isolation (mirrors learn-claude-code s12).

Each *task* (from ``task_graph``) can be bound to its own ``git worktree`` so
two parallel work streams never collide on the same files. Layout:

    .worktrees/
      <name>/                  <- the actual working tree on branch wt/<name>
      index.json               <- registry: name -> {branch, path, task_id, ...}
      events.jsonl             <- append-only lifecycle log (audit + recovery)

The worktree binds to a task by writing the worktree name into
``task["worktree"]`` and pushing the task to ``in_progress``. ``worktree_remove``
with ``complete_task=True`` will simultaneously delete the directory and mark
the bound task ``completed``.

This is the minimum that's actually useful for solo / small-team workflows.
Full multi-agent team coordination (s09 mailbox, s10 protocols, s11
autonomous claim) lives in ``team_mailbox.py``; the two are independent and
can be used together or separately.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

WORKTREES_DIRNAME = ".worktrees"
BRANCH_PREFIX = os.getenv("AGENT_WORKTREE_BRANCH_PREFIX", "wt/")
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._\-/]+$")


def worktrees_dir(root: Path) -> Path:
    return root / WORKTREES_DIRNAME


def index_path(root: Path) -> Path:
    return worktrees_dir(root) / "index.json"


def events_path(root: Path) -> Path:
    return worktrees_dir(root) / "events.jsonl"


# ---- registry I/O ---------------------------------------------------------


@dataclass
class WorktreeEntry:
    name: str
    path: str
    branch: str
    task_id: int | None = None
    status: str = "active"  # active | removed | kept
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_index(root: Path) -> dict[str, WorktreeEntry]:
    p = index_path(root)
    if not p.is_file():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, WorktreeEntry] = {}
    for name, entry in (raw.get("entries") or {}).items():
        try:
            out[name] = WorktreeEntry(**{k: entry[k] for k in entry if k in WorktreeEntry.__annotations__})
        except (TypeError, KeyError):
            continue
    return out


def _save_index(root: Path, entries: dict[str, WorktreeEntry]) -> None:
    p = index_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"entries": {n: e.to_dict() for n, e in entries.items()}}
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _emit_event(root: Path, event: str, **fields: Any) -> None:
    p = events_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {"event": event, "ts": time.time(), **fields}
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


# ---- git plumbing ---------------------------------------------------------


def _run_git(root: Path, args: list[str], *, timeout: float = 60.0) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()


def _ensure_clean_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise ValueError("worktree: name is required")
    if not _SAFE_NAME_RE.match(name):
        raise ValueError(f"worktree: invalid name {name!r} (allowed chars: letters, digits, . - _ /)")
    if name.startswith("/") or ".." in name.split("/"):
        raise ValueError(f"worktree: name must not be absolute or contain '..' ({name!r})")
    return name


# ---- mutations ------------------------------------------------------------


def worktree_create(
    root: Path,
    *,
    name: str,
    task_id: int | None = None,
    base: str = "HEAD",
) -> WorktreeEntry:
    name = _ensure_clean_name(name)
    if not (root / ".git").exists():
        raise RuntimeError(f"worktree_create: {root} is not a git repository")

    entries = _load_index(root)
    if name in entries and entries[name].status == "active":
        raise RuntimeError(f"worktree {name!r} already exists (status=active)")

    target_path = worktrees_dir(root) / name
    branch = f"{BRANCH_PREFIX}{name}"
    _emit_event(root, "worktree.create.before", name=name, branch=branch, task_id=task_id, base=base)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    code, out, err = _run_git(root, ["worktree", "add", "-b", branch, str(target_path), base])
    if code != 0:
        # Maybe the branch already exists; try without -b.
        code2, out2, err2 = _run_git(root, ["worktree", "add", str(target_path), branch])
        if code2 != 0:
            _emit_event(root, "worktree.create.failed", name=name, branch=branch, stderr=err or err2)
            raise RuntimeError(f"git worktree add failed: {err or err2}")
        out = out2

    entry = WorktreeEntry(
        name=name,
        path=str(target_path.resolve()),
        branch=branch,
        task_id=task_id,
        status="active",
    )
    entries[name] = entry
    _save_index(root, entries)
    _emit_event(root, "worktree.create.after", name=name, branch=branch, path=entry.path, task_id=task_id)

    if task_id is not None:
        try:
            from task_graph import task_update

            task_update(root, task_id, status="in_progress", owner=f"worktree:{name}")
        except Exception as exc:  # noqa: BLE001 — bookkeeping must not unwind a successful create
            _emit_event(root, "worktree.bind_task.failed", name=name, task_id=task_id, error=str(exc))
    return entry


def worktree_remove(
    root: Path,
    *,
    name: str,
    force: bool = False,
    complete_task: bool = False,
    delete_branch: bool = False,
) -> WorktreeEntry:
    name = _ensure_clean_name(name)
    entries = _load_index(root)
    entry = entries.get(name)
    if entry is None:
        raise KeyError(f"worktree_remove: {name!r} not in registry")
    _emit_event(root, "worktree.remove.before", name=name, task_id=entry.task_id, complete_task=complete_task)

    args = ["worktree", "remove", entry.path]
    if force:
        args.insert(2, "--force")
    code, _out, err = _run_git(root, args)
    if code != 0 and not force:
        # Try once more with force; record the original failure.
        _emit_event(root, "worktree.remove.failed", name=name, stderr=err, retry_with_force=True)
        code, _out, err = _run_git(root, ["worktree", "remove", "--force", entry.path])
    if code != 0:
        _emit_event(root, "worktree.remove.failed", name=name, stderr=err)
        # Best-effort directory cleanup so the registry isn't lying.
        if Path(entry.path).exists():
            shutil.rmtree(entry.path, ignore_errors=True)

    if delete_branch:
        _run_git(root, ["branch", "-D", entry.branch])

    entry.status = "removed"
    entry.updated_at = time.time()
    entries[name] = entry
    _save_index(root, entries)
    _emit_event(root, "worktree.remove.after", name=name, task_id=entry.task_id)

    if complete_task and entry.task_id is not None:
        try:
            from task_graph import task_update

            task_update(root, entry.task_id, status="completed")
            _emit_event(root, "task.completed", name=name, task_id=entry.task_id)
        except Exception as exc:  # noqa: BLE001
            _emit_event(root, "worktree.complete_task.failed", name=name, task_id=entry.task_id, error=str(exc))

    return entry


def worktree_keep(root: Path, *, name: str) -> WorktreeEntry:
    name = _ensure_clean_name(name)
    entries = _load_index(root)
    entry = entries.get(name)
    if entry is None:
        raise KeyError(f"worktree_keep: {name!r} not in registry")
    entry.status = "kept"
    entry.updated_at = time.time()
    entries[name] = entry
    _save_index(root, entries)
    _emit_event(root, "worktree.keep", name=name, task_id=entry.task_id)
    return entry


def worktree_list(root: Path) -> list[WorktreeEntry]:
    return sorted(_load_index(root).values(), key=lambda e: e.created_at)


def worktree_run(
    root: Path,
    *,
    name: str,
    command: str,
    timeout_sec: float = 300.0,
) -> str:
    name = _ensure_clean_name(name)
    entry = _load_index(root).get(name)
    if entry is None:
        return f"worktree_run: {name!r} not in registry"
    if entry.status != "active" and entry.status != "kept":
        return f"worktree_run: {name!r} status={entry.status}; refusing to run"
    p = Path(entry.path)
    if not p.is_dir():
        return f"worktree_run: {name!r} path {entry.path} no longer exists"
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(p),
            capture_output=True,
            text=True,
            timeout=max(1.0, min(timeout_sec, 1800.0)),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"worktree_run({name}): timed out after {timeout_sec}s"
    out = proc.stdout or ""
    err = proc.stderr or ""
    tail = f"\n--- stderr ---\n{err}" if err.strip() else ""
    return f"exit_code={proc.returncode}\n--- stdout ---\n{out}{tail}"


# ---- rendering / dispatch -------------------------------------------------


def render_worktree_summary(entries: list[WorktreeEntry]) -> str:
    if not entries:
        return "(no worktrees)"
    lines = []
    for e in entries:
        bind = f" task#{e.task_id}" if e.task_id is not None else ""
        lines.append(f"- {e.name} [{e.status}] branch={e.branch}{bind} path={e.path}")
    return "\n".join(lines)


def handle_worktree_tool(action: str, payload: dict[str, Any], *, root: Path) -> str:
    act = (action or "").strip().lower()
    try:
        if act == "create":
            tid = payload.get("task_id")
            entry = worktree_create(
                root,
                name=str(payload.get("name", "")),
                task_id=int(tid) if tid is not None else None,
                base=str(payload.get("base") or "HEAD"),
            )
            return f"Created worktree {entry.name!r} on branch {entry.branch} at {entry.path}"
        if act == "remove":
            entry = worktree_remove(
                root,
                name=str(payload.get("name", "")),
                force=bool(payload.get("force")),
                complete_task=bool(payload.get("complete_task")),
                delete_branch=bool(payload.get("delete_branch")),
            )
            return f"Removed worktree {entry.name!r} (task#{entry.task_id})"
        if act == "keep":
            entry = worktree_keep(root, name=str(payload.get("name", "")))
            return f"Marked worktree {entry.name!r} as kept"
        if act == "list":
            return render_worktree_summary(worktree_list(root))
        if act == "run":
            return worktree_run(
                root,
                name=str(payload.get("name", "")),
                command=str(payload.get("command", "")),
                timeout_sec=float(payload.get("timeout_sec") or 300.0),
            )
        if act == "events":
            n = int(payload.get("limit") or 20)
            p = events_path(root)
            if not p.is_file():
                return "(no events recorded)"
            lines = p.read_text(encoding="utf-8").splitlines()
            return "\n".join(lines[-n:]) if lines else "(no events recorded)"
        return f"worktree: unknown action {action!r}; use create/remove/keep/list/run/events"
    except (KeyError, ValueError, RuntimeError) as exc:
        return f"worktree error: {exc}"
