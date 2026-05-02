"""
Persistent in-task TODO list (mirrors learn-claude-code s03 + s07 persistence ideas).

- Disk file: ``.claude/todos/current.json`` under the project root.
- States: ``pending`` | ``in_progress`` | ``completed``.
- Invariant: at most one task may be ``in_progress`` at a time (focus pressure).

The agent calls a single tool ``todo_write`` with one of these actions:

- ``set``       — replace the entire list (each item: {id?, text|subject, status?})
- ``add``       — append a new item; auto-assigns id if missing
- ``update``    — change ``text``/``status`` of an existing item by id
- ``complete``  — shortcut for ``update id=... status=completed``
- ``clear``     — wipe the list (returns empty render)

Returns a markdown view + JSON snapshot so the model can re-orient quickly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

VALID_STATUSES = ("pending", "in_progress", "completed")


def todos_dir(root: Path) -> Path:
    return root / ".claude" / "todos"


def todos_file(root: Path) -> Path:
    return todos_dir(root) / "current.json"


@dataclass
class TodoItem:
    id: int
    text: str
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TodoState:
    items: list[TodoItem] = field(default_factory=list)

    # ---- persistence ----------------------------------------------------

    @classmethod
    def load(cls, root: Path) -> "TodoState":
        path = todos_file(root)
        if not path.is_file():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        items: list[TodoItem] = []
        for entry in raw.get("items", []) if isinstance(raw, dict) else []:
            try:
                items.append(
                    TodoItem(
                        id=int(entry.get("id")),
                        text=str(entry.get("text", "")).strip() or "(untitled)",
                        status=_normalize_status(entry.get("status")),
                    )
                )
            except (TypeError, ValueError):
                continue
        return cls(items=items)

    def save(self, root: Path) -> None:
        d = todos_dir(root)
        d.mkdir(parents=True, exist_ok=True)
        path = todos_file(root)
        payload = {"items": [i.to_dict() for i in self.items]}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- mutations ------------------------------------------------------

    def _next_id(self) -> int:
        return (max((i.id for i in self.items), default=0) + 1)

    def set_items(self, new_items: Iterable[dict[str, Any]]) -> None:
        items: list[TodoItem] = []
        seen_ids: set[int] = set()
        for entry in new_items:
            text = str(entry.get("text") or entry.get("subject") or "").strip()
            if not text:
                continue
            raw_id = entry.get("id")
            try:
                tid = int(raw_id) if raw_id is not None else 0
            except (TypeError, ValueError):
                tid = 0
            if tid <= 0 or tid in seen_ids:
                tid = max(self._next_id(), max(seen_ids, default=0) + 1)
            seen_ids.add(tid)
            items.append(TodoItem(id=tid, text=text, status=_normalize_status(entry.get("status"))))
        self.items = items
        self._enforce_single_in_progress()

    def add(self, text: str, status: str = "pending") -> TodoItem:
        text = text.strip()
        if not text:
            raise ValueError("todo add: text is required")
        item = TodoItem(id=self._next_id(), text=text, status=_normalize_status(status))
        self.items.append(item)
        self._enforce_single_in_progress()
        return item

    def update(
        self,
        item_id: int,
        *,
        text: str | None = None,
        status: str | None = None,
    ) -> TodoItem:
        for it in self.items:
            if it.id == item_id:
                if text is not None and text.strip():
                    it.text = text.strip()
                if status is not None:
                    it.status = _normalize_status(status)
                self._enforce_single_in_progress(prefer=item_id)
                return it
        raise KeyError(f"todo update: id {item_id} not found")

    def complete(self, item_id: int) -> TodoItem:
        return self.update(item_id, status="completed")

    def clear(self) -> None:
        self.items = []

    # ---- helpers --------------------------------------------------------

    def _enforce_single_in_progress(self, prefer: int | None = None) -> None:
        in_prog = [i for i in self.items if i.status == "in_progress"]
        if len(in_prog) <= 1:
            return
        # Keep ``prefer`` in_progress, demote others to pending.
        keep = prefer if prefer is not None else in_prog[0].id
        for it in self.items:
            if it.status == "in_progress" and it.id != keep:
                it.status = "pending"

    def render_markdown(self) -> str:
        if not self.items:
            return "(no todos)"
        lines: list[str] = []
        for it in self.items:
            box = {"completed": "[x]", "in_progress": "[>]"}.get(it.status, "[ ]")
            lines.append(f"- {box} #{it.id} {it.text}")
        counts = self.counts()
        lines.append(
            f"\n_summary: {counts['completed']} done, "
            f"{counts['in_progress']} in progress, {counts['pending']} pending_"
        )
        return "\n".join(lines)

    def counts(self) -> dict[str, int]:
        out = {"pending": 0, "in_progress": 0, "completed": 0}
        for it in self.items:
            out[it.status] = out.get(it.status, 0) + 1
        return out

    def snapshot(self) -> dict[str, Any]:
        return {"items": [i.to_dict() for i in self.items], "counts": self.counts()}


def _normalize_status(value: Any) -> str:
    s = (str(value or "pending")).strip().lower()
    if s not in VALID_STATUSES:
        return "pending"
    return s


# ---- tool entry point ----------------------------------------------------


def handle_todo_write(action: str, payload: dict[str, Any], *, root: Path) -> str:
    """Dispatch a single ``todo_write`` tool call. Always persists on success."""
    state = TodoState.load(root)
    act = (action or "").strip().lower()
    try:
        if act == "set":
            items = payload.get("items") or []
            if not isinstance(items, list):
                return "todo_write set: 'items' must be a JSON array"
            state.set_items(items)
        elif act == "add":
            text = str(payload.get("text") or payload.get("subject") or "").strip()
            status = str(payload.get("status") or "pending")
            state.add(text, status=status)
        elif act == "update":
            tid = _coerce_int(payload.get("id"), name="id")
            state.update(
                tid,
                text=(payload.get("text") if "text" in payload else None),
                status=(payload.get("status") if "status" in payload else None),
            )
        elif act == "complete":
            tid = _coerce_int(payload.get("id"), name="id")
            state.complete(tid)
        elif act == "clear":
            state.clear()
        else:
            return (
                f"todo_write: unknown action {action!r}. "
                "Use one of: set, add, update, complete, clear."
            )
    except (ValueError, KeyError) as exc:
        return f"todo_write error: {exc}"
    state.save(root)
    return state.render_markdown()


def _coerce_int(value: Any, *, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"todo_write: {name!r} must be an integer (got {value!r})")
