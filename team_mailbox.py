"""
Minimal file-based team mailbox (subset of learn-claude-code s09).

Use case: you want to leave structured notes for collaborators (other agent
sessions, the human, future you) that survive across runs but don't require
spinning up live teammate threads.

Layout::

    .team/
      roster.json                 <- {name: {role, created_at, ...}}
      inbox/
        <name>.jsonl              <- append-only; read drains by default

API surface (one tool, six actions; everything else is bookkeeping):

- ``team(action="register", name, role?)``     create a mailbox
- ``team(action="send", to, content, from_?, type?)``  append a message
- ``team(action="broadcast", content, from_?, type?)`` send to every roster name
- ``team(action="read", name, drain?=true)``   read & (by default) drain
- ``team(action="peek", name, limit?=20)``     read without draining
- ``team(action="list")``                       roster + inbox depths

Skipping for now: live teammate threads (s09's spawn), request/response
protocols (s10), autonomous claim loop (s11). Those need a real second
process to be useful — easy to bolt on later if you want them.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

TEAM_DIRNAME = ".team"
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._\-]+$")
MAX_PEEK = int(os.getenv("AGENT_TEAM_PEEK_MAX", "100"))


def team_dir(root: Path) -> Path:
    return root / TEAM_DIRNAME


def roster_path(root: Path) -> Path:
    return team_dir(root) / "roster.json"


def inbox_dir(root: Path) -> Path:
    return team_dir(root) / "inbox"


def _ensure_layout(root: Path) -> None:
    team_dir(root).mkdir(parents=True, exist_ok=True)
    inbox_dir(root).mkdir(parents=True, exist_ok=True)
    p = roster_path(root)
    if not p.is_file():
        p.write_text(json.dumps({"members": {}}, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_roster(root: Path) -> dict[str, dict[str, Any]]:
    p = roster_path(root)
    if not p.is_file():
        return {}
    try:
        return (json.loads(p.read_text(encoding="utf-8")) or {}).get("members", {})
    except (OSError, json.JSONDecodeError):
        return {}


def _save_roster(root: Path, members: dict[str, dict[str, Any]]) -> None:
    _ensure_layout(root)
    roster_path(root).write_text(
        json.dumps({"members": members}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _validate_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise ValueError("team: name is required")
    if not _SAFE_NAME_RE.match(name):
        raise ValueError(f"team: invalid name {name!r} (allowed: letters, digits, . - _)")
    return name


def _inbox_for(root: Path, name: str) -> Path:
    return inbox_dir(root) / f"{_validate_name(name)}.jsonl"


# ---- mutations ------------------------------------------------------------


def team_register(root: Path, name: str, role: str = "") -> dict[str, Any]:
    name = _validate_name(name)
    members = _load_roster(root)
    if name in members:
        members[name]["role"] = role.strip() or members[name].get("role", "")
        members[name]["updated_at"] = time.time()
    else:
        members[name] = {"role": role.strip(), "created_at": time.time(), "updated_at": time.time()}
    _save_roster(root, members)
    # Touch the inbox file so list() sees a depth=0 mailbox.
    inbox = _inbox_for(root, name)
    if not inbox.exists():
        inbox.parent.mkdir(parents=True, exist_ok=True)
        inbox.write_text("", encoding="utf-8")
    return members[name] | {"name": name}


def team_send(
    root: Path,
    *,
    to: str,
    content: str,
    from_: str = "",
    msg_type: str = "message",
) -> dict[str, Any]:
    to = _validate_name(to)
    if not content or not str(content).strip():
        raise ValueError("team_send: content is required")
    msg = {
        "type": msg_type or "message",
        "from": (from_ or "").strip(),
        "to": to,
        "content": str(content),
        "timestamp": time.time(),
    }
    _ensure_layout(root)
    p = _inbox_for(root, to)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(msg, ensure_ascii=False) + "\n")
    return msg


def team_broadcast(
    root: Path,
    *,
    content: str,
    from_: str = "",
    msg_type: str = "broadcast",
) -> int:
    members = _load_roster(root)
    sent = 0
    for name in list(members.keys()):
        if name == (from_ or "").strip():
            continue
        team_send(root, to=name, content=content, from_=from_, msg_type=msg_type)
        sent += 1
    return sent


def team_read(root: Path, *, name: str, drain: bool = True) -> list[dict[str, Any]]:
    p = _inbox_for(root, name)
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8")
    msgs = [json.loads(line) for line in text.splitlines() if line.strip()]
    if drain:
        p.write_text("", encoding="utf-8")
    return msgs


def team_peek(root: Path, *, name: str, limit: int = 20) -> list[dict[str, Any]]:
    msgs = team_read(root, name=name, drain=False)
    if limit <= 0:
        return msgs
    return msgs[-min(limit, MAX_PEEK):]


def team_list(root: Path) -> list[dict[str, Any]]:
    members = _load_roster(root)
    out = []
    for name, meta in sorted(members.items()):
        inbox = _inbox_for(root, name)
        depth = 0
        if inbox.exists():
            depth = sum(1 for _ in inbox.read_text(encoding="utf-8").splitlines() if _.strip())
        out.append({"name": name, "role": meta.get("role", ""), "inbox_depth": depth})
    return out


# ---- rendering / dispatch -------------------------------------------------


def render_team_summary(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(no team members registered)"
    lines = [
        f"- {r['name']} ({r.get('role','-')}) inbox={r['inbox_depth']}"
        for r in rows
    ]
    return "\n".join(lines)


def handle_team_tool(action: str, payload: dict[str, Any], *, root: Path) -> str:
    act = (action or "").strip().lower()
    try:
        if act == "register":
            m = team_register(root, str(payload.get("name", "")), str(payload.get("role", "")))
            return f"Registered teammate {m['name']!r} (role={m.get('role','')})"
        if act == "send":
            msg = team_send(
                root,
                to=str(payload.get("to", "")),
                content=str(payload.get("content", "")),
                from_=str(payload.get("from", "") or payload.get("from_", "")),
                msg_type=str(payload.get("type", "message")),
            )
            return f"Sent to {msg['to']} (type={msg['type']})"
        if act == "broadcast":
            n = team_broadcast(
                root,
                content=str(payload.get("content", "")),
                from_=str(payload.get("from", "") or payload.get("from_", "")),
                msg_type=str(payload.get("type", "broadcast")),
            )
            return f"Broadcast to {n} teammate(s)"
        if act == "read":
            msgs = team_read(
                root,
                name=str(payload.get("name", "")),
                drain=bool(payload.get("drain", True)),
            )
            return json.dumps(msgs, ensure_ascii=False, indent=2) if msgs else "(empty inbox)"
        if act == "peek":
            msgs = team_peek(
                root,
                name=str(payload.get("name", "")),
                limit=int(payload.get("limit") or 20),
            )
            return json.dumps(msgs, ensure_ascii=False, indent=2) if msgs else "(empty inbox)"
        if act == "list":
            return render_team_summary(team_list(root))
        return f"team: unknown action {action!r}; use register/send/broadcast/read/peek/list"
    except (KeyError, ValueError, RuntimeError) as exc:
        return f"team error: {exc}"
