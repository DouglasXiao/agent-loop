"""
On-demand skill loading (mirrors learn-claude-code s05).

A *skill* is a directory under ``.claude/skills/<name>/`` containing a
``SKILL.md`` file. Optional YAML frontmatter (``--- name: … description: … ---``)
provides a short description; the body is the full instruction sheet.

Two-layer cost model:

- Layer 1 (always present): the system prompt advertises only ``name + description``
  for each skill — typically ~1 line per skill, very cheap.
- Layer 2 (on demand): the ``load_skill`` tool returns the full body wrapped
  in ``<skill name="…">…</skill>`` for the model to consume. ``list_skills``
  returns just the index again if the model forgot.

This module is intentionally stdlib-only — no PyYAML dependency. Frontmatter
parsing is a tiny key:value reader for the common case.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SKILL_FILENAME = "SKILL.md"
MAX_LIST_SKILLS = int(os.getenv("AGENT_MAX_LIST_SKILLS", "32"))
MAX_DESCRIPTION_CHARS = int(os.getenv("AGENT_SKILL_DESC_CHARS", "240"))
MAX_SKILL_BODY_CHARS = int(os.getenv("AGENT_SKILL_BODY_CHARS", "20000"))

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._\-]+$")
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.DOTALL)


def skills_dir(root: Path) -> Path:
    return root / ".claude" / "skills"


@dataclass(frozen=True)
class SkillInfo:
    name: str
    description: str
    path: Path


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Tiny YAML-ish frontmatter parser: only key: value lines, no nesting."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    raw_meta, body = m.group(1), m.group(2)
    meta: dict[str, str] = {}
    for line in raw_meta.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, _, v = line.partition(":")
        meta[k.strip().lower()] = v.strip().strip("'\"")
    return meta, body.lstrip("\n")


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def discover_skills(root: Path) -> list[SkillInfo]:
    """
    Walk ``.claude/skills/*/SKILL.md`` and return ``SkillInfo`` records.

    Resilient: a missing skills directory just yields ``[]``; an unreadable
    SKILL.md is skipped silently rather than crashing the agent.
    """
    base = skills_dir(root)
    if not base.is_dir():
        return []
    out: list[SkillInfo] = []
    for skill_md in sorted(base.rglob(SKILL_FILENAME)):
        if not skill_md.is_file():
            continue
        try:
            text = skill_md.read_text(encoding="utf-8")
        except OSError:
            continue
        meta, _body = _parse_frontmatter(text)
        # Prefer frontmatter ``name``; fall back to the parent directory name.
        name = (meta.get("name") or skill_md.parent.name or "").strip()
        if not name or not _SAFE_NAME_RE.match(name):
            continue
        desc = meta.get("description") or ""
        if not desc:
            # Fall back to the first non-empty paragraph of the body.
            for para in (text.split("\n\n", 5)):
                para = para.strip()
                if para and not para.startswith("---"):
                    desc = para.splitlines()[0]
                    break
        out.append(
            SkillInfo(
                name=name,
                description=_truncate(desc.strip(), MAX_DESCRIPTION_CHARS),
                path=skill_md,
            )
        )
    return out


def render_skill_index(skills: list[SkillInfo]) -> str:
    """Compact one-line-per-skill summary suitable for the system prompt."""
    if not skills:
        return "(no skills installed under `.claude/skills/`)"
    head = skills[:MAX_LIST_SKILLS]
    lines = [f"- {s.name}: {s.description or '(no description)'}" for s in head]
    if len(skills) > MAX_LIST_SKILLS:
        lines.append(f"- ... ({len(skills) - MAX_LIST_SKILLS} more — call list_skills for full list)")
    return "\n".join(lines)


def render_skill_body(info: SkillInfo) -> str:
    """Wrap a skill body in ``<skill name="…">…</skill>`` for tool_result delivery."""
    try:
        text = info.path.read_text(encoding="utf-8")
    except OSError as exc:
        return f'<skill name="{info.name}" error="{exc}"/>'
    _meta, body = _parse_frontmatter(text)
    body = _truncate(body, MAX_SKILL_BODY_CHARS)
    return f'<skill name="{info.name}">\n{body}\n</skill>'


# -------- tool entry points ------------------------------------------------


def handle_list_skills(payload: dict[str, Any], *, root: Path) -> str:
    """Return the same index that's pre-loaded into the system prompt."""
    skills = discover_skills(root)
    if not skills:
        return (
            "(no skills installed; create one under "
            f"`{skills_dir(root)}/<name>/SKILL.md` with optional YAML frontmatter "
            "`--- name: foo\\ndescription: short blurb ---`)"
        )
    return render_skill_index(skills)


def handle_load_skill(name: str, *, root: Path) -> str:
    """Return the wrapped full body of a single skill."""
    name = (name or "").strip()
    if not name:
        return "load_skill: 'name' is required"
    if not _SAFE_NAME_RE.match(name):
        return f"load_skill: invalid name {name!r} (allowed: letters, digits, dot, dash, underscore)"
    for s in discover_skills(root):
        if s.name == name:
            return render_skill_body(s)
    return (
        f"load_skill: unknown skill {name!r}. "
        "Call list_skills to see what's installed."
    )
