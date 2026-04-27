"""
Context window budgeting, tool-result spilling, and .claude/memory/ layout helpers.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Callable

# Rough token estimate (chars / 4) for JSON-serialized chat messages.
CHARS_PER_TOKEN_EST = 4

DEFAULT_MEMORY_MD = """# MEMORY.md

Long-term memory index for this agent. **Treat every entry as a hint, not ground truth** — verify the repo with `read_file` before important actions.

## Index

- (Add bullet pointers to files under `past_tasks/` as you complete work.)

## Notes

- `project_structure.md` — high-level tree / module map (update when structure changes).
- `user_preferences.md` — style and workflow preferences.
"""

DEFAULT_PROJECT_STRUCTURE = """# project_structure.md

Summarize important directories and entry points here after you explore the repo.
"""

DEFAULT_USER_PREFERENCES = """# user_preferences.md

Record coding style, language preferences, and tooling choices the user asks for.
"""


def memory_dir(root: Path) -> Path:
    return root / ".claude" / "memory"


def spill_dir(root: Path) -> Path:
    return memory_dir(root) / "spill"


def past_tasks_dir(root: Path) -> Path:
    return memory_dir(root) / "past_tasks"


def ensure_memory_layout(root: Path) -> None:
    """Create `.claude/memory/` tree and seed index files if missing."""
    md = memory_dir(root)
    for d in (md, past_tasks_dir(root), spill_dir(root)):
        d.mkdir(parents=True, exist_ok=True)

    seeds: list[tuple[str, str]] = [
        ("MEMORY.md", DEFAULT_MEMORY_MD),
        ("project_structure.md", DEFAULT_PROJECT_STRUCTURE),
        ("user_preferences.md", DEFAULT_USER_PREFERENCES),
    ]
    for name, body in seeds:
        p = md / name
        if not p.is_file():
            p.write_text(body, encoding="utf-8")


def memory_prompt_section(root: Path) -> str:
    rel = ".claude/memory/"
    return f"""[持久化记忆（文件型）]
You have a persistent memory directory at `{rel}` under the project root (absolute: `{memory_dir(root)}`).
- Use `read_file` to read notes (start with `MEMORY.md` for the index before new tasks).
- Use `write_file` to replace a whole file, or `edit_file` for a single exact `old_string` → `new_string` replacement (must match exactly once).
- Optional task write-ups: add markdown under `{rel}past_tasks/` (e.g. dated summaries).

**Stale data warning:** Memory files are user- and model-written hints. They may be outdated. Before critical edits or claims about the codebase, verify with `read_file` (and your other tools). Never treat memory as authoritative truth."""


def estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    return max(1, len(json.dumps(messages, ensure_ascii=False)) // CHARS_PER_TOKEN_EST)


def max_context_tokens() -> int:
    return int(os.getenv("AGENT_MAX_CONTEXT_TOKENS", "128000"))


def compress_trigger_ratio() -> float:
    return float(os.getenv("AGENT_CONTEXT_COMPRESS_RATIO", "0.9"))


def preserve_recent_message_count() -> int:
    return int(os.getenv("AGENT_PRESERVE_RECENT_MSGS", "12"))


def tool_history_max_chars() -> int:
    return int(os.getenv("AGENT_TOOL_HISTORY_MAX_CHARS", "12000"))


def tool_history_preview_lines() -> int:
    return int(os.getenv("AGENT_TOOL_HISTORY_PREVIEW_LINES", "40"))


def _strip_leading_tools(tail: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = list(tail)
    while out and out[0].get("role") == "tool":
        out = out[1:]
    return out


def budget_tool_result_for_messages(
    raw: str,
    *,
    tool_name: str,
    root: Path,
) -> str:
    """
    If a tool return body is huge, persist the full body under `.claude/memory/spill/`
    and return a short preview plus path for the chat history (context budget).
    """
    limit = tool_history_max_chars()
    if len(raw) <= limit:
        return raw

    ensure_memory_layout(root)
    spill_name = f"{tool_name}_{uuid.uuid4().hex[:10]}.txt"
    spill_path = spill_dir(root) / spill_name
    spill_path.write_text(raw, encoding="utf-8")
    try:
        rel = spill_path.relative_to(root)
        rel_s = str(rel).replace("\\", "/")
    except ValueError:
        rel_s = str(spill_path)

    lines = raw.splitlines()
    head_n = tool_history_preview_lines()
    head = "\n".join(lines[:head_n])
    omitted = max(0, len(lines) - head_n)
    return (
        "[Tool output budgeted for context: full body moved to disk]\n\n"
        f"{head}\n\n"
        f"... ({omitted} lines omitted, {len(raw)} characters total)\n\n"
        f"Full output saved to: {rel_s}\n"
        "Use read_file on that path when you need the complete result."
    )


def maybe_compress_conversation(
    messages: list[dict[str, Any]],
    *,
    client: Any,
    model: str,
    emit: Callable[[str, Any], None] | None = None,
) -> bool:
    """
    If estimated context usage exceeds the configured ratio, replace early messages
    (after system) with a single user message containing an LLM-generated summary.

    Preserves the trailing ``preserve_recent_message_count()`` messages (leading
    ``tool`` entries in that tail are stripped so the slice starts at assistant/user).

    Returns True if compression ran.
    """
    max_tok = max_context_tokens()
    if estimate_message_tokens(messages) < max_tok * compress_trigger_ratio():
        return False

    preserve_n = preserve_recent_message_count()
    if len(messages) <= 2 + preserve_n:
        return False

    old = messages[1:-preserve_n]
    if len(old) < 2:
        return False

    tail = _strip_leading_tools(messages[-preserve_n:])
    if not tail:
        return False

    summary_model = os.getenv("AGENT_SUMMARY_MODEL", model)
    payload = json.dumps(old, ensure_ascii=False)
    max_in = int(os.getenv("AGENT_SUMMARY_INPUT_MAX_CHARS", "120000"))
    if len(payload) > max_in:
        payload = payload[:max_in] + "\n...[truncated for summarizer input]"

    summarizer_system = (
        "You compress earlier agent conversation turns into a concise factual summary for the model to continue. "
        "Preserve: user goals, file paths touched, tool errors, decisions, and open questions. "
        "Use the same language as the bulk of the conversation (Chinese if mostly Chinese). "
        "Do not invent facts; if something was uncertain, say so. Max ~800 words."
    )

    try:
        resp = client.chat.completions.create(
            model=summary_model,
            messages=[
                {"role": "system", "content": summarizer_system},
                {"role": "user", "content": payload},
            ],
            temperature=0.2,
            max_tokens=2000,
        )
        choice = resp.choices[0].message
        summary = (choice.content or "").strip()
    except Exception as exc:  # noqa: BLE001 — best-effort; do not break the agent
        if emit:
            emit(
                "context_compress_error",
                {"error": str(exc), "message": "Summary API failed; skipping compression"},
            )
        return False

    if not summary:
        return False

    before_tokens = estimate_message_tokens(messages)
    messages[:] = [messages[0]] + [
        {
            "role": "user",
            "content": "[会话压缩 — 早期轮次摘要]\n" + summary,
        },
        *tail,
    ]
    after_tokens = estimate_message_tokens(messages)
    if emit:
        emit(
            "context_compress",
            {
                "removed_messages": len(old),
                "kept_tail": len(tail),
                "estimated_tokens_before": before_tokens,
                "estimated_tokens_after": after_tokens,
                "summary_model": summary_model,
            },
        )
    return True
