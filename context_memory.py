"""
Context window budgeting, tool-result spilling, and .claude/memory/ layout helpers.

Three-layer compaction (mirrors learn-claude-code s06):

1. ``micro_compact_inplace`` — silently shrinks ``role=tool`` messages older
   than ``KEEP_RECENT_TOOL_RESULTS`` to a one-line placeholder. Preserves
   ``tool_call_id`` so the OpenAI chat protocol stays well-formed; the full
   body is still on disk via ``budget_tool_result_for_messages`` spill.
2. ``maybe_compress_conversation`` — when the budget is near the limit,
   summarizes earlier turns through the LLM and replaces them with one
   compact user message. Saves a complete transcript snapshot to
   ``.claude/memory/transcripts/`` first so nothing is irretrievably lost.
3. (Future) Manual ``compact`` tool — same summary pipeline triggered on
   demand by the model. Hook is in place but the tool is not yet exposed.
"""

from __future__ import annotations

import json
import os
import time
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


def transcripts_dir(root: Path) -> Path:
    return memory_dir(root) / "transcripts"


def ensure_memory_layout(root: Path) -> None:
    """Create `.claude/memory/` tree and seed index files if missing."""
    md = memory_dir(root)
    for d in (md, past_tasks_dir(root), spill_dir(root), transcripts_dir(root)):
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
- Older tool outputs may appear as `[micro-compacted: ...]` placeholders to keep
  the context budget under control. The full body is still on disk at the
  spill path mentioned in the placeholder — read it with `read_file` if needed.
- When a heavy summary compression runs, the **complete pre-compression
  transcript** is snapshotted to `{rel}transcripts/transcript_<ts>_<id>.jsonl`.
  You can reload any earlier turn from there.

**Stale data warning:** Memory files are user- and model-written hints. They may be outdated. Before critical edits or claims about the codebase, verify with `read_file` (and your other tools). Never treat memory as authoritative truth."""


def estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    return max(1, len(json.dumps(messages, ensure_ascii=False)) // CHARS_PER_TOKEN_EST)


def max_context_tokens() -> int:
    """
    Upper bound for our internal estimate of how many tokens a single API call
    can carry. Default 200k aligns with most modern long-context chat models
    (gpt-5, claude-3.x, gemini-1.5+). Override with ``AGENT_MAX_CONTEXT_TOKENS``
    if your model is smaller (or larger).
    """
    return int(os.getenv("AGENT_MAX_CONTEXT_TOKENS", "200000"))


def compress_trigger_ratio() -> float:
    """
    Trigger summary compression when our estimated token usage crosses this
    fraction of ``max_context_tokens``. Defaults to 0.7 (down from 0.9) so we
    have headroom for the next assistant turn instead of crashing on a 4xx
    after we're already at the cliff.
    """
    return float(os.getenv("AGENT_CONTEXT_COMPRESS_RATIO", "0.7"))


def preserve_recent_message_count() -> int:
    return int(os.getenv("AGENT_PRESERVE_RECENT_MSGS", "12"))


def emergency_compact_ratio() -> float:
    """
    If estimated tokens cross this fraction (default 0.95), do a non-LLM
    "drop oldest messages" emergency compaction before the next API call so
    we don't bounce off the upstream hard limit.
    """
    return float(os.getenv("AGENT_EMERGENCY_COMPACT_RATIO", "0.95"))


def tool_history_max_chars() -> int:
    return int(os.getenv("AGENT_TOOL_HISTORY_MAX_CHARS", "12000"))


def tool_history_preview_lines() -> int:
    return int(os.getenv("AGENT_TOOL_HISTORY_PREVIEW_LINES", "40"))


def keep_recent_tool_results() -> int:
    """How many recent role=tool messages to keep in full before micro-compacting."""
    return max(1, int(os.getenv("AGENT_KEEP_RECENT_TOOL_RESULTS", "6")))


def micro_compact_min_chars() -> int:
    """Skip messages already shorter than this when micro-compacting."""
    return max(80, int(os.getenv("AGENT_MICRO_COMPACT_MIN_CHARS", "400")))


_MICRO_COMPACT_MARK = "[micro-compacted: previous"


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


def _tool_name_for_call_id(messages: list[dict[str, Any]], call_id: str | None) -> str:
    """Look back through assistant messages to recover the tool name for a tool_call_id."""
    if not call_id:
        return "<unknown>"
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            if tc.get("id") == call_id:
                return (tc.get("function") or {}).get("name") or "<unknown>"
    return "<unknown>"


def _spill_path_hint(content: str) -> str | None:
    marker = "Full output saved to:"
    idx = content.find(marker)
    if idx < 0:
        return None
    rest = content[idx + len(marker):].strip().splitlines()
    return rest[0].strip() if rest else None


def micro_compact_inplace(
    messages: list[dict[str, Any]],
    *,
    emit: Callable[[str, Any], None] | None = None,
) -> int:
    """
    Replace the body of older ``role=tool`` messages with a one-line placeholder
    so they stop eating context. Returns the number of messages compacted.

    - Keeps the last ``keep_recent_tool_results()`` tool messages untouched.
    - Skips already-compacted messages and messages shorter than the threshold.
    - Preserves ``tool_call_id`` so the chat protocol stays valid.
    - Preserves the spill path (if any) so the model can still recover the full body.
    """
    keep = keep_recent_tool_results()
    threshold = micro_compact_min_chars()

    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if len(tool_indices) <= keep:
        return 0

    targets = tool_indices[:-keep]
    compacted = 0
    freed = 0
    for i in targets:
        msg = messages[i]
        content = msg.get("content") or ""
        if not isinstance(content, str):
            continue
        if content.startswith(_MICRO_COMPACT_MARK):
            continue
        if len(content) < threshold:
            continue
        tool_name = _tool_name_for_call_id(messages, msg.get("tool_call_id"))
        spill = _spill_path_hint(content)
        new_body = (
            f"{_MICRO_COMPACT_MARK} {tool_name} output, "
            f"{len(content)} chars elided to free context"
        )
        if spill:
            new_body += f"; full body still on disk at {spill}"
        new_body += "]"
        freed += len(content) - len(new_body)
        msg["content"] = new_body
        compacted += 1

    if compacted and emit is not None:
        emit(
            "micro_compact",
            {
                "replaced_count": compacted,
                "freed_chars": freed,
                "kept_recent": keep,
            },
        )
    return compacted


def _summarize_with_fallback(
    *,
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
) -> str:
    """
    Call ``client.chat.completions.create`` for summarization while tolerating
    the OpenAI parameter rename (newer reasoning models — gpt-5, o-series —
    require ``max_completion_tokens``; older chat models still take
    ``max_tokens``).

    Strategy: try ``max_completion_tokens`` first; on the specific
    ``unsupported_parameter`` 400 fall back to ``max_tokens``. Cache the
    winning shape on the client instance so we only pay the round-trip once.
    """
    cap = int(os.getenv("AGENT_SUMMARY_MAX_OUTPUT_TOKENS", "2000"))
    cache_attr = "_agent_summary_param_name"
    cached = getattr(client, cache_attr, None)

    def _try(param_name: str) -> str:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            param_name: cap,
        }
        # ``temperature`` is also rejected on some reasoning models; only set it
        # when we're already on the legacy parameter name (older chat models).
        if param_name == "max_tokens":
            kwargs["temperature"] = 0.2
        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0].message
        return (choice.content or "").strip()

    if cached:
        return _try(cached)

    try:
        out = _try("max_completion_tokens")
        try:
            setattr(client, cache_attr, "max_completion_tokens")
        except (AttributeError, TypeError):
            pass
        return out
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "max_completion_tokens" in msg or "max_tokens" in msg or "Unsupported parameter" in msg:
            out = _try("max_tokens")
            try:
                setattr(client, cache_attr, "max_tokens")
            except (AttributeError, TypeError):
                pass
            return out
        raise


def emergency_compact_inplace(
    messages: list[dict[str, Any]],
    *,
    emit: Callable[[str, Any], None] | None = None,
    target_ratio: float | None = None,
    force: bool = False,
) -> int:
    """
    Last-resort, **non-LLM** compaction: drop the oldest non-system / non-tail
    messages until the estimated token count is back under
    ``target_ratio * max_context_tokens()``. The dropped run is replaced with
    a single ``role=user`` "[emergency-compacted: N messages elided]" placeholder
    so the conversation flow stays valid (and tool_call/tool pairing in the
    surviving tail is preserved).

    This guarantees forward progress when the LLM summarizer is unavailable
    (network, 400, billing, …). Returns the number of messages elided.

    Pass ``force=True`` when the upstream API has already told us we're over the
    real limit (so even if our internal char/4 estimate is optimistic, we still
    need to evict). Without ``force``, we respect the estimator and skip when
    we look fine.
    """
    if len(messages) <= 4:
        return 0
    ratio = target_ratio if target_ratio is not None else compress_trigger_ratio()
    cap = int(max_context_tokens() * ratio)

    # Aggressive micro-compact first: drop ``keep`` to 2 so we squeeze every
    # tool result we can before throwing away whole turns.
    saved_env = os.environ.get("AGENT_KEEP_RECENT_TOOL_RESULTS")
    os.environ["AGENT_KEEP_RECENT_TOOL_RESULTS"] = "2"
    try:
        micro_compact_inplace(messages, emit=emit)
    finally:
        if saved_env is None:
            os.environ.pop("AGENT_KEEP_RECENT_TOOL_RESULTS", None)
        else:
            os.environ["AGENT_KEEP_RECENT_TOOL_RESULTS"] = saved_env

    if not force and estimate_message_tokens(messages) <= cap:
        return 0

    preserve_n = max(2, preserve_recent_message_count() // 2)
    if len(messages) <= 1 + preserve_n:
        return 0

    # Walk forward from the system message, dropping until either we're under
    # the cap or we've eaten everything that isn't the tail we promised to
    # preserve. Always end up with valid tool_call/tool pairing in the tail.
    tail = _strip_leading_tools(messages[-preserve_n:])
    if not tail:
        return 0

    elided_count = len(messages) - 1 - len(tail)
    if elided_count <= 0:
        return 0

    placeholder = {
        "role": "user",
        "content": (
            f"[emergency-compacted: {elided_count} early messages elided to free "
            "context after summary failed; spill / transcript history may still "
            "be readable via read_file]"
        ),
    }
    messages[:] = [messages[0], placeholder, *tail]
    if emit is not None:
        emit(
            "emergency_compact",
            {
                "elided_count": elided_count,
                "kept_tail": len(tail),
                "estimated_tokens_after": estimate_message_tokens(messages),
            },
        )
    return elided_count


def _save_transcript_snapshot(messages: list[dict[str, Any]], root: Path) -> Path:
    ensure_memory_layout(root)
    ts = int(time.time())
    path = transcripts_dir(root) / f"transcript_{ts}_{uuid.uuid4().hex[:6]}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg, ensure_ascii=False, default=str) + "\n")
    return path


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
    # Layer 1 (cheap): always try micro-compact first. Often pulls usage back
    # under the threshold without paying for a summarizer call.
    micro_compact_inplace(messages, emit=emit)

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

    # Layer 2 (expensive): persist a full transcript snapshot before we
    # collapse history into a summary. The model can `read_file` the path if
    # it ever needs the original turns back.
    transcript_path: Path | None = None
    try:
        transcript_path = _save_transcript_snapshot(messages, root=Path.cwd())
    except OSError as exc:
        if emit:
            emit("transcript_snapshot_error", {"error": str(exc)})

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

    summary = ""
    try:
        summary = _summarize_with_fallback(
            client=client,
            model=summary_model,
            messages=[
                {"role": "system", "content": summarizer_system},
                {"role": "user", "content": payload},
            ],
        )
    except Exception as exc:  # noqa: BLE001 — best-effort; do not break the agent
        if emit:
            emit(
                "context_compress_error",
                {"error": str(exc), "message": "Summary API failed; falling back to emergency_compact"},
            )
        # Don't crash — let emergency_compact below at least free room so the
        # next API call has a chance.
        emergency_compact_inplace(messages, emit=emit)
        return False

    if not summary:
        return False

    before_tokens = estimate_message_tokens(messages)
    summary_block = "[会话压缩 — 早期轮次摘要]\n" + summary
    if transcript_path is not None:
        try:
            rel = transcript_path.relative_to(Path.cwd())
            rel_s = str(rel).replace("\\", "/")
        except ValueError:
            rel_s = str(transcript_path)
        summary_block += (
            f"\n\n[Full pre-compression transcript saved to: {rel_s} — "
            "use read_file on that path if you need verbatim earlier turns.]"
        )
    messages[:] = [messages[0]] + [
        {"role": "user", "content": summary_block},
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
                "transcript_path": str(transcript_path) if transcript_path else None,
            },
        )
    return True
