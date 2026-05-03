import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from context_memory import (
    budget_tool_result_for_messages,
    emergency_compact_inplace,
    emergency_compact_ratio,
    ensure_memory_layout,
    estimate_message_tokens,
    max_context_tokens,
    maybe_compress_conversation,
    memory_prompt_section,
    micro_compact_inplace,
)
from bg_tasks import manager as bg_manager, render_drain_message
from skill_loader import discover_skills, render_skill_index, skills_dir
from task_graph import render_task_prompt_section, tasks_dir
from todo_manager import TodoState, todos_file
from tools_execution import execute_tool
from tools_registry import (
    STANDARD_TOOLS,
    TOOL_ACCESS,
    filter_tools_by_policy,
    tool_allowed,
    tool_policy_from_env,
)

# Risk classes safe to fan out within a single assistant turn (pure read or
# external-fetch with no local side effects). Mutating tools and delegating
# tools (sub-agents) stay strictly serial to avoid file-write races and to
# keep token-usage bookkeeping deterministic.
PARALLEL_TOOL_ACCESS_CLASSES = {"read", "network"}
MAX_TOOL_PARALLELISM = int(os.getenv("AGENT_MAX_TOOL_PARALLELISM", "4"))

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


if load_dotenv:
    load_dotenv()


def _ensure_utf8_stdio() -> None:
    """Best-effort UTF-8 for terminal so Chinese input/output displays correctly."""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


_emit_lock = threading.Lock()


def emit_sse(event: str, data: Any) -> None:
    """Print one Server-Sent Events block to the terminal (single-line data, UTF-8, flushed).

    Holds a global lock because parallel tool execution may emit from worker
    threads; without the lock, two events could interleave inside one SSE block
    and corrupt the stream.
    """
    if isinstance(data, (dict, list, str, int, float, bool)) or data is None:
        payload = json.dumps(data, ensure_ascii=False)
    else:
        payload = json.dumps(str(data), ensure_ascii=False)
    with _emit_lock:
        sys.stdout.write(f"event: {event}\ndata: {payload}\n\n")
        sys.stdout.flush()


def _build_main_client() -> tuple[OpenAI, str, str]:
    """
    Build the orchestrator's chat client + active model + provider label.

    Provider selection order:
      1. ``OPENROUTER_API_KEY`` set → OpenRouter (default base
         https://openrouter.ai/api/v1, default model openai/gpt-5.2).
      2. otherwise → plain OpenAI (or any OpenAI-compatible endpoint via
         ``OPENAI_BASE_URL``).

    Optional OpenRouter env vars:
      - ``OPENROUTER_BASE_URL`` (override the default base)
      - ``OPENROUTER_MODEL`` (overrides the default model; ``OPENAI_MODEL``
        is also honored as a secondary fallback for back-compat)
      - ``OPENROUTER_REFERER``  → sent as the ``HTTP-Referer`` header
      - ``OPENROUTER_TITLE``    → sent as the ``X-Title`` header
        (both headers are optional; OpenRouter uses them only for app rankings)
    """
    or_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if or_key:
        base = (os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").strip()
        model = (
            os.getenv("OPENROUTER_MODEL")
            or os.getenv("OPENAI_MODEL")
            or "openai/gpt-5.2"
        )
        headers: dict[str, str] = {}
        ref = (os.getenv("OPENROUTER_REFERER") or "").strip()
        if ref:
            headers["HTTP-Referer"] = ref
        title = (os.getenv("OPENROUTER_TITLE") or "").strip()
        if title:
            headers["X-Title"] = title
        kwargs: dict[str, Any] = {"api_key": or_key, "base_url": base}
        if headers:
            kwargs["default_headers"] = headers
        return OpenAI(**kwargs), model, "openrouter"

    # Fallback: original OpenAI / OpenAI-compatible path.
    return (
        OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        ),
        os.getenv("OPENAI_MODEL", "gpt-5"),
        "openai",
    )


client, MODEL, PROVIDER = _build_main_client()

CLAUDE_MD_FILENAME = "CLAUDE.md"


def project_root() -> Path:
    return Path(__file__).resolve().parent


def load_claude_md(root: Path | None = None) -> str | None:
    path = (root or project_root()) / CLAUDE_MD_FILENAME
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _render_current_todos(root: Path) -> str:
    """Inline a snapshot of the persisted todo list so the model re-orients fast."""
    try:
        state = TodoState.load(root)
    except Exception:  # noqa: BLE001 — defensive; never break startup on bad disk state
        return "(todo state unavailable)"
    if not state.items:
        return (
            "(no todos yet — for any multi-step task, plan with `todo_write` "
            "action=set before acting; storage: `" + str(todos_file(root)) + "`)"
        )
    return state.render_markdown()


# nag reminder: how many consecutive assistant rounds without a todo_write call
# before we inject a <reminder> into the next tool result.
TODO_NAG_AFTER_ROUNDS = int(os.getenv("AGENT_TODO_NAG_AFTER_ROUNDS", "3"))


def _tool_behavior_guidelines() -> str:
    return """- read_file: Fetch contents on demand; use offset/limit (1-based lines) for large files. Default whole-file reads cap at 100 lines unless you pass offset/limit (then up to 2000 lines per call). If a prior tool message points to a spill file under `.claude/memory/spill/`, read that path for full tool output.
- glob_files / grep_files: Prefer glob → grep(files_with_matches) → read_file for exploration; avoid loading huge trees in one read_file.
- write_file: Full create/replace; for small edits on existing files prefer edit_file. Never overwrite critical secrets without confirmation.
- edit_file: Surgical edits; default requires exactly one match—use replace_all for intentional multi replacements (e.g. renames).
- todo_write: For any non-trivial multi-step task, FIRST call todo_write(action="set", items=[...]) to plan the steps; mark exactly one item as in_progress while you work on it, then update its status to completed before moving on. State persists in `.claude/todos/current.json` across context compression. Skip todo_write only for one-shot questions or single-tool answers.
- task (graph): Use for goals that must outlive the current conversation, or that have explicit dependency edges (`blocked_by`). Stored under `.claude/tasks/`. Prefer `todo_write` for ephemeral plans, `task` for durable backlog items.
- list_skills / load_skill: Skills are domain workflows under `.claude/skills/<name>/SKILL.md`. The system prompt lists their names + short descriptions; call `load_skill(name=...)` to pull the full body before following a skill's instructions.
- bg_run / bg_check: Use bg_run for slow shell commands (tests, builds, installs) so the agent loop can keep thinking. Results are auto-injected as `<background-results>` before the next turn; you only need bg_check if you want to peek before the auto-drain.
- worktree: For parallel work streams that mustn't collide on the same files, create a git worktree (`worktree create name=foo task_id=N`); run commands inside it via `worktree run`; finish with `worktree remove name=foo complete_task=true` to delete the dir and mark the bound task done in the same step. Lifecycle is logged to `.worktrees/events.jsonl`.
- team: File-based mailbox for cross-session notes. `team register` once per teammate; `team send/broadcast` to drop messages; `team peek` to look without draining; `team read` to consume. Persisted under `.team/`.
- get_weather / web_fetch: Network tools—if disabled by policy or missing keys, explain to the user; never fabricate live data.
- run_terminal_cmd: Host shell; only available when AGENT_ALLOW_BASH=1. Chain `cd dir && cmd` when directory matters; avoid interactive commands.
- run_sub_agent: Isolated worker with its own context (no access to this chat). Pass a self-contained task; result is JSON with ok, final_text, error, token_usage. Use for heavy subtasks.
- run_sub_agents_parallel: Same as run_sub_agent but several tasks in parallel; input is a JSON array of {task, label?}; output JSON lists per-item results.
- Memory files under `.claude/memory/` may be stale hints—verify important facts with read_file on source code."""


def build_system_prompt(
    tools: list[dict[str, Any]],
    *,
    root: Path | None = None,
    extra_context: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Assemble a modular system prompt: personality, safety, tools + guidelines,
    optional CLAUDE.md, and dynamic environment lines.

    Returns (system_text, meta) where meta includes claude_md_loaded.
    """
    root = root or project_root()
    claude = load_claude_md(root)
    meta: dict[str, Any] = {"claude_md_loaded": claude is not None}

    tools_json = json.dumps(tools, ensure_ascii=False, indent=2)
    today = date.today().isoformat()
    cwd = str(Path.cwd().resolve())

    blocks: list[str] = [
        "[基础人格与行为规范]",
        "You are a helpful AI assistant. Reason step by step when the task is non-trivial. "
        "Answer in the same language the user uses when appropriate (e.g. Chinese for Chinese questions).",
        "",
        "[安全与防御规则]",
        "Do not execute harmful or destructive actions beyond what write_file allows for normal file edits. "
        "If a request is ambiguous, ask a short clarifying question before using tools. "
        "Do not invent file paths or API results; use tools to ground answers when needed.",
        "",
        "[惰性加载]",
        "Do not preload files or guess repository state. Fetch information only through explicit tool calls when the task requires it.",
        "",
        "[工具描述]",
        "<tools>",
        tools_json,
        "<tool_behavior_guidelines>",
        _tool_behavior_guidelines(),
        "</tool_behavior_guidelines>",
        "</tools>",
        "",
        "[项目上下文 (CLAUDE.md)]",
        claude if claude else "(No CLAUDE.md found at project root; use read_file on README or source as needed.)",
        "",
        memory_prompt_section(root),
        "",
        "[当前 TODO 列表（持久化）]",
        _render_current_todos(root),
        "",
        "[持久化任务图（.claude/tasks/）]",
        render_task_prompt_section(root),
        "",
        "[已安装的 skill（按需加载，调用 load_skill 拉全文）]",
        render_skill_index(discover_skills(root))
        + (
            f"\n_(skills directory: {skills_dir(root)})_"
            if discover_skills(root)
            else ""
        ),
        "",
        "[动态环境变量]",
        f"- Current working directory: {cwd}",
        f"- Project root (agent package): {root.resolve()}",
        f"- Current date: {today}",
    ]
    if extra_context and extra_context.strip():
        blocks.extend(["", "[附加说明]", extra_context.strip()])
    return "\n".join(blocks).strip(), meta


def tools_for_api() -> list[dict[str, Any]]:
    """Tool definitions sent to the chat API (filtered by ``AGENT_TOOL_MODE`` / ``AGENT_ALLOW_BASH``)."""
    policy = tool_policy_from_env(sub_agent=False)
    return filter_tools_by_policy(TOOLS, policy)


DELEGATION_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_sub_agent",
            "description": (
                "Run an isolated worker agent (separate short-lived chat, SUB_AGENT_* env credentials, "
                "typically Claude Sonnet via an OpenAI-compatible gateway). "
                "The worker only sees the task string you pass—not this conversation. "
                "Returns JSON: ok, final_text, label?, error?, error_category?, token_usage?, "
                "spill_paths?, rounds_used?, duration_ms?, tools_used?, tool_errors?. "
                "error_category ∈ {config_error, api_error, timeout, max_rounds, bad_finish, "
                "policy_denied, unknown}. Use for delegated exploration, drafting, or multi-step "
                "side work without bloating your main context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Standalone, self-contained instructions for the worker agent",
                    },
                    "label": {
                        "type": "string",
                        "description": "Optional correlation tag echoed back in the result",
                    },
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sub_agents_parallel",
            "description": (
                "Run multiple isolated worker agents in parallel (thread pool, up to 8 workers). "
                "Each worker has its own context and the same SUB_AGENT_* configuration as run_sub_agent. "
                "Returns JSON: { ok, count, results } where results align with the input order. "
                "Each result mirrors run_sub_agent (ok / final_text / label / error / "
                "error_category / token_usage / spill_paths / rounds_used / duration_ms / "
                "tools_used / tool_errors). Per-item overrides: model, timeout, max_tool_rounds."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "task": {"type": "string"},
                                "label": {
                                    "type": "string",
                                    "description": "Optional string echoed back in that item's result",
                                },
                                "model": {"type": "string", "description": "Override SUB_AGENT_MODEL"},
                                "timeout": {"type": "number", "description": "Wall-clock seconds"},
                                "max_tool_rounds": {"type": "integer"},
                            },
                            "required": ["task"],
                        },
                        "minItems": 1,
                        "description": "List of { task, label?, model?, timeout?, max_tool_rounds? } objects",
                    },
                },
                "required": ["tasks"],
            },
        },
    },
]

TOOLS: list[dict[str, Any]] = STANDARD_TOOLS + DELEGATION_TOOLS


def orchestrator_execute_tool(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Orchestrator-only tools + builtin file/weather tools (sub-agents never see this path)."""
    policy = tool_policy_from_env(sub_agent=False)
    if tool_name in TOOL_ACCESS and not tool_allowed(tool_name, policy):
        err = (
            f"Tool {tool_name!r} is not permitted by the current tool policy "
            f"(AGENT_TOOL_MODE / AGENT_ALLOW_BASH)."
        )
        if tool_name in ("run_sub_agent", "run_sub_agents_parallel"):
            return json.dumps({"ok": False, "error": err}, ensure_ascii=False)
        return err

    if tool_name == "run_sub_agent":
        from sub_agent import SubAgentOptions, run_sub_agent

        task = str(tool_input.get("task", "")).strip()
        label = tool_input.get("label")
        opts = SubAgentOptions(label=str(label) if label else None)
        res = run_sub_agent(task, opts)
        return json.dumps(res.to_dict(), ensure_ascii=False)
    if tool_name == "run_sub_agents_parallel":
        from sub_agent import run_sub_agents_parallel_for_tool

        tasks = tool_input.get("tasks")
        if not isinstance(tasks, list):
            return json.dumps({"ok": False, "error": "tasks must be a JSON array"}, ensure_ascii=False)
        return run_sub_agents_parallel_for_tool(tasks)
    return execute_tool(tool_name, tool_input, policy=policy)


_CONTEXT_LIMIT_HINTS = (
    "context_length_exceeded",
    "maximum context length",
    "input tokens exceed",
    "too many tokens",
    "reduce the length",
)


def _is_context_limit_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(hint in msg for hint in _CONTEXT_LIMIT_HINTS)


def _open_stream(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        stream=True,
    )


def _stream_one_completion(
    messages: list[dict[str, Any]],
    *,
    client: OpenAI,
    model: str,
    tools: list[dict[str, Any]],
    emit: Callable[[str, Any], None],
) -> tuple[dict[str, Any], str | None]:
    """
    One model call with streaming. Returns assistant message dict and finish_reason.

    Recovery contract:
    - On upstream ``context_length_exceeded`` (4xx), run an emergency in-place
      compaction (drop oldest history, no LLM needed) and retry **once**.
      Emit ``upstream_context_overflow`` so callers can observe the recovery.
    - Any other 4xx/5xx is re-raised so the orchestrator can decide what to do.

    Emits SSE: thinking, content_delta, finish (tool 完整参数在 tool_call 事件中输出).
    """
    emit("thinking", {"model": model, "message": "模型推理中…"})

    try:
        stream = _open_stream(client, model=model, messages=messages, tools=tools)
    except Exception as exc:  # noqa: BLE001
        if _is_context_limit_error(exc):
            emit(
                "upstream_context_overflow",
                {
                    "error": str(exc),
                    "estimated_tokens": estimate_message_tokens(messages),
                    "action": "emergency_compact_and_retry",
                },
            )
            # ``force=True``: trust the upstream's verdict over our local
            # char/4 estimate, which can be wildly off for non-Latin text.
            elided = emergency_compact_inplace(messages, emit=emit, target_ratio=0.4, force=True)
            if elided <= 0:
                # We can't free room (already at minimum) — surface upstream error.
                raise
            stream = _open_stream(client, model=model, messages=messages, tools=tools)
        else:
            raise

    content_parts: list[str] = []
    tool_calls_acc: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None

    for chunk in stream:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if choice.finish_reason:
            finish_reason = choice.finish_reason

        delta = choice.delta
        if delta is None:
            continue

        if delta.content:
            emit("content_delta", {"chunk": delta.content})
            content_parts.append(delta.content)

        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index if tc.index is not None else 0
                if idx not in tool_calls_acc:
                    tool_calls_acc[idx] = {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                if tc.id:
                    tool_calls_acc[idx]["id"] = tc.id
                fn = tc.function
                if fn is not None:
                    if fn.name:
                        tool_calls_acc[idx]["function"]["name"] += fn.name
                    if fn.arguments:
                        tool_calls_acc[idx]["function"]["arguments"] += fn.arguments

    text = "".join(content_parts)
    content: str | None = text if text else None

    tool_calls_list: list[dict[str, Any]] | None = None
    if tool_calls_acc:
        tool_calls_list = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]

    assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls_list:
        assistant_msg["tool_calls"] = tool_calls_list

    emit(
        "finish",
        {
            "reason": finish_reason,
            "has_tool_calls": bool(tool_calls_list),
            "content_length": len(text),
        },
    )

    return assistant_msg, finish_reason


def _run_one_tool_call(
    tc: dict[str, Any],
    *,
    root: Path,
    emit: Callable[[str, Any], None],
) -> dict[str, Any]:
    """
    Execute a single ``tool_call`` end-to-end: parse args, dispatch, budget result, emit SSE.

    Returns the ``role=tool`` history message for ``messages.extend(...)``. Errors are
    captured into the ``content`` field so the model can recover instead of crashing the loop.
    """
    tool_name = (tc.get("function") or {}).get("name") or "<unknown>"
    raw_args = (tc.get("function") or {}).get("arguments") or "{}"
    tool_call_id = tc.get("id")
    started = time.monotonic()
    try:
        tool_args = json.loads(raw_args)
    except json.JSONDecodeError as e:
        err = f"Invalid JSON arguments for {tool_name}: {e}; raw={raw_args!r}"
        emit("tool_error", {"tool_call_id": tool_call_id, "tool": tool_name, "error": err})
        return {"tool_call_id": tool_call_id, "role": "tool", "content": err}

    emit(
        "tool_call",
        {"tool_call_id": tool_call_id, "name": tool_name, "arguments": tool_args},
    )
    try:
        function_response = orchestrator_execute_tool(
            tool_name=tool_name,
            tool_input=tool_args,
        )
    except Exception as exc:  # noqa: BLE001 — never let one bad tool kill the loop
        err = f"Unhandled exception in {tool_name}: {exc!s}"
        emit(
            "tool_error",
            {"tool_call_id": tool_call_id, "tool": tool_name, "error": err},
        )
        return {"tool_call_id": tool_call_id, "role": "tool", "content": err}

    content_for_history = budget_tool_result_for_messages(
        function_response,
        tool_name=tool_name,
        root=root,
    )
    duration_ms = int((time.monotonic() - started) * 1000)
    tr_payload: dict[str, Any] = {
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "result": content_for_history,
        "duration_ms": duration_ms,
    }
    if len(content_for_history) < len(function_response):
        tr_payload["history_budgeted_from_chars"] = len(function_response)
    emit("tool_result", tr_payload)
    return {"tool_call_id": tool_call_id, "role": "tool", "content": content_for_history}


def _run_all_tool_calls(
    tool_calls: list[dict[str, Any]],
    *,
    root: Path,
    emit: Callable[[str, Any], None],
) -> list[dict[str, Any]]:
    """
    Execute one assistant turn's worth of ``tool_calls``.

    Read- and network-class calls fan out across a thread pool (bounded by
    ``AGENT_MAX_TOOL_PARALLELISM``). Mutating, system, and delegating calls
    stay serial so they cannot race on the workspace, on the persisted todo
    list, or on sub-agent token bookkeeping.

    Output order **always** matches input order — required by the OpenAI
    chat protocol so each ``role=tool`` message lines up with its
    ``tool_call_id`` from the previous assistant turn.
    """
    if not tool_calls:
        return []

    def parallelizable(tc: dict[str, Any]) -> bool:
        name = (tc.get("function") or {}).get("name") or ""
        return TOOL_ACCESS.get(name) in PARALLEL_TOOL_ACCESS_CLASSES

    results: list[dict[str, Any] | None] = [None] * len(tool_calls)

    # Group consecutive parallelizable calls so the global ordering of
    # observable side effects (writes, sub-agent spawns) stays the same as
    # the model intended.
    i = 0
    while i < len(tool_calls):
        if parallelizable(tool_calls[i]):
            j = i
            while j < len(tool_calls) and parallelizable(tool_calls[j]):
                j += 1
            batch = list(range(i, j))
            if len(batch) == 1:
                results[batch[0]] = _run_one_tool_call(tool_calls[batch[0]], root=root, emit=emit)
            else:
                emit("tools_parallel", {"count": len(batch), "indices": batch})
                with ThreadPoolExecutor(max_workers=min(MAX_TOOL_PARALLELISM, len(batch))) as pool:
                    futs = {
                        pool.submit(_run_one_tool_call, tool_calls[k], root=root, emit=emit): k
                        for k in batch
                    }
                    for fut in as_completed(futs):
                        k = futs[fut]
                        results[k] = fut.result()
            i = j
        else:
            results[i] = _run_one_tool_call(tool_calls[i], root=root, emit=emit)
            i += 1

    return [r for r in results if r is not None]


def init_conversation_messages(root: Path | None = None) -> list[dict[str, Any]]:
    """Start a new chat session: system prompt only. Emits ``system_prompt`` SSE once."""
    root = root or project_root()
    ensure_memory_layout(root)
    system_text, prompt_meta = build_system_prompt(tools_for_api(), root=root)
    emit_sse(
        "system_prompt",
        {
            **prompt_meta,
            "length": len(system_text),
            "project_root": str(root.resolve()),
            "provider": PROVIDER,
            "model": MODEL,
        },
    )
    return [{"role": "system", "content": system_text}]


def core_agent_loop_streaming(messages: list[dict[str, Any]]) -> str | None:
    """
    Run one user turn in place: ``messages`` must already end with the new user message.

    Appends assistant messages (with optional ``tool_calls``), then ``tool`` role messages,
    until the model returns a final assistant reply without tools. The same ``messages`` list
    is reused across turns in interactive mode, so history grows monotonically.
    """
    if not messages:
        emit_sse("error", {"message": "core_agent_loop_streaming: empty messages"})
        return None
    last = messages[-1]
    if last.get("role") != "user":
        emit_sse(
            "error",
            {
                "message": "core_agent_loop_streaming: last message must be role=user",
                "got_role": last.get("role"),
            },
        )
        return None
    user_text = last.get("content") if isinstance(last.get("content"), str) else ""
    emit_sse("user", {"text": user_text})

    root = project_root()
    rounds_since_todo = 0
    while True:
        # Drain any background tasks that finished since the last LLM call so
        # the model can react in this turn instead of next.
        notifs = bg_manager().drain_notifications()
        if notifs:
            drained = render_drain_message(notifs)
            if drained is not None:
                messages.append(drained)
                emit_sse("bg_drain", {"count": len(notifs)})

        # Cheap, silent: shrink stale tool_results before every model call so the
        # context budget grows linearly even on long sessions.
        micro_compact_inplace(messages, emit=emit_sse)
        # Expensive (LLM summary) only when the budget threshold is crossed.
        maybe_compress_conversation(messages, client=client, model=MODEL, emit=emit_sse)
        # Pre-flight: if we're still alarmingly close to the upstream limit
        # (e.g. summary failed silently), do a non-LLM emergency compact so
        # the next API call doesn't 4xx-out.
        est = estimate_message_tokens(messages)
        if est >= max_context_tokens() * emergency_compact_ratio():
            emergency_compact_inplace(messages, emit=emit_sse)

        try:
            assistant_msg, finish_reason = _stream_one_completion(
                messages,
                client=client,
                model=MODEL,
                tools=tools_for_api(),
                emit=emit_sse,
            )
        except Exception as exc:  # noqa: BLE001 — don't traceback out of the loop
            emit_sse(
                "upstream_error",
                {
                    "error": str(exc),
                    "type": type(exc).__name__,
                    "message": "模型调用失败；本轮中止。建议：检查 API key/额度，或减少 max context、调整 AGENT_CONTEXT_COMPRESS_RATIO。",
                    "estimated_tokens": estimate_message_tokens(messages),
                },
            )
            return None
        tool_calls = assistant_msg.get("tool_calls")

        if not tool_calls:
            emit_sse("final", {"text": assistant_msg.get("content")})
            return assistant_msg.get("content")

        messages.append({k: v for k, v in assistant_msg.items() if v is not None})

        # Track whether this assistant turn touched the planning tool;
        # used downstream to inject a <reminder> on long stretches without planning.
        called_todo_this_turn = any(
            (tc.get("function") or {}).get("name") == "todo_write" for tc in tool_calls
        )
        if called_todo_this_turn:
            rounds_since_todo = 0
        else:
            rounds_since_todo += 1

        tool_outputs = _run_all_tool_calls(tool_calls, root=root, emit=emit_sse)

        messages.extend(tool_outputs)
        emit_sse("tools_done", {"count": len(tool_outputs), "message": "工具已执行，继续推理…"})

        # nag reminder: if the model has gone too many rounds without updating the
        # plan, glue a one-line <reminder> onto the most recent tool result so it
        # stays in the immediate attention window.
        if (
            tool_outputs
            and rounds_since_todo >= TODO_NAG_AFTER_ROUNDS
        ):
            last_tool = tool_outputs[-1]
            reminder = (
                "\n\n<reminder>You haven't updated `todo_write` for "
                f"{rounds_since_todo} rounds. If this is a multi-step task, "
                "call `todo_write` (set/update/complete) to keep the plan current. "
                "Skip only if the task is genuinely a one-shot.</reminder>"
            )
            last_tool["content"] = (last_tool.get("content") or "") + reminder
            emit_sse("todo_nag", {"rounds_since_todo": rounds_since_todo})
            rounds_since_todo = 0  # one nag per stretch, not every round

        if finish_reason not in (None, "tool_calls"):
            emit_sse("abort", {"reason": finish_reason, "message": "非正常结束"})
            return assistant_msg.get("content")


def main() -> None:
    _ensure_utf8_stdio()

    parser = argparse.ArgumentParser(
        description="Agent loop：支持中文；流式 SSE 风格输出；无参数时进入交互输入。",
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="单次问题（提供则运行一次后退出；省略则进入交互循环）",
    )
    args = parser.parse_args()

    one_shot = " ".join(args.prompt).strip()
    if one_shot:
        session = init_conversation_messages()
        session.append({"role": "user", "content": one_shot})
        core_agent_loop_streaming(session)
        return

    print("交互模式：多轮对话；上下文会保留。输入 exit / quit / 退出 结束。\n")
    session_messages = init_conversation_messages()
    while True:
        try:
            user_input = input("你: ").strip()
        except EOFError:
            emit_sse("session", {"status": "eof"})
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q") or user_input in ("退出",):
            emit_sse("session", {"status": "goodbye"})
            break
        session_messages.append({"role": "user", "content": user_input})
        core_agent_loop_streaming(session_messages)


if __name__ == "__main__":
    main()
