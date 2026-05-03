"""
Isolated sub-agents (Agent-as-a-Tool): own messages, dedicated client, no SSE.

Returns a structured ``SubAgentResult`` with classified errors so the
orchestrator can react differently to retryable vs. fatal failures.

Error categories (``SubAgentResult.error_category``):

- ``config_error``    — credentials / model / base_url missing or invalid.
- ``api_error``       — upstream API call raised (network, auth, schema).
- ``timeout``         — sub_agent run exceeded ``options.timeout`` wall-clock.
- ``max_rounds``      — exhausted ``max_tool_rounds`` without a final answer.
- ``bad_finish``      — model returned with ``finish_reason`` other than
                        ``stop`` / ``tool_calls`` (e.g. ``length``).
- ``policy_denied``   — a tool call was rejected by the sub-agent ToolPolicy.
- ``unknown``         — anything else; the message carries the raw exception.
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

from context_memory import budget_tool_result_for_messages, ensure_memory_layout, maybe_compress_conversation
from tools_execution import execute_tool
from tools_registry import STANDARD_TOOLS, filter_tools_by_policy, tool_policy_from_env

# User fills these in .env (placeholders — no real secret in repo).
# Typical setup: LiteLLM or another OpenAI-compatible proxy that routes to Claude.
SUB_AGENT_API_KEY_PLACEHOLDER = os.getenv("SUB_AGENT_API_KEY", "")
SUB_AGENT_OPENAI_BASE_URL = os.getenv(
    "SUB_AGENT_OPENAI_BASE_URL",
    "https://replace-with-your-openai-compatible-base.example/v1",
)
SUB_AGENT_MODEL = os.getenv("SUB_AGENT_MODEL", "")

_SPILL_PATH_RE = re.compile(r"Full output saved to:\s*(\S+?)(?:\s|$)")
_POLICY_DENIED_RE = re.compile(
    r"is not permitted by the current tool policy", re.IGNORECASE
)


# -------- result types -----------------------------------------------------


ERROR_CATEGORIES = (
    "config_error",
    "api_error",
    "timeout",
    "max_rounds",
    "bad_finish",
    "policy_denied",
    "unknown",
)


@dataclass
class SubAgentOptions:
    """Configuration for a single sub-agent run (isolated from the orchestrator)."""

    model: str = field(default_factory=lambda: SUB_AGENT_MODEL)
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 120.0
    max_tool_rounds: int = 24
    root: Path | None = None
    label: str | None = None  # echoed back in result for correlation


@dataclass
class SubAgentResult:
    ok: bool
    final_text: str
    label: str | None = None
    error: str | None = None
    error_category: str | None = None
    token_usage: dict[str, int] | None = None
    spill_paths: list[str] = field(default_factory=list)
    rounds_used: int = 0
    duration_ms: int = 0
    tools_used: list[str] = field(default_factory=list)
    tool_errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = asdict(self)
        # Drop empty / None fields so the JSON the orchestrator sees stays terse.
        # Keep ``ok`` and ``final_text`` always (False / "" are meaningful).
        keep_always = {"ok", "final_text"}
        out: dict[str, Any] = {}
        for k, v in d.items():
            if k in keep_always:
                out[k] = v
                continue
            if v is None or v == [] or v == {}:
                continue
            # int 0 is meaningful for tool_errors=0; only skip the rounds_used /
            # duration_ms zeroes when both fields have no signal at all.
            if isinstance(v, int) and v == 0 and k in ("rounds_used", "duration_ms", "tool_errors"):
                continue
            out[k] = v
        return out


# -------- helpers ----------------------------------------------------------


def _merge_usage(acc: dict[str, int], usage: Any) -> None:
    if usage is None:
        return
    acc["prompt_tokens"] = acc.get("prompt_tokens", 0) + (getattr(usage, "prompt_tokens", None) or 0)
    acc["completion_tokens"] = acc.get("completion_tokens", 0) + (getattr(usage, "completion_tokens", None) or 0)
    acc["total_tokens"] = acc.get("total_tokens", 0) + (getattr(usage, "total_tokens", None) or 0)


def _collect_spill_paths(text: str, bucket: list[str]) -> None:
    for m in _SPILL_PATH_RE.finditer(text):
        p = m.group(1).strip()
        if p and p not in bucket:
            bucket.append(p)


def _classify_api_error(exc: BaseException) -> tuple[str, str]:
    """Map an exception during ``client.chat.completions.create`` to (category, message)."""
    name = type(exc).__name__.lower()
    text = str(exc)
    if "timeout" in name or "timeout" in text.lower():
        return "timeout", f"sub_agent upstream timeout: {text}"
    return "api_error", f"sub_agent API error: {text}"


def _build_assistant_msg(msg: Any) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": msg.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": getattr(tc, "type", None) or "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                },
            }
            for tc in (getattr(msg, "tool_calls", None) or [])
        ],
    }


# -------- main entry -------------------------------------------------------


def run_sub_agent(task: str, options: SubAgentOptions | None = None) -> SubAgentResult:
    """
    Run a standalone worker loop: fresh system + one user task, standard tools only, no SSE.

    Uses ``SUB_AGENT_API_KEY`` / ``SUB_AGENT_OPENAI_BASE_URL`` / ``SUB_AGENT_MODEL``
    (see module constants). Result is always a fully populated ``SubAgentResult``;
    on error, ``error_category`` indicates which retry strategy is appropriate.
    """
    opts = options or SubAgentOptions()
    started = time.monotonic()

    # Import after app startup to avoid circular import while ``agent_loop`` loads.
    from agent_loop import build_system_prompt, project_root

    root = opts.root or project_root()
    ensure_memory_layout(root)

    api_key = opts.api_key if opts.api_key is not None else SUB_AGENT_API_KEY_PLACEHOLDER
    if not api_key or not api_key.strip():
        return SubAgentResult(
            ok=False,
            final_text="",
            label=opts.label,
            error=(
                "SUB_AGENT_API_KEY is not set. Add it to your environment or .env "
                "(placeholder in code: set SUB_AGENT_API_KEY, SUB_AGENT_OPENAI_BASE_URL "
                "for an OpenAI-compatible Claude route)."
            ),
            error_category="config_error",
            duration_ms=int((time.monotonic() - started) * 1000),
        )
    if not opts.model or not str(opts.model).strip():
        return SubAgentResult(
            ok=False,
            final_text="",
            label=opts.label,
            error="SUB_AGENT_MODEL is not set; cannot dispatch sub-agent.",
            error_category="config_error",
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    base_url = opts.base_url or SUB_AGENT_OPENAI_BASE_URL
    sub_client = OpenAI(api_key=api_key.strip(), base_url=base_url.strip(), timeout=opts.timeout)

    sub_policy = tool_policy_from_env(sub_agent=True)
    sub_tools = filter_tools_by_policy(STANDARD_TOOLS, sub_policy)
    system_text, _meta = build_system_prompt(sub_tools, root=root)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": task.strip() or "(empty task)"},
    ]

    usage_acc: dict[str, int] = {}
    spill_paths: list[str] = []
    tools_used: list[str] = []
    tool_errors = 0
    model = opts.model
    rounds_used = 0

    def _build_result(**overrides: Any) -> SubAgentResult:
        return SubAgentResult(
            ok=overrides.get("ok", False),
            final_text=overrides.get("final_text", ""),
            label=opts.label,
            error=overrides.get("error"),
            error_category=overrides.get("error_category"),
            token_usage=usage_acc or None,
            spill_paths=spill_paths,
            rounds_used=rounds_used,
            duration_ms=int((time.monotonic() - started) * 1000),
            tools_used=tools_used,
            tool_errors=tool_errors,
        )

    for _ in range(opts.max_tool_rounds):
        if (time.monotonic() - started) * 1000 >= opts.timeout * 1000:
            return _build_result(
                error=f"sub_agent wall-clock timeout after {opts.timeout}s "
                f"(rounds_used={rounds_used})",
                error_category="timeout",
            )

        rounds_used += 1
        maybe_compress_conversation(messages, client=sub_client, model=model, emit=None)

        try:
            resp = sub_client.chat.completions.create(
                model=model,
                messages=messages,
                tools=sub_tools,
                tool_choice="auto",
                stream=False,
            )
        except Exception as exc:  # noqa: BLE001 — classify and surface
            cat, msg = _classify_api_error(exc)
            return _build_result(error=msg, error_category=cat)

        _merge_usage(usage_acc, getattr(resp, "usage", None))
        choice = resp.choices[0]
        msg = choice.message
        finish = choice.finish_reason

        tcalls = getattr(msg, "tool_calls", None) or []
        if not tcalls:
            text = (msg.content or "").strip()
            return _build_result(ok=True, final_text=text)

        messages.append(_build_assistant_msg(msg))

        for tc in tcalls:
            name = tc.function.name
            tools_used.append(name)
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                tool_errors += 1
                err_body = f"Invalid JSON arguments for {name}: {exc}"
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": err_body})
                continue
            raw = execute_tool(name, args, policy=sub_policy)
            if _POLICY_DENIED_RE.search(raw or ""):
                tool_errors += 1
            budgeted = budget_tool_result_for_messages(raw, tool_name=name, root=root)
            _collect_spill_paths(budgeted, spill_paths)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": budgeted})

        if finish not in (None, "tool_calls"):
            return _build_result(
                final_text=msg.content or "",
                error=f"sub_agent stopped with finish_reason={finish!r}",
                error_category="bad_finish",
            )

    return _build_result(
        error=f"sub_agent exceeded max_tool_rounds={opts.max_tool_rounds}",
        error_category="max_rounds",
    )


# -------- parallel + tool entry points -------------------------------------


def run_sub_agents_parallel_for_tool(tasks: list[dict[str, Any]]) -> str:
    """
    Run multiple ``run_sub_agent`` calls in parallel (thread pool). Each item must have key ``task``;
    optional ``label`` for correlation, optional ``model`` / ``timeout`` / ``max_tool_rounds``
    overrides.
    """
    if not tasks:
        return json.dumps({"ok": False, "error": "no tasks provided"}, ensure_ascii=False)

    def _work(item: dict[str, Any]) -> dict[str, Any]:
        task = str(item.get("task", "")).strip()
        label = item.get("label") or None
        opts = SubAgentOptions(
            label=label,
            model=str(item.get("model") or SUB_AGENT_MODEL),
            timeout=float(item.get("timeout") or 120.0),
            max_tool_rounds=int(item.get("max_tool_rounds") or 24),
        )
        return run_sub_agent(task, opts).to_dict()

    max_workers = min(8, len(tasks))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_work, t) for t in tasks]
        results = [f.result() for f in futs]

    overall_ok = all(r.get("ok") for r in results)
    return json.dumps(
        {"ok": overall_ok, "count": len(results), "results": results},
        ensure_ascii=False,
    )
