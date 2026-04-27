"""
Isolated sub-agents (Agent-as-a-Tool): own messages, Claude Sonnet via dedicated client, no SSE.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

from context_memory import budget_tool_result_for_messages, ensure_memory_layout, maybe_compress_conversation
from tools_execution import execute_tool
from tools_registry import STANDARD_TOOLS

# User fills these in .env (placeholders — no real secret in repo).
# Typical setup: LiteLLM or another OpenAI-compatible proxy that routes to Claude.
SUB_AGENT_API_KEY_PLACEHOLDER = os.getenv("SUB_AGENT_API_KEY", "")
SUB_AGENT_OPENAI_BASE_URL = os.getenv(
    "SUB_AGENT_OPENAI_BASE_URL",
    "https://replace-with-your-openai-compatible-base.example/v1",
)
SUB_AGENT_MODEL = os.getenv("SUB_AGENT_MODEL", "")

_SPILL_PATH_RE = re.compile(r"Full output saved to:\s*(\S+?)(?:\s|$)")


@dataclass
class SubAgentOptions:
    """Configuration for a single sub-agent run (isolated from the orchestrator)."""

    model: str = field(default_factory=lambda: SUB_AGENT_MODEL)
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 120.0
    max_tool_rounds: int = 24
    root: Path | None = None


@dataclass
class SubAgentResult:
    ok: bool
    final_text: str
    error: str | None = None
    token_usage: dict[str, int] | None = None
    spill_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = asdict(self)
        return {k: v for k, v in d.items() if v not in (None, [], {})}


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


def run_sub_agent(task: str, options: SubAgentOptions | None = None) -> SubAgentResult:
    """
    Run a standalone worker loop: fresh system + one user task, standard tools only, no SSE.

    Uses ``SUB_AGENT_API_KEY`` / ``SUB_AGENT_OPENAI_BASE_URL`` / ``SUB_AGENT_MODEL`` (see module constants).
    """
    opts = options or SubAgentOptions()
    # Import after app startup to avoid circular import while ``agent_loop`` loads.
    from agent_loop import build_system_prompt, project_root

    root = opts.root or project_root()
    ensure_memory_layout(root)

    api_key = opts.api_key if opts.api_key is not None else SUB_AGENT_API_KEY_PLACEHOLDER
    if not api_key or not api_key.strip():
        return SubAgentResult(
            ok=False,
            final_text="",
            error=(
                "SUB_AGENT_API_KEY is not set. Add it to your environment or .env "
                "(placeholder in code: set SUB_AGENT_API_KEY, SUB_AGENT_OPENAI_BASE_URL for an OpenAI-compatible Claude route)."
            ),
        )

    base_url = opts.base_url or SUB_AGENT_OPENAI_BASE_URL
    sub_client = OpenAI(api_key=api_key.strip(), base_url=base_url.strip(), timeout=opts.timeout)

    system_text, _meta = build_system_prompt(STANDARD_TOOLS, root=root)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": task.strip() or "(empty task)"},
    ]

    usage_acc: dict[str, int] = {}
    spill_paths: list[str] = []
    model = opts.model

    for _ in range(opts.max_tool_rounds):
        maybe_compress_conversation(messages, client=sub_client, model=model, emit=None)

        try:
            resp = sub_client.chat.completions.create(
                model=model,
                messages=messages,
                tools=STANDARD_TOOLS,
                tool_choice="auto",
                stream=False,
            )
        except Exception as exc:  # noqa: BLE001
            return SubAgentResult(
                ok=False,
                final_text="",
                error=f"sub_agent API error: {exc!s}",
                token_usage=usage_acc or None,
                spill_paths=spill_paths,
            )

        _merge_usage(usage_acc, getattr(resp, "usage", None))
        choice = resp.choices[0]
        msg = choice.message
        finish = choice.finish_reason

        tcalls = getattr(msg, "tool_calls", None) or []
        if not tcalls:
            text = (msg.content or "").strip()
            return SubAgentResult(
                ok=True,
                final_text=text,
                token_usage=usage_acc or None,
                spill_paths=spill_paths,
            )

        assistant_msg: dict[str, Any] = {
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
                for tc in tcalls
            ],
        }
        messages.append(assistant_msg)

        for tc in tcalls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                err_body = f"Invalid JSON arguments for {name}: {exc}"
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": err_body})
                continue
            raw = execute_tool(name, args)
            budgeted = budget_tool_result_for_messages(raw, tool_name=name, root=root)
            _collect_spill_paths(budgeted, spill_paths)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": budgeted})

        if finish not in (None, "tool_calls"):
            return SubAgentResult(
                ok=False,
                final_text=msg.content or "",
                error=f"sub_agent stopped with finish_reason={finish!r}",
                token_usage=usage_acc or None,
                spill_paths=spill_paths,
            )

    return SubAgentResult(
        ok=False,
        final_text="",
        error=f"sub_agent exceeded max_tool_rounds={opts.max_tool_rounds}",
        token_usage=usage_acc or None,
        spill_paths=spill_paths,
    )


def run_sub_agents_parallel_for_tool(tasks: list[dict[str, Any]]) -> str:
    """
    Run multiple ``run_sub_agent`` calls in parallel (thread pool). Each item must have key ``task``;
    optional ``label`` for correlation in the JSON result.
    """
    if not tasks:
        return json.dumps({"ok": False, "error": "no tasks provided"}, ensure_ascii=False)

    def _work(item: dict[str, Any]) -> dict[str, Any]:
        task = str(item.get("task", "")).strip()
        label = item.get("label", "")
        res = run_sub_agent(task, SubAgentOptions())
        out = res.to_dict()
        out["label"] = label
        out["ok"] = res.ok
        return out

    max_workers = min(8, len(tasks))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_work, t) for t in tasks]
        results = [f.result() for f in futs]
    return json.dumps({"ok": True, "results": results}, ensure_ascii=False)
