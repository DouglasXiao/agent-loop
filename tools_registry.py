"""Shared OpenAI-style tool schemas for the main agent and isolated sub-agents."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

ToolAccess = Literal["read", "mutate", "network", "system", "delegate"]

# Aligns with Claude Managed Agents / agent_toolset: read-class tools are "safe" defaults;
# mutate/network/system/delegate map to permission gates in this repo (env + filtering).
TOOL_ACCESS: dict[str, ToolAccess] = {
    "read_file": "read",
    "glob_files": "read",
    "grep_files": "read",
    "write_file": "mutate",
    "edit_file": "mutate",
    "get_weather": "network",
    "web_fetch": "network",
    "run_terminal_cmd": "system",
    "run_sub_agent": "delegate",
    "run_sub_agents_parallel": "delegate",
}


@dataclass(frozen=True)
class ToolPolicy:
    """Which risk classes may be exposed to the model and executed."""

    allow_read: bool = True
    allow_mutate: bool = True
    allow_network: bool = True
    allow_system: bool = False
    allow_delegate: bool = True


def _truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def tool_policy_from_env(*, sub_agent: bool) -> ToolPolicy:
    """
    Build a policy from environment.

    ``AGENT_TOOL_MODE`` (orchestrator): ``full`` | ``safe_only`` | ``no_network`` | ``no_mutate``.
    ``SUB_AGENT_TOOL_MODE``: same values; if unset, falls back to ``AGENT_TOOL_MODE``.

    ``AGENT_ALLOW_BASH=1``: enables ``run_terminal_cmd`` (system) when mode allows it.
    """
    if sub_agent:
        sub_raw = os.environ.get("SUB_AGENT_TOOL_MODE")
        if sub_raw is None or not str(sub_raw).strip():
            mode = os.getenv("AGENT_TOOL_MODE", "full").strip().lower()
        else:
            mode = str(sub_raw).strip().lower()
    else:
        mode = os.getenv("AGENT_TOOL_MODE", "full").strip().lower()

    if mode == "safe_only":
        return ToolPolicy(
            allow_read=True,
            allow_mutate=False,
            allow_network=False,
            allow_system=False,
            allow_delegate=False if sub_agent else True,
        )
    if mode == "no_network":
        return ToolPolicy(
            allow_read=True,
            allow_mutate=True,
            allow_network=False,
            allow_system=_truthy("AGENT_ALLOW_BASH"),
            allow_delegate=not sub_agent,
        )
    if mode == "no_mutate":
        return ToolPolicy(
            allow_read=True,
            allow_mutate=False,
            allow_network=True,
            allow_system=_truthy("AGENT_ALLOW_BASH"),
            allow_delegate=not sub_agent,
        )
    # full (default)
    return ToolPolicy(
        allow_read=True,
        allow_mutate=True,
        allow_network=True,
        allow_system=_truthy("AGENT_ALLOW_BASH"),
        allow_delegate=not sub_agent,
    )


def _access_allowed(policy: ToolPolicy, access: ToolAccess) -> bool:
    if access == "read":
        return policy.allow_read
    if access == "mutate":
        return policy.allow_mutate
    if access == "network":
        return policy.allow_network
    if access == "system":
        return policy.allow_system
    if access == "delegate":
        return policy.allow_delegate
    return False


def filter_tools_by_policy(tools: list[dict[str, Any]], policy: ToolPolicy) -> list[dict[str, Any]]:
    """Drop tool schemas whose risk class is disabled by ``policy``."""
    out: list[dict[str, Any]] = []
    for entry in tools:
        fn = entry.get("function") if isinstance(entry, dict) else None
        name = fn.get("name") if isinstance(fn, dict) else None
        if not isinstance(name, str) or name not in TOOL_ACCESS:
            continue
        access = TOOL_ACCESS[name]
        if _access_allowed(policy, access):
            out.append(entry)
    return out


def tool_allowed(name: str, policy: ToolPolicy) -> bool:
    """Whether ``name`` is permitted for registration and execution under ``policy``."""
    access = TOOL_ACCESS.get(name)
    if access is None:
        return False
    return _access_allowed(policy, access)


STANDARD_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a UTF-8 text file and return contents. Prefer absolute paths when possible. "
                "Optional offset/limit (1-based lines) for large files—read a header slice first, then narrow. "
                "When limit is omitted, very large files still cap at a server max; the reply notes truncation. "
                "Output uses optional line-number prefixes (cat -n style). "
                "Use when you need ground truth from the repo."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file to read"},
                    "offset": {
                        "type": "integer",
                        "description": "1-based line number to start reading from (optional)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to return from offset (optional)",
                    },
                    "line_numbers": {
                        "type": "boolean",
                        "description": "If true (default), prefix each line with line number and tab",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob_files",
            "description": (
                "List file paths matching a glob pattern under a directory (default: cwd). "
                "Use for repo exploration—lighter than shell find. Examples: **/*.py, src/**/*.ts. "
                "Returns a bounded list; refine the pattern if truncated."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern, e.g. **/*.py or src/**/*.test.js",
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional root directory to search (default: current working directory)",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_files",
            "description": (
                "Search file contents with a regular expression (ripgrep when available, else Python scan). "
                "Use to find symbols, errors, or references. Prefer bounded output_mode=files_with_matches first, "
                "then read_file on hits."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression (Rust regex if using rg)"},
                    "path": {
                        "type": "string",
                        "description": "Directory to search (default: cwd)",
                    },
                    "glob": {
                        "type": "string",
                        "description": "Optional glob filter for paths, e.g. *.ts",
                    },
                    "output_mode": {
                        "type": "string",
                        "enum": ["content", "files_with_matches", "count"],
                        "description": "content: matching lines; files_with_matches: paths only; count: per-file counts",
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Lines of context before/after each match (content mode only, default 0)",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write full text to a file; creates parent directories if needed. "
                "Heavy operation: for small edits on existing files prefer edit_file. "
                "Use only when the user wants a file created or fully replaced."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file to write"},
                    "content": {"type": "string", "description": "Full file content"},
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace occurrences of old_string with new_string in a UTF-8 text file. "
                "Default: exactly one match (safest). Set replace_all true to rename across the file. "
                "Ideal for surgical edits; if match count is wrong, narrow old_string or use write_file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file to edit"},
                    "old_string": {"type": "string", "description": "Literal substring or pattern context to replace"},
                    "new_string": {"type": "string", "description": "Replacement text"},
                    "replace_all": {
                        "type": "boolean",
                        "description": "If true, replace every occurrence; if false/omitted, require exactly one match",
                    },
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get current weather for a global city via OpenWeather (host must have OPENWEATHER_API_KEY). "
                "City may include country code, e.g. London,GB or Tokyo,JP. "
                "If the tool returns a missing-key or location error, explain it to the user—do not fabricate weather."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The global city name, optionally with country code",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": (
                "Fetch a public HTTP(S) URL and return decoded text (truncated if very large). "
                "Use after you have a real URL (e.g. from docs)—do not guess URLs. "
                "The prompt field reminds you what to extract from the page in your reply; execution returns raw text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL including https://"},
                    "prompt": {
                        "type": "string",
                        "description": "What you intend to extract or summarize from this page (echoed in output header)",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_terminal_cmd",
            "description": (
                "Run a shell command on the host (blocking, one-shot). Requires AGENT_ALLOW_BASH=1. "
                "Use for tests, builds, git status—combine cd and command with && when a specific directory matters. "
                "Avoid interactive prompts; prefer non-interactive flags (-y). High risk: do not use for destructive "
                "operations unless the user explicitly asked."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"},
                    "description": {
                        "type": "string",
                        "description": "Short human-readable intent for logs",
                    },
                    "timeout_sec": {
                        "type": "number",
                        "description": "Timeout in seconds (default 120, max 300)",
                    },
                },
                "required": ["command"],
            },
        },
    },
]
