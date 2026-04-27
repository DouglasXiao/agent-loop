"""Shared OpenAI-style tool schemas for the main agent and isolated sub-agents."""

from __future__ import annotations

from typing import Any

STANDARD_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a UTF-8 text file and return its contents. "
                "For very large files, only the first 100 lines are returned and the result notes truncation—summarize and offer next steps. "
                "Use when the user references a path or when you need ground truth from the repo."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "The path to the file to read"},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write full text to a file; creates parent directories if needed. "
                "Use only when the user wants a file created or replaced. "
                "Prefer clear, complete content; warn before overwriting important files if the user did not ask to replace them."
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
                "Replace exactly one occurrence of old_string with new_string in a UTF-8 text file. "
                "Ideal for updating `.claude/memory/*.md` without rewriting the whole file. "
                "If old_string is missing or not unique, the tool fails—adjust the snippet or use write_file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file to edit"},
                    "old_string": {"type": "string", "description": "Literal substring to replace (must occur exactly once)"},
                    "new_string": {"type": "string", "description": "Replacement text"},
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
]
