import argparse
import json
import os
import sys
from typing import Any

from openai import OpenAI

from tools_execution import execute_tool

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


def emit_sse(event: str, data: Any) -> None:
    """Print one Server-Sent Events block to the terminal (single-line data, UTF-8, flushed)."""
    if isinstance(data, (dict, list, str, int, float, bool)) or data is None:
        payload = json.dumps(data, ensure_ascii=False)
    else:
        payload = json.dumps(str(data), ensure_ascii=False)
    sys.stdout.write(f"event: {event}\ndata: {payload}\n\n")
    sys.stdout.flush()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))

MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file and return the contents",
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
            "description": "Write text to a file. Creates the file and parent directories if needed.",
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
            "name": "get_weather",
            "description": "Get current weather for a global city. City can include country code, for example: London,GB or Tokyo,JP.",
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


def _stream_one_completion(
    messages: list[dict[str, Any]],
) -> tuple[dict[str, Any], str | None]:
    """
    One model call with streaming. Returns assistant message dict and finish_reason.
    Emits SSE: thinking, content_delta, finish (tool 完整参数在 tool_call 事件中输出).
    """
    emit_sse("thinking", {"model": MODEL, "message": "模型推理中…"})

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        stream=True,
    )

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
            emit_sse("content_delta", {"chunk": delta.content})
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

    emit_sse(
        "finish",
        {
            "reason": finish_reason,
            "has_tool_calls": bool(tool_calls_list),
            "content_length": len(text),
        },
    )

    return assistant_msg, finish_reason


def core_agent_loop_streaming(user_input: str) -> str | None:
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_input}]
    emit_sse("user", {"text": user_input})

    while True:
        assistant_msg, finish_reason = _stream_one_completion(messages)
        tool_calls = assistant_msg.get("tool_calls")

        if not tool_calls:
            emit_sse("final", {"text": assistant_msg.get("content")})
            return assistant_msg.get("content")

        messages.append({k: v for k, v in assistant_msg.items() if v is not None})

        tool_outputs: list[dict[str, Any]] = []
        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            raw_args = tc["function"]["arguments"] or "{}"
            try:
                tool_args = json.loads(raw_args)
            except json.JSONDecodeError as e:
                tool_args = {}
                err = f"Invalid JSON arguments for {tool_name}: {e}; raw={raw_args!r}"
                emit_sse("tool_error", {"tool": tool_name, "error": err})
                function_response = err
            else:
                emit_sse(
                    "tool_call",
                    {"tool_call_id": tc.get("id"), "name": tool_name, "arguments": tool_args},
                )
                function_response = execute_tool(
                    tool_name=tool_name,
                    tool_input=tool_args,
                )
                emit_sse(
                    "tool_result",
                    {"tool_call_id": tc.get("id"), "name": tool_name, "result": function_response},
                )

            tool_outputs.append(
                {
                    "tool_call_id": tc["id"],
                    "role": "tool",
                    "content": function_response,
                }
            )

        messages.extend(tool_outputs)
        emit_sse("tools_done", {"count": len(tool_outputs), "message": "工具已执行，继续推理…"})

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
        core_agent_loop_streaming(one_shot)
        return

    print("交互模式：输入问题后回车；支持中文。输入 exit / quit / 退出 结束。\n")
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
        core_agent_loop_streaming(user_input)


if __name__ == "__main__":
    main()
