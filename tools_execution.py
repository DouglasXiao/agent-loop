import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from tools_registry import ToolPolicy, tool_allowed

OPENWEATHER_GEO_URL = "https://api.openweathermap.org/geo/1.0/direct"
OPENWEATHER_CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"

READ_FILE_MAX_LINES = 100
READ_FILE_HARD_MAX_LINES = 2000
GLOB_MAX_RESULTS = 500
GREP_MAX_LINES = 200
GREP_MAX_FILES_LIST = 80
WEB_FETCH_MAX_CHARS = 400_000


def _resolve_path(file_path: str) -> Path:
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def read_file(
    file_path: str,
    *,
    offset: int | None = None,
    limit: int | None = None,
    line_numbers: bool | None = None,
) -> str:
    path = _resolve_path(file_path)
    if not path.exists():
        return f"File not found: {path}"
    if not path.is_file():
        return f"Path is not a file: {path}"

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    total = len(lines)
    numbering = True if line_numbers is None else bool(line_numbers)

    def format_line(line_no: int, content: str) -> str:
        if numbering:
            return f"{line_no}\t{content}"
        return content

    # Whole-file path: keep the legacy 100-line cap unless offset/limit are used.
    if offset is None and limit is None:
        if total <= READ_FILE_MAX_LINES:
            return "\n".join(format_line(i + 1, lines[i]) for i in range(total))
        head = "\n".join(format_line(i + 1, lines[i]) for i in range(READ_FILE_MAX_LINES))
        return (
            f"{head}\n\n"
            f"[truncated: showing first {READ_FILE_MAX_LINES} of {total} lines; "
            f"use offset/limit to read more]"
        )

    off = offset if offset is not None else 1
    if off < 1:
        return "read_file: offset must be >= 1 (1-based line numbers)."
    start_idx = off - 1
    if start_idx >= total:
        return f"read_file: offset {off} is past end of file ({total} lines)."

    if limit is not None and limit < 1:
        return "read_file: limit must be >= 1."
    max_lines = READ_FILE_HARD_MAX_LINES if limit is None else min(limit, READ_FILE_HARD_MAX_LINES)

    end_idx = min(start_idx + max_lines, total)
    body = "\n".join(format_line(i + 1, lines[i]) for i in range(start_idx, end_idx))
    if end_idx < total:
        return (
            f"{body}\n\n"
            f"[truncated: lines {start_idx + 1}-{end_idx} of {total}; "
            f"increase limit or adjust offset to continue]"
        )
    return body


def write_file(file_path: str, content: str) -> str:
    path = _resolve_path(file_path)
    if path.exists() and not path.is_file():
        return f"Path is not a file: {path}"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"File written: {path}"


def edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
    *,
    replace_all: bool | None = None,
) -> str:
    """Replace ``old_string`` with ``new_string`` (once or all occurrences)."""
    path = _resolve_path(file_path)
    if not path.exists():
        return f"File not found: {path}"
    if not path.is_file():
        return f"Path is not a file: {path}"

    text = path.read_text(encoding="utf-8")
    if old_string not in text:
        return "edit_file: old_string not found in file (no changes written)."
    count = text.count(old_string)
    do_all = bool(replace_all)
    if not do_all and count != 1:
        return (
            f"edit_file: old_string must match exactly once (found {count} occurrences); "
            "set replace_all=true or narrow old_string."
        )
    if do_all:
        new_text = text.replace(old_string, new_string)
        path.write_text(new_text, encoding="utf-8")
        return f"File updated ({count} replacements, replace_all): {path}"
    path.write_text(text.replace(old_string, new_string, 1), encoding="utf-8")
    return f"File updated (single replacement): {path}"


def _glob_base(search_path: str | None) -> Path:
    raw = (search_path or ".").strip() or "."
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        path = path.resolve()
    except OSError as exc:
        raise ValueError(f"Invalid path: {exc}") from exc
    cwd = Path.cwd().resolve()
    try:
        path.relative_to(cwd)
    except ValueError:
        try:
            anchor = Path(__file__).resolve().parent
            path.relative_to(anchor)
        except ValueError as exc:
            raise ValueError(
                "glob_files: path must stay under the current working directory or the agent package root."
            ) from exc
    return path


def glob_files(pattern: str, path: str | None = None) -> str:
    try:
        base = _glob_base(path)
    except ValueError as exc:
        return str(exc)
    if not base.is_dir():
        return f"glob_files: not a directory: {base}"

    try:
        matches = sorted({str(p.resolve()) for p in base.glob(pattern) if p.is_file()})
    except OSError as exc:
        return f"glob_files: pattern error: {exc}"

    if len(matches) > GLOB_MAX_RESULTS:
        head = matches[:GLOB_MAX_RESULTS]
        return (
            "\n".join(head)
            + f"\n\n[truncated: {len(matches)} matches, showing first {GLOB_MAX_RESULTS}; narrow the pattern]"
        )
    if not matches:
        return f"(no files matched pattern {pattern!r} under {base})"
    return "\n".join(matches)


_SKIP_DIR_NAMES = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox", "dist", "build"}


def _grep_with_rg(
    pattern: str,
    base: Path,
    file_glob: str | None,
    output_mode: str,
    context_lines: int,
) -> str | None:
    rg = shutil.which("rg")
    if not rg:
        return None
    cmd: list[str] = [
        rg,
        "--color",
        "never",
        "--regexp",
        pattern,
    ]
    if file_glob:
        cmd.extend(["--glob", file_glob])
    if output_mode == "files_with_matches":
        cmd.append("--files-with-matches")
    elif output_mode == "count":
        cmd.append("--count")
    else:
        if context_lines > 0:
            cmd.extend(["--context", str(context_lines)])
        cmd.append("--line-number")

    cmd.append(str(base))
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode not in (0, 1):
        return f"grep_files (rg): exit {proc.returncode}: {err or out or 'no output'}"
    if not out:
        return "(no matches)"
    lines = out.splitlines()
    if len(lines) > GREP_MAX_LINES and output_mode == "content":
        return (
            "\n".join(lines[:GREP_MAX_LINES])
            + f"\n\n[truncated grep output to {GREP_MAX_LINES} lines; narrow pattern or path]"
        )
    if output_mode == "files_with_matches" and len(lines) > GREP_MAX_FILES_LIST:
        return (
            "\n".join(lines[:GREP_MAX_FILES_LIST])
            + f"\n\n[truncated to {GREP_MAX_FILES_LIST} paths; refine glob or pattern]"
        )
    return out


def _grep_python_scan(
    pattern: str,
    base: Path,
    file_glob: str | None,
    output_mode: str,
    context_lines: int,
) -> str:
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return f"grep_files: invalid regular expression: {exc}"

    glob_pat = re.compile(fnmatch_translate(file_glob) if file_glob else ".*")

    def rel_ok(p: Path) -> bool:
        try:
            rel = p.relative_to(base)
        except ValueError:
            return False
        parts = rel.parts
        if any(part in _SKIP_DIR_NAMES for part in parts):
            return False
        return p.is_file() and glob_pat.search(p.name)

    file_counts: dict[str, int] = {}
    content_lines: list[str] = []
    files_hit: set[str] = set()

    for dirpath, dirnames, filenames in os.walk(base, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIR_NAMES]
        for name in filenames:
            fp = Path(dirpath) / name
            if not rel_ok(fp):
                continue
            try:
                data = fp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            lines = data.splitlines()
            matches_for_file = 0
            for i, line in enumerate(lines):
                if regex.search(line):
                    matches_for_file += 1
                    rel = str(fp.resolve())
                    if output_mode == "content":
                        lo = max(0, i - context_lines)
                        hi = min(len(lines), i + context_lines + 1)
                        for j in range(lo, hi):
                            prefix = f"{rel}:{j + 1}:"
                            content_lines.append(f"{prefix}{lines[j]}")
                            if len(content_lines) >= GREP_MAX_LINES:
                                return (
                                    "\n".join(content_lines)
                                    + f"\n\n[truncated to {GREP_MAX_LINES} lines; use files_with_matches or rg]"
                                )
                    files_hit.add(rel)
            if matches_for_file:
                file_counts[str(fp.resolve())] = matches_for_file

    if output_mode == "count":
        if not file_counts:
            return "(no matches)"
        lines_out = [f"{k}:{v}" for k, v in sorted(file_counts.items())]
        if len(lines_out) > GREP_MAX_FILES_LIST:
            return "\n".join(lines_out[:GREP_MAX_FILES_LIST]) + "\n\n[truncated file list]"
        return "\n".join(lines_out)
    if output_mode == "files_with_matches":
        if not files_hit:
            return "(no matches)"
        ordered = sorted(files_hit)
        if len(ordered) > GREP_MAX_FILES_LIST:
            return "\n".join(ordered[:GREP_MAX_FILES_LIST]) + "\n\n[truncated file list]"
        return "\n".join(ordered)
    if not content_lines:
        return "(no matches)"
    return "\n".join(content_lines)


def fnmatch_translate(glob_pat: str) -> str:
    """Turn a simple *.ext glob into a regex for matching basenames."""
    escaped = re.escape(glob_pat).replace(r"\*", ".*").replace(r"\?", ".")
    return f"^{escaped}$"


def grep_files(
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
    output_mode: str | None = None,
    context_lines: int | None = None,
) -> str:
    mode = output_mode or "content"
    if mode not in ("content", "files_with_matches", "count"):
        return "grep_files: output_mode must be content, files_with_matches, or count."
    ctx = max(0, int(context_lines or 0))
    try:
        base = _glob_base(path)
    except ValueError as exc:
        return str(exc)
    if not base.is_dir():
        return f"grep_files: not a directory: {base}"

    rg_out = _grep_with_rg(pattern, base, glob, mode, ctx)
    if rg_out is not None:
        return rg_out
    return _grep_python_scan(pattern, base, glob, mode, ctx)


def fetch_json(url: str, params: dict[str, Any]) -> Any:
    request_url = f"{url}?{urlencode(params)}"
    with urlopen(request_url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def web_fetch(url: str, prompt: str | None = None) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return "web_fetch: only http(s) URLs with a host are allowed."
    header_note = f"[fetch intent: {prompt}]\n\n" if prompt else ""
    req = Request(
        url,
        headers={
            "User-Agent": "agent-loop-web_fetch/1.0 (+https://github.com/)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.5",
        },
        method="GET",
    )
    try:
        with urlopen(req, timeout=20) as resp:
            raw = resp.read()
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")[:8000]
        return f"web_fetch HTTP {error.code}: {body}"
    except URLError as error:
        return f"web_fetch network error: {error.reason}"

    text = raw.decode("utf-8", errors="replace")
    if len(text) > WEB_FETCH_MAX_CHARS:
        text = text[:WEB_FETCH_MAX_CHARS] + "\n\n[truncated response body]"
    return header_note + text


def run_terminal_cmd(command: str, description: str | None = None, timeout_sec: float | None = None) -> str:
    if os.getenv("AGENT_ALLOW_BASH", "").strip() != "1":
        return (
            "run_terminal_cmd is disabled. Set AGENT_ALLOW_BASH=1 in the environment to enable "
            "(high risk: runs shell on the host)."
        )
    timeout = 120.0 if timeout_sec is None else float(timeout_sec)
    timeout = max(1.0, min(timeout, 300.0))
    meta = f"[intent: {description}]\n" if description else ""
    shell = True if os.name == "nt" else False
    argv: str | list[str]
    if os.name == "nt":
        argv = command
    else:
        argv = ["/bin/sh", "-c", command]
    try:
        proc = subprocess.run(
            argv,
            shell=shell,
            cwd=str(Path.cwd()),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return meta + f"run_terminal_cmd: timed out after {timeout}s"
    except OSError as exc:
        return meta + f"run_terminal_cmd: failed to start: {exc}"

    out = proc.stdout or ""
    err = proc.stderr or ""
    tail = f"\n--- stderr ---\n{err}" if err.strip() else ""
    status = f"exit_code={proc.returncode}"
    combined = meta + f"{status}\n--- stdout ---\n{out}{tail}"
    max_chars = 120_000
    if len(combined) > max_chars:
        return combined[:max_chars] + f"\n\n[truncated combined output to {max_chars} chars]"
    return combined


def get_weather(city: str) -> str:
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return (
            "Missing OPENWEATHER_API_KEY. Get a free API key from "
            "https://openweathermap.org/api and add it to your .env file."
        )

    try:
        locations = fetch_json(
            OPENWEATHER_GEO_URL,
            {
                "q": city,
                "limit": 1,
                "appid": api_key,
            },
        )
        if not locations:
            return f"Could not find weather location for city: {city}"

        location = locations[0]
        weather = fetch_json(
            OPENWEATHER_CURRENT_URL,
            {
                "lat": location["lat"],
                "lon": location["lon"],
                "appid": api_key,
                "units": "metric",
                "lang": "zh_cn",
            },
        )
    except HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        return f"Weather API HTTP error {error.code}: {error_body}"
    except URLError as error:
        return f"Weather API network error: {error.reason}"
    except (KeyError, IndexError, json.JSONDecodeError) as error:
        return f"Weather API response parse error: {error}"

    location_name = location.get("local_names", {}).get("zh") or location.get("name", city)
    country = location.get("country", "")
    weather_item = weather["weather"][0]
    main = weather["main"]
    wind = weather.get("wind", {})

    return (
        f"{location_name}, {country} 当前天气: {weather_item.get('description', 'unknown')}; "
        f"温度 {main['temp']}°C, 体感 {main['feels_like']}°C, "
        f"湿度 {main['humidity']}%, 风速 {wind.get('speed', 'unknown')} m/s."
    )


def execute_tool(tool_name: str, tool_input: dict[str, Any], policy: ToolPolicy | None = None) -> str:
    print(f"[execute_tool] name={tool_name}")
    print(f"[execute_tool] input={json.dumps(tool_input, ensure_ascii=False)}")

    pol = policy or ToolPolicy()
    if not tool_allowed(tool_name, pol):
        result = (
            f"Tool {tool_name!r} is not permitted by the current tool policy "
            f"(AGENT_TOOL_MODE / AGENT_ALLOW_BASH / SUB_AGENT_TOOL_MODE)."
        )
        print(f"[execute_tool] output={result}")
        return result

    if tool_name == "read_file":
        result = read_file(
            file_path=tool_input["file_path"],
            offset=tool_input.get("offset"),
            limit=tool_input.get("limit"),
            line_numbers=tool_input.get("line_numbers"),
        )
    elif tool_name == "get_weather":
        result = get_weather(city=tool_input["city"])
    elif tool_name == "write_file":
        result = write_file(file_path=tool_input["file_path"], content=tool_input["content"])
    elif tool_name == "edit_file":
        result = edit_file(
            file_path=tool_input["file_path"],
            old_string=tool_input["old_string"],
            new_string=tool_input["new_string"],
            replace_all=tool_input.get("replace_all"),
        )
    elif tool_name == "glob_files":
        result = glob_files(pattern=tool_input["pattern"], path=tool_input.get("path"))
    elif tool_name == "grep_files":
        result = grep_files(
            pattern=tool_input["pattern"],
            path=tool_input.get("path"),
            glob=tool_input.get("glob"),
            output_mode=tool_input.get("output_mode"),
            context_lines=tool_input.get("context_lines"),
        )
    elif tool_name == "web_fetch":
        result = web_fetch(url=tool_input["url"], prompt=tool_input.get("prompt"))
    elif tool_name == "run_terminal_cmd":
        result = run_terminal_cmd(
            command=tool_input["command"],
            description=tool_input.get("description"),
            timeout_sec=tool_input.get("timeout_sec"),
        )
    else:
        result = f"Unknown tool: {tool_name}"

    print(f"[execute_tool] output={result}")
    return result
