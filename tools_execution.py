import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


OPENWEATHER_GEO_URL = "https://api.openweathermap.org/geo/1.0/direct"
OPENWEATHER_CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"


READ_FILE_MAX_LINES = 100


def read_file(file_path: str) -> str:
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path

    if not path.exists():
        return f"File not found: {path}"
    if not path.is_file():
        return f"Path is not a file: {path}"

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if len(lines) <= READ_FILE_MAX_LINES:
        return text
    head = "\n".join(lines[:READ_FILE_MAX_LINES])
    return (
        f"{head}\n\n"
        f"[truncated: showing first {READ_FILE_MAX_LINES} of {len(lines)} lines; "
        f"ask for a smaller range or a different path if you need more]"
    )


def write_file(file_path: str, content: str) -> str:
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path

    if path.exists() and not path.is_file():
        return f"Path is not a file: {path}"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"File written: {path}"


def edit_file(file_path: str, old_string: str, new_string: str) -> str:
    """Replace exactly one occurrence of ``old_string`` with ``new_string``."""
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path

    if not path.exists():
        return f"File not found: {path}"
    if not path.is_file():
        return f"Path is not a file: {path}"

    text = path.read_text(encoding="utf-8")
    if old_string not in text:
        return "edit_file: old_string not found in file (no changes written)."
    count = text.count(old_string)
    if count != 1:
        return (
            f"edit_file: old_string must match exactly once (found {count} occurrences); "
            "narrow old_string or use write_file for full replacement."
        )
    path.write_text(text.replace(old_string, new_string, 1), encoding="utf-8")
    return f"File updated (single replacement): {path}"


def fetch_json(url: str, params: dict[str, Any]) -> Any:
    request_url = f"{url}?{urlencode(params)}"
    with urlopen(request_url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


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


def execute_tool(tool_name: str, tool_input: dict[str, Any]) -> str:
    print(f"[execute_tool] name={tool_name}")
    print(f"[execute_tool] input={json.dumps(tool_input, ensure_ascii=False)}")

    if tool_name == "read_file":
        result = read_file(file_path=tool_input["file_path"])
    elif tool_name == "get_weather":
        result = get_weather(city=tool_input["city"])
    elif tool_name == "write_file":
        result = write_file(file_path=tool_input["file_path"], content=tool_input["content"])
    elif tool_name == "edit_file":
        result = edit_file(
            file_path=tool_input["file_path"],
            old_string=tool_input["old_string"],
            new_string=tool_input["new_string"],
        )
    else:
        result = f"Unknown tool: {tool_name}"

    print(f"[execute_tool] output={result}")
    return result
