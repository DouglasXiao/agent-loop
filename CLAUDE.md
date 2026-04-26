# Project manual (agent-loop)

## What this is

A minimal **Python agent loop** using the OpenAI-compatible chat API with **streaming**, **function tools**, and **SSE-style** logs on stdout. Entry point: `agent_loop.py`. Tool implementations: `tools_execution.py`.

## Layout

- `agent_loop.py` — client, modular system prompt (`build_system_prompt`), tool schemas, loop.
- `tools_execution.py` — `read_file`, `write_file`, `get_weather` (OpenWeather; needs `OPENWEATHER_API_KEY`).
- `requirements.txt` — dependencies.
- `.env` — local secrets (not committed); use `OPENAI_API_KEY`, optional `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENWEATHER_API_KEY`.

## Commands

- One-shot: `python agent_loop.py your question here`
- Interactive: `python agent_loop.py`

## Conventions

- Prefer **UTF-8**; the CLI reconfigures stdio when possible.
- **`read_file`**: returns at most the **first 100 lines** for large files; the reply includes a truncation notice.
- **`write_file`**: overwrites the target path; use only when the user wants a file written.
- **`CLAUDE.md`** (this file) is injected into the system prompt on each run; keep it short and factual.

## Style

Match existing code: type hints, small helpers, no unnecessary abstractions.
