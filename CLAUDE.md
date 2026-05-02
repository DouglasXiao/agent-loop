# Project manual (agent-loop)

## What this is

A minimal **Python agent loop** using the OpenAI-compatible chat API with **streaming**, **function tools**, and **SSE-style** logs on stdout. Entry point: `agent_loop.py`. Tool implementations: `tools_execution.py`.

## Layout

- `agent_loop.py` — client, modular system prompt (`build_system_prompt`), tool schemas, streaming loop, nag reminder.
- `tools_execution.py` / `tools_registry.py` — central tool dispatch + risk-class policy (read / mutate / network / system / delegate).
- `context_memory.py` — context budgeting, tool-result spilling under `.claude/memory/spill/`, summary-based compression.
- `todo_manager.py` — persistent in-task todo list under `.claude/todos/current.json`.
- `sub_agent.py` — isolated worker agents (`run_sub_agent` / parallel) using `SUB_AGENT_*` env vars.
- `PLAN.md` — improvement roadmap aligned with shareAI-lab/learn-claude-code.
- `requirements.txt` — dependencies.
- `.env` — local secrets (not committed); use `OPENAI_API_KEY`, optional `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENWEATHER_API_KEY`, `SUB_AGENT_*`.

## Commands

- One-shot: `python agent_loop.py your question here`
- Interactive: `python agent_loop.py`

## Conventions

- Prefer **UTF-8**; the CLI reconfigures stdio when possible.
- **`read_file`**: returns at most the **first 100 lines** for large files; the reply includes a truncation notice.
- **`write_file`**: overwrites the target path; use only when the user wants a file written.
- **`CLAUDE.md`** (this file) is injected into the system prompt on each run; keep it short and factual.
- **`todo_write`** is the planning tool: for any multi-step task, call `action="set"` first; keep at most one `in_progress`. The current list is also rendered into the system prompt every turn.
- A nag `<reminder>` is appended to the last tool result if the model goes `AGENT_TODO_NAG_AFTER_ROUNDS` (default 3) consecutive turns without calling `todo_write`.

## Style

Match existing code: type hints, small helpers, no unnecessary abstractions.
