# Project manual (agent-loop)

## What this is

A minimal **Python agent loop** using the OpenAI-compatible chat API with **streaming**, **function tools**, and **SSE-style** logs on stdout. Entry point: `agent_loop.py`. Tool implementations: `tools_execution.py`.

## Layout

- `agent_loop.py` — client, modular system prompt (`build_system_prompt`), tool schemas, streaming loop, nag reminder.
- `tools_execution.py` / `tools_registry.py` — central tool dispatch + risk-class policy (read / mutate / network / system / delegate).
- `context_memory.py` — context budgeting, tool-result spilling under `.claude/memory/spill/`, summary-based compression.
- `todo_manager.py` — persistent in-task todo list under `.claude/todos/current.json`.
- `task_graph.py` — durable task graph under `.claude/tasks/task_<id>.json` (status + blockedBy DAG; survives restart).
- `bg_tasks.py` — in-process background shell runner (`bg_run` / `bg_check`); finished tasks auto-injected as `<background-results>` before the next agent turn.
- `worktree.py` — git worktree task isolation (`worktree create/remove/keep/list/run/events`); registry under `.worktrees/index.json`, lifecycle log under `.worktrees/events.jsonl`; can bind to a `task_id` so create→in_progress and remove(complete_task)→completed flip in lockstep.
- `team_mailbox.py` — file-based team mailbox under `.team/` (register / send / broadcast / read / peek / list). Append-only JSONL inboxes per teammate.
- `skill_loader.py` — on-demand skills under `.claude/skills/<name>/SKILL.md` (`list_skills` / `load_skill`).
- `sub_agent.py` — isolated worker agents (`run_sub_agent` / parallel) with structured `SubAgentResult` (label, error_category, rounds_used, duration_ms, tools_used, tool_errors).
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
- Skills live at `.claude/skills/<name>/SKILL.md` with optional YAML frontmatter `--- name: foo\ndescription: short blurb ---`. The system prompt advertises name + description per skill (cheap); call `load_skill(name=...)` to pull the full body.
- **`task`** is the persistent task graph (vs. `todo_write`'s in-session list). Use it when work needs to outlive the conversation or has explicit dependency edges. Completing a task auto-strips its id from every other task's `blockedBy`.
- **`bg_run` / `bg_check`** spawn slow shell work in a daemon thread (requires `AGENT_ALLOW_BASH=1`, system risk class). The orchestrator drains finished tasks at the top of every loop turn into a `<background-results>` user message so the model reacts on the next decision.
- **`worktree`** isolates parallel work in `git worktree` directories on `wt/<name>` branches. Pair with `task` via `task_id` so the task lifecycle and the worktree lifecycle stay in sync; `worktree remove ... complete_task=true` marks the bound task done atomically. Requires `AGENT_ALLOW_BASH=1`.
- **`team`** is a file-based mailbox for cross-session notes (no live agent threads). Use `register` once, then `send/broadcast/peek/read`. `read` drains by default — call `peek` if you want to look without consuming.
- Three-layer context compaction (defaults: `AGENT_MAX_CONTEXT_TOKENS=200000`, `AGENT_CONTEXT_COMPRESS_RATIO=0.7`, `AGENT_EMERGENCY_COMPACT_RATIO=0.95`):
  1. Per-turn `micro_compact_inplace` shrinks `role=tool` messages older than `AGENT_KEEP_RECENT_TOOL_RESULTS` (default 6) to a one-line placeholder; spill path is preserved so the body stays readable via `read_file`.
  2. When estimated tokens cross `AGENT_CONTEXT_COMPRESS_RATIO * AGENT_MAX_CONTEXT_TOKENS`, an LLM summary collapses early turns. A full pre-compression transcript is first snapshotted to `.claude/memory/transcripts/transcript_<ts>_<id>.jsonl`; the path is included in the summary block. The summarizer call auto-detects whether the model wants `max_completion_tokens` (gpt-5 / o-series) or `max_tokens` (older chat models) and caches the winner per client.
  3. `emergency_compact_inplace` — non-LLM, deterministic last resort. Drops the oldest non-system / non-tail messages and replaces them with a single placeholder. Triggered automatically (a) before each LLM call when usage crosses `AGENT_EMERGENCY_COMPACT_RATIO`, (b) after a summary attempt fails, and (c) inside `_stream_one_completion` when the upstream returns `context_length_exceeded` (it then retries the call once).
- `_stream_one_completion` re-raises non-context-limit errors so the outer loop can SSE-emit `upstream_error` and return `None` instead of tracebacking out of the process.

## Style

Match existing code: type hints, small helpers, no unnecessary abstractions.
