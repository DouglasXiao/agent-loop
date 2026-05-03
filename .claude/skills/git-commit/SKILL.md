---
name: git-commit
description: Conventional-commit workflow for this repo (feat/fix/refactor/docs/chore + body explaining why).
---

# git-commit skill

Use this skill when you need to land a change.

1. **Stage only what's relevant.** `git add <files>` — never `git add -A` unless the user explicitly asked to commit everything.
2. **Pick a type prefix:** `feat:` (new behavior or new tool), `fix:` (bug fix), `refactor:` (no behavior change), `docs:` (md only), `chore:` (build/deps/format), `test:`.
3. **Subject ≤ 72 chars** in imperative mood, scoped if useful — e.g. `feat(parallel): fan out read tools within a turn`.
4. **Body** (encouraged for non-trivial changes): what changed, *why* it changed, and the self-check that was run.
5. **Push:** `git push origin main`. If push fails on auth from a sandbox, leave the commit local and ask Douglas to push from his terminal.

## Anti-patterns

- Don't commit `.venv/`, `.DS_Store`, `.claude/memory/spill/`, or `.env`.
- Don't squash unrelated changes into one commit.
- Don't write subjects that just restate file paths (`update agent_loop.py` — say *what* and *why*).
