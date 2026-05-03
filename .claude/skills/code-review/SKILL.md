---
name: code-review
description: Self-review checklist before declaring an edit "done" — types, errors, tests, side effects.
---

# code-review skill

Run this before saying "done" on any code change you authored.

## 1. Read your diff

`git diff` (or `git diff --cached`) — read every changed line as if a stranger wrote it. Flag: unused imports, debug prints, magic numbers, hardcoded paths, mutating shared state without a lock.

## 2. Failure modes

For every new branch, ask: what input makes this raise?

- Network calls → timeout, non-2xx, malformed JSON.
- File ops → file missing, file is a directory, permission denied, encoding errors.
- Concurrency → two workers writing the same file; reorder of independent work.

If the agent loop should keep going after a failure, catch and surface it as a tool-result string instead of letting the exception bubble.

## 3. Tests / smoke check

Pure helpers: a 5-line `python3 -c "..."` script exercising happy path + one edge case. Anything that touches OpenAI: confirm imports succeed and new tool schemas are valid JSON.

## 4. CLAUDE.md / PLAN.md drift

If you added a new tool, env var, or file under `.claude/`, update `CLAUDE.md` in the same commit so the next session's system prompt is accurate.

## 5. Commit hygiene

Defer to the `git-commit` skill for prefix, subject, and body conventions.
