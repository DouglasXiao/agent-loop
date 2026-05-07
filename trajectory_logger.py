"""
Trajectory persistence for the agent loop.

Each user turn (one user input → one final assistant answer, possibly with
multiple intermediate tool rounds) is appended as a single JSONL record to
``.claude/trajectories/<task_id>.jsonl``.

Record schema::

    {
      "task_id":      "sess_<unix_ts>_<uuid8>",   # session-level
      "turn_index":   0,                          # 0-based, per task_id
      "timestamp":    1730000000.123,             # turn started_at (epoch sec)
      "duration_ms":  3450,
      "user_input":   "...",
      "thought":      "...",                      # see "thought sources" below
      "tool_calls":   [{"id","name","arguments"}, ...],   # input order
      "tool_results": [{"tool_call_id","name","result","duration_ms","error"}, ...],
      "final_answer": "...",                      # visible content, after stripping <thinking>
      "rounds":       N,                          # how many LLM round-trips this turn used
      "model": {"requested": "openrouter/auto",
                "actual":    ["google/gemini-2.5-flash", ...]},   # one entry per round
      "usage": {"prompt_tokens": …, "completion_tokens": …, "total_tokens": …}
    }

This is the substrate for any future SFT / DPO / RL pipeline — every
trajectory has a clean (input, thought, action, observation, output) shape
without extra parsing.

**Thought sources** (merged in order, deduplicated, joined with blank line):
  1. ``delta.reasoning_content`` from streaming — populated when
     OpenRouter routed to an o-series / R1 / extended-thinking model.
  2. ``<thinking>...</thinking>`` blocks parsed out of the assistant
     content. Universal — any instruction-following model can be asked
     to wrap its reasoning this way.

Stdlib only — no extra deps.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Tag matching is case-insensitive and tolerant of whitespace; non-greedy so
# back-to-back blocks parse as separate records, not one giant span.
_THINKING_RE = re.compile(r"<\s*thinking\s*>(.*?)<\s*/\s*thinking\s*>", re.DOTALL | re.IGNORECASE)


def trajectories_dir(root: Path) -> Path:
    return root / ".claude" / "trajectories"


def new_session_id() -> str:
    return f"sess_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def extract_thought(content: str | None) -> tuple[str, str]:
    """
    Pull ``<thinking>...</thinking>`` blocks out of ``content``.
    Returns ``(thought, visible)`` — both stripped.

    - No tags found ⇒ ``("", content_stripped)``.
    - Multiple blocks ⇒ joined with ``\\n\\n``.
    - Tag matching is case-insensitive (``<Thinking>`` works).
    - Trailing blank-line runs that the strip leaves behind are collapsed.
    """
    if not content:
        return "", ""
    thoughts: list[str] = []

    def _grab(m: re.Match) -> str:
        thoughts.append(m.group(1).strip())
        return ""

    visible = _THINKING_RE.sub(_grab, content).strip()
    visible = re.sub(r"\n{3,}", "\n\n", visible)
    return "\n\n".join(t for t in thoughts if t), visible


@dataclass
class TurnAccumulator:
    """In-memory state for one user turn while it's still being executed."""

    user_input: str
    started_at: float = field(default_factory=time.time)
    thoughts: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    rounds: int = 0
    requested_model: str | None = None
    actual_models: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )
    ttft_ms_per_round: list[int] = field(default_factory=list)
    duration_ms_per_round: list[int] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ---- per-round bookkeeping ------------------------------------------

    def add_round(
        self,
        *,
        thought: str = "",
        routed: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self.rounds += 1
            if thought.strip():
                self.thoughts.append(thought.strip())
            if routed:
                actual = routed.get("actual")
                if actual:
                    self.actual_models.append(actual)
                req = routed.get("requested")
                if req and not self.requested_model:
                    self.requested_model = req
                u = routed.get("usage") or {}
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    v = u.get(k)
                    if isinstance(v, int):
                        self.usage[k] = self.usage.get(k, 0) + v
                # Per-round latency profile — useful for spotting which round
                # was the slow one when reviewing a trajectory after the fact.
                ttft = routed.get("ttft_ms")
                if isinstance(ttft, int):
                    self.ttft_ms_per_round.append(ttft)
                dur = routed.get("duration_ms")
                if isinstance(dur, int):
                    self.duration_ms_per_round.append(dur)

    def add_tool_call(self, *, call_id: str | None, name: str | None, arguments: Any) -> None:
        with self._lock:
            self.tool_calls.append({"id": call_id, "name": name, "arguments": arguments})

    def add_tool_result(
        self,
        *,
        tool_call_id: str | None,
        name: str | None,
        result: str,
        duration_ms: int | None = None,
        error: str | None = None,
    ) -> None:
        rec: dict[str, Any] = {
            "tool_call_id": tool_call_id,
            "name": name,
            "result": result,
        }
        if duration_ms is not None:
            rec["duration_ms"] = duration_ms
        if error:
            rec["error"] = error
        with self._lock:
            self.tool_results.append(rec)

    def set_final_answer(self, text: str) -> None:
        with self._lock:
            self.final_answer = text or ""

    # ---- emit -----------------------------------------------------------

    def to_record(self, *, task_id: str, turn_index: int) -> dict[str, Any]:
        with self._lock:
            return {
                "task_id": task_id,
                "turn_index": turn_index,
                "timestamp": self.started_at,
                "duration_ms": int((time.time() - self.started_at) * 1000),
                "user_input": self.user_input,
                "thought": "\n\n".join(self.thoughts).strip(),
                "tool_calls": list(self.tool_calls),
                "tool_results": list(self.tool_results),
                "final_answer": self.final_answer,
                "rounds": self.rounds,
                "model": {
                    "requested": self.requested_model,
                    "actual": list(self.actual_models),
                },
                "usage": dict(self.usage),
                "latency": {
                    "ttft_ms_per_round": list(self.ttft_ms_per_round),
                    "duration_ms_per_round": list(self.duration_ms_per_round),
                },
            }


class TrajectoryLogger:
    """
    Owns the JSONL file for one ``task_id`` and a turn-index counter.

    Disabled (no I/O) when ``enabled=False`` so unit tests can construct one
    without touching the filesystem.
    """

    def __init__(
        self,
        root: Path,
        *,
        task_id: str | None = None,
        enabled: bool = True,
    ) -> None:
        self.root = root
        self.task_id = task_id or new_session_id()
        self.enabled = enabled
        self.path: Path | None = None
        self._turn_index = 0
        self._lock = threading.Lock()
        if enabled:
            d = trajectories_dir(root)
            d.mkdir(parents=True, exist_ok=True)
            self.path = d / f"{self.task_id}.jsonl"

    @classmethod
    def from_env(cls, root: Path) -> "TrajectoryLogger":
        """Build from ``AGENT_TRAJECTORY`` (default on) and ``AGENT_TRAJECTORY_ID``."""
        enabled = os.getenv("AGENT_TRAJECTORY", "1").strip().lower() not in (
            "0", "false", "no", "off", ""
        )
        tid = (os.getenv("AGENT_TRAJECTORY_ID") or "").strip() or None
        return cls(root, task_id=tid, enabled=enabled)

    def start_turn(self, user_input: str) -> TurnAccumulator:
        return TurnAccumulator(user_input=user_input or "")

    def finish_turn(self, accum: TurnAccumulator) -> dict[str, Any] | None:
        if not self.enabled or self.path is None:
            return None
        rec = accum.to_record(task_id=self.task_id, turn_index=self._turn_index)
        line = json.dumps(rec, ensure_ascii=False, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._turn_index += 1
        return rec
