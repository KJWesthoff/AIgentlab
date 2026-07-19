"""Structured run tracing for the live viewer.

The runtime and workflow emit JSON events describing exactly what enters
each agent's context window and every policy/budget decision along the
way. Events append to a JSON-lines file so a reader (the live viewer's
HTTP server, or a human with jq) can follow a run while it happens.

Tracing is opt-in and passive: the default ``Tracer`` is a no-op, and no
event ever feeds back into orchestration decisions.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..llm.types import Message
    from ..orchestration.state import BudgetTracker

# Documents in the corpus can be large; cap what a single event carries so
# the trace file and the browser stay responsive.
MAX_CONTENT_CHARS = 20_000


class Tracer:
    """No-op base tracer — the default when tracing is disabled."""

    def emit(self, event: str, **payload: Any) -> None:
        return None


class TraceWriter(Tracer):
    """Appends one JSON object per line to ``path``.

    The file is truncated on construction, so one file describes one run.
    Only the run's event-loop thread writes; readers tolerate a partial
    final line.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", encoding="utf-8")
        self._seq = 0

    def emit(self, event: str, **payload: Any) -> None:
        self._seq += 1
        record = {
            "seq": self._seq,
            "time": time.time(),
            "event": event,
            **payload,
        }
        self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def truncate(text: str) -> str:
    if len(text) <= MAX_CONTENT_CHARS:
        return text
    omitted = len(text) - MAX_CONTENT_CHARS
    return text[:MAX_CONTENT_CHARS] + f"\n… [truncated {omitted} chars]"


def serialize_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """The context window as the viewer shows it: one dict per message.

    ``untrusted`` marks tool results that re-entered the context under the
    untrusted-data label, so the viewer can highlight exactly which parts
    of the context the security model treats as data rather than
    instructions.
    """
    from ..agents.runtime import UNTRUSTED_PREFIX  # avoid an import cycle

    return [
        {
            "role": message.role.value,
            "content": truncate(message.content),
            "name": message.name,
            "tool_call_id": message.tool_call_id,
            "tool_calls": [
                {"id": c.id, "name": c.name, "arguments": c.arguments}
                for c in message.tool_calls
            ],
            "untrusted": message.content.startswith(UNTRUSTED_PREFIX),
        }
        for message in messages
    ]


def budget_snapshot(tracker: BudgetTracker) -> dict[str, Any]:
    budget = tracker.budget
    return {
        "model_calls": tracker.model_calls,
        "maximum_model_calls": budget.maximum_model_calls,
        "tool_calls": tracker.tool_calls,
        "maximum_tool_calls": budget.maximum_tool_calls,
        "input_tokens": tracker.input_tokens,
        "maximum_input_tokens": budget.maximum_input_tokens,
        "output_tokens": tracker.output_tokens,
        "maximum_output_tokens": budget.maximum_output_tokens,
        "cost_usd": round(tracker.accumulated_cost_usd, 6),
        "maximum_cost_usd": budget.maximum_cost_usd,
    }
