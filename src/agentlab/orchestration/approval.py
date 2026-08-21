"""Human-in-the-loop approval for write-capable tools.

``policy.py`` decides that a call *needs* a human. This module is how the
human answers. Approval is a real gate: with no approver configured
nothing write-capable executes, which is why ``DenyingApprover`` is the
default rather than a permissive one.

It also models the way these gates fail in practice. Every real
implementation offers "don't ask me again", because being asked on every
call is unusable — and the moment that option is taken, a per-call
control silently becomes a per-session one. The rest of the run is
unattended, but the trace still says "approved". That is not a flaw in
this implementation; it is the thing being demonstrated, so the session
grant is recorded distinctly and ``graph/analysis.py`` reports it.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TextIO


class ApprovalScope(str, Enum):
    #: Approved for this call only — the gate stays a gate.
    ONCE = "once"
    #: Approved for every later call to this tool in this run.
    SESSION = "session"
    #: No approval was given.
    NONE = "none"


@dataclass(frozen=True)
class ApprovalDecision:
    approved: bool
    scope: ApprovalScope
    reason: str


class Approver(ABC):
    @abstractmethod
    def request(
        self, *, agent: str, tool: str, arguments: dict[str, Any], preview: str
    ) -> ApprovalDecision:
        """Decide whether one write-capable call may proceed."""


class DenyingApprover(Approver):
    """The default: nobody is at the keyboard, so nothing is approved.

    Failing closed matters more than convenience here — an unattended run
    that silently gained write access would be the worst outcome.
    """

    def request(
        self, *, agent: str, tool: str, arguments: dict[str, Any], preview: str
    ) -> ApprovalDecision:
        return ApprovalDecision(
            approved=False,
            scope=ApprovalScope.NONE,
            reason=(
                "No approver is configured; write-capable tools require a "
                "human. Re-run with --approve-writes to be prompted."
            ),
        )


@dataclass
class ConsoleApprover(Approver):
    """Prompts a human on the terminal before each write.

    ``session_grants`` holds the tools for which someone chose "don't ask
    again". A granted tool is auto-approved for the rest of the run — the
    prompt still prints, so the transcript shows what happened, but no
    human sees it in time to object.
    """

    input_stream: TextIO = field(default_factory=lambda: sys.stdin)
    output_stream: TextIO = field(default_factory=lambda: sys.stdout)
    session_grants: set[str] = field(default_factory=set)

    def request(
        self, *, agent: str, tool: str, arguments: dict[str, Any], preview: str
    ) -> ApprovalDecision:
        if tool in self.session_grants:
            self._write(
                f"\n  ✓ {tool} auto-approved for {agent} "
                "(session grant — not shown to a human)\n"
            )
            return ApprovalDecision(
                approved=True,
                scope=ApprovalScope.SESSION,
                reason="Previously approved for the rest of this run.",
            )

        self._prompt(agent, tool, arguments, preview)
        answer = self._read()

        if answer == "a":
            self.session_grants.add(tool)
            return ApprovalDecision(
                approved=True,
                scope=ApprovalScope.SESSION,
                reason="Approved for the rest of this run.",
            )
        if answer == "y":
            return ApprovalDecision(
                approved=True,
                scope=ApprovalScope.ONCE,
                reason="Approved for this call.",
            )
        return ApprovalDecision(
            approved=False,
            scope=ApprovalScope.NONE,
            reason="Denied by the human approver.",
        )

    def _prompt(
        self, agent: str, tool: str, arguments: dict[str, Any], preview: str
    ) -> None:
        rule = "─" * 66
        lines = [
            f"\n╭{rule}╮",
            "  Approval required — write-capable tool",
            "",
            f"  agent:  {agent}",
            f"  tool:   {tool}",
        ]
        width = max((len(k) for k in arguments), default=0) + 2
        for key, value in arguments.items():
            rendered = str(value).replace("\n", " ")
            if len(rendered) > 52:
                rendered = rendered[:52] + "…"
            lines.append(f"  {key + ':':<{width}}{rendered}")
        if preview:
            lines += ["", "  preview:"]
            lines += [f"    {line}" for line in preview.splitlines()[:8]]
        lines += [
            f"╰{rule}╯",
            "  [y] approve once   [a] approve for the rest of this run   "
            "[N] deny",
            "  > ",
        ]
        self._write("\n".join(lines))

    def _read(self) -> str:
        line = self.input_stream.readline()
        return line.strip().lower()[:1]

    def _write(self, text: str) -> None:
        self.output_stream.write(text)
        self.output_stream.flush()
