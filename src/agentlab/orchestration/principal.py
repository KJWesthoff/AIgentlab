"""The human on whose authority a run executes.

One principal, many agents. Every agent in a workflow acts for the same
human, and authorization derives from that human's identity rather than
from the agent that happens to be asking. Per-agent authority would make
each delegation a privilege-escalation step, because an agent could then
reach something the person who started the run never could.

Two properties matter, and both are structural rather than advisory:

- **Carried with the delegation, never in the context.** The principal is
  passed as a parameter through the workflow and into the policy check.
  It is never serialized into a message, because anything in the context
  window is data the model can be steered into rewriting — an identity
  the model can edit is not an identity.
- **Checked at the gate.** ``authorize_tool_call`` asks whether *this
  principal's scopes* permit the call, not merely whether the calling
  agent's allowlist does. That is the difference between "a trusted peer
  asked" and "the person who started this authorized it".

Scopes are coarse on purpose: a lab needs enough to show the boundary
exists, not a production authorization model.
"""

from __future__ import annotations

import getpass
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Principal(BaseModel):
    """An authenticated human and the authority they carry."""

    name: str
    scopes: frozenset[str] = Field(default_factory=frozenset)

    def authorizes(self, scope: str | None) -> bool:
        """Whether this principal's authority covers ``scope``.

        A tool declaring no scope is not implicitly authorized — it is
        unscoped, which the graph reports rather than quietly allows.
        """
        if scope is None:
            return False
        return scope in self.scopes

    def describe(self) -> str:
        return f"{self.name} [{', '.join(sorted(self.scopes)) or 'no scopes'}]"


def resolve_name(configured: str | None) -> str:
    """Identity for this run.

    ``local-user`` in config means "whoever is actually running this",
    resolved from the OS rather than invented, so the trace names a real
    account.
    """
    if configured and configured != "local-user":
        return configured
    try:
        return getpass.getuser()
    except Exception:  # pragma: no cover - depends on the environment
        return "unknown-user"


def load_principal(
    path: Path | str, scopes: list[str] | None = None
) -> Principal:
    """Load the principal, optionally overriding its scopes for one run.

    Narrowing scopes on the command line is how a demo shows the
    difference between approving a call and being authorized to make it.
    """
    data = yaml.safe_load(Path(path).read_text()) or {}
    configured = data.get("principal") or {}

    return Principal(
        name=resolve_name(configured.get("name")),
        scopes=frozenset(
            scopes if scopes is not None else (configured.get("scopes") or [])
        ),
    )
