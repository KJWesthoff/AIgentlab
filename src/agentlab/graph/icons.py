"""Custom node-kind icons for BloodHound CE.

Custom OpenGraph kinds all render with the same default glyph until they
are registered, which flattens the visual story exactly where it should
be strongest — you cannot tell a Document from a Tool at a glance.

The palette carries meaning rather than decoration:

- **warm** (documents, the corpus, artifacts) — content that is or may
  become attacker-influenced
- **gold** (tools) — privilege, the thing a path is trying to reach
- **green** (the approval gate) — a control standing in the way
- **violet** (the principal and its scopes) — authority: whose say-so a
  call rests on
- **cool** (agents, profiles, models, providers) — infrastructure

so a rendered path reads red → warm → gold, and any green on it is a
control the attacker has to get through.

Icon names are Font Awesome **free solid**, written without the ``fa-``
prefix, as BloodHound's API requires.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bloodhound import BloodHoundClient
from .model import NodeKind

#: Where BloodHound CE registers custom node kinds.
ICON_ENDPOINT = "/api/v2/custom-nodes"



@dataclass(frozen=True)
class Icon:
    name: str
    color: str


ICONS: dict[NodeKind, Icon] = {
    # Untrusted content and anything carrying it.
    NodeKind.DOCUMENT: Icon("file-lines", "#E8663D"),
    NodeKind.CORPUS: Icon("folder-open", "#D98324"),
    NodeKind.ARTIFACT: Icon("box", "#B5651D"),
    # Privilege: what a path is trying to reach.
    NodeKind.TOOL: Icon("wrench", "#C9A227"),
    # A control standing on the path.
    NodeKind.APPROVAL_GATE: Icon("user-shield", "#2E9E5B"),
    # Authority: the human a run acts for, and what they actually granted.
    NodeKind.PRINCIPAL: Icon("fingerprint", "#6E4FD1"),
    NodeKind.SCOPE: Icon("id-card", "#8B78DE"),
    # Infrastructure.
    NodeKind.AGENT: Icon("robot", "#4A90D9"),
    NodeKind.MODEL_PROFILE: Icon("layer-group", "#7B68A6"),
    NodeKind.MODEL: Icon("microchip", "#5B8C7B"),
    NodeKind.PROVIDER: Icon("cloud", "#6B7280"),
    NodeKind.CAPABILITY: Icon("key", "#8899A6"),
}


def _icon_entry(icon: Icon) -> dict[str, str]:
    return {"type": "font-awesome", "name": icon.name, "color": icon.color}


def _payload_for(icons: dict[NodeKind, Icon]) -> dict[str, Any]:
    return {
        "custom_types": {
            kind.value: {"icon": _icon_entry(icon)}
            for kind, icon in icons.items()
        }
    }


def icon_payload() -> dict[str, Any]:
    """The body BloodHound's custom-nodes collection endpoint expects."""
    return _payload_for(ICONS)


def write_icons(path: Path) -> Path:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(icon_payload(), indent=2), encoding="utf-8")
    return path


@dataclass(frozen=True)
class Registration:
    """What registering actually changed, so the CLI can say so."""

    created: tuple[str, ...]
    updated: tuple[str, ...]


def register_icons(
    base_url: str, client: BloodHoundClient | None = None
) -> Registration:
    """Register or refresh every node-kind icon on a running BloodHound.

    Idempotent, because the collection endpoint only creates. Ingesting a
    graph already registers its kinds — without icons — so a plain POST
    comes back ``409 duplicate kind name``, and since the batch is atomic
    one existing kind rejects all of them. There is no batch update
    either: ``PUT`` lives at ``/api/v2/custom-nodes/{kind_name}``, one
    kind at a time, and a ``PUT`` to the collection is a ``405``. So this
    lists what exists, creates the rest in one POST, and PUTs the
    remainder individually.
    """
    client = client or BloodHoundClient.from_environment(base_url)

    listing = client.request("GET", ICON_ENDPOINT) or {}
    known = {
        entry.get("kindName")
        for entry in (listing.get("data") or [])
        if isinstance(entry, dict)
    }

    missing = {k: v for k, v in ICONS.items() if k.value not in known}
    present = {k: v for k, v in ICONS.items() if k.value in known}

    if missing:
        client.request("POST", ICON_ENDPOINT, _payload_for(missing))

    for kind, icon in present.items():
        client.request(
            "PUT",
            f"{ICON_ENDPOINT}/{kind.value}",
            {"config": {"icon": _icon_entry(icon)}},
        )

    return Registration(
        created=tuple(sorted(k.value for k in missing)),
        updated=tuple(sorted(k.value for k in present)),
    )
