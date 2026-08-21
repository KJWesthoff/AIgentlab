"""Custom node-kind icons for BloodHound CE.

Custom OpenGraph kinds all render with the same default glyph until they
are registered, which flattens the visual story exactly where it should
be strongest — you cannot tell a Document from a Tool at a glance.

The palette carries meaning rather than decoration:

- **warm** (documents, the corpus, artifacts) — content that is or may
  become attacker-influenced
- **gold** (tools) — privilege, the thing a path is trying to reach
- **green** (the approval gate) — a control standing in the way
- **cool** (agents, profiles, models, providers) — infrastructure

so a rendered path reads red → warm → gold, and any green on it is a
control the attacker has to get through.

Icon names are Font Awesome **free solid**, written without the ``fa-``
prefix, as BloodHound's API requires.
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import hmac
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model import NodeKind

#: Where BloodHound CE registers custom node kinds.
ICON_ENDPOINT = "/api/v2/custom-nodes"

#: BloodHound issues a token as an *id* plus a *key*. The id identifies the
#: token in the Authorization header; the key never leaves this process —
#: it signs the request instead. Both are read from the environment or
#: .env so neither lands in the repo or in shell history.
TOKEN_ID_VARIABLE = "BLOODHOUND_TOKEN_ID"
TOKEN_KEY_VARIABLE = "BLOODHOUND_TOKEN_KEY"


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


def sign_request(
    method: str,
    uri: str,
    body: bytes,
    token_key: str,
    when: datetime.datetime | None = None,
) -> tuple[str, str]:
    """Compute BloodHound's ``bhesignature`` for one request.

    Three HMAC-SHA256 digests chained, each keyed by the previous digest:
    the method and URI path, then the timestamp truncated to the hour,
    then the exact request body. Chaining means changing any part of the
    request invalidates the signature, and the hour truncation is what
    bounds replay — a signature is only good for the hour it was made, so
    a clock skewed past that window fails with a token error rather than
    anything more descriptive.

    Returns ``(RequestDate header, Signature header)``.
    """
    when = when or datetime.datetime.now().astimezone()
    formatted = when.isoformat("T")

    digester = hmac.new(token_key.encode(), None, hashlib.sha256)
    digester.update(f"{method}{uri}".encode())

    digester = hmac.new(digester.digest(), None, hashlib.sha256)
    digester.update(formatted[:13].encode())

    digester = hmac.new(digester.digest(), None, hashlib.sha256)
    digester.update(body)

    return formatted, base64.b64encode(digester.digest()).decode()


@dataclass(frozen=True)
class Registration:
    """What registering actually changed, so the CLI can say so."""

    created: tuple[str, ...]
    updated: tuple[str, ...]


def register_icons(
    base_url: str,
    token_id: str | None = None,
    token_key: str | None = None,
) -> Registration:
    """Register or refresh every node-kind icon on a running BloodHound.

    Idempotent, because the collection endpoint only creates. Ingesting a
    graph already registers its kinds — without icons — so a plain POST
    comes back ``409 duplicate kind name``, and since the batch is atomic
    one existing kind rejects all of them. Updating is a per-kind ``PUT``
    to ``/api/v2/custom-nodes/{kind_name}``; there is no batch update, so
    this lists what exists, creates the rest in one POST, and PUTs the
    remainder one at a time.

    BloodHound's API does not accept bearer tokens — the key signs each
    request and is never transmitted.
    """
    import json

    import httpx

    token_id = token_id or os.environ.get(TOKEN_ID_VARIABLE)
    token_key = token_key or os.environ.get(TOKEN_KEY_VARIABLE)
    if not token_id or not token_key:
        raise SystemExit(
            f"Set {TOKEN_ID_VARIABLE} and {TOKEN_KEY_VARIABLE} (in the "
            "environment or .env). BloodHound shows both when you create a "
            "token under Administration → API Tokens; the key is only "
            "displayed once. Both are required — the API signs requests "
            "with the key rather than sending it."
        )

    def send(method: str, uri: str, payload: Any | None = None) -> Any:
        # Body and URI are both signed, so serialize once and send those
        # exact bytes; re-encoding would invalidate the signature.
        body = b"" if payload is None else json.dumps(payload).encode()
        request_date, signature = sign_request(method, uri, body, token_key)
        response = httpx.request(
            method,
            base_url.rstrip("/") + uri,
            content=body,
            headers={
                "Authorization": f"bhesignature {token_id}",
                "RequestDate": request_date,
                "Signature": signature,
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        if response.status_code >= 400:
            raise SystemExit(
                f"BloodHound rejected {method} {uri} "
                f"({response.status_code}): {response.text[:500]}"
            )
        return response.json() if response.content else None

    listing = send("GET", ICON_ENDPOINT) or {}
    known = {
        entry.get("kindName")
        for entry in (listing.get("data") or [])
        if isinstance(entry, dict)
    }

    missing = {k: v for k, v in ICONS.items() if k.value not in known}
    present = {k: v for k, v in ICONS.items() if k.value in known}

    if missing:
        send("POST", ICON_ENDPOINT, _payload_for(missing))

    for kind, icon in present.items():
        send(
            "PUT",
            f"{ICON_ENDPOINT}/{kind.value}",
            {"config": {"icon": _icon_entry(icon)}},
        )

    return Registration(
        created=tuple(sorted(k.value for k in missing)),
        updated=tuple(sorted(k.value for k in present)),
    )
