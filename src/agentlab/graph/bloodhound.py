"""Signed HTTP client for the BloodHound CE API.

BloodHound does not accept bearer tokens. A token is an *id* and a *key*:
the id names the token in the ``Authorization`` header, and the key signs
the request without ever being transmitted. The signature is three
chained HMAC-SHA256 digests — over the method and URI, then the
timestamp truncated to the hour, then the exact request body — so
changing any part of a request invalidates it.

The hour truncation is what bounds replay. It also means a clock skewed
by more than an hour produces a signature BloodHound computes
differently, and the failure arrives as a *token* error that never
mentions time.
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from typing import Any

#: Read from the environment or .env so no credential reaches the repo,
#: a traceback, or shell history.
TOKEN_ID_VARIABLE = "BLOODHOUND_TOKEN_ID"
TOKEN_KEY_VARIABLE = "BLOODHOUND_TOKEN_KEY"


def sign_request(
    method: str,
    uri: str,
    body: bytes,
    token_key: str,
    when: datetime.datetime | None = None,
) -> tuple[str, str]:
    """Compute BloodHound's ``bhesignature`` for one request.

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
class BloodHoundClient:
    base_url: str
    token_id: str
    token_key: str

    @classmethod
    def from_environment(cls, base_url: str) -> BloodHoundClient:
        token_id = os.environ.get(TOKEN_ID_VARIABLE)
        token_key = os.environ.get(TOKEN_KEY_VARIABLE)
        if not token_id or not token_key:
            raise SystemExit(
                f"Set {TOKEN_ID_VARIABLE} and {TOKEN_KEY_VARIABLE} (in the "
                "environment or .env). BloodHound shows both when you create "
                "a token under Administration → API Tokens; the key is only "
                "displayed once. Both are required — the API signs requests "
                "with the key rather than sending it."
            )
        return cls(base_url.rstrip("/"), token_id, token_key)

    def request(
        self, method: str, uri: str, payload: Any | None = None
    ) -> Any:
        """Send one signed request, returning the decoded body.

        Raises ``SystemExit`` with the server's own message on failure —
        BloodHound's errors are specific enough to act on, and far more
        useful than a traceback from this layer.
        """
        import httpx

        # Body and URI are both signed, so serialize once and send exactly
        # those bytes; re-encoding could invalidate the signature.
        body = b"" if payload is None else json.dumps(payload).encode()
        request_date, signature = sign_request(
            method, uri, body, self.token_key
        )

        response = httpx.request(
            method,
            self.base_url + uri,
            content=body,
            headers={
                "Authorization": f"bhesignature {self.token_id}",
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
