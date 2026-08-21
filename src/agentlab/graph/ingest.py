"""Upload an OpenGraph payload to BloodHound CE.

File ingest is three calls, not one: create a job, upload files into it,
then end the job. Nothing is processed until the job ends, so a client
that uploads and exits leaves the data sitting in an open job — the
symptom is an ingest that "succeeded" while the graph stays empty.

Ending the job only queues the work. This module then waits for the job
to leave its running states, so that a single command finishes with the
graph actually queryable rather than merely accepted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .bloodhound import BloodHoundClient
from .export import SOURCE_KIND

UPLOAD_ENDPOINT = "/api/v2/file-upload"
SOURCE_KINDS_ENDPOINT = "/api/v2/graphs/source-kinds"
CLEAR_ENDPOINT = "/api/v2/clear-database"
DATAPIPE_ENDPOINT = "/api/v2/datapipe/status"

#: Job status codes, from BloodHound's OpenAPI enum.job-status.
STATUS_NAMES = {
    -1: "invalid",
    0: "ready",
    1: "running",
    2: "complete",
    3: "canceled",
    4: "timed out",
    5: "failed",
    6: "ingesting",
    7: "analyzing",
    8: "partially complete",
}
RUNNING_STATUSES = frozenset({0, 1, 6, 7})
SUCCESS_STATUSES = frozenset({2, 8})

DEFAULT_TIMEOUT_SECONDS = 180.0
POLL_INTERVAL_SECONDS = 2.0


@dataclass(frozen=True)
class IngestResult:
    job_id: int
    status: int
    waited: bool
    cleared: bool = False

    @property
    def status_name(self) -> str:
        return STATUS_NAMES.get(self.status, f"unknown ({self.status})")

    @property
    def succeeded(self) -> bool:
        return self.status in SUCCESS_STATUSES


def clear_source_kind(
    client: BloodHoundClient,
    source_kind: str = SOURCE_KIND,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> bool:
    """Delete only the nodes attributed to ``source_kind``.

    Ingest adds; it never replaces. Re-uploading after the corpus changed
    leaves both document sets in the graph, which reads as a graph that
    is simply wrong.

    Deliberately narrow: this passes ``deleteSourceKinds`` with a single
    id resolved by name, never ``deleteCollectedGraphData``. Any AD or
    Azure data in the same instance is untouched, and a source kind that
    is not present is a no-op rather than an error.
    """
    listing = client.request("GET", SOURCE_KINDS_ENDPOINT) or {}
    kinds = (listing.get("data") or {}).get("kinds") or []
    matched = [
        k.get("id")
        for k in kinds
        if isinstance(k, dict) and k.get("name") == source_kind
    ]
    if not matched:
        return False

    client.request("POST", CLEAR_ENDPOINT, {"deleteSourceKinds": matched})

    # Clearing is asynchronous, and a clear in progress cancels in-flight
    # ingest jobs. Waiting for the datapipe to report "idle" is not enough:
    # immediately after the request it is still idle because the work has
    # not started, so the wait returns at once and the upload that follows
    # gets cancelled. Wait for the observable end state instead — the
    # source kind disappearing — then let the pipeline settle.
    _await_source_kind_gone(client, source_kind, timeout)
    _await_idle(client, timeout)
    return True


def _source_kind_names(client: BloodHoundClient) -> set[str]:
    listing = client.request("GET", SOURCE_KINDS_ENDPOINT) or {}
    return {
        k.get("name")
        for k in ((listing.get("data") or {}).get("kinds") or [])
        if isinstance(k, dict)
    }


def _await_source_kind_gone(
    client: BloodHoundClient, source_kind: str, timeout: float
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if source_kind not in _source_kind_names(client):
            return True
        time.sleep(POLL_INTERVAL_SECONDS)
    raise SystemExit(
        f"Timed out waiting for {source_kind!r} to be cleared; not "
        "uploading, because an in-flight clear cancels the ingest job."
    )


def _await_idle(client: BloodHoundClient, timeout: float) -> str:
    deadline = time.monotonic() + timeout
    status = "unknown"
    while time.monotonic() < deadline:
        response = client.request("GET", DATAPIPE_ENDPOINT) or {}
        status = (response.get("data") or {}).get("status", status)
        if status == "idle":
            return status
        time.sleep(POLL_INTERVAL_SECONDS)
    return status


def ingest_graph(
    base_url: str,
    path: Path,
    client: BloodHoundClient | None = None,
    wait: bool = True,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    replace: bool = False,
) -> IngestResult:
    """Upload ``path`` as one file-ingest job and wait for it to finish."""
    client = client or BloodHoundClient.from_environment(base_url)

    cleared = clear_source_kind(client, timeout=timeout) if replace else False

    started = client.request("POST", f"{UPLOAD_ENDPOINT}/start") or {}
    job_id = (started.get("data") or {}).get("id")
    if job_id is None:
        raise SystemExit(
            "BloodHound did not return a file-upload job id; cannot continue."
        )

    # Sent verbatim: the signature covers these exact bytes.
    client.request(
        "POST", f"{UPLOAD_ENDPOINT}/{job_id}", body=path.read_bytes()
    )
    # Without this the job stays open and nothing is ever processed.
    client.request("POST", f"{UPLOAD_ENDPOINT}/{job_id}/end")

    if not wait:
        return IngestResult(job_id, 0, waited=False, cleared=cleared)

    return IngestResult(
        job_id,
        _await_job(client, job_id, timeout),
        waited=True,
        cleared=cleared,
    )


def _await_job(
    client: BloodHoundClient, job_id: int, timeout: float
) -> int:
    """Poll until the job leaves its running states, or time out.

    Returns the last status seen. A timeout is reported as the status at
    that moment rather than raising: the upload did happen, and saying
    "still analyzing" is more useful than an exception.
    """
    deadline = time.monotonic() + timeout
    status = 0

    while time.monotonic() < deadline:
        listing = client.request("GET", UPLOAD_ENDPOINT) or {}
        for entry in listing.get("data") or []:
            if not isinstance(entry, dict) or entry.get("id") != job_id:
                continue
            status = entry.get("status", status)
            if status not in RUNNING_STATUSES:
                return status
            break
        time.sleep(POLL_INTERVAL_SECONDS)

    return status
