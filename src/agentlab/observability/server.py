"""Local HTTP server for the live trace viewer.

Stdlib only, bound to localhost. It serves two things: the single-file
viewer page, and the trace events as JSON (``/events?after=N`` returns
every event with a sequence number greater than N, which is all the
viewer needs to poll a run live). The server only ever reads the trace
file — it cannot influence the run.
"""

from __future__ import annotations

import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path


def _read_events(trace_path: Path, after: int) -> list[dict]:
    events: list[dict] = []
    if not trace_path.exists():
        return events
    with trace_path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # partial final line while the writer is mid-write
            if record.get("seq", 0) > after:
                events.append(record)
    return events


class TraceServer:
    def __init__(
        self,
        *,
        trace_path: Path,
        host: str = "127.0.0.1",
        port: int = 8642,
    ) -> None:
        viewer_html = (
            resources.files("agentlab.observability")
            .joinpath("viewer.html")
            .read_text(encoding="utf-8")
            .encode("utf-8")
        )

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path in ("/", "/index.html"):
                    self._reply(200, "text/html; charset=utf-8", viewer_html)
                elif parsed.path == "/events":
                    query = urllib.parse.parse_qs(parsed.query)
                    try:
                        after = int(query.get("after", ["0"])[0])
                    except ValueError:
                        after = 0
                    body = json.dumps(
                        {"events": _read_events(trace_path, after)}
                    ).encode("utf-8")
                    self._reply(200, "application/json", body)
                else:
                    self._reply(404, "text/plain", b"not found")

            def _reply(self, status: int, content_type: str, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: object) -> None:
                return None  # keep the CLI output clean

        try:
            self._server = ThreadingHTTPServer((host, port), Handler)
        except OSError:
            # Requested port taken — fall back to an ephemeral one.
            self._server = ThreadingHTTPServer((host, 0), Handler)
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/"

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
