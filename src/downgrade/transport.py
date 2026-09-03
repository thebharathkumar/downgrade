"""The seam between the runner and the network.

Every HTTP call the sweep makes goes through a Transport. There are four:

  HttpTransport       real calls to the API
  RecordingTransport  wraps HttpTransport and writes cassettes to disk
  ReplayTransport     serves those cassettes back, with no network
  FakeTransport       scripted responses, and a record of every request made

This exists for two reasons beyond testability. A sweep costs money, so being
able to replay one is the difference between debugging the classifier for free
and paying for every iteration. And the conversation-isolation guarantee is
only checkable if something records what was actually sent: FakeTransport
keeps every request, and the isolation test reads them back.

The interface deliberately collects the whole SSE stream rather than yielding
chunks. Token usage arrives only in the final chunk, so nothing downstream can
act on a partial stream anyway, and a collected list is far easier to record,
replay and assert against.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx

DONE_SENTINEL = "[DONE]"


@dataclass(frozen=True)
class TransportRequest:
    url: str
    headers: dict[str, str]
    payload: dict[str, Any]

    def cassette_key(self) -> str:
        """Stable key for record/replay.

        Covers the payload and the routing header but not Authorization, so a
        cassette recorded with one key replays with another, and a committed
        cassette never embeds a credential in its filename.
        """
        routing = {
            k.lower(): v for k, v in self.headers.items() if k.lower().startswith("x-routing")
        }
        blob = json.dumps(
            {"url": self.url, "routing": routing, "payload": self.payload},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]

    def redacted_headers(self) -> dict[str, str]:
        return {
            k: ("<redacted>" if k.lower() == "authorization" else v)
            for k, v in self.headers.items()
        }


@dataclass
class TransportResponse:
    status_code: int
    headers: dict[str, str] = field(default_factory=dict)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and 200 <= self.status_code < 300


class Transport(Protocol):
    """Anything that can turn a request into a collected streaming response."""

    def post_stream(self, request: TransportRequest) -> TransportResponse: ...


def parse_sse_line(line: str) -> dict[str, Any] | None:
    """Parse one `data:` line. Returns None for keep-alives, blanks and [DONE]."""
    stripped = line.strip()
    if not stripped or not stripped.startswith("data:"):
        return None
    body = stripped[len("data:") :].strip()
    if not body or body == DONE_SENTINEL:
        return None
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


class HttpTransport:
    """Real streaming calls. The only class in the project that opens a socket."""

    def __init__(self, timeout: float = 120.0, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(timeout=timeout)

    def post_stream(self, request: TransportRequest) -> TransportResponse:
        try:
            with self._client.stream(
                "POST", request.url, headers=request.headers, json=request.payload
            ) as response:
                headers = dict(response.headers)
                if response.status_code >= 400:
                    response.read()
                    return TransportResponse(
                        status_code=response.status_code,
                        headers=headers,
                        error=response.text[:2000],
                    )
                chunks = [
                    parsed
                    for line in response.iter_lines()
                    if (parsed := parse_sse_line(line)) is not None
                ]
                return TransportResponse(
                    status_code=response.status_code, headers=headers, chunks=chunks
                )
        except httpx.HTTPError as exc:
            # A transport failure is a run outcome, not a crash: the runner
            # records it as an errored trajectory so the arm keeps its
            # replicate count and the sweep stays balanced.
            return TransportResponse(status_code=0, error=f"{type(exc).__name__}: {exc}")

    def close(self) -> None:
        self._client.close()


class RecordingTransport:
    """Wraps a real transport and writes each response to a cassette."""

    def __init__(self, inner: Transport, cassette_dir: Path) -> None:
        self._inner = inner
        self._dir = cassette_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def post_stream(self, request: TransportRequest) -> TransportResponse:
        response = self._inner.post_stream(request)
        target = self._dir / f"{request.cassette_key()}.json"
        target.write_text(
            json.dumps(
                {
                    "request": {
                        "url": request.url,
                        "headers": request.redacted_headers(),
                        "payload": request.payload,
                    },
                    "response": {
                        "status_code": response.status_code,
                        "headers": response.headers,
                        "chunks": response.chunks,
                        "error": response.error,
                    },
                },
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        return response


class CassetteMissingError(RuntimeError):
    """Raised when replay is asked for a request that was never recorded."""


class ReplayTransport:
    """Serves recorded cassettes. Never touches the network.

    A missing cassette raises rather than falling through to a live call: a
    replay run that silently reached the network would spend money and, worse,
    produce a trajectory that is not reproducible.
    """

    def __init__(self, cassette_dir: Path) -> None:
        self._dir = cassette_dir

    def post_stream(self, request: TransportRequest) -> TransportResponse:
        target = self._dir / f"{request.cassette_key()}.json"
        if not target.is_file():
            raise CassetteMissingError(
                f"No cassette for {request.cassette_key()} in {self._dir}. "
                "Record one with RecordingTransport before replaying."
            )
        data = json.loads(target.read_text(encoding="utf-8"))["response"]
        return TransportResponse(
            status_code=int(data["status_code"]),
            headers=dict(data.get("headers") or {}),
            chunks=list(data.get("chunks") or []),
            error=data.get("error"),
        )


class FakeTransport:
    """Scripted responses plus a full record of what was asked.

    `requests` is the evidence the conversation-isolation test reads: it can
    assert that no two runs shared a conversation, that no run's first request
    carried messages from a previous task, and that the routing header was
    present on every single call rather than only the first.
    """

    def __init__(self, responses: list[TransportResponse] | None = None) -> None:
        self.responses = list(responses or [])
        self.requests: list[TransportRequest] = []
        self._index = 0

    def post_stream(self, request: TransportRequest) -> TransportResponse:
        self.requests.append(request)
        if self._index >= len(self.responses):
            raise AssertionError(
                f"FakeTransport ran out of scripted responses at call "
                f"{self._index + 1}; {len(self.responses)} were provided."
            )
        response = self.responses[self._index]
        self._index += 1
        return response

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def payloads(self) -> list[dict[str, Any]]:
        return [r.payload for r in self.requests]


__all__ = [
    "DONE_SENTINEL",
    "CassetteMissingError",
    "FakeTransport",
    "HttpTransport",
    "RecordingTransport",
    "ReplayTransport",
    "Transport",
    "TransportRequest",
    "TransportResponse",
    "parse_sse_line",
]
