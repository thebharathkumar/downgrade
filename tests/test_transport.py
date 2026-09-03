"""Transport tests.

The record/replay path is what makes iterating on the classifier free after
one paid sweep, so the properties that matter are that a cassette key is
stable across runs, that credentials never reach disk, and that a replay
which cannot find its cassette fails loudly instead of quietly reaching the
network and spending money.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from downgrade.transport import (
    CassetteMissingError,
    FakeTransport,
    HttpTransport,
    RecordingTransport,
    ReplayTransport,
    TransportRequest,
    TransportResponse,
    parse_sse_line,
)

URL = "https://api.fireworks.ai/inference/v1/chat/completions"


def request(**kw: object) -> TransportRequest:
    defaults: dict[str, object] = {
        "url": URL,
        "headers": {"Authorization": "Bearer secret-key", "x-routing-preference": "5"},
        "payload": {"model": "firerouter/a/b", "messages": [{"role": "user", "content": "hi"}]},
    }
    defaults.update(kw)
    return TransportRequest(**defaults)  # type: ignore[arg-type]


class TestParseSse:
    def test_parses_a_data_line(self) -> None:
        assert parse_sse_line('data: {"id": "x"}') == {"id": "x"}

    @pytest.mark.parametrize(
        "line",
        ["", "   ", "data: [DONE]", "data:", ": keep-alive", "event: ping", "data: not-json"],
    )
    def test_returns_none_for_non_payload_lines(self, line: str) -> None:
        assert parse_sse_line(line) is None

    def test_returns_none_for_a_non_object_payload(self) -> None:
        assert parse_sse_line("data: [1, 2]") is None


class TestCassetteKey:
    def test_is_stable_for_the_same_request(self) -> None:
        assert request().cassette_key() == request().cassette_key()

    def test_ignores_the_authorization_header(self) -> None:
        """A cassette recorded with one key must replay with another."""
        other = request(headers={"Authorization": "Bearer different", "x-routing-preference": "5"})
        assert request().cassette_key() == other.cassette_key()

    def test_distinguishes_routing_preference(self) -> None:
        """Same prompt at preference 1 and 5 are different requests."""
        other = request(headers={"x-routing-preference": "1"})
        assert request().cassette_key() != other.cassette_key()

    def test_distinguishes_payloads(self) -> None:
        other = request(payload={"model": "firerouter/a/b", "messages": []})
        assert request().cassette_key() != other.cassette_key()

    def test_redacts_credentials(self) -> None:
        redacted = request().redacted_headers()
        assert redacted["Authorization"] == "<redacted>"
        assert redacted["x-routing-preference"] == "5"


class TestTransportResponse:
    def test_ok_for_a_2xx_with_no_error(self) -> None:
        assert TransportResponse(status_code=200).ok

    def test_not_ok_for_an_error_status(self) -> None:
        assert not TransportResponse(status_code=429).ok

    def test_not_ok_when_an_error_is_set(self) -> None:
        assert not TransportResponse(status_code=200, error="boom").ok


class TestHttpTransport:
    @respx.mock
    def test_collects_streamed_chunks(self) -> None:
        body = (
            'data: {"id":"c","model":"m","choices":[{"delta":{"content":"hi"}}]}\n'
            "\n"
            'data: {"id":"c","model":"m","usage":{"prompt_tokens":5,"completion_tokens":2}}\n'
            "data: [DONE]\n"
        )
        respx.post(URL).mock(
            return_value=httpx.Response(200, text=body, headers={"x-ratelimit-remaining": "9"})
        )
        response = HttpTransport().post_stream(request())
        assert response.ok
        assert len(response.chunks) == 2
        assert response.headers["x-ratelimit-remaining"] == "9"

    @respx.mock
    def test_an_error_status_becomes_an_error_response(self) -> None:
        respx.post(URL).mock(return_value=httpx.Response(429, text="slow down"))
        response = HttpTransport().post_stream(request())
        assert not response.ok
        assert response.status_code == 429
        assert "slow down" in (response.error or "")

    @respx.mock
    def test_a_connection_failure_is_returned_not_raised(self) -> None:
        """A dropped run would unbalance the arm's replicate count."""
        respx.post(URL).mock(side_effect=httpx.ConnectError("refused"))
        response = HttpTransport().post_stream(request())
        assert not response.ok
        assert "ConnectError" in (response.error or "")

    @respx.mock
    def test_close_releases_the_client(self) -> None:
        transport = HttpTransport()
        transport.close()
        assert transport is not None


class TestRecordAndReplay:
    @respx.mock
    def test_round_trips_through_a_cassette(self, tmp_path: Path) -> None:
        body = 'data: {"id":"c","model":"served-model","choices":[]}\ndata: [DONE]\n'
        respx.post(URL).mock(return_value=httpx.Response(200, text=body))

        recorded = RecordingTransport(HttpTransport(), tmp_path).post_stream(request())
        replayed = ReplayTransport(tmp_path).post_stream(request())

        assert replayed.status_code == recorded.status_code
        assert replayed.chunks == recorded.chunks

    @respx.mock
    def test_cassettes_never_contain_the_api_key(self, tmp_path: Path) -> None:
        respx.post(URL).mock(return_value=httpx.Response(200, text="data: [DONE]\n"))
        RecordingTransport(HttpTransport(), tmp_path).post_stream(request())
        written = next(tmp_path.glob("*.json")).read_text(encoding="utf-8")
        assert "secret-key" not in written
        assert "<redacted>" in written

    def test_a_missing_cassette_raises_rather_than_reaching_the_network(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(CassetteMissingError, match="No cassette"):
            ReplayTransport(tmp_path).post_stream(request())

    @respx.mock
    def test_an_error_response_is_replayed_as_an_error(self, tmp_path: Path) -> None:
        respx.post(URL).mock(return_value=httpx.Response(500, text="upstream"))
        RecordingTransport(HttpTransport(), tmp_path).post_stream(request())
        replayed = ReplayTransport(tmp_path).post_stream(request())
        assert not replayed.ok
        assert replayed.status_code == 500


class TestFakeTransport:
    def test_records_every_request(self) -> None:
        transport = FakeTransport([TransportResponse(status_code=200) for _ in range(2)])
        transport.post_stream(request())
        transport.post_stream(request(payload={"model": "x", "messages": []}))
        assert transport.call_count == 2
        assert transport.payloads()[1]["model"] == "x"

    def test_serves_responses_in_order(self) -> None:
        transport = FakeTransport(
            [TransportResponse(status_code=200), TransportResponse(status_code=429)]
        )
        assert transport.post_stream(request()).status_code == 200
        assert transport.post_stream(request()).status_code == 429

    def test_running_out_of_responses_fails_loudly(self) -> None:
        """A silent empty response would look like a model that said nothing."""
        transport = FakeTransport([])
        with pytest.raises(AssertionError, match="ran out of scripted responses"):
            transport.post_stream(request())
