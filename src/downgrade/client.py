"""OpenAI-compatible streaming client for the Fireworks inference API.

Three details of this API shape the code and are each easy to get wrong.

Usage arrives only in the FINAL streaming chunk. Summing per-chunk usage
yields zero, and defaulting a missing usage block to zero yields a confident
looking wrong number, so `Usage.complete` records whether the final chunk was
actually seen and the cost model can tell the difference.

Tool calls arrive as deltas indexed by position: the id and function name come
in one chunk, and the arguments arrive as a string split across many. They
have to be accumulated by index and parsed once at the end.

The served model is the route. `response.model` is the only per-request
attribution the API reliably offers, so it is read from the chunks and the
response headers are captured wholesale in case routing metadata appears there
later. `RouteObservation.source` records which of the two answered.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from downgrade.models import RouteObservation, RouteSource, ToolCall, Usage
from downgrade.transport import Transport, TransportRequest, TransportResponse

CHAT_COMPLETIONS_PATH = "/chat/completions"

# Header names that would carry routing metadata if the API ever emits any.
# None is documented; they are checked so that if one appears it is used and
# recorded as the attribution source rather than silently ignored.
ROUTE_HEADER_CANDIDATES = (
    "x-fireworks-served-model",
    "x-served-model",
    "x-routing-selected-model",
    "x-model",
)


@dataclass
class _ToolCallAccumulator:
    """Reassembles one streamed tool call from its deltas."""

    call_id: str = ""
    name: str = ""
    arguments: str = ""

    def absorb(self, delta: dict[str, Any]) -> None:
        if delta.get("id"):
            self.call_id = str(delta["id"])
        function = delta.get("function") or {}
        if function.get("name"):
            self.name = str(function["name"])
        if function.get("arguments"):
            self.arguments += str(function["arguments"])

    def finish(self, fallback_index: int) -> ToolCall | None:
        if not self.name:
            return None
        try:
            parsed = json.loads(self.arguments) if self.arguments.strip() else {}
        except json.JSONDecodeError:
            # A model that emitted malformed JSON made a call we cannot
            # replay. Keep it visible as a call with the raw text rather than
            # dropping it, because a vanished call is exactly what the
            # missing_tool_call detector looks for and a parse bug must not
            # masquerade as a routing finding.
            parsed = {"__unparsed__": self.arguments}
        if not isinstance(parsed, dict):
            parsed = {"__value__": parsed}
        return ToolCall(
            call_id=self.call_id or f"call_{fallback_index}",
            name=self.name,
            arguments=parsed,
        )


@dataclass
class ChatResult:
    """One completed assistant turn."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    route: RouteObservation | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class FireworksClient:
    """Sends one chat completion and normalises the streamed response."""

    def __init__(
        self,
        transport: Transport,
        *,
        api_key: str = "",
        base_url: str = "https://api.fireworks.ai/inference/v1",
    ) -> None:
        self._transport = transport
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        turn_index: int,
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
        seed: int | None = None,
        preference: int | None = None,
        extra_headers: dict[str, str] | None = None,
        primary_model: str | None = None,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": model,
            # Copied, not referenced. The runner appends to its message list
            # as the conversation grows, and storing the live list here would
            # make every recorded request mutate into the final state:
            # cassettes would replay the wrong turn and the isolation test
            # would pass vacuously by inspecting the same object each time.
            "messages": [dict(m) for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            # Without this Fireworks omits usage from the stream entirely.
            "stream_options": {"include_usage": True},
        }
        if seed is not None:
            payload["seed"] = seed
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            **(extra_headers or {}),
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        request = TransportRequest(
            url=f"{self._base_url}{CHAT_COMPLETIONS_PATH}",
            headers=headers,
            payload=payload,
        )

        started = time.monotonic()
        response = self._transport.post_stream(request)
        latency_ms = int((time.monotonic() - started) * 1000)

        route = self._build_route(
            response=response,
            model=model,
            turn_index=turn_index,
            temperature=temperature,
            seed=seed,
            preference=preference,
            primary_model=primary_model,
            latency_ms=latency_ms,
        )

        if not response.ok:
            return ChatResult(
                content="",
                route=route,
                error=response.error or f"HTTP {response.status_code}",
            )

        content, tool_calls, finish_reason = self._collect(response.chunks)
        route.finish_reason = finish_reason
        return ChatResult(content=content, tool_calls=tool_calls, route=route)

    def _collect(self, chunks: list[dict[str, Any]]) -> tuple[str, list[ToolCall], str | None]:
        parts: list[str] = []
        accumulators: dict[int, _ToolCallAccumulator] = {}
        finish_reason: str | None = None

        for chunk in chunks:
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    parts.append(str(delta["content"]))
                for call_delta in delta.get("tool_calls") or []:
                    index = int(call_delta.get("index", 0))
                    accumulators.setdefault(index, _ToolCallAccumulator()).absorb(call_delta)
                if choice.get("finish_reason"):
                    finish_reason = str(choice["finish_reason"])

        tool_calls = [
            call
            for index in sorted(accumulators)
            if (call := accumulators[index].finish(index)) is not None
        ]
        return "".join(parts), tool_calls, finish_reason

    def _build_route(
        self,
        *,
        response: TransportResponse,
        model: str,
        turn_index: int,
        temperature: float,
        seed: int | None,
        preference: int | None,
        primary_model: str | None,
        latency_ms: int,
    ) -> RouteObservation:
        served, source = self._attribute_route(response)
        downgraded: bool | None = None
        if served is not None and primary_model is not None:
            downgraded = served.rsplit("/", 1)[-1] != primary_model.rsplit("/", 1)[-1]

        return RouteObservation(
            turn_index=turn_index,
            requested_model=model,
            served_model=served,
            preference=preference,
            downgraded=downgraded,
            source=source,
            temperature=temperature,
            seed=seed,
            response_id=self._first_value(response.chunks, "id"),
            response_headers=dict(response.headers),
            usage=self._extract_usage(response.chunks),
            latency_ms=latency_ms,
        )

    @staticmethod
    def _attribute_route(response: TransportResponse) -> tuple[str | None, RouteSource]:
        """Prefer an explicit header if one ever exists; fall back to the body.

        No routing header is documented, so `response.model` is expected to be
        the answer in practice. Checking headers first costs nothing and means
        the sweep picks up better attribution automatically if it appears,
        with `source` recording which one was used.
        """
        lowered = {k.lower(): v for k, v in response.headers.items()}
        for candidate in ROUTE_HEADER_CANDIDATES:
            value = lowered.get(candidate)
            if value:
                return value, "header"
        for chunk in response.chunks:
            model = chunk.get("model")
            if model:
                return str(model), "response_model"
        return None, "unknown"

    @staticmethod
    def _first_value(chunks: list[dict[str, Any]], key: str) -> str | None:
        for chunk in chunks:
            value = chunk.get(key)
            if value:
                return str(value)
        return None

    @staticmethod
    def _extract_usage(chunks: list[dict[str, Any]]) -> Usage:
        """Read usage from the last chunk that carries it.

        Scanning from the end rather than assuming the very last chunk,
        because some servers send a trailing empty chunk after the usage one.
        A stream with no usage block yields complete=False and zeros.
        """
        for chunk in reversed(chunks):
            usage = chunk.get("usage")
            if isinstance(usage, dict) and usage:
                return Usage(
                    input_tokens=int(usage.get("prompt_tokens") or 0),
                    output_tokens=int(usage.get("completion_tokens") or 0),
                    complete=True,
                )
        return Usage()


__all__ = [
    "CHAT_COMPLETIONS_PATH",
    "ROUTE_HEADER_CANDIDATES",
    "ChatResult",
    "FireworksClient",
]
