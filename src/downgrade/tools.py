"""The tool layer the agent loop calls into.

Tools are hermetic: they read the committed corpus and nothing else. No
network, no clock, no randomness. That is not a convenience, it is what makes
the experiment interpretable. If a tool could return different results between
the baseline arm and a downgraded arm run ten minutes later, every finding the
classifier produced would be confounded by the tool layer.

It is also what makes value provenance exact. Because the registry sees every
tool return, it can record the set of values the run actually observed, and a
number in the final answer that is not in that set was not computed from
evidence. That is the fabricated_value detector, and it is structural rather
than a judge call only because this layer is closed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from downgrade.models import ToolCall, ToolResult


class ToolError(Exception):
    """Raised by a tool for an error the model is expected to see and handle."""


@dataclass(frozen=True)
class ToolDef:
    """A tool's public contract, in the shape the API expects."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@runtime_checkable
class Tool(Protocol):
    """Anything the registry can invoke."""

    definition: ToolDef

    def __call__(self, **kwargs: Any) -> Any: ...


class ToolRegistry:
    """Holds the tools for one task and invokes them by name.

    Never raises out of `invoke`. A tool failure is an observation the model
    receives and may recover from, and turning it into an exception would end
    the run rather than record how the model handled it. An unknown tool name
    is treated the same way: hallucinating a tool is a finding, not a crash.
    """

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        name = tool.definition.name
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered")
        self._tools[name] = tool

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def definitions(self) -> list[dict[str, Any]]:
        return [self._tools[name].definition.to_openai() for name in sorted(self._tools)]

    def invoke(self, call: ToolCall) -> ToolResult:
        started = time.monotonic()
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                ok=False,
                error=f"No such tool '{call.name}'. Available: {', '.join(self.names)}",
                latency_ms=int((time.monotonic() - started) * 1000),
            )
        try:
            content = tool(**call.arguments)
        except ToolError as exc:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                ok=False,
                error=str(exc),
                latency_ms=int((time.monotonic() - started) * 1000),
            )
        except TypeError as exc:
            # Wrong or missing arguments. The model can correct this, so it
            # comes back as an observation rather than ending the run.
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                ok=False,
                error=f"Invalid arguments for '{call.name}': {exc}",
                latency_ms=int((time.monotonic() - started) * 1000),
            )
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            content=content,
            ok=True,
            latency_ms=int((time.monotonic() - started) * 1000),
        )


@dataclass
class FunctionTool:
    """Adapts a plain callable into a Tool."""

    definition: ToolDef
    func: Any

    def __call__(self, **kwargs: Any) -> Any:
        return self.func(**kwargs)


__all__ = ["FunctionTool", "Tool", "ToolDef", "ToolError", "ToolRegistry"]
