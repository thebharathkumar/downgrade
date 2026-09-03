"""The agent loop, and the conversation-isolation guarantee it exists to hold.

FireRouter caches its routing decision within a conversation, so a preference
change takes several turns to take effect. A runner that reused a conversation
across tasks or across arms would spend part of every run being served by the
previous arm's cached route, and the whole sweep would be noise.

The guarantee is structural rather than a convention: `AgentLoop.run` accepts
no message list, no history and no session object. It builds the conversation
from the task prompt alone and mints a fresh id for it. There is no parameter
through which prior state can enter, so the isolation cannot be broken by a
caller passing the wrong thing. `ConversationGuard` then catches the remaining
way to get it wrong, which is minting an id twice.

A run never raises. Transport failures, tool failures and step-budget
exhaustion all become a Trajectory with a status, because an arm that drops
runs on error would end up with fewer replicates than its baseline and every
rate computed against it would be wrong.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from downgrade.arms import Arm
from downgrade.client import FireworksClient
from downgrade.models import Step, SweepConfig, Trajectory
from downgrade.suite.spec import TaskSpec
from downgrade.tools import ToolRegistry

SYSTEM_PROMPT = (
    "You are a careful financial analyst working with SEC filings.\n"
    "Use the provided tools to gather evidence before answering. Every figure "
    "in your final answer must come from a tool result; do not estimate, "
    "recall, or infer numbers you have not looked up.\n"
    "When you have enough evidence, reply with your final answer as plain "
    "text and make no further tool calls."
)


class ConversationReuseError(RuntimeError):
    """Raised when a conversation id is used for a second run."""


class ConversationGuard:
    """Refuses to let two runs share a conversation.

    The runner already mints a fresh uuid per run, so this exists to catch a
    future refactor that threads an id in from outside, which is the change
    that would silently reintroduce cached routing without failing any other
    test.
    """

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def claim(self, conversation_id: str) -> None:
        if conversation_id in self._seen:
            raise ConversationReuseError(
                f"Conversation '{conversation_id}' was already used. Routing "
                "decisions cache per conversation, so every run must start a "
                "fresh one."
            )
        self._seen.add(conversation_id)

    @property
    def count(self) -> int:
        return len(self._seen)


class AgentLoop:
    """Runs one task under one arm, in its own conversation."""

    def __init__(
        self,
        client: FireworksClient,
        registry: ToolRegistry,
        config: SweepConfig,
        guard: ConversationGuard | None = None,
    ) -> None:
        self._client = client
        self._registry = registry
        self._config = config
        self._guard = guard or ConversationGuard()

    @property
    def guard(self) -> ConversationGuard:
        return self._guard

    def run(self, task: TaskSpec, arm: Arm, replicate: int = 0) -> Trajectory:
        """Execute one replicate. Note what this does NOT accept: any history.

        The signature is the guarantee. Adding a `messages` or `session`
        parameter here would be the change that breaks the experiment.
        """
        conversation_id = f"conv-{uuid.uuid4()}"
        self._guard.claim(conversation_id)

        trajectory = Trajectory(
            run_id=f"{task.task_id}:{arm.name}:{replicate}:{uuid.uuid4().hex[:8]}",
            task_id=task.task_id,
            arm=arm.name,
            replicate=replicate,
            conversation_id=conversation_id,
            config_fingerprint=self._config.fingerprint,
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task.prompt},
        ]
        tool_definitions = self._registry.definitions()
        model = arm.model_for(self._config)
        headers = arm.request_headers()
        max_steps = task.max_steps or self._config.max_steps
        started = time.monotonic()

        for index in range(max_steps):
            result = self._client.chat(
                messages=messages,
                model=model,
                turn_index=index,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
                tools=tool_definitions,
                seed=self._config.seed,
                preference=arm.preference,
                extra_headers=headers,
                primary_model=self._config.primary_model,
            )
            assert result.route is not None  # the client always builds one

            step = Step(index=index, thought=result.content, route=result.route)

            if not result.ok:
                step.error = result.error
                trajectory.steps.append(step)
                trajectory.status = "errored"
                trajectory.error = result.error
                break

            step.tool_calls = list(result.tool_calls)
            messages.append(self._assistant_message(result.content, result.tool_calls))

            if not result.tool_calls:
                trajectory.steps.append(step)
                trajectory.final_answer = result.content.strip()
                trajectory.status = "completed"
                break

            for call in result.tool_calls:
                tool_result = self._registry.invoke(call)
                step.tool_results.append(tool_result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "content": _render_tool_content(tool_result.content)
                        if tool_result.ok
                        else f"ERROR: {tool_result.error}",
                    }
                )
            trajectory.steps.append(step)
        else:
            # Loop finished without a final answer: the model kept calling
            # tools until the budget ran out. That is a loud finding, not an
            # error, and it must stay distinguishable from a crash.
            trajectory.status = "no_termination"

        trajectory.wall_ms = int((time.monotonic() - started) * 1000)
        return trajectory

    @staticmethod
    def _assistant_message(content: str, tool_calls: list[Any]) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": content or None}
        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": _dump_arguments(call.arguments),
                    },
                }
                for call in tool_calls
            ]
        return message


def _dump_arguments(arguments: dict[str, Any]) -> str:
    import json

    return json.dumps(arguments, sort_keys=True, default=str)


def _render_tool_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    import json

    return json.dumps(content, default=str)


__all__ = [
    "SYSTEM_PROMPT",
    "AgentLoop",
    "ConversationGuard",
    "ConversationReuseError",
]
