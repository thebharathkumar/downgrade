"""Tests for the pre-sweep probes and the CLI.

These cover the decisions that happen before any money is spent: which wire
format the API accepts, whether the preference dial changes anything, and
what the sweep will cost.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from click.testing import CliRunner

from downgrade.arms import ALL_ARMS, BASELINE_DIRECT, arm_by_name
from downgrade.cli import PROBE_TASK, _probe_registry, main
from downgrade.client import FireworksClient
from downgrade.models import SweepConfig
from downgrade.probe import SmokeReport, probe_router_format, run_smoke
from downgrade.transport import FakeTransport, TransportResponse

PRIMARY = "accounts/fireworks/models/kimi-k3"
CHEAP = "accounts/fireworks/models/glm-5p2"
URL = "https://api.fireworks.ai/inference/v1/chat/completions"


def config(**kw: Any) -> SweepConfig:
    defaults: dict[str, Any] = {
        "primary_model": PRIMARY,
        "secondary_model": CHEAP,
        "short_slugs": True,
    }
    defaults.update(kw)
    return SweepConfig(**defaults)


def answer(text: str, model: str) -> TransportResponse:
    return TransportResponse(
        status_code=200,
        chunks=[
            {"id": "c", "model": model, "choices": [{"delta": {"content": text}}]},
            {"id": "c", "model": model, "choices": [{"finish_reason": "stop"}]},
            {"id": "c", "model": model, "usage": {"prompt_tokens": 120, "completion_tokens": 15}},
        ],
    )


def rejected(detail: str = "model not found") -> TransportResponse:
    return TransportResponse(status_code=404, error=detail)


class TestFormatProbe:
    def test_short_form_accepted(self) -> None:
        transport = FakeTransport([answer("ok", CHEAP), answer("ok", CHEAP)])
        result = probe_router_format(FireworksClient(transport, api_key="k"), config())
        assert result.short_slugs is True
        assert result.resolved
        assert "short slugs" in result.summary()

    def test_falls_through_to_the_full_form(self) -> None:
        transport = FakeTransport([rejected(), answer("ok", CHEAP)])
        result = probe_router_format(FireworksClient(transport, api_key="k"), config())
        assert result.short_slugs is False
        assert "fully-qualified" in result.summary()

    def test_neither_form_accepted_stays_unresolved(self) -> None:
        """Better than silently picking one and 404ing the whole sweep."""
        transport = FakeTransport([rejected(), rejected()])
        result = probe_router_format(FireworksClient(transport, api_key="k"), config())
        assert result.short_slugs is None
        assert not result.resolved
        assert "UNRESOLVED" in result.summary()

    def test_both_forms_are_actually_sent(self) -> None:
        transport = FakeTransport([rejected(), rejected()])
        probe_router_format(FireworksClient(transport, api_key="k"), config())
        models = [r.payload["model"] for r in transport.requests]
        assert models[0] == "firerouter/kimi-k3/glm-5p2"
        assert models[1] == f"firerouter/{PRIMARY}/{CHEAP}"

    def test_the_probe_records_the_failure_detail(self) -> None:
        transport = FakeTransport([rejected("no such model: kimi-k3"), rejected()])
        result = probe_router_format(FireworksClient(transport, api_key="k"), config())
        assert "no such model" in result.attempts[0].detail


def smoke_with(models_per_arm: dict[str, str], replicates: int = 2) -> SmokeReport:
    """Run the smoke sweep with each arm served by a chosen model."""
    responses: list[TransportResponse] = []
    for arm in ALL_ARMS:
        served = models_per_arm[arm.name]
        responses.extend(answer("1000", served) for _ in range(replicates))
    transport = FakeTransport(responses)
    return run_smoke(
        FireworksClient(transport, api_key="k"),
        _probe_registry(),
        config(),
        PROBE_TASK,
        replicates=replicates,
    )


class TestSmokeSweep:
    def test_runs_every_arm(self) -> None:
        report = smoke_with({a.name: CHEAP for a in ALL_ARMS})
        assert [a.arm for a in report.arms] == [a.name for a in ALL_ARMS]
        assert all(a.replicates == 2 for a in report.arms)

    def test_the_control_arm_runs_first(self) -> None:
        """If the sweep is interrupted, the one arm the others are meaningless
        without should already exist."""
        report = smoke_with({a.name: CHEAP for a in ALL_ARMS})
        assert report.arms[0].arm == BASELINE_DIRECT.name

    def test_downgrade_rate_per_arm(self) -> None:
        report = smoke_with(
            {
                "baseline_direct": PRIMARY,
                "pref_1": PRIMARY,
                "pref_2": PRIMARY,
                "pref_3": CHEAP,
                "pref_4": CHEAP,
                "pref_5": CHEAP,
            }
        )
        assert report.arm("pref_1") is not None
        assert report.arm("pref_1").downgrade_rate == 0.0  # type: ignore[union-attr]
        assert report.arm("pref_5").downgrade_rate == 1.0  # type: ignore[union-attr]

    def test_routing_responds_when_arms_differ(self) -> None:
        report = smoke_with(
            {
                "baseline_direct": PRIMARY,
                "pref_1": PRIMARY,
                "pref_2": PRIMARY,
                "pref_3": CHEAP,
                "pref_4": CHEAP,
                "pref_5": CHEAP,
            }
        )
        assert report.routing_responds_to_preference

    def test_routing_flat_across_preferences_is_reported_not_hidden(self) -> None:
        """If the dial does nothing there is no downgrade to measure, and that
        is a real result rather than a bug to work around."""
        report = smoke_with({a.name: PRIMARY for a in ALL_ARMS})
        assert not report.routing_responds_to_preference

    def test_baseline_agreement_is_one_when_the_control_is_stable(self) -> None:
        report = smoke_with({a.name: PRIMARY for a in ALL_ARMS})
        assert report.baseline_answer_agreement == 1.0

    def test_baseline_agreement_falls_when_the_control_disagrees(self) -> None:
        """The noise floor: the control arm varying on its own bounds every
        other number in the report."""
        responses = [answer("1000", PRIMARY), answer("999", PRIMARY)]
        transport = FakeTransport(responses)
        report = run_smoke(
            FireworksClient(transport, api_key="k"),
            _probe_registry(),
            config(),
            PROBE_TASK,
            arms=[BASELINE_DIRECT],
            replicates=2,
        )
        assert report.baseline_answer_agreement == 0.5
        assert report.arms[0].distinct_answers == 2

    def test_served_model_counts(self) -> None:
        report = smoke_with({a.name: CHEAP for a in ALL_ARMS})
        assert report.arm("pref_5").served_model_counts == {CHEAP: 2}  # type: ignore[union-attr]

    def test_status_counts_are_tracked(self) -> None:
        report = smoke_with({a.name: CHEAP for a in ALL_ARMS})
        assert report.arm("pref_5").status_counts == {"completed": 2}  # type: ignore[union-attr]

    def test_extrapolation_scales_linearly(self) -> None:
        report = smoke_with({a.name: CHEAP for a in ALL_ARMS}, replicates=2)
        small = report.extrapolate(tasks=1, replicates=2)
        large = report.extrapolate(tasks=24, replicates=5)
        assert large["input_tokens"] == pytest.approx(small["input_tokens"] * 60)
        assert large["runs"] == 6 * 24 * 5

    def test_extrapolation_of_an_empty_report_is_zero(self) -> None:
        empty = SmokeReport(config=config(), task_id="t")
        assert empty.extrapolate(tasks=24, replicates=5)["runs"] == 0.0
        assert empty.baseline_answer_agreement == 0.0

    def test_unknown_arm_lookup_returns_none(self) -> None:
        report = smoke_with({a.name: CHEAP for a in ALL_ARMS})
        assert report.arm("nope") is None

    def test_usage_completeness_is_surfaced(self) -> None:
        report = smoke_with({a.name: CHEAP for a in ALL_ARMS})
        assert all(a.usage_complete for a in report.arms)


def sse(text: str, model: str) -> str:
    return (
        f'data: {{"id":"c","model":"{model}","choices":[{{"delta":{{"content":"{text}"}}}}]}}\n'
        f'data: {{"id":"c","model":"{model}","choices":[{{"finish_reason":"stop"}}]}}\n'
        f'data: {{"id":"c","model":"{model}","usage":'
        f'{{"prompt_tokens":100,"completion_tokens":10}}}}\n'
        "data: [DONE]\n"
    )


class TestCli:
    def test_requires_an_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
        result = CliRunner().invoke(main, ["doctor", "--primary", "a", "--secondary", "b"])
        assert result.exit_code != 0
        assert "No API key" in result.output

    @respx.mock
    def test_doctor_reports_the_accepted_form(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIREWORKS_API_KEY", "k")
        respx.post(URL).mock(return_value=httpx.Response(200, text=sse("ok", CHEAP)))
        result = CliRunner().invoke(main, ["doctor", "--primary", PRIMARY, "--secondary", CHEAP])
        assert result.exit_code == 0
        assert "short slugs" in result.output
        assert "--short-slugs=true" in result.output

    @respx.mock
    def test_doctor_exits_nonzero_when_unresolved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIREWORKS_API_KEY", "k")
        respx.post(URL).mock(return_value=httpx.Response(404, text="no such model"))
        result = CliRunner().invoke(main, ["doctor", "--primary", PRIMARY, "--secondary", CHEAP])
        assert result.exit_code == 1
        assert "UNRESOLVED" in result.output

    @respx.mock
    def test_smoke_reports_noise_floor_first(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIREWORKS_API_KEY", "k")
        respx.post(URL).mock(return_value=httpx.Response(200, text=sse("1000", CHEAP)))
        result = CliRunner().invoke(
            main,
            [
                "smoke",
                "--primary",
                PRIMARY,
                "--secondary",
                CHEAP,
                "--short-slugs",
                "--replicates",
                "2",
            ],
        )
        assert result.exit_code == 0
        floor_at = result.output.index("Baseline noise floor")
        routing_at = result.output.index("Per-arm routing")
        assert floor_at < routing_at
        assert "control arm agrees with itself" in result.output

    @respx.mock
    def test_smoke_warns_when_the_dial_does_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIREWORKS_API_KEY", "k")
        respx.post(URL).mock(return_value=httpx.Response(200, text=sse("1000", PRIMARY)))
        result = CliRunner().invoke(
            main,
            ["smoke", "--primary", PRIMARY, "--secondary", CHEAP, "--short-slugs"],
        )
        assert "did not change which model served" in result.output

    @respx.mock
    def test_smoke_writes_raw_trajectories(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("FIREWORKS_API_KEY", "k")
        respx.post(URL).mock(return_value=httpx.Response(200, text=sse("1000", CHEAP)))
        out = tmp_path / "runs" / "smoke.json"
        result = CliRunner().invoke(
            main,
            [
                "smoke",
                "--primary",
                PRIMARY,
                "--secondary",
                CHEAP,
                "--short-slugs",
                "--replicates",
                "1",
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == 0
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert len(payload["arms"]) == 6
        assert payload["config"]["temperature"] == 0.0
        assert payload["arms"][0]["trajectories"][0]["conversation_id"].startswith("conv-")

    def test_verify_without_a_manifest_explains_how_to_build_one(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(main, ["verify", "--corpus", str(tmp_path)])
        assert result.exit_code != 0
        assert "fetch_corpus.py" in result.output

    def test_verify_passes_on_a_clean_corpus(self, tmp_path: Path) -> None:
        from downgrade.suite.corpus import CorpusFile, CorpusManifest, sha256_bytes

        data = b"risk factors"
        (tmp_path / "a.txt").write_bytes(data)
        manifest = CorpusManifest(
            generated_at="t",
            files=[
                CorpusFile(
                    path="a.txt",
                    kind="document",
                    sha256=sha256_bytes(data),
                    bytes=len(data),
                    source_url="https://www.sec.gov/x",
                    retrieved_at="t",
                    title="a",
                )
            ],
        )
        (tmp_path / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
        result = CliRunner().invoke(main, ["verify", "--corpus", str(tmp_path)])
        assert result.exit_code == 0
        assert "corpus OK" in result.output

    def test_verify_fails_on_a_tampered_corpus(self, tmp_path: Path) -> None:
        from downgrade.suite.corpus import CorpusFile, CorpusManifest, sha256_bytes

        manifest = CorpusManifest(
            generated_at="t",
            files=[
                CorpusFile(
                    path="a.txt",
                    kind="document",
                    sha256=sha256_bytes(b"original"),
                    bytes=8,
                    source_url="https://www.sec.gov/x",
                    retrieved_at="t",
                    title="a",
                )
            ],
        )
        (tmp_path / "a.txt").write_bytes(b"tampered")
        (tmp_path / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
        result = CliRunner().invoke(main, ["verify", "--corpus", str(tmp_path)])
        assert result.exit_code != 0
        assert "CORRUPTED" in result.output


class TestProbeTask:
    def test_the_probe_task_needs_real_multi_step_tool_use(self) -> None:
        registry = _probe_registry()
        assert registry.names == ["get_figure", "list_companies"]
        assert PROBE_TASK.answer_check.value == 1000.0

    def test_the_expected_answer_is_actually_correct(self) -> None:
        """(1200-800) + (900-300) = 1000. If the fixture changes, this catches it."""
        from downgrade.cli import PROBE_TABLE

        equity = sum(row["assets"] - row["liabilities"] for row in PROBE_TABLE.values())
        assert equity == PROBE_TASK.answer_check.value

    def test_unknown_company_is_a_recoverable_tool_error(self) -> None:
        from downgrade.models import ToolCall

        result = _probe_registry().invoke(
            ToolCall(
                call_id="c", name="get_figure", arguments={"company": "X", "concept": "assets"}
            )
        )
        assert not result.ok
        assert "No such company" in (result.error or "")

    def test_unknown_concept_is_a_recoverable_tool_error(self) -> None:
        from downgrade.models import ToolCall

        result = _probe_registry().invoke(
            ToolCall(call_id="c", name="get_figure", arguments={"company": "ACME", "concept": "x"})
        )
        assert not result.ok
        assert "No such concept" in (result.error or "")


class TestArmsUnused:
    def test_arm_by_name_round_trips(self) -> None:
        assert arm_by_name("pref_5").preference == 5
