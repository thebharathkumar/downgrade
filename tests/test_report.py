"""Report, loader, judge client and `downgrade report` tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from _builders import CHEAP, GOOD_CALLS, arm_runs, config, task, trajectory
from click.testing import CliRunner

from downgrade.classify import classify_sweep
from downgrade.classify.judge import (
    AnthropicJudge,
    JudgeAssessment,
    JudgeIssue,
    render_evidence,
)
from downgrade.cli import PROBE_TASK, main
from downgrade.models import SweepConfig, Trajectory
from downgrade.report import RunBundle, headline, load_bundle, render_markdown, save_bundle
from downgrade.stats import analyze
from downgrade.stats.analysis import SweepAnalysis


def analysis_for(**pref5: Any) -> SweepAnalysis:
    specs = {f"t{i}": task(f"t{i}") for i in range(6)}
    runs: list[Trajectory] = []
    for tid in specs:
        runs += arm_runs("baseline_direct", task_id=tid)
        runs += arm_runs("pref_5", task_id=tid, served=CHEAP, **pref5)
    return analyze(classify_sweep(specs, runs, config=config()), config())


class TestMarkdown:
    def test_null_result_is_reported_as_such(self) -> None:
        analysis = analysis_for()
        assert headline(analysis).startswith("No regression detected")
        text = render_markdown(analysis)
        assert "No downgraded arm drew any finding." in text
        assert "## 5. Evidence" not in text

    def test_noise_floor_comes_before_any_arm_result(self) -> None:
        text = render_markdown(analysis_for())
        assert text.index("Noise floor") < text.index("Per-arm outcomes") < text.index("Tests")

    def test_silent_discovery_has_headline_table_and_evidence(self) -> None:
        text = render_markdown(analysis_for(calls=GOOD_CALLS[:3]))
        assert text.splitlines()[2].startswith("**SILENT regression detected")
        assert "| pref_5 | missing_tool_call | structural | 30/30 |" in text
        assert "### pref_5: missing_tool_call (30/30 runs)" in text
        assert "... and 27 more" in text

    def test_loud_discovery(self) -> None:
        analysis = analysis_for(status="errored", error="HTTP 500")
        assert "LOUD regression detected" in headline(analysis)

    def test_structural_mode_says_judge_rows_are_absent(self) -> None:
        assert "were not run. Those rows are absent, not zero." in render_markdown(analysis_for())

    def test_unprobed_router_format_still_renders(self) -> None:
        analysis = analysis_for()
        analysis.config = analysis.config.model_copy(update={"short_slugs": None})
        assert "wire format not recorded" in render_markdown(analysis)

    def test_baseline_floor_note(self) -> None:
        specs = {"t0": task("t0")}
        runs = arm_runs("baseline_direct", n=4, task_id="t0") + [
            trajectory(task_id="t0", replicate=4, calls=GOOD_CALLS[:2])
        ]
        runs += arm_runs("pref_5", task_id="t0")
        analysis = analyze(classify_sweep(specs, runs, config=config()), config())
        assert "The classifier fires on 1 of 5 baseline runs" in render_markdown(analysis)

    def test_no_floor_warning(self) -> None:
        specs = {"t0": task("t0")}
        runs = arm_runs("baseline_direct", n=1, task_id="t0") + arm_runs("pref_5", task_id="t0")
        analysis = analyze(classify_sweep(specs, runs, config=config()), config())
        assert "No floor could be computed" in render_markdown(analysis)

    def test_table_cells_escape_pipes_and_truncate(self) -> None:
        from downgrade.report.markdown import _cell

        assert _cell("a | b") == "a \\| b"
        assert _cell("x" * 100, limit=10) == "xxxxxxx..."
        assert _cell(None) == ""


class TestLoader:
    def bundle(self) -> RunBundle:
        return RunBundle(config=config(), trajectories=arm_runs("baseline_direct"))

    def test_round_trip(self, tmp_path: Path) -> None:
        save_bundle(self.bundle(), tmp_path / "runs.json")
        loaded = load_bundle(tmp_path / "runs.json")
        assert loaded.config.fingerprint == config().fingerprint
        assert loaded.task_ids == ["equity"] and loaded.arms == ["baseline_direct"]

    def test_reads_smoke_output(self, tmp_path: Path) -> None:
        payload = {
            "config": json.loads(config().model_dump_json()),
            "arms": [{"trajectories": [json.loads(t.model_dump_json()) for t in arm_runs("x")]}],
        }
        (tmp_path / "smoke.json").write_text(json.dumps(payload))
        assert len(load_bundle(tmp_path / "smoke.json").trajectories) == 5

    def test_merges_a_directory(self, tmp_path: Path) -> None:
        save_bundle(self.bundle(), tmp_path / "a.json")
        save_bundle(RunBundle(config(), arm_runs("pref_5")), tmp_path / "b.json")
        assert len(load_bundle(tmp_path).trajectories) == 10

    def test_edited_config_is_refused(self, tmp_path: Path) -> None:
        save_bundle(self.bundle(), tmp_path / "runs.json")
        data = json.loads((tmp_path / "runs.json").read_text())
        data["config"]["fdr_q"] = 0.2
        (tmp_path / "runs.json").write_text(json.dumps(data))
        with pytest.raises(ValueError, match="different config"):
            load_bundle(tmp_path / "runs.json")

    def test_directory_with_mixed_fingerprints_is_refused(self, tmp_path: Path) -> None:
        save_bundle(self.bundle(), tmp_path / "a.json")
        other = config(seed=1)
        runs = arm_runs("pref_5", fingerprint=other.fingerprint)
        save_bundle(RunBundle(other, runs), tmp_path / "b.json")
        with pytest.raises(ValueError, match="disagree"):
            load_bundle(tmp_path)

    def test_duplicate_runs_are_refused(self, tmp_path: Path) -> None:
        save_bundle(self.bundle(), tmp_path / "a.json")
        save_bundle(self.bundle(), tmp_path / "b.json")
        with pytest.raises(ValueError, match="duplicate run ids"):
            load_bundle(tmp_path)

    def test_malformed_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.json").write_text("{}")
        with pytest.raises(ValueError, match="no 'config'"):
            load_bundle(tmp_path / "a.json")
        (tmp_path / "a.json").write_text(
            json.dumps({"config": json.loads(config().model_dump_json())})
        )
        with pytest.raises(ValueError, match="neither"):
            load_bundle(tmp_path / "a.json")
        with pytest.raises(FileNotFoundError):
            load_bundle(_empty(tmp_path))


def _empty(tmp_path: Path) -> Path:
    target = tmp_path / "empty"
    target.mkdir()
    return target


class FakeResponse:
    def __init__(self, parsed: Any, stop_reason: str = "end_turn") -> None:
        self.parsed_output = parsed
        self.stop_reason = stop_reason


class FakeMessages:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.kwargs: dict[str, Any] = {}

    def parse(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, messages: FakeMessages) -> None:
        self.messages = messages


class TestAnthropicJudge:
    def test_sends_evidence_and_returns_parsed_output(self) -> None:
        verdict = JudgeAssessment(unsupported_claims=[JudgeIssue(claim="c", reason="r")])
        messages = FakeMessages(FakeResponse(verdict))
        judge = AnthropicJudge(FakeClient(messages))
        assert judge.assess(task(), trajectory()) is verdict
        assert messages.kwargs["model"] == "claude-opus-5-5"
        assert messages.kwargs["output_format"] is JudgeAssessment
        assert "FINAL ANSWER" in messages.kwargs["messages"][0]["content"]

    @pytest.mark.parametrize(
        ("messages", "note"),
        [
            (FakeMessages(error=RuntimeError("down")), "RuntimeError"),
            (FakeMessages(FakeResponse(None, "refusal")), "refused"),
            (FakeMessages(FakeResponse("not a model")), "no structured output"),
        ],
    )
    def test_failures_leave_the_run_unjudged(self, messages: FakeMessages, note: str) -> None:
        result = AnthropicJudge(FakeClient(messages)).assess(task(), trajectory())
        assert not result.available and note in result.note

    def test_runs_without_an_answer_are_not_sent(self) -> None:
        messages = FakeMessages()
        result = AnthropicJudge(FakeClient(messages)).assess(task(), trajectory(status="errored"))
        assert not result.available and messages.kwargs == {}

    def test_builds_a_default_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        assert AnthropicJudge(model="m").model == "m"

    def test_evidence_shows_errors_missing_results_and_no_calls(self) -> None:
        failed = render_evidence(task(), trajectory(failed={0}))
        assert "ERROR: boom" in failed
        bare = render_evidence(task(), trajectory(calls=[]))
        assert "(none)" in bare
        orphan = trajectory()
        orphan.steps[0].tool_results = []
        assert "(no result)" in render_evidence(task(), orphan)


class TestReportCommand:
    def write_runs(self, tmp_path: Path, cfg: SweepConfig | None = None) -> Path:
        cfg = cfg or config()
        runs = [
            trajectory(task_id="probe_equity", arm=arm, replicate=r, fingerprint=cfg.fingerprint)
            for arm in ("baseline_direct", "pref_5")
            for r in range(3)
        ]
        path = tmp_path / "runs.json"
        save_bundle(RunBundle(cfg, runs), path)
        return path

    def test_writes_markdown_and_json(self, tmp_path: Path) -> None:
        runs = self.write_runs(tmp_path)
        result = CliRunner().invoke(
            main,
            [
                "report",
                "--runs",
                str(runs),
                "--out",
                str(tmp_path / "r.md"),
                "--json",
                str(tmp_path / "r.json"),
            ],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "r.md").read_text().startswith("# downgrade: sweep report")
        assert json.loads((tmp_path / "r.json").read_text())["family_size"] == 7

    def test_prints_to_stdout_by_default(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(main, ["report", "--runs", str(self.write_runs(tmp_path))])
        assert result.exit_code == 0 and "Noise floor" in result.output

    def test_suite_tasks_are_loaded(self, tmp_path: Path) -> None:
        # The suite overrides the probe task's expected answer, so every run in
        # both arms is now wrong. Baseline is wrong too, so the floor absorbs
        # it and the report must NOT call this a routing regression.
        suite = tmp_path / "suite.yaml"
        suite.write_text(
            "task_id: probe_equity\nfamily: tabular_aggregation\nprompt: p\n"
            "answer_check: {kind: numeric, value: 5000}\n"
        )
        args = ["report", "--runs", str(self.write_runs(tmp_path)), "--suite", str(suite)]
        result = CliRunner().invoke(main, args)
        assert result.exit_code == 0, result.output
        assert "No regression detected" in result.output
        assert "| wrong_answer | structural | 3/3 | 100.0% |" in result.output

    def test_unknown_tasks_and_bad_runs_fail_cleanly(self, tmp_path: Path) -> None:
        cfg = config()
        path = tmp_path / "runs.json"
        save_bundle(RunBundle(cfg, arm_runs("baseline_direct", task_id="mystery")), path)
        result = CliRunner().invoke(main, ["report", "--runs", str(path)])
        assert result.exit_code != 0 and "no task spec" in result.output
        (tmp_path / "bad.json").write_text("{}")
        result = CliRunner().invoke(main, ["report", "--runs", str(tmp_path / "bad.json")])
        assert result.exit_code != 0 and "no 'config'" in result.output

    @pytest.mark.parametrize("mode", ["judge", "both"])
    def test_judge_modes_build_a_judge(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
    ) -> None:
        built: list[str] = []

        class StubJudge:
            def __init__(self, model: str) -> None:
                built.append(model)

            def assess(self, task: Any, trajectory: Any) -> JudgeAssessment:
                return JudgeAssessment()

        monkeypatch.setattr("downgrade.classify.judge.AnthropicJudge", StubJudge)
        args = ["report", "--runs", str(self.write_runs(tmp_path)), "--mode", mode]
        result = CliRunner().invoke(main, [*args, "--judge-model", "claude-haiku-5-5"])
        assert result.exit_code == 0, result.output
        assert built == ["claude-haiku-5-5"]

    def test_probe_task_is_always_available(self) -> None:
        assert PROBE_TASK.task_id == "probe_equity"
