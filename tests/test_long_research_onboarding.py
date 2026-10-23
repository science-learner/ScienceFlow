from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scienceflow.research.onboarding import (
    LongResearchSession,
    OnboardingState,
    PreflightStatus,
    build_parallel_manifest,
    extract_task_text,
    parse_cpu_list,
    parse_duration,
    parse_gpu_selection,
    run_preflight,
    write_onboarding_files,
)
from scienceflow.runtime.parallel.execution.runner import ParallelRunner
from scienceflow.runtime.task_package import find_task_package


def _complete_unknown_session(tmp_path: Path) -> LongResearchSession:
    data = tmp_path / "data"
    data.mkdir()
    (data / "sample.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    session = LongResearchSession.start(
        (("user", "Find a robust model for this new dataset"),),
        workspace=tmp_path,
        constraints=(
            f"data={data} metric=accuracy direction=max "
            "artifact=artifacts/result.json command='python evaluate.py' "
            "gpu=cpu workers=2 cpu=0-7 duration=2h"
        ),
    )
    assert session.state == OnboardingState.PREFLIGHT
    return session


def test_detaches_conversation_and_extracts_latest_non_command_user_task() -> None:
    source = [
        ["user", "first"],
        ["assistant", "response"],
        ["user", "build the best model"],
    ]
    session = LongResearchSession.start(source, workspace=".")
    source[-1][1] = "mutated"

    assert session.conversation[-1] == ("user", "build the best model")
    assert session.draft.task_text == "build the best model"
    assert (
        extract_task_text((("user", "real objective"), ("user", "/long-research")))
        == "real objective"
    )


def test_registered_task_autofills_authoritative_contract(tmp_path: Path) -> None:
    session = LongResearchSession.start(
        (("user", "Continue sci-modeling-bench-tfbind8"),),
        workspace=tmp_path,
        constraints="gpu=cpu workers=2 cpu=0-3 duration=10m",
    )

    assert session.draft.registered_task
    assert not session.draft.exploratory
    package = find_task_package("sci-modeling-bench-tfbind8")
    assert package is not None
    assert session.draft.metric_name == package.metric_name
    assert session.draft.lower_is_better is package.lower_is_better
    assert session.draft.artifact_path == package.artifact_path
    assert session.draft.evaluator["backend"] == "task_package"
    assert session.next_question().field == "input_data_dir"  # type: ignore[union-attr]
    assert session.answer(str(tmp_path)).action == "preflight"


def test_session_asks_only_missing_fields_and_supports_cancel_confirm_run(
    tmp_path: Path,
) -> None:
    session = _complete_unknown_session(tmp_path)
    assert session.next_question() is None
    assert session.answer("run").action == "not_ready"

    files = write_onboarding_files(
        session.draft, tmp_path, conversation=session.conversation
    )
    report = run_preflight(files.manifest_path)
    update = session.apply_preflight(report)
    assert report.status == PreflightStatus.EXPLORATORY
    assert update.state == OnboardingState.CONFIRM
    assert session.answer("run").state == OnboardingState.RUNNING

    cancelled = LongResearchSession.start((), workspace=tmp_path).answer("cancel")
    assert cancelled.state == OnboardingState.CANCELLED


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("cpu", "cpu"), ("AUTO", "auto"), ("auto:2", "auto:2"), ("1,0", "1,0")],
)
def test_gpu_parser(raw: str, expected: str) -> None:
    assert parse_gpu_selection(raw) == expected


def test_resource_and_duration_parsers_reject_ambiguous_values() -> None:
    assert parse_cpu_list("0-3, 6, 8-9") == "0-3,6,8-9"
    assert parse_duration("600") == 600
    assert parse_duration("10m") == 600
    assert parse_duration("2h") == 7200
    with pytest.raises(ValueError):
        parse_gpu_selection("0,0")
    with pytest.raises(ValueError):
        parse_cpu_list("4-2")


@pytest.mark.parametrize("raw", ("2", "599s", "9m"))
def test_duration_parser_enforces_ten_minute_minimum(raw: str) -> None:
    with pytest.raises(ValueError, match="at least 10 minutes"):
        parse_duration(raw)


def test_unknown_manifest_is_unique_canonical_and_runner_loadable(
    tmp_path: Path,
) -> None:
    session = _complete_unknown_session(tmp_path)
    first = build_parallel_manifest(session.draft)
    first_run = first["tasks"][0]["run_id"]  # type: ignore[index]
    session.draft.run_id = ""
    second = build_parallel_manifest(session.draft)
    second_run = second["tasks"][0]["run_id"]  # type: ignore[index]
    assert first_run != second_run

    files = write_onboarding_files(
        session.draft, tmp_path, conversation=session.conversation
    )
    manifest = yaml.safe_load(files.manifest_path.read_text(encoding="utf-8"))
    task = manifest["tasks"][0]
    assert task["evaluator"]["backend"] == "artifact_command"
    assert task["evaluator"]["command"]["evaluator_command"] == "python evaluate.py"
    assert task["gpu_list"] == "cpu"
    assert task["lnr"] == {"wall_clock_budget_sec": 7200, "num_workers": 2}
    assert ParallelRunner(files.manifest_path).manifest_path == files.manifest_path

    provenance = json.loads(files.onboarding_path.read_text(encoding="utf-8"))
    assert provenance["classification"] == "exploratory"
    assert provenance["source"]["kind"] == "detached_conversation"


def test_registered_task_preflight_is_verified_and_reported(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "inputs.json").write_text("{}\n", encoding="utf-8")
    session = LongResearchSession.start(
        (("user", "Run sci-modeling-bench-tfbind8"),),
        workspace=tmp_path,
        constraints=f"data={data} gpu=cpu workers=2 cpu=0-3 duration=10m",
    )
    assert session.state == OnboardingState.PREFLIGHT
    files = write_onboarding_files(
        session.draft, tmp_path, conversation=session.conversation
    )

    report = run_preflight(files.manifest_path)

    assert report.status == PreflightStatus.VERIFIED
    assert all(check.ok for check in report.checks)
    persisted = json.loads(
        (tmp_path / ".scienceflow" / "preflight_report.json").read_text()
    )
    assert persisted["status"] == "verified"
    assert persisted["evaluator_backend"] == "task_package"


def test_preflight_fails_closed_for_missing_dataset_and_unknown_backend(
    tmp_path: Path,
) -> None:
    session = _complete_unknown_session(tmp_path)
    files = write_onboarding_files(session.draft, tmp_path)
    manifest = yaml.safe_load(files.manifest_path.read_text(encoding="utf-8"))
    manifest["tasks"][0]["input_data_dir"] = str(tmp_path / "missing")
    manifest["tasks"][0]["evaluator"]["backend"] = "invented"
    files.manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )

    report = run_preflight(files.manifest_path)

    assert report.status == PreflightStatus.FAILED
    failures = {check.name for check in report.checks if not check.ok}
    assert {"parallel_loader", "dataset_scan", "evaluator_registry"} <= failures


def test_preflight_rejects_resource_config_that_cannot_slice_workers(
    tmp_path: Path,
) -> None:
    session = _complete_unknown_session(tmp_path)
    files = write_onboarding_files(session.draft, tmp_path)
    manifest = yaml.safe_load(files.manifest_path.read_text(encoding="utf-8"))
    manifest["tasks"][0]["cpu_list"] = "0"
    manifest["tasks"][0]["lnr"]["num_workers"] = 2
    files.manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )

    report = run_preflight(files.manifest_path)

    resource = next(check for check in report.checks if check.name == "resource_config")
    assert report.status == PreflightStatus.FAILED
    assert not resource.ok
    assert "one" in resource.detail


@pytest.mark.parametrize("status", ["verified", "exploratory", "failed"])
def test_run_requires_successful_preflight_but_no_separate_confirmation(tmp_path, status):
    from types import SimpleNamespace

    session = _complete_unknown_session(tmp_path)
    session.apply_preflight(SimpleNamespace(status=status))
    result = session.answer("run")
    assert result.action == ("not_ready" if status == "failed" else "run")


def test_chinese_argument_separators_and_minute_shorthand_preserve_quoted_values():
    from scienceflow.research.onboarding.support.answers import parse_constraints

    values = parse_constraints('circle packing workers=2，cpu=8，20min')
    assert values['workers'] == 2
    assert values['cpu_list'] == '8'  # Explicit CPU IDs retain their existing meaning.
    assert values['wall_clock_sec'] == 1200
    assert values['task_text'] == 'circle packing'
    quoted = parse_constraints('task="math，with punctuation" cpu=0-3,8-11；duration=20分钟')
    assert quoted['task_text'] == 'math，with punctuation'
    assert quoted['cpu_list'] == '0-3,8-11'
    assert quoted['wall_clock_sec'] == 1200
