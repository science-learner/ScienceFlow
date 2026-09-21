"""Final reports remain local, deterministic and readable after task termination."""

import json
from pathlib import Path

from scienceflow.research.reporting import (
    generate_task_report,
    read_report_manifest,
)
from scienceflow.research.reporting.charts import lineage_svg
from scienceflow.research.reporting.lineage import lineage_layout


def _record(root: Path, **updates):
    value = {
        "run_id": "run-1",
        "attempt": 2,
        "status": "failed",
        "alive": False,
        "started_at": 10,
        "finished_at": 70,
        "error": "API_KEY=sk-abcdefghijklmnop at https://private.example/v1",
        "task_roots": [str(root)],
        "draft": {
            "exp_id": "circle-packing",
            "wall_clock_sec": 120,
            "metric_name": "radius",
            "lower_is_better": False,
        },
    }
    value.update(updates)
    return value


def test_failed_task_generates_all_formats_archive_and_redacts_secrets(tmp_path):
    root = tmp_path / "task"
    logs = root / "task_logs"
    notes = root / "workers/w00/workspace"
    logs.mkdir(parents=True)
    notes.mkdir(parents=True)
    (logs / "state.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "failure_kind": "provider_error",
                "error": "Authorization: Bearer hidden-token",
                "resume_retriable": True,
            }
        ),
        encoding="utf-8",
    )
    (notes / "result.md").write_text(
        "Candidate notes <script>alert(1)</script> https://private.example/result",
        encoding="utf-8",
    )

    result = generate_task_report(_record(root), root)

    assert result["status"] == "ready"
    assert result["task_status"] == "failed"
    assert all(result["formats"].values())
    assert (root / "report.pdf").read_bytes().startswith(b"%PDF-")
    markdown = (root / "report.md").read_text(encoding="utf-8")
    html = (root / "report.html").read_text(encoding="utf-8")
    assert "Bearer <redacted>" in markdown
    assert "hidden-token" not in markdown
    assert "private.example" not in markdown
    assert "&lt;script&gt;" in html and "<script>alert" not in html
    archive = root / "task_logs/reports/attempt-002"
    assert (archive / "report.md").is_file()
    assert (archive / "report_assets/metric-history.svg").is_file()
    assert read_report_manifest(root) == result


def test_corrupt_monitor_sources_still_produce_a_readable_report(
    tmp_path,
    monkeypatch,
):
    from scienceflow.research.reporting import collect

    root = tmp_path / "task"
    (root / "task_logs").mkdir(parents=True)
    (root / "task_logs/state.json").write_text("not-json", encoding="utf-8")

    def broken(*_args, **_kwargs):
        raise ValueError("broken evidence")

    monkeypatch.setattr(collect.TaskProjection, "item", broken)
    monkeypatch.setattr(collect, "build_task_trace", broken)

    result = generate_task_report(_record(root, status="stopped"), root)

    assert result["status"] == "ready"
    assert result["task_status"] == "stopped"
    assert "No comparable metric points" in (
        root / "report_assets/metric-history.svg"
    ).read_text(encoding="utf-8")


def test_manifest_only_advertises_files_that_still_exist(tmp_path):
    root = tmp_path / "task"
    root.mkdir()
    generate_task_report(_record(root), root)
    (root / "report.pdf").unlink()

    result = read_report_manifest(root)

    assert result is not None
    assert result["formats"]["report.md"] is True
    assert result["formats"]["report.pdf"] is False


def test_scientific_report_collects_task_finalists_and_artifact_figure(tmp_path):
    root = tmp_path / "task"
    description = root / "task_runtime/description.md"
    final = root / "merge/finals/final_00"
    artifact = final / "artifacts/best_solution.json"
    description.parent.mkdir(parents=True)
    artifact.parent.mkdir(parents=True)
    description.write_text(
        "# Circle packing\n\n## Problem\nPlace two circles in a unit square.\n",
        encoding="utf-8",
    )
    artifact.write_text(
        json.dumps({"circles": [[0.25, 0.5, 0.2], [0.75, 0.5, 0.2]]}),
        encoding="utf-8",
    )
    (final / "eval_result.json").write_text(
        json.dumps(
            {
                "candidate_id": "final_00",
                "metric_value": 0.4,
                "validation_ok": True,
                "selection_eligible": True,
                "evaluator_status": "ok",
                "artifact_path": "artifacts/best_solution.json",
                "artifact_sha": "abcdef1234567890",
            }
        ),
        encoding="utf-8",
    )
    (final / "merge_report.md").write_text(
        "Input: candidate (W00:L01:S02).\n", encoding="utf-8"
    )

    result = generate_task_report(_record(root, status="completed", error=""), root)

    markdown = (root / "report.md").read_text(encoding="utf-8")
    html = (root / "report.html").read_text(encoding="utf-8")
    assert result["document_schema_version"] == 3
    assert "## Abstract" in markdown
    assert "## Research question and objective" in markdown
    assert "Place two circles" in markdown
    assert "W00:L01:S02" in markdown
    assert "Scientific Research Report" in html
    assert "Final candidate comparison" in html
    assert (root / "report_assets/candidate-comparison.svg").is_file()
    artifact_svg = (root / "report_assets/result-artifact.svg").read_text(
        encoding="utf-8"
    )
    assert artifact_svg.count("<circle ") == 2


def test_related_research_and_agent_analysis_render_without_inventing_metrics(
    tmp_path,
    monkeypatch,
):
    from dataclasses import replace

    from scienceflow.research.reporting import service

    root = tmp_path / "task"
    root.mkdir()

    def enriched(_record, report, **_kwargs):
        return replace(
            report,
            related_works=[
                {
                    "id": "R1",
                    "title": "A related optimization study",
                    "authors": ["A. Researcher"],
                    "year": 2024,
                    "venue": "Journal of Examples",
                    "doi": "10.1000/example",
                    "url": "https://doi.org/10.1000/example",
                    "citations": 12,
                    "retrieval_score": 8.0,
                    "type": "journal-article",
                }
            ],
            research_analysis={
                "related_work": "R1 provides related optimization context.",
                "interpretation": "The persisted evidence is insufficient for comparison.",
                "limitations": ["Only one retrieval candidate was available."],
            },
            report_agent={"status": "ready", "model": "feedback-model"},
        )

    monkeypatch.setattr(service, "enrich_report", enriched)
    result = generate_task_report(_record(root), root)

    markdown = (root / "report.md").read_text(encoding="utf-8")
    html = (root / "report.html").read_text(encoding="utf-8")
    assert result["related_work_count"] == 1
    assert "## Related research" in markdown
    assert "https://doi.org/10.1000/example" in markdown
    assert "Related research landscape" in html
    assert (root / "report_assets/related-work.svg").is_file()


def test_report_collects_hierarchical_stage_ancestry(tmp_path):
    import csv

    from scienceflow.research.reporting.collect import collect_report

    root = tmp_path / "task"
    logs = root / "task_logs"
    logs.mkdir(parents=True)
    with (logs / "lhr_stage_performance.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "worker_id",
                "stage_id",
                "node_uid",
                "parent_stage_id",
                "lineage_id",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "worker_id": "W00",
                "stage_id": "S01",
                "node_uid": "W00:L01:S01",
                "lineage_id": "L01",
            }
        )
        writer.writerow(
            {
                "worker_id": "W00",
                "stage_id": "S02",
                "node_uid": "W00:L01:S02",
                "parent_stage_id": "S01",
                "lineage_id": "L01",
            }
        )

    report = collect_report(_record(root), root)

    assert report.stages[1]["parent"] == "W00:L01:S01"
    svg = lineage_svg(report)
    assert 'width="204" height="48"' in svg
    assert ">L01</text>" in svg
    assert "Depth " not in svg
    assert "report-parent-arrow" in svg

    positions, _, lanes = lineage_layout(
        {
            "a": {"lineage": "L01", "timestamp": 1},
            "b": {"lineage": "L01", "timestamp": 2, "parent": "a"},
            "c": {"lineage": "L02", "timestamp": 3, "restored": "b"},
        }
    )
    assert lanes == ["L01", "L02"]
    assert positions["a"][1] == positions["b"][1]
    assert positions["a"][0] < positions["b"][0]
    assert positions["b"][0] == positions["c"][0]


def test_report_agent_plans_retrieval_then_synthesizes_from_evidence(
    tmp_path,
    monkeypatch,
):
    import asyncio

    from scienceflow.research.reporting import analysis
    from scienceflow.research.reporting.collect import collect_report

    root = tmp_path / "task"
    root.mkdir()
    report = collect_report(_record(root), root)
    responses = iter(
        [
            '{"query":"circle packing unit square"}',
            json.dumps(
                {
                    "summary": "Evidence-grounded summary.",
                    "related_work": "R1 is relevant context.",
                    "interpretation": "No causal claim is supported.",
                    "limitations": ["The search set is small."],
                }
            ),
        ]
    )

    class FakeLLM:
        async def ask(self, **_kwargs):
            return next(responses)

    captured = []

    def search(_report, *, query="", timeout=8):
        captured.append((query, timeout))
        return [
            {
                "id": "R1",
                "title": "Circle packing in a square",
                "authors": [],
                "year": 2020,
                "venue": "Examples",
                "doi": "10.1000/example",
                "url": "https://doi.org/10.1000/example",
                "citations": 3,
                "retrieval_score": 1.0,
                "type": "article",
            }
        ]

    async def close(_llm):
        return None

    monkeypatch.setattr(
        analysis, "build_stage_llm", lambda *_args, **_kwargs: FakeLLM()
    )
    monkeypatch.setattr(analysis, "search_related_works", search)
    monkeypatch.setattr(analysis, "aclose_llm_clients", close)
    record = _record(
        root,
        manifest_payload={
            "tasks": [
                {
                    "run_id": report.run_id,
                    "agent": {"feedback": {"model_aliases": ["feedback-test"]}},
                }
            ]
        },
    )

    audit_dir = root / "task_logs" / "report_agent_runtime_audit"
    result, status, related = asyncio.run(
        analysis._ask_report_agent(record, report, audit_dir=audit_dir)
    )

    assert captured == [("circle packing unit square", 8)]
    assert status == {
        "status": "ready",
        "model": "feedback-test",
        "query": "circle packing unit square",
    }
    assert related[0]["id"] == "R1"
    assert result["limitations"] == ["The search set is small."]
    assert len(list((audit_dir / "agent_runtime_operations").glob("*.jsonl"))) == 2
    provider = [
        json.loads(line)
        for line in (audit_dir / "agent_provider_calls.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["turn_kind"] for row in provider] == [
        "report_search_query",
        "report_analysis",
    ]
    assert all(row["llm_role"] == "feedback" for row in provider)
    assert (audit_dir / "agent_runtime_events.jsonl").is_file()


def test_successful_report_analysis_is_reused_for_same_evidence(
    tmp_path,
    monkeypatch,
):
    from scienceflow.research.reporting import analysis
    from scienceflow.research.reporting.collect import collect_report

    root = tmp_path / "task"
    root.mkdir()
    record = _record(root, manifest_payload={"tasks": []})
    report = collect_report(record, root)
    calls = []

    async def ask(_record, _report, **_kwargs):
        calls.append(True)
        return (
            {"summary": "cached analysis"},
            {"status": "ready", "model": "feedback-test", "query": "test query"},
            [{"id": "R1", "title": "Related work"}],
        )

    monkeypatch.setattr(analysis, "_ask_report_agent", ask)
    cache = root / "task_logs/report_analysis.json"
    first = analysis.enrich_report(record, report, cache_path=cache)
    second = analysis.enrich_report(record, report, cache_path=cache)

    assert first.report_agent["status"] == "ready"
    assert second.report_agent["status"] == "cached"
    assert second.research_analysis == {"summary": "cached analysis"}
    assert len(calls) == 1
    assert cache.stat().st_mode & 0o777 == 0o600
