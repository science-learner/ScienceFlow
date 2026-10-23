"""Atomic final-report generation and manifest validation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from scienceflow.runtime.parallel.location.local import validate_record_location

from .analysis import enrich_report
from .charts import report_svgs
from .collect import collect_report
from .html import render_html
from .markdown import render_markdown
from .storage.publication import publish_bundle

_FORMATS = ("report.md", "report.html")


def read_report_manifest(root: str | Path) -> dict[str, Any] | None:
    root = Path(root).expanduser().absolute()
    if root.resolve() != root or not (root / "report.json").resolve().is_relative_to(
        root
    ):
        return None
    try:
        value = json.loads((root / "report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        return None
    formats = value.get("formats")
    if not isinstance(formats, dict):
        return None
    value["formats"] = {
        name: bool(
            formats.get(name)
            and (root / name).resolve().is_relative_to(root)
            and (root / name).is_file()
        )
        for name in _FORMATS
    }
    value.pop("pdf_error", None)
    if value.get("status") in {"ready", "partial"}:
        value["status"] = "ready" if all(value["formats"].values()) else "partial"
    return value


def generate_task_report(
    record: dict[str, Any], root: str | Path, *, enrich: bool = True
) -> dict[str, Any]:
    root = Path(root).expanduser().absolute()
    validate_record_location(dict(record, task_roots=[str(root)]))
    if root.resolve() != root:
        raise ValueError("Report task root has moved or contains a symlink")
    # A launch failure can precede worker workspace creation. Preserve a report
    # at the authoritative task root even when no research artifacts exist yet.
    root.mkdir(parents=True, exist_ok=True)
    report = collect_report(record, root)
    if enrich:
        report = enrich_report(
            record, report, cache_path=root / "task_logs/report_analysis.json"
        )
    else:
        report = replace(
            report, report_agent={"status": "local_evidence_only", "model": ""}
        )
    with TemporaryDirectory(prefix=".report-", dir=root) as temporary_name:
        temporary = Path(temporary_name)
        assets = temporary / "report_assets"
        assets.mkdir()
        svgs = report_svgs(report)
        for name, content in svgs.items():
            (assets / name).write_text(content, encoding="utf-8")
        markdown = render_markdown(report)
        (temporary / "report.md").write_text(markdown, encoding="utf-8")
        (temporary / "report.html").write_text(
            render_html(markdown, svgs), encoding="utf-8"
        )
        formats = {name: True for name in _FORMATS}
        manifest = {
            "schema_version": 1,
            "document_schema_version": report.schema_version,
            "status": "ready",
            "generated_at_utc": report.generated_at_utc,
            "run_id": report.run_id,
            "attempt": report.attempt,
            "task_status": report.status,
            "report_agent": report.report_agent,
            "related_work_count": len(report.related_works),
            "formats": formats,
        }
        (temporary / "report.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        archive = root / "task_logs/reports" / f"attempt-{report.attempt:03d}"
        publish_bundle(temporary, archive)
        publish_bundle(temporary, root)
    return manifest


def generate_run_reports(
    record: dict[str, Any], *, enrich: bool = True
) -> list[dict[str, Any]]:
    results = []
    for root in record.get("task_roots") or ():
        try:
            result = generate_task_report(record, root, enrich=enrich)
        # One malformed task root must not suppress reports for sibling tasks.
        except Exception as exc:  # noqa: BLE001
            result = {
                "schema_version": 1,
                "status": "generation_failed",
                "run_id": str(record.get("run_id") or ""),
                "attempt": int(record.get("attempt") or 1),
                "task_status": str(record.get("status") or "unknown"),
                "formats": {name: False for name in _FORMATS},
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(dict(result, task_root=str(Path(root).resolve())))
    return results
