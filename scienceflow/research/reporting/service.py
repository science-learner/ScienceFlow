"""Atomic final-report generation and manifest validation."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .analysis import enrich_report
from .charts import report_svgs
from .collect import collect_report
from .html import render_html
from .markdown import render_markdown
from .pdf import render_pdf

_FORMATS = ("report.md", "report.html", "report.pdf")


def _replace(source: Path, target: Path) -> None:
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def read_report_manifest(root: str | Path) -> dict[str, Any] | None:
    root = Path(root).expanduser().resolve()
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
        name: bool(formats.get(name) and (root / name).is_file()) for name in _FORMATS
    }
    return value


def generate_task_report(record: dict[str, Any], root: str | Path) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Task root does not exist: {root}")
    report = enrich_report(
        record,
        collect_report(record, root),
        cache_path=root / "task_logs/report_analysis.json",
    )
    with TemporaryDirectory(prefix=".report-", dir=root) as temporary_name:
        temporary = Path(temporary_name)
        assets = temporary / "report_assets"
        assets.mkdir()
        svgs = report_svgs(report)
        for name, content in svgs.items():
            (assets / name).write_text(content, encoding="utf-8")
        (temporary / "report.md").write_text(render_markdown(report), encoding="utf-8")
        (temporary / "report.html").write_text(
            render_html(report, svgs), encoding="utf-8"
        )
        pdf_error = ""
        try:
            render_pdf(report, temporary / "report.pdf")
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            pdf_error = f"{type(exc).__name__}: {exc}"
        formats = {
            "report.md": True,
            "report.html": True,
            "report.pdf": (temporary / "report.pdf").is_file(),
        }
        manifest = {
            "schema_version": 1,
            "document_schema_version": report.schema_version,
            "status": "ready" if formats["report.pdf"] else "partial",
            "generated_at_utc": report.generated_at_utc,
            "run_id": report.run_id,
            "attempt": report.attempt,
            "task_status": report.status,
            "report_agent": report.report_agent,
            "related_work_count": len(report.related_works),
            "formats": formats,
            "pdf_error": pdf_error,
        }
        (temporary / "report.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        target_assets = root / "report_assets"
        target_assets.mkdir(exist_ok=True)
        for name in svgs:
            _replace(assets / name, target_assets / name)
        for name in _FORMATS:
            source = temporary / name
            if source.is_file():
                _replace(source, root / name)
            elif (root / name).is_file():
                (root / name).unlink()
        archive = root / "task_logs/reports" / f"attempt-{report.attempt:03d}"
        if not archive.exists():
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(temporary, archive)
        _replace(temporary / "report.json", root / "report.json")
    return manifest


def generate_run_reports(record: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for root in record.get("task_roots") or ():
        try:
            result = generate_task_report(record, root)
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
