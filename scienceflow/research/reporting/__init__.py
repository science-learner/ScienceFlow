"""Evidence-backed terminal reports for local and Web consumers."""

from .service import generate_run_reports, generate_task_report, read_report_manifest

__all__ = ("generate_run_reports", "generate_task_report", "read_report_manifest")
