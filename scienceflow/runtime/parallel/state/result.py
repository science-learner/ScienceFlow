"""ParallelRunner responsibility: result."""

from __future__ import annotations

from scienceflow.runtime.parallel.config.models import TaskResult


@staticmethod
def format_summary(results: list[TaskResult]) -> str:
    lines = [
        f"{'run_id':<32} {'Status':<10} {'Time':>8} {'Exit':>5}",
        "-" * 60,
    ]
    for r in results:
        time_str = f"{r.elapsed_sec:.0f}s" if r.elapsed_sec else "-"
        exit_str = str(r.exit_code) if r.exit_code is not None else "-"
        rid = r.run_id or r.exp_id
        lines.append(f"{rid:<32} {r.status:<10} {time_str:>8} {exit_str:>5}")
        if r.error:
            lines.append(f"  error: {r.error}")
    return "\n".join(lines)
