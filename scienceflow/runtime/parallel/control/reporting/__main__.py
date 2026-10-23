"""Write a complete evidence report before attempting optional model analysis."""

import sys


def main() -> None:
    from scienceflow.research.reporting import generate_run_reports
    from scienceflow.runtime.parallel.control.registry import read_record, run_dir

    record = read_record(run_dir(sys.argv[1]))
    if record["launch_token"] != sys.argv[2]:
        return
    record = dict(
        record,
        status=record["research_status"],
        finished_at=record["research_finished_at"],
    )
    baseline = generate_run_reports(record, enrich=False)
    print("Evidence reports: " + ", ".join(r["status"] for r in baseline), flush=True)
    if (
        record["status"] == "completed"
        and baseline
        and all(r["status"] == "ready" for r in baseline)
    ):
        generate_run_reports(record)


if __name__ == "__main__":
    from . import report_budget
    from .guard import report_guard

    with report_guard(
        int(sys.argv[3]), report_budget({"report_budget_sec": sys.argv[4]})
    ):
        main()
