"""Resource summary command."""

import json

import click

@click.command("resource-summary")
@click.argument("root", type=click.Path(exists=True))
@click.option("--max-files", default=256, show_default=True, help="Maximum resource_events.jsonl files to scan.")
@click.option("--max-events-per-file", default=None, type=int, help="Optional cap for events read per file.")
@click.option("--json", "json_output", is_flag=True, help="Print machine-readable JSON instead of a table.")
def resource_summary_cmd(root, max_files, max_events_per_file, json_output):
    """Print a run-level summary from resource_events.jsonl files."""
    from scienceflow.research.solver.lnr.resources.runtime.execution.state.unified_store import (
        format_resource_run_summary,
        summarize_resource_run,
    )

    summary = summarize_resource_run(root, max_files=max_files, max_events_per_file=max_events_per_file)
    if json_output:
        click.echo(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    else:
        click.echo(format_resource_run_summary(summary))
