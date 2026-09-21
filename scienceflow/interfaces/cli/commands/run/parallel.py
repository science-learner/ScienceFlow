"""Parallel task command."""

import asyncio

import click

@click.command()
@click.option("--manifest", "-m", required=True, help="Task manifest YAML path")
@click.option("--max-concurrent", "-j", default=None, type=int, help="Override max concurrent tasks")
@click.option(
    "--log-dir",
    default=None,
    help="Legacy external subprocess log directory; by default logs stay in each task workspace.",
)
def parallel(manifest, max_concurrent, log_dir):
    """Run multiple tasks in parallel with per-task CPU/GPU isolation."""
    from scienceflow.runtime.parallel.service import run_manifest

    try:
        summary = asyncio.run(
            run_manifest(
                manifest,
                max_concurrent=max_concurrent,
                log_dir=log_dir,
                on_started=lambda count, concurrency: click.echo(
                    f"Launching {count} tasks (max_concurrent={concurrency})"
                ),
            )
        )
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    click.echo("\n" + summary.text)

    failed = [r for r in summary.results if r.status not in ("success", "skipped")]
    if failed:
        raise SystemExit(1)
