"""Task preparation and managed research commands."""

import asyncio
from pathlib import Path

import click

from scienceflow.interfaces.cli import (
    _SCIENCEFLOW_REPO_ROOT,
    _apply_parallel_manifest_cfg_env,
    _apply_task_workspace,
    _repl_allow_input_data_root,
    _repl_apply_dataset_symlinks,
)
from scienceflow.foundation.config.schema.settings import (
    apply_parallel_manifest_agent_overrides,
    apply_profile_overrides,
    load_cfg,
    prep_cfg,
)
from scienceflow.runtime.observability.logging import setup_logging


@click.command()
@click.option("--task", "-t", required=True, help="Task description")
@click.option("--config", "-c", default=None, help="Config YAML path")
@click.option(
    "--workspace",
    "-w",
    default=".",
    help="Task output root (LNR uses task_logs/ plus worker or inner workspace directories)",
)
@click.option(
    "--input-data-dir",
    "-d",
    "input_data_dir",
    default=None,
    type=click.Path(exists=False),
    help="Override config input_data_dir (read-only raw competition input)",
)
@click.option(
    "--show-preview/--no-show-preview",
    default=False,
    help="Print truncated data layout after preparing dataset",
)
@click.option(
    "--agent/--link-only",
    "run_agent",
    default=True,
    help="Run the data-prep agent after exposing input data; --link-only only symlinks.",
)
def prep(task, config, workspace, input_data_dir, show_preview, run_agent):
    """Prepare a REPL workspace dataset view, optionally via the data-prep agent."""
    cfg = load_cfg(config)
    _apply_parallel_manifest_cfg_env(cfg)
    _apply_task_workspace(cfg, workspace)
    if input_data_dir is not None:
        cfg.input_data_dir = Path(input_data_dir).expanduser().resolve(strict=False)
    apply_profile_overrides(cfg)
    apply_parallel_manifest_agent_overrides(cfg)
    prep_cfg(cfg)
    setup_logging(
        cfg.log_dir, interaction_log_color=cfg.scienceflow_interaction_log_color
    )
    if run_agent:
        _repl_allow_input_data_root(cfg)
        ws_dataset = Path(cfg.workspace_dir) / "dataset"
        if ws_dataset.is_symlink():
            ws_dataset.unlink()
        ws_dataset.mkdir(parents=True, exist_ok=True)
        from scienceflow.research.solver.data_prep import run_data_prep_agent

        asyncio.run(run_data_prep_agent(cfg, task, repo_root=_SCIENCEFLOW_REPO_ROOT))
        click.echo("prep agent ok")
    else:
        _repl_apply_dataset_symlinks(cfg)
    click.echo(f"prep ok: dataset={Path(cfg.workspace_dir) / 'dataset'}")
    if show_preview:
        from scienceflow.research.state.dataset.discovery.scan import scan_data_dir

        preview = scan_data_dir(
            Path(cfg.workspace_dir) / "dataset",
            max_chars=int(getattr(cfg, "data_preview_max_chars", 32000) or 32000),
            max_items_per_dir=int(
                getattr(cfg, "data_preview_max_items_per_dir", 40) or 40
            ),
            walk_budget_dirs=getattr(cfg, "data_scan_walk_budget_dirs", None),
            walk_budget_files=getattr(cfg, "data_scan_walk_budget_files", None),
            probe_binary_dirs_budget=int(
                getattr(cfg, "data_scan_probe_binary_dirs_budget", 8) or 8
            ),
            preview_raw_dirs_budget=int(
                getattr(cfg, "data_scan_preview_raw_dirs_budget", 12) or 12
            ),
            meta_sample_max_bytes=int(
                getattr(cfg, "data_scan_meta_sample_max_bytes", 50000) or 50000
            ),
            csv_max_rows_to_scan=int(
                getattr(cfg, "data_scan_csv_max_rows_to_scan", 200000) or 200000
            ),
        )
        click.echo("--- data preview ---")
        click.echo(preview)


@click.command()
@click.argument("target", required=False)
@click.option(
    "--execution-root",
    default=None,
    help="Local allocation root; creates a unique task directory without moving chat sessions.",
)
@click.option(
    "--manifest", "-m", type=click.Path(exists=True, dir_okay=False), default=None
)
@click.option(
    "--tui",
    "with_tui",
    is_flag=True,
    help="Prepare and view a managed research task in TUI.",
)
@click.option("--workers", type=click.IntRange(min=1), default=None)
@click.option(
    "--duration",
    default=None,
    help="Research time, minimum 10m; for example 20m or 2h.",
)
@click.option("--cpu", default=None, help="Total CPU pool, for example 0-7.")
@click.option("--gpu", default=None, help="cpu, auto or explicit GPU IDs.")
@click.option("--config", "-c", default=None, help="Config YAML path")
@click.option(
    "--workspace",
    "-w",
    default=".",
    help="Management workspace for task discovery and default execution; chat sessions stay here.",
)
@click.option(
    "--input-data-dir",
    "-d",
    "input_data_dir",
    default=None,
    type=click.Path(exists=False),
    help="Override config input_data_dir (read-only raw competition input)",
)
def run(
    target,
    execution_root,
    manifest,
    with_tui,
    workers,
    duration,
    cpu,
    gpu,
    config,
    workspace,
    input_data_dir,
):
    """Run TARGET through the single managed research lifecycle."""
    from .control import managed_run

    if target is not None and manifest is not None:
        raise click.UsageError("Use TARGET or --manifest, not both.")
    if manifest is not None:
        overrides = {
            "--execution-root": execution_root,
            "--workers": workers,
            "--duration": duration,
            "--cpu": cpu,
            "--gpu": gpu,
            "--config": config,
            "--input-data-dir": input_data_dir,
        }
        supplied = [name for name, value in overrides.items() if value is not None]
        if supplied:
            raise click.UsageError(
                "Cannot combine --manifest with "
                + ", ".join(supplied)
                + ". Set task resources and configuration in the manifest."
            )
    if target is None and manifest is None:
        from .control.guided import choose_task

        target = choose_task(Path(workspace).expanduser().resolve(strict=False))
    return managed_run(
        target,
        manifest=manifest,
        execution_root=execution_root,
        with_tui=with_tui,
        workspace=workspace,
        input_data_dir=input_data_dir,
        workers=workers,
        duration=duration,
        cpu=cpu,
        gpu=gpu,
        config=config,
    )
