"""Interactive REPL command."""

import asyncio
from pathlib import Path
from typing import Any

import click
import yaml

from scienceflow.interfaces.cli import (
    _apply_task_workspace,
    _is_repl_exit_line,
    _manifest_bool,
    _manifest_str,
    _read_manifest_text_file,
    _repl_apply_dataset_symlinks,
    _repl_build_first_user_query,
    _repl_materialize_task_desc,
    _repl_resolve_runtime_options,
    _repl_resolve_task_text,
)
from scienceflow.agent.core.runtime.run_policy import AutoContinuePolicy
from scienceflow.foundation.config.schema.settings import (
    apply_profile_overrides,
    apply_repl_manifest_defaults,
    load_cfg,
    prep_cfg,
)
from scienceflow.runtime.observability.logging import setup_logging
from scienceflow.research.state.workspace.storage.git import (
    ensure_workspace_source_git,
    normalize_workspace_git_track_globs,
)

@click.command()
@click.option("--config", "-c", default=None, help="Config YAML path")
@click.option(
    "--workspace",
    "-w",
    default=None,
    type=click.Path(exists=False),
    help="Task output root (omit when using --manifest for this command).",
)
@click.option(
    "--manifest",
    "-m",
    "manifest_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Single-task YAML: derive -w and input_data_dir like parallel (mutually exclusive with -w).",
)
@click.option(
    "--exp-id",
    "exp_id_cli",
    default=None,
    help="Override cfg.exp_id (task description lookup via tasks/**/<exp_id>/task.yaml).",
)
@click.option("--plain", is_flag=True, default=False, help="Disable rich UI (plain text mode)")
@click.option(
    "--show-code",
    type=click.IntRange(min=-1),
    default=5,
    help="Code preview lines per step: 5 (default), 0=hide, -1=full code",
)
@click.option(
    "--enable-sandbox/--no-enable-sandbox",
    default=True,
    help="Naive agent: restrict read/write/edit/grep/glob/ls to the execution workspace (default on). "
    "Use --no-enable-sandbox for arbitrary paths (e.g. external datasets).",
)
@click.option(
    "--input-data-dir",
    "-d",
    "input_data_dir",
    default=None,
    type=click.Path(exists=False),
    help="Read-only competition data root; auto-symlinked to workspace/dataset/ (same layout as run/prep).",
)
@click.option(
    "--auto-first-user/--no-auto-first-user",
    default=None,
    help="Automatically run one REPL turn using the manifest first-user query or a REPL-native task prompt.",
)
@click.option(
    "--exit-after-auto/--no-exit-after-auto",
    default=None,
    help="Exit after --auto-first-user completes instead of entering the interactive prompt.",
)
def repl(
    config,
    workspace,
    manifest_path,
    exp_id_cli,
    plain,
    show_code,
    enable_sandbox,
    input_data_dir,
    auto_first_user,
    exit_after_auto,
):
    """Interactive REPL mode."""
    from scienceflow.runtime.parallel.execution.runner import (
        _manifest_input_data_dir,
        _manifest_task_exp_id,
        resolve_manifest_task_workspace,
    )

    manifest_task_raw: Any = None
    defaults: dict[str, Any] = {}
    task0: dict[str, Any] = {}
    manifest_dir: Path | None = None
    repl_first_user_query_text = ""
    link_data = False
    if manifest_path:
        if workspace is not None:
            raise click.ClickException("Use either --manifest or --workspace, not both.")
        manifest_file = Path(manifest_path).expanduser().resolve(strict=False)
        manifest_dir = manifest_file.parent
        with open(manifest_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        tasks = data.get("tasks") or []
        if len(tasks) != 1:
            raise click.ClickException("repl --manifest requires exactly one task in tasks[]")
        defaults = data.get("defaults") or {}
        task0 = tasks[0]
        exp_id = _manifest_task_exp_id(task0, 0)
        resolved_ws = resolve_manifest_task_workspace(defaults, task0, 0, exp_id)
        cfg_yaml = config
        if cfg_yaml is None:
            tcfg = task0.get("config")
            dcfg = defaults.get("config")
            cfg_yaml = tcfg if (tcfg is not None and str(tcfg).strip()) else dcfg
        cfg = load_cfg(cfg_yaml)
        apply_repl_manifest_defaults(cfg, defaults, task0)
        apply_profile_overrides(cfg)
        _apply_task_workspace(cfg, resolved_ws)
        idd_m = _manifest_input_data_dir(defaults, task0)
        if idd_m.strip():
            cfg.input_data_dir = Path(idd_m).expanduser().resolve(strict=False)
            link_data = True
        cfg.exp_id = exp_id
        manifest_task_raw = task0.get("task")
        first_user_file = _manifest_str(defaults, task0, "repl_first_user_query_file")
        if first_user_file:
            repl_first_user_query_text = _read_manifest_text_file(
                first_user_file,
                manifest_dir=manifest_dir,
            )
    else:
        ws = workspace if workspace is not None else "."
        cfg = load_cfg(config)
        apply_profile_overrides(cfg)
        _apply_task_workspace(cfg, ws)

    if input_data_dir is not None:
        cfg.input_data_dir = Path(input_data_dir).expanduser().resolve(strict=False)
        link_data = True

    if exp_id_cli is not None and str(exp_id_cli).strip():
        cfg.exp_id = str(exp_id_cli).strip()

    prep_cfg(cfg)
    cfg.scienceflow_tools_sandbox = enable_sandbox

    if link_data:
        _repl_apply_dataset_symlinks(cfg)

    td_text, td_src = _repl_resolve_task_text(cfg, manifest_task_raw=manifest_task_raw)
    _repl_materialize_task_desc(cfg, td_text, td_src)
    git_track_globs = normalize_workspace_git_track_globs(
        getattr(cfg, "repl_workspace_git_track_globs", None),
    )
    cfg.repl_workspace_git_track_globs = list(git_track_globs)
    git_result = ensure_workspace_source_git(
        cfg.workspace_dir,
        enabled=bool(getattr(cfg, "repl_workspace_git_enabled", True)),
        track_globs=git_track_globs,
        initial_commit=bool(getattr(cfg, "repl_workspace_git_initial_commit", True)),
    )
    if git_result.enabled and not git_result.ready:
        cfg.repl_workspace_git_enabled = False
        if git_result.message:
            click.echo(f"[repl-git] workspace git unavailable: {git_result.message}", err=True)
    auto_first_user_enabled = (
        _manifest_bool(
            defaults,
            task0,
            "repl_auto_first_user",
            bool(repl_first_user_query_text),
        )
        if auto_first_user is None
        else bool(auto_first_user)
    )
    exit_after_auto_enabled = (
        _manifest_bool(defaults, task0, "repl_exit_after_auto", False)
        if exit_after_auto is None
        else bool(exit_after_auto)
    )
    repl_runtime = _repl_resolve_runtime_options(
        cfg,
        auto_first_user_enabled=auto_first_user_enabled,
    )

    # File-only: avoid interleaving INFO lines with Rich REPL output
    setup_logging(
        cfg.log_dir,
        console=False,
        interaction_log_color=cfg.scienceflow_interaction_log_color,
    )

    from scienceflow.foundation.config.llm.llm_http import aclose_llm_clients as _aclose_llm_clients
    from scienceflow.runtime.workflow import Orchestrator
    orch = Orchestrator(cfg)

    ui = None
    if not plain:
        from scienceflow.interfaces.ui import RichUI
        ui = RichUI(show_code=show_code)

    if ui:
        ui.welcome()
    else:
        click.echo("ScienceFlow REPL (type exit, quit, or exit() to quit)")

    async def _close_llm(llm) -> None:
        await _aclose_llm_clients(llm)

    repl_run_id = 0
    system_prompt_override = None
    system_prompt_hook = None
    repl_teleport_mode = "off"
    llm_trace_hook = orch.make_llm_call_tracer(
        node_id="repl",
        process_id=task0.get("run_id") if isinstance(task0, dict) else None,
        detail_prefix=(
            f"mode=repl;profile={repl_runtime['profile']};"
            f"tool_preset={repl_runtime['tool_preset']};teleport={repl_teleport_mode}"
        ),
    )
    science_agent = orch.create_science_agent(
        ui=ui,
        task_description=td_text or None,
        run_policy=AutoContinuePolicy(max_text_only_retries=2),
        max_steps_override=int(repl_runtime["max_steps"]),
        system_prompt=system_prompt_override,
        system_prompt_hook=system_prompt_hook,
        append_repl_system_prompt=True,
        pin_task_description=bool(repl_runtime["pin_task_description"]),
        on_llm_call=llm_trace_hook,
        teleport_mode=repl_teleport_mode,
        repl_bash_write_mode=bool(repl_runtime["repl_bash_write_mode"]),
        stable_system_prompt=bool(repl_runtime["stable_system_prompt"]),
        pin_environment_context=bool(repl_runtime["pin_environment_context"]),
        code_organization_hint=str(repl_runtime["code_organization_hint"]),
        workspace_git_enabled=bool(repl_runtime["workspace_git_enabled"]),
        workspace_git_track_globs=repl_runtime["workspace_git_track_globs"],
        workspace_git_auto_review=bool(repl_runtime["workspace_git_auto_review"]),
        workspace_git_auto_checkpoint=bool(
            repl_runtime["workspace_git_auto_checkpoint"],
        ),
        bash_max_output_chars_override=int(repl_runtime["bash_max_output_chars"]),
        bash_max_stream_line_chars_override=int(
            repl_runtime["bash_max_stream_line_chars"],
        ),
        bash_observation_summary_override=bool(
            repl_runtime["bash_observation_summary"],
        ),
    )
    try:
        if auto_first_user_enabled:
            auto_task = repl_first_user_query_text or _repl_build_first_user_query(td_text)
            msg = "file first-user query" if repl_first_user_query_text else "REPL first-user query"
            click.echo(f"[repl-auto] running {msg}")
            repl_run_id += 1
            science_agent.set_repl_session_run_index(repl_run_id)
            asyncio.run(science_agent.run(auto_task))
            if exit_after_auto_enabled:
                return

        while True:
            if ui:
                task = ui.prompt_input()
                if task is None:
                    break
            else:
                try:
                    task = click.prompt("scienceflow", prompt_suffix="> ")
                except (EOFError, KeyboardInterrupt):
                    break

            if _is_repl_exit_line(task):
                break

            stripped = task.strip()
            if stripped == "/compact":
                out = asyncio.run(science_agent.compact())
                click.echo(out)
                continue
            if stripped == "/files":
                click.echo(science_agent.file_state_summary)
                continue

            repl_run_id += 1
            science_agent.set_repl_session_run_index(repl_run_id)
            asyncio.run(science_agent.run(task))
    finally:
        asyncio.run(_close_llm(science_agent.llm))

    if ui:
        ui.goodbye()
