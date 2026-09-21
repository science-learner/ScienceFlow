"""Public BashTool facade."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pydantic import ConfigDict, Field

from scienceflow.runtime.safety.execution.agent_runtime.tool_base import BaseTool
from scienceflow.runtime.safety.tooling.bash.runtime import tool_runtime
from scienceflow.runtime.safety.tooling.bash.process import pipeline as process_pipeline

class BashTool(BaseTool):
    """Execute a bash one-liner / pipeline in the workspace directory."""

    name: str = "bash"
    description: str = (
        "Execute a shell command with cwd already set to the task workspace. "
        "Prefer this for shell execution, package installs, quick listings, "
        "validation/training runs, artifact preservation, and one-off shell tasks. "
        "Use workspace-relative paths (e.g. dataset/) or run `pwd` first; do not "
        "assume external notebook paths or Kaggle paths like /mnt/data exist on "
        "this machine. IMPORTANT: cwd is already the workspace, so do not prefix "
        "commands with an absolute-path cd. Run workspace commands directly, such "
        "as `python3 solution.py` or `python3 train.py` when relevant. For large "
        "observations, print compact summaries or write verbose logs to workspace "
        "files instead of dumping raw output. "
        "For verbose training or validation, save a full workspace log first; do not "
        "use `command | tail` as the only output record."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": (
                    "Shell command to run (bash -c semantics via subprocess shell). "
                    "cwd is already set to the workspace; do not use an absolute "
                    "path prefix before running scripts. Use workspace-relative "
                    "paths (e.g. dataset/) or run `pwd` to confirm location. If "
                    "output may be large, write the full log to a workspace file "
                    "and print a compact summary/tail; do not pipe verbose "
                    "training or validation directly to `tail` as the only output record. "
                    "For long jobs, use SCIENCEFLOW_RUN_ARTIFACT_DIR and "
                    "SCIENCEFLOW_RUN_STATE_PATH when present: atomically save "
                    "periodic checkpoints/partials and handle SIGUSR1 by writing "
                    "status=interrupted before clean exit."
                ),
            },
        },
        "required": ["command"],
    }
    workspace_dir: Path = Field(...)
    max_output_chars: int = Field(default=8000)
    # Cap each stdout/stderr line sent to on_output (REPL live stream); 0 = no per-line cap.
    max_stream_line_chars: int = Field(default=2400)
    bash_timeout_sec: float = Field(default=1800.0)
    bash_timeout_slow_sec: float = Field(default=1800.0)
    # Optional task-level hard fuse. When set, BashTool caps command runtime by
    # remaining task wall clock minus finalization reserve; resource arbiter still
    # receives deadline events before the deterministic timeout fires.
    bash_hard_fuse_deadline_monotonic: float = Field(default=0.0)
    bash_hard_fuse_finalization_reserve_sec: float = Field(default=0.0)
    # REPL live stream: after this many seconds with no stdout/stderr line, emit progress updates.
    bash_heartbeat_idle_sec: float = Field(default=5.0)
    # Seconds between heartbeat updates; 0 disables. Each new subprocess line resets idle (see plan).
    bash_heartbeat_interval_sec: float = Field(default=10.0)
    # Minimum seconds between resource progress_heartbeat events for noisy training logs.
    # The first and final meaningful progress summaries are still emitted.
    resource_progress_heartbeat_min_interval_sec: float = Field(default=30.0)
    # Minimum seconds between workspace artifact scans while a command is running.
    resource_artifact_heartbeat_scan_interval_sec: float = Field(default=30.0)
    # Minimum age for a same-size/same-mtime artifact to be considered stable.
    resource_artifact_recoverable_settle_sec: float = Field(default=5.0)
    # Recoverable resource stops send SIGUSR1 first and wait for a clean marker before hard kill.
    resource_recoverable_stop_enabled: bool = Field(default=True)
    resource_recoverable_stop_sigusr1_grace_sec: float = Field(default=60.0)
    resource_recoverable_stop_marker_exit_grace_sec: float = Field(default=10.0)
    resource_recoverable_stop_sigterm_grace_sec: float = Field(default=5.0)
    # Optional extra environment variables injected into every subprocess (e.g. CUDA_VISIBLE_DEVICES).
    extra_env: Optional[dict[str, str]] = Field(default=None)
    # Basenames under workspace_dir that must not be used as execution targets (e.g. parent_workspace).
    readonly_dirs: list[str] = Field(default_factory=list)
    # Extra absolute roots (dataset symlink targets, pretrained_models_dir, etc.) that are also
    # allowed as ``cd`` targets — mirrors PathGuard.extra_roots used by read/write/edit/grep/glob/ls.
    path_guard_extra_roots: list[Path] = Field(default_factory=list)
    path_guard_denied_prefixes: list[str] = Field(default_factory=list)
    # asyncio StreamReader buffer limit (bytes).  Default 64 KiB is too small for
    # processes that produce very long lines (tqdm \r, large JSON, etc.).
    subprocess_stream_limit: int = Field(default=50 * 1024 * 1024)
    # Collapse consecutive repeated stdout/stderr line-blocks (1–4 lines) when a block
    # repeats at least this many times. 0 = disable (keep raw subprocess output).
    dedup_min_repeat: int = Field(default=3)
    # Collapse site-packages / non-workspace frames inside Python tracebacks in bash output.
    distill_tracebacks: bool = Field(default=True)
    # Strip symlink arrow targets (`` -> /real/path``) from bash output so the LLM does not
    # see host-specific absolute paths that leak from ``ls -la`` / ``find`` output.
    strip_symlink_targets: bool = Field(default=True)
    # When enabled, large successful observation outputs (ls/find/cat/head/...) are
    # represented as exact head/tail summaries while the full output is carried in
    # ToolResult.system for raw artifact storage. The command itself is never rewritten.
    observation_summary_enabled: bool = Field(default=False)
    # Opt-in guard for solvers that require model-visible workspace-relative
    # paths only. Blocks host absolute paths and parent-directory escapes in
    # bash commands before execution.
    forbid_host_absolute_paths: bool = Field(default=False)
    # Optional observe-only resource hook used by LNR. It must never affect command
    # execution or the agent-visible tool schema.
    resource_observer: Any | None = Field(default=None, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)
    _apply_hard_fuse_timeout = tool_runtime._apply_hard_fuse_timeout
    _deadline_state = tool_runtime._deadline_state
    execute = tool_runtime.execute
    _execute_with_stream_monitoring = tool_runtime._execute_with_stream_monitoring
    _execute_process_pipeline = process_pipeline._execute_process_pipeline
