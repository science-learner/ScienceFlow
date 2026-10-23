"""Bash responsibility: streaming."""

from __future__ import annotations

from scienceflow.runtime.safety.tooling.bash.core.shared import (
    Any,
    Callable,
    Path,
    RESOURCE_HEAVY_GPU_CANDIDATE,
    ShadowWorkspaceManager,
    ToolResult,
    asyncio,
    logger,
    spawn_shell,
    terminate_tree,
    time,
)
from scienceflow.runtime.safety.tooling.bash.policy.admission import (
    _resource_call,
)


async def _read_stream_limited(
    stream: asyncio.StreamReader | None,
    *,
    limit_chars: int = 4000,
) -> str:
    if stream is None:
        return ""
    chunks: list[str] = []
    seen = 0
    while True:
        data = await stream.readline()
        if not data:
            break
        text = data.decode(errors="replace")
        if seen < limit_chars:
            room = max(0, limit_chars - seen)
            chunks.append(text[:room])
            seen += len(text)
    return "".join(chunks)


def _create_observation_shadow(
    *,
    observer: Any | None,
    job_id: str | None,
    workspace_dir: Path,
    on_output: Callable[[str], None] | None,
) -> tuple[ShadowWorkspaceManager, Any | None]:
    manager = ShadowWorkspaceManager(
        copy_file_max_bytes=20 * 1024 * 1024,
        readonly_dir_names=("dataset", "data", "input"),
        protected_globs=(
            "submission.csv",
            "result.md",
            "*.ckpt",
            "*.joblib",
            "*.npy",
            "*.npz",
            "*.pkl",
            "*.pt",
            "*.pth",
            "*logit*",
            "*pred*.csv",
            "checkpoints/**",
            "models/**",
            "dataset/**",
            "data/**",
            "input/**",
        ),
    )
    try:
        shadow = manager.create(workspace_dir, job_id or "unknown")
    except Exception as exc:
        _resource_call(
            observer,
            "observation_event",
            job_id,
            state="shadow_create_failed",
            reason=type(exc).__name__,
        )
        return manager, None
    if on_output is not None:
        on_output("[bash observing] running isolated trial before replay\n")
    _resource_call(
        observer,
        "observation_event",
        job_id,
        state="started",
        shadow_workspace=str(shadow.shadow_workspace),
    )
    return manager, shadow


async def _observe_unknown_command(
    *,
    observer: Any | None,
    job_id: str | None,
    inferred_class: str,
    gpu_ids: list[str],
    cmd: str,
    workspace_dir: Path,
    proc_env: dict[str, str],
    stream_limit: int,
    on_output: Callable[[str], None] | None,
) -> tuple[str | None, ToolResult | None]:
    policy = _resource_call(
        observer,
        "observation_policy",
        job_id,
        inferred_class=inferred_class,
        gpu_ids=gpu_ids,
    )
    if not isinstance(policy, dict) or not policy.get("enabled"):
        return None, None

    window_sec = max(0.05, float(policy.get("window_sec") or 30.0))
    manager, shadow = _create_observation_shadow(
        observer=observer,
        job_id=job_id,
        workspace_dir=workspace_dir,
        on_output=on_output,
    )
    if shadow is None:
        return None, None

    start = time.time()
    timed_out = False
    proc = None
    try:
        proc = await spawn_shell(
            cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(shadow.shadow_workspace),
            env=proc_env,
            limit=int(stream_limit),
        )
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    _read_stream_limited(proc.stdout),
                    _read_stream_limited(proc.stderr),
                ),
                timeout=window_sec,
            )
            await proc.wait()
        except asyncio.TimeoutError:
            timed_out = True
            await terminate_tree(proc)
            await proc.wait()

        elapsed = time.time() - start
        ok, changed = shadow.validate_real_workspace_unchanged()
        shadow_ok, shadow_changed = shadow.validate_shadow_protected_unchanged()
        if not ok or not shadow_ok:
            prefix = "real" if not ok else "shadow"
            details = changed if not ok else shadow_changed
            reason = "shadow_isolation_failed:" + prefix + ":" + ",".join(details[:5])
            _resource_call(
                observer,
                "observation_event",
                job_id,
                state="shadow_isolation_failed",
                reason=reason,
                elapsed_sec=elapsed,
                shadow_workspace=str(shadow.shadow_workspace),
            )
            _resource_call(
                observer,
                "job_finished",
                job_id,
                status="shadow_isolation_failed",
                returncode=getattr(proc, "returncode", None),
                elapsed_sec=elapsed,
                reason=reason,
            )
            return None, ToolResult(error=reason)

        if timed_out:
            reason = f"observation_window_exceeded_{window_sec:.0f}s"
            _resource_call(
                observer,
                "observation_promoted",
                job_id,
                reason=reason,
                elapsed_sec=elapsed,
            )
            if on_output is not None:
                on_output("[bash observing] command promoted to queued heavy replay\n")
            return RESOURCE_HEAVY_GPU_CANDIDATE, None

        _resource_call(
            observer,
            "observation_event",
            job_id,
            state="completed_replay_real",
            reason="trial_completed",
            elapsed_sec=elapsed,
            shadow_workspace=str(shadow.shadow_workspace),
        )
        return None, None
    except asyncio.CancelledError:
        if proc is not None:
            await terminate_tree(proc)
        raise
    except Exception as exc:
        elapsed = time.time() - start
        _resource_call(
            observer,
            "observation_event",
            job_id,
            state="trial_failed",
            reason=type(exc).__name__,
            elapsed_sec=elapsed,
            shadow_workspace=str(shadow.shadow_workspace),
        )
        return None, None
    finally:
        try:
            shadow.cleanup()
        except Exception:
            logger.debug("[bash-resource] shadow cleanup failed", exc_info=True)


__all__ = tuple(name for name in globals() if not name.startswith("__"))
