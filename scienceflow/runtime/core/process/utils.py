"""Process lifecycle API bridge for the pinned and next InquiryCraft releases."""

try:
    from inquirycraft.runtime import (
        kill_all_live_pgids,
        live_process_group_ids,
        run_in_process_group,
        spawn_exec,
        spawn_shell,
        terminate_process_tree,
        terminate_process_tree_recoverable,
        terminate_process_tree_sync,
        unregister_process_group,
    )
except ImportError:  # Remove once ScienceFlow pins InquiryCraft >=0.7.2.
    from inquirycraft.runtime.subprocess import (
        _LIVE_PGIDS,
        kill_all_live_pgids,
        spawn_exec,
        spawn_shell,
        terminate_tree as terminate_process_tree,
        terminate_tree_recoverable as terminate_process_tree_recoverable,
    )
    from inquirycraft.runtime.subprocess_sync import (
        _terminate_tree_sync as terminate_process_tree_sync,
        run_in_process_group,
    )

    def live_process_group_ids() -> tuple[int, ...]:
        return tuple(sorted(_LIVE_PGIDS))

    def unregister_process_group(pgid: int) -> None:
        _LIVE_PGIDS.discard(int(pgid))

__all__ = [
    "kill_all_live_pgids",
    "live_process_group_ids",
    "run_in_process_group",
    "spawn_exec",
    "spawn_shell",
    "terminate_process_tree",
    "terminate_process_tree_recoverable",
    "terminate_process_tree_sync",
    "unregister_process_group",
]
