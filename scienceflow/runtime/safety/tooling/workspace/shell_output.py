"""Shell-output API bridge for the pinned and next InquiryCraft releases."""

try:
    from inquirycraft.tools import (
        deduplicate_repeated_blocks as _dedup_repeated_blocks,
        distill_tracebacks as _distill_tracebacks,
        has_masked_python_traceback as _has_masked_python_traceback,
        maybe_lossless_observation_summary as _maybe_lossless_observation_summary,
        sanitize_shell_output_paths as _sanitize_model_visible_output_paths,
        trim_shell_output as _trim_output,
    )
except ImportError:  # Remove once ScienceFlow pins InquiryCraft >=0.7.2.
    from inquirycraft.tools.shell_reducer import (
        _dedup_repeated_blocks,
        _distill_tracebacks,
        _has_masked_python_traceback,
        _maybe_lossless_observation_summary,
        _sanitize_model_visible_output_paths,
        _trim_output,
    )
from inquirycraft.tools.shell_output import (
    sanitize_host_absolute_paths as _sanitize_host_absolute_paths,
    sanitize_python_env_paths as _sanitize_python_env_paths,
    sanitize_root_prefix as _sanitize_root_prefix_in_output,
    sanitize_workspace_prefix as _sanitize_workspace_prefix_in_output,
    strip_symlink_targets as _strip_symlink_targets,
)

__all__ = [
    "_dedup_repeated_blocks",
    "_distill_tracebacks",
    "_has_masked_python_traceback",
    "_maybe_lossless_observation_summary",
    "_sanitize_host_absolute_paths",
    "_sanitize_model_visible_output_paths",
    "_sanitize_python_env_paths",
    "_sanitize_root_prefix_in_output",
    "_sanitize_workspace_prefix_in_output",
    "_strip_symlink_targets",
    "_trim_output",
]
