"""Local execution allocation, independent of chat and worker layout."""

from .local import LocalExecutionLocation, allocate_local_workspace, resolve_local_root

__all__ = ["LocalExecutionLocation", "allocate_local_workspace", "resolve_local_root"]
