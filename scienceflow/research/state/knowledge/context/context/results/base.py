# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Imports, constants, and value types shared by memory-context components."""

from __future__ import annotations
import logging
from dataclasses import dataclass
logger = logging.getLogger("scienceflow")
_TOOL_FEEDBACK_TRIMMED_PREFIX = "[Tool feedback trimmed for LLM context:"
_READ_OVERLAP_COACHING = (
    "\n\n[file unchanged since last read; sha matches]"
)
_WRITE_REPEAT_TO_SAME_FILE_COACHING = (
    "\n\n[Guard] This is write #{n} to `{rel}` in this session with no `bash "
    "python3 solution.py` between writes. Previous write(s) already landed on disk "
    "(not a placeholder). **Next turn must be bash**: run `python3 solution.py` "
    "(or equivalent validation). Do NOT write this file again until you have run it."
)
COMPACTED_CONVERSATION_SUMMARY_MARKER = "[Compacted conversation summary]"
_PINNED_TRUNC_SUFFIX = "[... pinned content truncated for budget ...]"

@dataclass
class FileSnapshotInfo:
    lines: int
    sha256_short: str

__all__ = tuple(name for name in globals() if not name.startswith("__"))
