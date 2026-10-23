"""MemoryContextManager responsibility: state."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from inquirycraft.memory import Memory, Message
from inquirycraft.tools import PathGuard

from scienceflow.research.state.knowledge.context.context.results.base import FileSnapshotInfo
from scienceflow.research.state.knowledge.context.source_snapshot import (
    SymbolReadCoverage,
    _build_redundant_symbol_read_coverage_summary,
    _merge_read_intervals,
    _parse_python_source_symbols,
    _range_fully_covered_by_intervals,
    _sha256_short_bytes,
    _symbol_sha,
)


def __init__(
    self,
    memory: Memory,
    workspace_dir: Path,
    *,
    budget_chars: int = 60_000,
    file_snapshot_max: int = 8,
    path_guard_extra_roots: Sequence[str | Path] | None = None,
    tool_memory_compression: bool = True,
    bash_success_tail_lines: int = 20,
    bash_success_tail_lines_solution: int = 8,
    bash_success_tail_lines_test: int = 120,
    bash_success_tail_lines_readonly: int = 30,
    bash_success_tail_lines_install: int = 5,
    read_success_max_lines: int = 200,
    sliding_window_priority_enabled: bool = True,
    pinned_budget_ratio: float = 0.45,
    bash_output_dedup_apply_to_memory: bool = True,
    bash_output_dedup_min_repeat: int = 3,
    bash_output_dedup_summary_prefix: str = "[log-dedup]",
    write_auto_snapshot_enabled: bool = True,
    write_auto_snapshot_paths: Sequence[str] | None = None,
    write_auto_snapshot_code_extensions: Sequence[str] | None = None,
    write_auto_snapshot_max_lines: int = 400,
    write_auto_snapshot_max_chars: int = 8_000,
    write_auto_snapshot_changed_context_lines: int = 10,
    write_auto_snapshot_symbol_body_lines: int = 3,
    read_overlap_guard_enabled: bool = True,
    msg0_compress_body: bool = False,
) -> None:
    self._memory = memory
    self._workspace = Path(workspace_dir).resolve()
    self._guard = PathGuard(self._workspace, extra_roots=path_guard_extra_roots)
    self._budget_chars = int(budget_chars)
    self._pinned_budget_ratio = float(pinned_budget_ratio)
    self._file_snapshot_max = int(file_snapshot_max)
    self._file_snapshots: dict[str, FileSnapshotInfo] = {}
    self._edit_fail_counter: dict[str, int] = {}
    self._edit_fail_total_counter: dict[str, int] = {}
    self._edit_fail_escalation_level: dict[str, int] = {}
    self._write_counter_by_path: dict[str, int] = {}
    self._tool_memory_compression = bool(tool_memory_compression)
    self._bash_success_tail_lines = int(bash_success_tail_lines)
    self._bash_success_tail_lines_solution = int(bash_success_tail_lines_solution)
    self._bash_success_tail_lines_test = int(bash_success_tail_lines_test)
    self._bash_success_tail_lines_readonly = int(bash_success_tail_lines_readonly)
    self._bash_success_tail_lines_install = int(bash_success_tail_lines_install)
    self._read_success_max_lines = int(read_success_max_lines)
    self._sliding_window_priority_enabled = bool(sliding_window_priority_enabled)
    self._bash_output_dedup_apply_to_memory = bool(bash_output_dedup_apply_to_memory)
    self._bash_output_dedup_min_repeat = int(bash_output_dedup_min_repeat)
    self._bash_output_dedup_summary_prefix = str(bash_output_dedup_summary_prefix)
    self._write_auto_snapshot_enabled = bool(write_auto_snapshot_enabled)
    _wpaths = list(write_auto_snapshot_paths) if write_auto_snapshot_paths else []
    self._write_auto_snapshot_paths: tuple[str, ...] = tuple(
        (p or "").replace("\\", "/").lstrip("/").strip() for p in _wpaths if (p or "").strip()
    )
    _code_exts = (
        list(write_auto_snapshot_code_extensions)
        if write_auto_snapshot_code_extensions is not None
        else [".py"]
    )
    self._write_auto_snapshot_code_extensions: tuple[str, ...] = tuple(
        e if e.startswith(".") else f".{e}"
        for e in (str(x).strip().lower() for x in _code_exts)
        if e
    ) or (".py",)
    self._write_auto_snapshot_max_lines = max(1, int(write_auto_snapshot_max_lines))
    self._write_auto_snapshot_max_chars = max(256, int(write_auto_snapshot_max_chars))
    self._write_auto_snapshot_changed_context_lines = max(
        1,
        int(write_auto_snapshot_changed_context_lines),
    )
    self._write_auto_snapshot_symbol_body_lines = max(
        0,
        int(write_auto_snapshot_symbol_body_lines),
    )
    self._read_overlap_guard_enabled = bool(read_overlap_guard_enabled)
    # When False (default), msg[0] is preserved verbatim by both the in-session
    # mechanical compactor and the LLM-fallback ``compact()`` — the entire user
    # pin (head + ``## Task description`` body) survives every compress pass.
    # When True, only the head is preserved verbatim and the body is replaced
    # with ``MSG0_BODY_TRUNCATED_PLACEHOLDER`` to save tokens.
    self._msg0_compress_body = bool(msg0_compress_body)
    # Optional LHR hook: preserve early EDA/tool evidence verbatim across compact.
    # This is an absolute message index in the current compacted memory layout;
    # messages after the leading user/task pin and before this index are never summarized.
    self._protected_raw_prefix_end_index: int = 0
    self._protected_raw_prefix_warn_chars: int = 0
    self._protected_raw_prefix_label: str = "protected raw prefix"
    # rel path -> (sha256~short, sorted merged (lo, hi) intervals) for read-overlap nudges
    self._read_coverage: dict[str, tuple[str, list[tuple[int, int]]]] = {}
    # rel path -> symbol key -> last complete read coverage for that Python symbol.
    self._read_symbol_coverage: dict[str, dict[str, SymbolReadCoverage]] = {}
    # rel path -> last sha256 short for which an auto-snapshot was injected.
    # Used to dedup repeated ``[auto-snapshot after successful write]`` blocks
    # when the on-disk content does not change between successive writes.
    self._last_auto_snapshot_sha_by_path: dict[str, str] = {}
    # rel path -> last code text used for semantic changed-range snapshot diffing.
    self._last_auto_snapshot_text_by_path: dict[str, str] = {}
    # Tier 2 (silent read interception): rel path -> count of redundant reads (range fully
    # covered by prior reads at the same on-disk sha). Used by teleport health checks and
    # tests to verify the silent-intercept path actually fired.
    self._redundant_read_count_by_path: dict[str, int] = {}
    self._pinned_messages: list[Message] = []
    # Monotone lower bound on the suffix start index for prefix-cache stability.
    self._last_window_k: int = 0
    self._seen_tool_signatures: dict[tuple[str, str], int] = {}
    self._released_full_signatures: set[tuple[str, str]] = set()


def _norm_rel(self, r: str) -> str:
    return (r or "").replace("\\", "/").lstrip("/").strip()


def _path_matches_write_snapshot(self, rel: str) -> bool:
    """True if *rel* is configured for code auto-snapshots."""
    nr = self._norm_rel(rel)
    if not nr:
        return False
    name = PurePosixPath(nr).name
    for pat in self._write_auto_snapshot_paths:
        p = self._norm_rel(pat)
        if not p:
            continue
        if nr == p or name == PurePosixPath(p).name:
            return True
    suffix = PurePosixPath(nr).suffix.lower()
    return bool(suffix and suffix in self._write_auto_snapshot_code_extensions)


def _current_snapshot_sha_for_rel(self, rel: str) -> str:
    nr = self._norm_rel(rel)
    if not nr or not self._path_matches_write_snapshot(nr):
        return ""
    expected = self._last_auto_snapshot_sha_by_path.get(nr, "")
    if not expected:
        return ""
    try:
        resolved = self._guard.resolve(nr)
        if not resolved.is_file():
            return ""
        text = resolved.read_text(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        return ""
    cur = _sha256_short_bytes(text.encode("utf-8", errors="replace"))
    return cur if cur == expected else ""


def _compress_grep_snapshot_refs_for_memory(self, obs: str, *, raw_id: str = "") -> str:
    if not obs:
        return obs
    groups: dict[str, list[tuple[int, str]]] = {}
    order: list[str] = []
    passthrough: list[str] = []
    for line in obs.splitlines():
        m = re.match(r"^(.+?):(\d+):(.*)$", line)
        if not m:
            passthrough.append(line)
            continue
        rel = self._norm_rel(m.group(1))
        if not rel:
            passthrough.append(line)
            continue
        if rel not in groups:
            groups[rel] = []
            order.append(rel)
        groups[rel].append((int(m.group(2)), m.group(3)))
    if not groups:
        return obs
    replaced_any = False
    out: list[str] = [
        (
            f"[tool-output compressed: reducer=snapshot_ref_v1 "
            f"raw_chars={len(obs)} raw_lines={len(obs.splitlines())}]"
        )
    ]
    for rel in order:
        matches = groups[rel]
        sha = self._current_snapshot_sha_for_rel(rel)
        if sha and len(matches) > 5:
            nums = ",".join(str(n) for n, _ in matches[:10])
            if len(matches) > 10:
                nums += ",..."
            out.append(
                f"{rel}: {len(matches)} matches (lines {nums}); "
                f"[see current snapshot: {rel} sha~{sha}]"
            )
            replaced_any = True
            continue
        for lineno, text in matches:
            out.append(f"{rel}:{lineno}:{text}")
    if passthrough:
        out.extend(passthrough[-3:])
    if raw_id:
        out.append(f"[exact raw output: {raw_id}]")
    return "\n".join(out) if replaced_any else obs


def _update_read_coverage_and_is_redundant(
    self,
    rel: str,
    args: dict[str, Any],
    *,
    raw_id: str = "",
) -> tuple[bool, str | None]:
    """Update read coverage; return redundant flag and optional replacement summary."""
    if not self._read_overlap_guard_enabled or not rel:
        return False, None
    try:
        resolved = self._guard.resolve(rel)
    except (ValueError, OSError):
        return False, None
    if not resolved.is_file():
        return False, None
    try:
        text = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False, None
    lines = text.splitlines(keepends=False)
    n = len(lines)
    if n == 0:
        return False, None
    start0 = max(0, int(args.get("offset") or 1) - 1)
    lim = int(args.get("limit") or 200)
    if lim <= 0:
        lo, hi = 1, n
    else:
        lo = start0 + 1
        hi = min(n, start0 + lim)
    if lo > hi:
        return False, None
    raw = text.encode("utf-8", errors="replace")
    short = _sha256_short_bytes(raw)

    # Python code gets a finer-grained symbol coverage index. Once a complete
    # function/class has been read at a specific symbol sha, repeated reads inside
    # that same symbol can be summarized even if unrelated parts of the file changed.
    symbols = (
        _parse_python_source_symbols(text)
        if PurePosixPath(rel).suffix.lower() == ".py"
        else []
    )
    symbol_redundant_summary: str | None = None
    if symbols:
        cov_by_key = self._read_symbol_coverage.setdefault(rel, {})
        for sym in symbols:
            cov = cov_by_key.get(sym.key)
            if (
                cov is not None
                and cov.symbol_sha == _symbol_sha(text, sym)
                and lo >= cov.start
                and hi <= cov.end
            ):
                symbol_redundant_summary = _build_redundant_symbol_read_coverage_summary(
                    rel,
                    requested=(lo, hi),
                    coverage=cov,
                )
                break
        for sym in symbols:
            if lo <= sym.start and hi >= sym.end:
                previous_cov = cov_by_key.get(sym.key)
                cov_by_key[sym.key] = SymbolReadCoverage(
                    symbol_name=sym.name,
                    symbol_kind=sym.kind,
                    start=sym.start,
                    end=sym.end,
                    symbol_sha=_symbol_sha(text, sym),
                    raw_id=raw_id or (previous_cov.raw_id if previous_cov else ""),
                )
        if symbol_redundant_summary:
            return True, symbol_redundant_summary

    ent = self._read_coverage.get(rel)
    merged_prior = _merge_read_intervals(list(ent[1])) if ent and ent[0] == short else []
    redundant = bool(
        ent is not None
        and ent[0] == short
        and merged_prior
        and _range_fully_covered_by_intervals(lo, hi, merged_prior),
    )
    if ent is None or ent[0] != short:
        self._read_coverage[rel] = (short, _merge_read_intervals([(lo, hi)]))
    else:
        self._read_coverage[rel] = (short, _merge_read_intervals([*ent[1], (lo, hi)]))
    return redundant, None


def _prune_symbol_read_coverage_after_write(self, rel: str, text: str) -> None:
    cov_by_key = self._read_symbol_coverage.get(rel)
    if not cov_by_key:
        return
    symbols = {
        sym.key: (sym, _symbol_sha(text, sym))
        for sym in _parse_python_source_symbols(text)
    }
    if not symbols:
        self._read_symbol_coverage.pop(rel, None)
        return
    kept: dict[str, SymbolReadCoverage] = {}
    for key, cov in cov_by_key.items():
        current = symbols.get(key)
        if current is None:
            continue
        sym, sym_sha = current
        if cov.symbol_sha != sym_sha:
            continue
        kept[key] = SymbolReadCoverage(
            symbol_name=sym.name,
            symbol_kind=sym.kind,
            start=sym.start,
            end=sym.end,
            symbol_sha=sym_sha,
            raw_id=cov.raw_id,
        )
    if kept:
        self._read_symbol_coverage[rel] = kept
    else:
        self._read_symbol_coverage.pop(rel, None)


def _trim_snapshots(self) -> None:
    if len(self._file_snapshots) <= self._file_snapshot_max:
        return
    for key in sorted(self._file_snapshots.keys())[
        : -self._file_snapshot_max
    ]:
        self._file_snapshots.pop(key, None)
