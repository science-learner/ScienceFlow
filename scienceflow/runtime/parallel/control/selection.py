"""Resolve exact run IDs or task directories without silently choosing a run."""

from __future__ import annotations

from pathlib import Path

from .service import list_runs


class AmbiguousRun(ValueError):
    def __init__(self, candidates):
        self.candidates = candidates
        listing = '\n'.join(f"{r['run_id']} · {r['status']}" for r in candidates)
        super().__init__('Multiple runs match this directory; choose an explicit run ID:\n' + listing)


def workspace_aliases(record: dict) -> set[Path]:
    roots = {Path(p).resolve() for p in record.get('task_roots', ())}
    draft = record.get('draft') or {}
    structured = {p for p in roots if p.name == draft.get('exp_id') and p.parent.name == draft.get('run_id')}
    aliases = roots | {p.parent for p in structured}
    workspace = record.get('control_workspace')
    if workspace:
        aliases.add(Path(workspace).resolve())
    source = Path(record.get('source_manifest', ''))
    if source.parent.name == '.scienceflow':
        aliases.add(source.parent.parent.resolve())
    base = (record.get('draft') or {}).get('workspace_base')
    if base and (structured or workspace):
        aliases.add(Path(base).resolve())
    return aliases


def resolve_run(target: str | None = None, *, cwd: Path | None = None) -> dict:
    rows = list_runs()
    working = Path(cwd or Path.cwd()).resolve()
    candidate = Path(target or '.').expanduser()
    candidate = (candidate if candidate.is_absolute() else working / candidate).resolve()
    id_matches = [r for r in rows if target and r['run_id'] == target]
    path_matches = []
    if candidate.is_dir():
        for row in rows:
            if candidate in workspace_aliases(row) or any(candidate.is_relative_to(Path(p).resolve()) for p in row.get('task_roots', ())):
                path_matches.append(row)
    matches = {r['run_id']: r for r in (*id_matches, *path_matches)}
    if not matches:
        raise ValueError(f'No managed run matches {target or str(working)}. Use scienceflow status to list runs.')
    if len(matches) != 1:
        raise AmbiguousRun(list(matches.values()))
    return next(iter(matches.values()))


def overlaps(first, second) -> bool:
    """Canonical path overlap includes nested task roots and symlink aliases."""
    left = [Path(p).resolve() for p in first]
    right = [Path(p).resolve() for p in second]
    return any(a == b or a.is_relative_to(b) or b.is_relative_to(a) for a in left for b in right)


def workspace_runs(workspace: Path) -> list[dict]:
    """List only records belonging to this directory, never global recent runs."""
    candidate = Path(workspace).expanduser().resolve()
    return [row for row in list_runs() if candidate in workspace_aliases(row)
            or any(candidate.is_relative_to(Path(p).resolve()) for p in row.get('task_roots', ()))]


def select_workspace_run(action: str, target: str | None = None, *, cwd: Path | None = None) -> dict:
    """Prefer the active local run; mutations require an unambiguous eligible target."""
    if target:
        try:
            return resolve_run(target, cwd=cwd)
        except AmbiguousRun as exc:
            rows = exc.candidates
    else:
        rows = workspace_runs(Path(cwd or Path.cwd()))
    active = [row for row in rows if row['alive']]
    if action in {'stop', 'attach'}:
        eligible = active
    elif action == 'resume':
        eligible = active or [r for r in rows if r['status'] != 'completed']
    else:
        eligible = active or sorted(rows, key=lambda r: r.get('created_at', 0), reverse=True)[:1]
    if not eligible:
        raise ValueError(f'No {"active" if action in {"stop", "attach"} else "eligible"} research task in this workspace.')
    if len(eligible) > 1:
        raise AmbiguousRun(eligible)
    return eligible[0]


def run_label(row: dict) -> str:
    """Human-readable directory label; IDs remain available for advanced addressing."""
    workspace = row.get('control_workspace') or row.get('cwd')
    return Path(workspace).name if workspace else 'Research task'
