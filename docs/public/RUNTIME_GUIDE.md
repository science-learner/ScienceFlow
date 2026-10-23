# Runtime guide

[Quick start](../../README.md) · [中文快速开始](README_CN.md)

## Start a task

```bash
scienceflow tui
# Or prepare a managed task from the CLI:
scienceflow run circle-packing --workers 2 --gpu cpu --duration 1h
```

In TUI, enter `/long-research circle-packing` or describe a new task, then enter
`run` after preflight passes. Use `/models` to choose the chat model.

`scienceflow run TARGET` accepts a task ID, local task folder, or quoted new-task
description. Omit TARGET for an interactive task picker; non-interactive calls
require TARGET or `--manifest FILE`. With a manifest, put all run settings in that
file. See `scienceflow run --help` for options.

## Choose a workspace

The current directory holds chat sessions and the task board by default.
`--workspace PATH` selects another management workspace. To put a task on a
different local disk:

```bash
scienceflow run circle-packing --duration 1h --execution-root /data/experiments
```

In TUI, add `workspace=/data/experiments` to `/long-research`. Each launch gets a
unique task subdirectory; datasets are not copied. Relative paths use the
management workspace. Resume keeps the task's recorded directory. This preview
supports local execution; `--execution-root` cannot be combined with `--manifest`.

## Monitor, stop, and resume

- TUI: `/tasks`, `/status N`, `/stop N`, `/resume N`; `/web` opens the monitor.
- CLI: `scienceflow status`, `scienceflow stop TARGET`, `scienceflow resume TARGET`.
  Here TARGET is a run ID or task directory.
- `scienceflow tui --resume` restores **chat**; `/resume N` resumes **research**.
- Closing TUI leaves managed research running. Chat shell commands have no default
  timeout and run with your user permissions, without an OS sandbox. Esc cancels
  the active chat; exiting TUI also stops its foreground commands and children.

## Find results and reports

The task directory contains `report.md`, `report.html`, and `report_assets/`;
previous attempts are under `task_logs/reports/attempt-NNN/`.

Reports get a separate **5-minute budget**, outside research time, and finish early
when ready. Set `report_budget_sec` in a manifest to change it. If reporting fails
after research has ended, resume retries the report without restarting exploration.
Open the report in Web and use browser printing to save a PDF.
