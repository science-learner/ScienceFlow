"""Read-only progress projection for a TUI-owned long research run."""

from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path

from inquirycraft.tui.formatting import duration

from .research_events import ResearchEvents
from .research_usage import ResearchUsage
from .telemetry.formatting import worker_resources
from .telemetry.sampler import SAMPLER


def _read_state(path: Path) -> dict:
    try:
        if path.stat().st_size > 4_000_000:
            return {}
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _clock(seconds: float) -> str:
    return duration(seconds)


def _best_metric_value(best: object, *, metric: str, lower_is_better: bool) -> float | None:
    """Return one comparable, evaluator-backed best value."""
    if not isinstance(best, dict):
        return None
    value = best.get("metric_value")
    if not (
        best.get("validation_ok") is True
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and best.get("metric_name") in (None, "", metric)
        and best.get("lower_is_better") in (None, lower_is_better)
    ):
        return None
    return float(value)


def _select_best(values: list[float], *, lower_is_better: bool) -> float | None:
    return (min(values) if lower_is_better else max(values)) if values else None


def _stage_count(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _runtime_metric_direction(
    *,
    draft_direction: object,
    metric: str,
    task_best: dict,
    worker_states: list[dict],
    events: ResearchEvents,
) -> bool | None:
    """Resolve a deferred direction from evaluator-backed runtime state."""
    sources = [task_best]
    sources.extend(
        best
        for worker in worker_states
        if isinstance(best := worker.get("global_best"), dict)
    )
    for best in sources:
        lower = best.get("lower_is_better")
        if isinstance(lower, bool) and best.get("metric_name") in (None, "", metric):
            return lower
    event_direction = events.metric_direction(metric)
    if isinstance(event_direction, bool):
        return event_direction
    return draft_direction if isinstance(draft_direction, bool) else None


class LongResearchProgress:
    def __init__(self, draft) -> None:
        self.draft = draft
        self.events = ResearchEvents()
        self.usage = ResearchUsage(getattr(draft, "model_config_path", ""))
        self.rows = ()
        self.resource_samples = {}
        self.worker_details = {}
        self.summary = ""
        self.root = Path(draft.workspace_base) / draft.run_id / draft.exp_id
        self.budget = max(1, int(draft.wall_clock_sec or 1))
        self.started = time.monotonic()
        self.active = True
        self.final_elapsed = None
        self.final_state = "stopped"
        self.status = "Starting worker processes"

    def snapshot(self) -> tuple[str, str]:
        elapsed = self.final_elapsed if self.final_elapsed is not None else time.monotonic() - self.started
        fraction = min(1, elapsed / self.budget)
        filled = int(fraction * 10)
        bar = " ".join("■" if index < filled else "□" for index in range(10))
        state = _read_state(self.root / "task_logs" / "lhr_state.json")
        worker_root = self.root / "workers"
        workers = sorted(set(worker_root.glob("*/logs/lhr_state.json")) |
                         {path.with_name("lhr_state.json") for path in worker_root.glob("*/logs/lhr_events.jsonl")})
        rows = []
        self.resource_samples = {}
        self.worker_details = {}
        bests = []
        metric = getattr(self.draft, "metric_name", "metric")
        task_best = state.get("global_best") or {}
        if not isinstance(task_best, dict):
            task_best = {}
        worker_states = [(path, _read_state(path)) for path in workers]
        for path, _worker in worker_states:
            self.events.read(path.with_name("lhr_events.jsonl"))
        lower = _runtime_metric_direction(
            draft_direction=getattr(self.draft, "lower_is_better", None),
            metric=metric,
            task_best=task_best,
            worker_states=[worker for _path, worker in worker_states],
            events=self.events,
        )
        self.lower_is_better = lower
        task_best_value = _best_metric_value(
            task_best,
            metric=metric,
            lower_is_better=lower,
        ) if isinstance(lower, bool) else None
        task_best_worker = str(task_best.get("worker_id") or "").casefold()
        if task_best_value is not None:
            bests.append(task_best_value)
        counts = {"running": 0, "waiting": 0, "completed": 0, "failed": 0, "stopped": 0}
        names = set()
        updated_at = []
        phases = {
            "artifact_update": "updating candidate artifacts",
            "running": "executing",
            "waiting": "waiting",
        }
        task_usage = self.usage.read_task(self.root)
        for path, worker in worker_states:
            if not task_usage:
                self.usage.read_worker(path.parent)
            best = worker.get("global_best") or {}
            value = (
                _best_metric_value(best, metric=metric, lower_is_better=lower)
                if isinstance(lower, bool)
                else None
            )
            event_best = (
                self.events.best_value(
                    path.parents[1].name,
                    metric_name=metric,
                    lower_is_better=lower,
                )
                if isinstance(lower, bool)
                else None
            )
            scores = [score for score in (value, event_best) if score is not None]
            name = path.parents[1].name
            stages = max(
                _stage_count(worker.get("stage_count")),
                self.events.stage_count(name),
            )
            if task_best_worker and name.casefold() == task_best_worker and task_best_value is not None:
                scores.append(task_best_value)
            value = _select_best(scores, lower_is_better=lower) if isinstance(lower, bool) else None
            if value is not None:
                bests.append(value)
            progress = worker.get("progress_last") or {}
            if not isinstance(progress, dict):
                progress = {}
            phase = str(progress.get("phase") or "")
            description = phases.get(phase, phase) or "waiting for progress report"
            names.add(name)
            worker_status = str(worker.get("status") or phase).lower()
            category = ("failed" if any(word in worker_status for word in ("fail", "error")) else
                        "completed" if worker_status in {"completed", "success", "finished"} else
                        "waiting" if not phase or any(word in worker_status for word in ("wait", "pending")) else "running")
            if worker_status in {'stopped', 'cancelled', 'canceled', 'interrupted', 'stopped_by_user'}:
                category = 'stopped'
            elif not self.active and category in {'running', 'waiting'}:
                category = 'failed' if self.final_state in {'failed', 'interrupted'} else ('completed' if self.final_state == 'completed' else 'stopped')
            counts[category] += 1
            phase_text = self.events.activity.get(name.lower(), description)
            if category in {"failed", "completed", "stopped"}:
                phase_text = "needs review" if category == "failed" else category
            age = max(0, time.time() - float(worker.get("generated_at") or time.time()))
            updated_at.append(worker.get("generated_at") or time.time())
            score = f"{value:.6g}" if value is not None else "—"
            live = self.active and category in {"running", "waiting"}
            assigned = {gpu for (owner, _), (_, ids) in self.events.resources.leases.items()
                        if owner.casefold() == name.casefold() for gpu in ids}
            sample = SAMPLER.worker(path.parents[1], assigned=assigned, active=live)
            self.resource_samples[name] = sample
            base = (
                f"{name}: {phase_text[:16] if phase_text != 'waiting for progress report' else phase_text}"
                f" · best {score} · Stages {stages}"
            )
            age_text = f" · {duration(age, relative=True)} ago" if live else ""
            rows.append(base + ' · ' + worker_resources(sample) + age_text)
            self.worker_details[name] = (
                f"{name}: {phase_text} · best {score} · Stages {stages} · "
                + worker_resources(sample, compact=False)
                + age_text
            )
        detail = "; ".join(list(self.worker_details.values())[:5]) + (f"; +{len(rows) - 5} workers (use panel pages)" if len(rows) > 5 else "") or (
            "Workers starting; waiting for first progress report"
            if state else "Preparing configuration and starting worker processes"
        )
        if self.active and elapsed >= self.budget:
            detail = "Time budget reached; waiting for evaluation and cleanup. " + detail
        self.status = (
            f"Time [{bar}] {fraction:.0%} · {_clock(elapsed)}/{_clock(self.budget)}"
            f" · remaining {_clock(self.budget - elapsed)}"
        )
        for index in range(int(getattr(self.draft, "workers", 0) or 0)):
            name = f"w{index:02d}"
            if name not in names:
                category = "waiting" if self.active else ("completed" if self.final_state == "completed" else
                           "failed" if self.final_state in {"failed", "interrupted"} else "stopped")
                counts[category] += 1
                final_label = "needs review" if self.final_state in {"failed", "interrupted"} else self.final_state
                value = task_best_value if name.casefold() == task_best_worker else None
                score = f"{value:.6g}" if value is not None else "—"
                rows.append(
                    f"{name}: "
                    f"{'waiting for first report' if self.active else 'task ' + final_label + '; no report'}"
                    f" · best {score} · Stages 0"
                )
        self.rows = tuple(rows)
        best_value = _select_best(bests, lower_is_better=lower) if isinstance(lower, bool) else None
        best_text = f"{best_value:.6g}" if best_value is not None else "—"
        valid = sum(value[0] for value in self.events.candidates.values())
        metric = getattr(self.draft, "metric_name", "metric")
        arrow = "↓" if lower is True else "↑" if lower is False else "?"
        self.best_text = best_text
        self.worker_counts = counts
        self.summary = (f"{self.status}\n"
                        f"{Path(self.draft.workspace_base).parent.name} · Workers {len(rows)} · running {counts['running']} · waiting {counts['waiting']} · issues {counts['failed']} · done {counts['completed']} · stopped {counts['stopped']}\n"
                        f"Result  {metric} {arrow} · Best {best_text} · Evaluated {len(self.events.candidates)} / valid {valid}\n"
                        + self.events.control_summary() + "\n" + self.usage.summary())
        updated = max(updated_at or [time.time()])
        if workers and self.active:
            detail += f" · last worker report {duration(time.time() - updated, relative=True)} ago"
        return self.status, detail

    async def watch(self, host) -> None:
        last_detail = ""
        last_notice = 0.0
        while True:
            status, detail = await asyncio.to_thread(self.snapshot)
            host.set_host_status(status)
            now = time.monotonic()
            panel = getattr(host, "set_host_panel", None)
            if panel is not None:
                panel(self.summary, self.rows)
                host.set_host_status("long research · running · /status details")
                for event in self.events.new_events[-5:]:
                    await host.notice(event)
                self.events.new_events.clear()
            elif now - last_notice >= 60 or (detail != last_detail and now - last_notice >= 15):
                await host.notice(f"{status}\n{detail}")
                last_notice, last_detail = now, detail
            await asyncio.sleep(2)
