"""Incremental projection of evaluator events, without model-generated summaries."""

import hashlib
import json
import math
from collections import Counter, defaultdict


class ResearchEvents:
    def __init__(self):
        from .control.tasks.metrics import ResourceFacts
        self.resources = ResourceFacts()
        self.offsets = {}
        self.counts = Counter()
        self.control_seen = set()
        self.candidates = {}
        self.candidate_metrics = {}
        self.activity = {}
        self.stage_ids = defaultdict(set)
        self.estra_fallback_count = 0
        self.legacy_estra_fallback_count = 0
        self.new_events = []

    def read(self, path):
        try:
            offset = self.offsets.get(path, 0)
            if path.stat().st_size < offset:
                offset = 0
            with path.open('rb') as stream:
                stream.seek(offset)
                for _ in range(250):
                    line = stream.readline(1_000_000)
                    if not line or not line.endswith(b'\n'):
                        break
                    self.offsets[path] = stream.tell()
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    payload = event.get('payload') or {}
                    if not isinstance(payload, dict):
                        continue
                    worker = str(event.get('worker_id') or path.parents[1].name)
                    name = event.get('event', '')
                    if isinstance(name, str) and name.startswith(('estra_', 'resource_')):
                        identity = hashlib.sha256((worker + json.dumps(event, sort_keys=True)).encode()).digest()
                        if identity in self.control_seen:
                            continue
                        self.control_seen.add(identity)
                    self.resources.observe(event, worker)
                    if isinstance(name, str):
                        self.counts[name] += 1
                    if name == 'estra_decision':
                        decision_mode = str(
                            payload.get('decision_mode')
                            or event.get('decision_mode')
                            or ''
                        )
                        if decision_mode in {'safe_fallback', 'deterministic_fallback'}:
                            self.estra_fallback_count += 1
                    elif name == 'estra_deterministic_fallback':
                        # Pre-safe-fallback logs used a standalone event and did not
                        # emit the canonical estra_decision record.
                        self.legacy_estra_fallback_count += 1
                    if name == 'stage_captured':
                        stage_id = payload.get('node_uid') or payload.get('stage_id')
                        if stage_id:
                            self.stage_ids[worker.casefold()].add(str(stage_id))
                    elif name == 'evaluator_metric_event':
                        candidate = payload.get('candidate_id') or payload.get('artifact_sha')
                        if not candidate:
                            continue
                        key = (worker, str(candidate), str(payload.get('artifact_sha', '')))
                        value = payload.get('metric_value')
                        valid = (payload.get('evaluator_status') == 'ok'
                                 and payload.get('selection_eligible') is True
                                 and payload.get("validation_ok") is not False
                                 and not isinstance(value, bool)
                                 and isinstance(value, (float, int)) and math.isfinite(value))
                        if key not in self.candidates:
                            previous = [score for ok, score in self.candidates.values() if ok]
                            lower = payload.get("lower_is_better")
                            improved = (
                                valid
                                and isinstance(lower, bool)
                                and (
                                    not previous
                                    or (value < min(previous) if lower else value > max(previous))
                                )
                            )
                            label = "New best" if improved else "Evaluated"
                            stamp = str(event.get("timestamp_utc") or "")[11:19]
                            self.new_events.append(f"{stamp} {worker} · {label} {candidate}: " + (f"{value:.6g}" if valid else "invalid candidate"))
                        self.candidates[key] = (valid, value)
                        self.candidate_metrics[key] = (payload.get("metric_name"), payload.get("lower_is_better"))
                        self.activity[worker.lower()] = 'evaluation complete' if valid else 'invalid candidate'
                    elif name == 'progress_heartbeat':
                        self.activity[worker.lower()] = {'artifact_update': 'updating candidate', 'running': 'running'}.get(payload.get('phase'), 'progress report')
                    elif name == 'stage_commit_text_requested':
                        self.activity[worker.lower()] = 'waiting for stage summary'
                    elif name == 'stage_commit_text_parse_retry':
                        self.activity[worker.lower()] = 'retrying stage summary'
        except OSError:
            pass

    def best_value(self, worker, *, metric_name, lower_is_better):
        """Project eligible evaluator evidence before a stage/state commit completes."""
        values = []
        for key, (valid, value) in self.candidates.items():
            if not valid or key[0].casefold() != worker.casefold():
                continue
            metric, lower = self.candidate_metrics.get(key, (None, None))
            if metric and metric != metric_name:
                continue
            if lower is not None and lower != lower_is_better:
                continue
            values.append(value)
        return (min(values) if lower_is_better else max(values)) if values else None

    def metric_direction(self, metric_name):
        """Return the runtime evaluator direction once the evidence agrees."""
        directions = {
            lower
            for metric, lower in self.candidate_metrics.values()
            if isinstance(lower, bool) and metric in (None, "", metric_name)
        }
        return directions.pop() if len(directions) == 1 else None

    def stage_count(self, worker):
        """Return distinct formally captured stages observed for one worker."""
        return len(self.stage_ids[str(worker).casefold()])

    def control_summary(self):
        counts = self.counts
        fallback_count = (
            self.estra_fallback_count + self.legacy_estra_fallback_count
        )
        decision_count = counts['estra_decision'] + self.legacy_estra_fallback_count
        fallback = f" (fallback {fallback_count})" if fallback_count else ""
        return (
            f"ESTRA · Decisions {decision_count}{fallback} · Switches {counts['estra_stage_switched']}"
            f" · Compactions {counts['estra_keep_current_compacted']} · Invalid {counts['estra_invalid']}\n"
            + self.resources.summary()
        )
