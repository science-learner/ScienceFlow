# Workspace Web monitor

From an experiment workspace, run `scienceflow web`, or enter `/web` in its TUI.
The launcher prints the actual URL and port, and reuses a running monitor for the
same workspace. If the default port is occupied, a free port is selected.

```bash
scienceflow web --workspace /path/to/workspace
scienceflow web --workspace /path/to/workspace --port 0 --open
scienceflow web stop --workspace /path/to/workspace
```

The task overview is paginated. Click a task for its own page with the shared TUI
statistics, worker pages, metric curve, and stage lineage. Parent and restoration
edges come from explicit candidate IDs in stage logs; unknown relationships are
not inferred from row order. Off-page references remain visible in the lineage
table. Curves show stage history; the Best card uses the current TUI projection.

## Remote access

The default listener is `127.0.0.1`. In an SSH session, copy the printed port-forward
command and run it on your local computer, replacing `user@server` with your SSH
login. Then open the printed URL locally. VS Code Remote users can forward the
printed port using the Ports panel. `--open` skips automatic browser launch in SSH
sessions. Explicit `--host 0.0.0.0` allows listening on all IPv4 interfaces.

The URL includes a random access token. The server serves only the monitor page
and cached JSON, not arbitrary workspace files. Do not publish the access URL.

## Refresh and lifecycle

Web runs in a separate process with reduced scheduling priority. Log scans run
asynchronously in a background thread; HTTP requests receive the previous cached
snapshot while a scan is running. No LLM requests are made and no research state
is changed. Scans do consume a small amount of CPU and disk I/O.

Overview refresh defaults to five seconds and automatically slows down when scans
take longer. Only task pages requested by a browser load curve/lineage details;
these are cached for at least 15 seconds and unchanged source files are not reread.
Background browser tabs pause polling. Closing TUI does not stop Web; `/web stop`
or `scienceflow web stop` stops Web only. Scan time and last update are displayed.

The existing `scienceflow monitor-trace --manifest ... --output ...` exporter
remains available for manifest-based offline reports.

## Frontend structure and development checks

`web_monitor/page.py` assembles packaged HTML, CSS and JavaScript into one inline
page. No CDN or browser build step is required when running an installed wheel.
The assets are split by responsibility:

- `core.js`: DOM helpers and shared display formatting.
- `metric.js`: elapsed-time metric chart.
- `lineage_layout.js`: source-aligned graph coordinates, without DOM access.
- `lineage_viewport.js`: worker viewport state, wheel zoom and pointer dragging.
- `lineage.js`: worker selection and graph rendering.
- `view.js`: task sections and preservation of scroll/expanded records.
- `refresh.js`: single request chain, adaptive polling and interaction guards.

Python `data.py` reads/caches logs; `projection.py` transforms recorded timestamps
and ancestry. Missing/non-finite timestamps stay unknown. `TaskIndex.snapshot()`
is a read-only API and does not synchronize or create research records.

Run the browser DOM regression suite using Node 24 (development only):

```bash
npm --prefix tests/web_monitor ci
npm --prefix tests/web_monitor test
npm --prefix tests/web_monitor run format:check
python -m pytest tests/test_web_monitor.py tests/test_monitor_trace_html.py
```

The DOM suite covers worker navigation, source alignment, wheel anchoring, drag
lifecycle, scroll preservation, timestamp axes and coalescing repeated refresh
requests. It does not replace visual checks in a real browser. After changing
packaged assets, rebuild the wheel and restart the Web service.

## Worker utilization

Worker rows share a cached sampler with the one-line host resource summary.
Process/CPU snapshots are refreshed at most every five seconds and GPU queries
at most every ten seconds, per monitor process. CPU reflects processes whose
working directory belongs to the worker plus their descendants, normalized by
observed CPU affinity; shared coordinator overhead cannot be attributed to a
single worker. First samples and inaccessible processes display `—`.

GPU IDs are physical device indices, identified from active leases or worker
process mappings. Utilization and VRAM marked `dev` are device-wide readings,
not an estimate of exclusive worker use. `shared` indicates other observed GPU
processes on that device; unavailable process attribution does not imply exclusive
use. Many devices are summarized with ID ranges and averages. Click a worker row
in the expanded TUI task card to see its full list, or use `/status N`.
