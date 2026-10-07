import { useEffect, useRef, useState } from 'react';
import clsx from 'clsx';
import { useGatewayStore } from '../store/useGatewayStore';
import { useT } from '../i18n/useT';
import { useMonitorSources } from '../hooks/useMonitorSources';
import {
  gatewayFetchMonitor,
  gatewayAgentStatus,
  getSystemStats,
  type GatewayMonitorMetrics,
  type GatewayAgentStatusResult,
  type SystemStats,
} from '../api/gateway';
import type { FileNode } from '../types';

// ── formatting helpers ─────────────────────────────────────────────────────

function formatMonitorBytes(bytes: number): string {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index++;
  }
  return `${value >= 10 || index === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[index]}`;
}

function fmtCompact(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return '—';
  if (Math.abs(n) >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (Math.abs(n) >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(Math.round(n));
}

function fmtPct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—';
  return `${(v * 100).toFixed(digits)}%`;
}

function fmtSec(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v) || v <= 0) return '—';
  return v >= 1 ? `${v.toFixed(2)}s` : `${Math.round(v * 1000)}ms`;
}

function fmtClock(ts: number | string): string {
  if (!ts) return '';
  const d = typeof ts === 'number' ? new Date(ts * 1000) : new Date(ts);
  if (isNaN(d.getTime())) return '';
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function fileSizeBySuffix(tree: FileNode[], suffix: string): number | null {
  let best: FileNode | null = null;
  for (const n of tree) {
    if (n.type === 'file' && n.path.endsWith(suffix) && (!best || n.mtime > best.mtime)) {
      best = n;
    }
  }
  return best ? best.size : null;
}

function firstLine(text: string): string {
  const line = (text || '').split('\n')[0].trim();
  return line.length > 120 ? line.slice(0, 120) + '…' : line;
}

function MonitorLane({
  label,
  value,
  total,
  danger,
}: {
  label: string;
  value: number;
  total: number;
  danger?: boolean;
}) {
  const pct = total > 0 ? Math.round((value / total) * 100) : 0;
  return (
    <div className="lane">
      <span>{label}</span>
      <div
        className={clsx('bar', danger && 'danger')}
        style={{ '--w': `${pct}%` } as React.CSSProperties}
      >
        <i></i>
      </div>
      <span>{total > 0 ? `${value}/${total}` : '—'}</span>
    </div>
  );
}

// ── monitor panel ──────────────────────────────────────────────────────────

// Data sources: gateway /sessions/{id}/monitor + /agent plus per-run telemetry
// files read through the workspace APIs (time trace, provider calls, runtime
// events, compression ledger, productivity, context events).
export function MonitorPanel() {
  const t = useT();
  const token = useGatewayStore((s) => s.token);
  const sessionId = useGatewayStore((s) => s.sessionId);
  const fileTree = useGatewayStore((s) => s.fileTree);

  const [gatewayMonitor, setGatewayMonitor] = useState<GatewayMonitorMetrics | null>(null);
  const [agentStatus, setAgentStatus] = useState<GatewayAgentStatusResult | null>(null);
  const [sysStats, setSysStats] = useState<SystemStats | null>(null);
  const [loading, setLoading] = useState(false);
  const sources = useMonitorSources(Boolean(token && sessionId));

  const prevRef = useRef<Record<string, unknown>>({});
  const [changedKeys, setChangedKeys] = useState<Set<string>>(new Set());

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      if (!token || !sessionId) {
        setGatewayMonitor(null);
        setAgentStatus(null);
        return;
      }
      setLoading(true);
      const statsPromise = getSystemStats();
      try {
        const [mon, status] = await Promise.all([
          gatewayFetchMonitor(token, sessionId).catch(() => null),
          gatewayAgentStatus(token, sessionId).catch(() => null),
        ]);
        if (!cancelled) {
          if (mon) setGatewayMonitor(mon);
          if (status) setAgentStatus(status);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
      const stats = await statsPromise;
      if (!cancelled && stats) setSysStats(stats);
    };
    void tick();
    const timer = window.setInterval(() => void tick(), 5000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [token, sessionId]);

  // ── derived core metrics ──
  const trace = sources.trace;
  const events = sources.events;
  const provider = sources.provider;
  const task = agentStatus?.task;
  const queue = agentStatus?.queue_stats;

  const gatewayTokens = gatewayMonitor?.tokens;
  const tokensIn = trace ? trace.tokensIn : gatewayTokens?.input ?? 0;
  const tokensOut = trace ? trace.tokensOut : gatewayTokens?.output ?? 0;
  const tokensCached = trace ? trace.tokensCached : gatewayTokens?.cached ?? 0;
  const cacheRate = trace?.cacheRate ?? (tokensIn > 0 ? tokensCached / tokensIn : null);
  const calls = trace?.calls ?? provider?.calls ?? 0;
  const runtimeStatus = task?.status || gatewayMonitor?.runtime.status || agentStatus?.status || 'idle';
  const elapsed = gatewayMonitor?.runtime.elapsed || '--';
  const resources = gatewayMonitor?.resources;
  const sourceLabel = gatewayMonitor?.sources?.length
    ? gatewayMonitor.sources.join(' · ')
    : t.common.dash;

  // ── context metrics ──
  const lastCall = sources.providerCalls.length > 0
    ? sources.providerCalls[sources.providerCalls.length - 1]
    : null;
  const lastTraceRow = sources.traceRows.length > 0
    ? sources.traceRows[sources.traceRows.length - 1]
    : null;
  const contextLength = lastCall?.tokensInput ?? lastTraceRow?.tokensInput ?? null;
  const contextMessages = lastCall?.providerMessageCount ?? null;
  const compactionGen = lastCall?.compactionGeneration ?? null;
  const longTermBytes = fileSizeBySuffix(fileTree, 'agent_memory/ScienceAgent/long_term.jsonl');
  const shortTermBytes = fileSizeBySuffix(fileTree, 'agent_memory/ScienceAgent/short_term.json');
  const ctxEvents = sources.contextEvents;

  // ── activity / efficiency / risks ──
  const toolStarted = events?.toolStarted ?? 0;
  const toolCompleted = events?.toolCompleted ?? 0;
  const toolFailed = events?.toolFailed ?? 0;
  const llmRequested = events?.llmRequested ?? 0;
  const llmCompleted = events?.llmCompleted ?? 0;
  const turnsStarted = events?.turnsStarted ?? 0;
  const turnsCompleted = events?.turnsCompleted ?? 0;
  const abnormalFinishes = Object.entries(provider?.finishReasons || {}).filter(
    ([reason]) => reason !== 'stop' && reason !== 'tool_calls',
  );
  const abnormalCount = abnormalFinishes.reduce((n, [, count]) => n + count, 0);
  const riskCount =
    toolFailed + abnormalCount + (trace?.failovers ?? 0) + (trace?.errors ?? 0);

  // ── recent call rows (provider calls preferred, trace fallback) ──
  type CallRow = {
    key: string;
    time: string;
    model: string;
    role: string;
    tin: number;
    tout: number;
    cache: number | null;
    ttft: string;
    tpot: string;
    finish: string;
  };
  const callRows: CallRow[] = sources.providerCalls.length > 0
    ? sources.providerCalls.slice(-12).reverse().map((c, i) => ({
        key: `p${c.callSeq}-${i}`,
        time: fmtClock(c.timestamp),
        model: c.model,
        role: c.llmRole || c.turnKind || c.callKind,
        tin: c.tokensInput,
        tout: c.tokensOutput,
        cache: c.cacheRate ?? (c.tokensInput > 0 ? c.tokensCached / c.tokensInput : null),
        ttft: fmtSec(c.ttftSec),
        tpot: c.tpotSec !== null ? `${Math.round(c.tpotSec * 1000)}ms` : '—',
        finish: c.attempt > 1 ? `attempt ${c.attempt}` : c.finishReason,
      }))
    : sources.traceRows.slice(-12).reverse().map((r, i) => ({
        key: `t${i}`,
        time: r.timestamp.slice(11, 19),
        model: r.model,
        role: r.llmRole,
        tin: r.tokensInput,
        tout: r.tokensOutput,
        cache: r.cacheRate ?? (r.tokensInput > 0 ? r.tokensCached / r.tokensInput : null),
        ttft: fmtSec(r.ttftSec),
        tpot: r.tpotMs !== null ? `${Math.round(r.tpotMs)}ms` : '—',
        finish: r.status,
      }));

  // ── change highlighting ──
  useEffect(() => {
    const flat: Record<string, unknown> = {
      cost: trace?.costUsd ?? null,
      tokens: tokensIn + tokensOut,
      cache: cacheRate,
      queue: queue ? queue.queued + queue.running : null,
      tools: events ? events.toolCompleted : null,
      failed: events ? events.toolFailed : null,
      ttft: trace?.avgTtftSec ?? null,
      tpot: trace?.avgTpotMs ?? null,
      ctx: contextLength,
    };
    const prev = prevRef.current;
    const changed = new Set<string>();
    for (const key of Object.keys(flat)) {
      if (prev[key] !== undefined && prev[key] !== flat[key]) changed.add(key);
      prev[key] = flat[key];
    }
    if (changed.size > 0) {
      setChangedKeys(changed);
      const timer = setTimeout(() => setChangedKeys(new Set()), 1200);
      return () => clearTimeout(timer);
    }
    return undefined;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sources.lastUpdated, agentStatus, gatewayMonitor]);

  if (!gatewayMonitor && sources.lastUpdated === 0) {
    return (
      <div className="dim monitor-empty">
        {loading ? t.l1Workspace.loadingMonitor : t.l1Workspace.noMonitorData}
      </div>
    );
  }

  const sysCpu = sysStats ? Math.round(sysStats.cpu_percent) : null;
  const riskSummary = riskCount > 0
    ? `${riskCount} ${t.l1Workspace.activeAlerts}`
    : t.l1Workspace.noActiveAlerts;
  const costValue = trace?.costUsd !== null && trace?.costUsd !== undefined
    ? `$${trace.costUsd.toFixed(2)}`
    : t.common.dash;

  return (
    <div>
      <div className="monitor-grid">
        {/* ── runtime, scheduler & activity ── */}
        <div className="viz-card">
          <div className="runtime-status-grid">
            <div className="monitor-subpanel">
              <div className="section-title">{t.l1Workspace.scheduler}</div>
              <div className="kv">
                <div className="kv-row">
                  <span>{t.l1Workspace.taskState}</span>
                  <span
                    className={clsx(
                      runtimeStatus === 'running' && 'monitor-status-running',
                      runtimeStatus === 'failed' && 'monitor-status-failed',
                    )}
                    title={task?.error || runtimeStatus}
                  >
                    {runtimeStatus}
                  </span>
                </div>
                <div className="kv-row">
                  <span>{t.l1Workspace.queue}</span>
                  <span title={`${t.l1Workspace.queued}: ${queue?.queued ?? 0} / running: ${queue?.running ?? 0}`}>
                    {queue ? `${queue.queued} / ${queue.running}` : t.common.dash}
                  </span>
                </div>
                <div className="kv-row">
                  <span>{t.l1Workspace.exitCode}</span>
                  <span>{task?.exit_code !== undefined && task?.exit_code !== null ? task.exit_code : t.common.dash}</span>
                </div>
                <div className="kv-row">
                  <span>{t.l1Workspace.rounds}</span>
                  <span>{trace ? trace.rounds : t.common.dash}</span>
                </div>
                <div className="kv-row">
                  <span>{t.l1Workspace.modelsLabel}</span>
                  <span title={trace?.models.join(', ') || ''}>
                    {trace?.models.length ? trace.models.join(', ') : t.common.dash}
                  </span>
                </div>
                <div className="kv-row">
                  <span>{t.l1Workspace.monitorFile}</span>
                  <span title={sourceLabel}>{sourceLabel}</span>
                </div>
              </div>
            </div>
            <div className="monitor-subpanel">
              <div className="section-title">{t.l1Workspace.activity}</div>
              <div className="lane-grid">
                <MonitorLane label={t.l1Workspace.toolsLabel} value={toolCompleted} total={toolStarted} />
                <MonitorLane label={t.l1Workspace.failuresLabel} value={toolFailed} total={toolStarted} danger />
                <MonitorLane label={t.l1Workspace.llmCallsLabel} value={llmCompleted} total={llmRequested} />
                <MonitorLane label={t.l1Workspace.turnsLabel} value={turnsCompleted} total={turnsStarted} />
              </div>
            </div>
            <div className="monitor-subpanel monitor-wide">
              <div className="runtime-wide-grid">
                <div>
                  <div className="section-title">{t.l1Workspace.budget}</div>
                  <div className="bar-row">
                    <span>{t.l1Workspace.wall}</span>
                    <div className="bar" style={{ '--w': '100%' } as React.CSSProperties}><i></i></div>
                    <span>{elapsed}</span>
                  </div>
                </div>
                <div>
                  <div className="section-title">{t.l1Workspace.riskHints}</div>
                  <div className="kv">
                    <div className="kv-row">
                      <span>{t.l1Workspace.status}</span>
                      <span className={clsx(riskCount > 0 && 'monitor-status-failed')}>{riskSummary}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* ── cost, latency & local machine ── */}
        <div className="viz-card">
          <div className="section-title">{t.l1Workspace.costAndLatency}</div>
          <div className="donut-row">
            <div
              className="donut"
              style={{ '--dp': `${cacheRate !== null ? Math.round(cacheRate * 100) : 0}%` } as React.CSSProperties}
            >
              {fmtPct(cacheRate, 0)}
            </div>
            <div className="kv">
              <div className={clsx('kv-row', changedKeys.has('cost') && 'value-changed')}>
                <span>{t.l1Workspace.cost}</span>
                <span title={trace?.costKnownCalls ? `${trace.costKnownCalls} calls with cost` : ''}>{costValue}</span>
              </div>
              <div className={clsx('kv-row', changedKeys.has('tokens') && 'value-changed')}>
                <span>{t.l1Workspace.tokensIn}</span>
                <span title={String(tokensIn)}>{fmtCompact(tokensIn)}</span>
              </div>
              <div className={clsx('kv-row', changedKeys.has('tokens') && 'value-changed')}>
                <span>{t.l1Workspace.tokensOut}</span>
                <span title={String(tokensOut)}>{fmtCompact(tokensOut)}</span>
              </div>
              <div className={clsx('kv-row', changedKeys.has('cache') && 'value-changed')}>
                <span>{t.l1Workspace.cacheHit}</span>
                <span title={String(tokensCached)}>{`${fmtCompact(tokensCached)} · ${fmtPct(cacheRate)}`}</span>
              </div>
              <div className="kv-row">
                <span>{t.l1Workspace.llmCallsLabel}</span>
                <span>{calls || t.common.dash}</span>
              </div>
            </div>
          </div>
          <div className="section-title">{t.l1Workspace.latencyHealth}</div>
          <div className="latency-metrics">
            <div className={clsx('latency-metric', changedKeys.has('ttft') && 'value-changed')}>
              <span>{t.l1Workspace.firstResponse}</span>
              <strong>{fmtSec(trace?.avgTtftSec)}</strong>
              <small>{`max ${fmtSec(trace?.maxTtftSec)}`}</small>
            </div>
            <div className={clsx('latency-metric', changedKeys.has('tpot') && 'value-changed')}>
              <span>{t.l1Workspace.tpotLabel}</span>
              <strong>{trace?.avgTpotMs !== null && trace?.avgTpotMs !== undefined ? `${Math.round(trace.avgTpotMs)}ms` : '—'}</strong>
              <small>{`max ${trace?.maxTpotMs !== null && trace?.maxTpotMs !== undefined ? `${Math.round(trace.maxTpotMs)}ms` : '—'}`}</small>
            </div>
            <div className={clsx('latency-metric', changedKeys.has('failed') && 'value-changed')}>
              <span>{t.l1Workspace.failuresLabel}</span>
              <strong>{riskCount}</strong>
              <small>{`failover ${trace?.failovers ?? 0}`}</small>
            </div>
            <div className="latency-metric">
              <span>{t.l1Workspace.queue}</span>
              <strong>{queue ? queue.queued + queue.running : 0}</strong>
              <small>{t.l1Workspace.queued}</small>
            </div>
          </div>
          <div className="section-title">{t.l1Workspace.localResources}</div>
          <div className="kv">
            <div className="kv-row">
              <span>{t.l1Workspace.cpu}</span>
              <span>{sysCpu !== null ? `${sysCpu}%` : t.common.dash}</span>
            </div>
            <div className="kv-row">
              <span>{t.l1Workspace.memory}</span>
              <span>{sysStats ? `${formatMonitorBytes(sysStats.memory_used_bytes)} / ${formatMonitorBytes(sysStats.memory_total_bytes)}` : t.common.dash}</span>
            </div>
            <div className="kv-row">
              <span>{t.l1Workspace.storage}</span>
              <span>{sysStats ? `${formatMonitorBytes(sysStats.disk_used_bytes)} / ${formatMonitorBytes(sysStats.disk_total_bytes)}` : (resources?.storage_display || t.common.dash)}</span>
            </div>
          </div>
        </div>
      </div>

      <div className="monitor-grid monitor-grid-row">
        {/* ── context & memory ── */}
        <div className="viz-card">
          <div className="section-title">{t.l1Workspace.contextAndMemory}</div>
          <div className="kv">
            <div className={clsx('kv-row', changedKeys.has('ctx') && 'value-changed')}>
              <span>{t.l1Workspace.contextLength}</span>
              <span title={contextLength !== null ? `${contextLength} tokens` : ''}>
                {contextLength !== null ? `${fmtCompact(contextLength)} tok` : t.common.dash}
              </span>
            </div>
            <div className="kv-row">
              <span>{t.l1Workspace.contextMessages}</span>
              <span>{contextMessages !== null ? contextMessages : t.common.dash}</span>
            </div>
            <div className="kv-row">
              <span>{t.l1Workspace.compactions}</span>
              <span>{compactionGen !== null ? compactionGen : t.common.dash}</span>
            </div>
            <div className="kv-row">
              <span>{t.l1Workspace.contextTriggers}</span>
              <span>
                {ctxEvents
                  ? `${ctxEvents.checks} / ${ctxEvents.triggers}`
                  : t.common.dash}
              </span>
            </div>
            <div className="kv-row">
              <span>{t.l1Workspace.longTermMemory}</span>
              <span>{longTermBytes !== null ? formatMonitorBytes(longTermBytes) : t.common.dash}</span>
            </div>
            <div className="kv-row">
              <span>{t.l1Workspace.shortTermMemory}</span>
              <span>{shortTermBytes !== null ? formatMonitorBytes(shortTermBytes) : t.common.dash}</span>
            </div>
          </div>
        </div>

        {/* ── efficiency & risks ── */}
        <div className="viz-card">
          <div className="section-title">{t.l1Workspace.efficiencyAndRisk}</div>
          <div className="kv">
            <div className="kv-row">
              <span>{t.l1Workspace.outputCompression}</span>
              <span title={sources.toolIndex ? `${sources.toolIndex.rawChars} → ${sources.toolIndex.compressedChars} chars` : ''}>
                {sources.toolIndex
                  ? `${sources.toolIndex.reductionPct !== null ? `${sources.toolIndex.reductionPct.toFixed(0)}%` : '—'} · ${sources.toolIndex.calls}`
                  : t.common.dash}
              </span>
            </div>
            <div className="kv-row">
              <span>{t.l1Workspace.writesEdits}</span>
              <span>
                {sources.productivity
                  ? `${sources.productivity.writeSuccessCount}/${sources.productivity.writeCount} · ${sources.productivity.editSuccessCount}/${sources.productivity.editCount}`
                  : t.common.dash}
              </span>
            </div>
            <div className="kv-row">
              <span>{t.l1Workspace.metricProduced}</span>
              <span>{sources.productivity ? (sources.productivity.hadMetric ? 'yes' : 'no') : t.common.dash}</span>
            </div>
          </div>
          <div className="section-title">{t.l1Workspace.riskHints}</div>
          <div className="monitor-risk-list">
            {(events?.toolFailures || []).slice(0, 4).map((f, i) => (
              <div key={`${f.name}-${i}`} className="monitor-risk-item">
                <span className="monitor-risk-tag danger">{f.name}</span>
                <span className="monitor-risk-text" title={f.preview}>{firstLine(f.preview)}</span>
              </div>
            ))}
            {abnormalFinishes.map(([reason, count]) => (
              <div key={reason} className="monitor-risk-item">
                <span className="monitor-risk-tag warn">{reason}</span>
                <span className="monitor-risk-text">×{count}</span>
              </div>
            ))}
            {(trace?.failovers ?? 0) > 0 && (
              <div className="monitor-risk-item">
                <span className="monitor-risk-tag warn">failover</span>
                <span className="monitor-risk-text">×{trace?.failovers}</span>
              </div>
            )}
            {riskCount === 0 && <div className="dim">{t.l1Workspace.riskNone}</div>}
          </div>
        </div>
      </div>

      {/* ── recent LLM calls ── */}
      <div className="section-title">{t.l1Workspace.recentCalls}</div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>{t.l1Workspace.wall}</th>
              <th>{t.l1Workspace.modelLabel}</th>
              <th>{t.l1Workspace.roleLabel}</th>
              <th>{t.l1Workspace.tokensIn}</th>
              <th>{t.l1Workspace.tokensOut}</th>
              <th>{t.l1Workspace.cacheRateLabel}</th>
              <th>{t.l1Workspace.firstResponse}</th>
              <th>{t.l1Workspace.tpotLabel}</th>
              <th>{t.l1Workspace.finishLabel}</th>
            </tr>
          </thead>
          <tbody>
            {callRows.length > 0 ? (
              callRows.map((row) => (
                <tr key={row.key}>
                  <td className="dim">{row.time}</td>
                  <td title={row.model}>{row.model || t.common.dash}</td>
                  <td className="dim">{row.role || t.common.dash}</td>
                  <td>{fmtCompact(row.tin)}</td>
                  <td>{fmtCompact(row.tout)}</td>
                  <td>{fmtPct(row.cache, 0)}</td>
                  <td>{row.ttft}</td>
                  <td>{row.tpot}</td>
                  <td className="dim">{row.finish || t.common.dash}</td>
                </tr>
              ))
            ) : (
              <tr>
                <td className="dim" colSpan={9}>{t.l1Workspace.noLatencySamples}</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
