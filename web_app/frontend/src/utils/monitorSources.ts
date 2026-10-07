// Parsers for the ScienceFlow per-run telemetry files surfaced in the L1
// monitor. All sources live inside the session workspace (read via the
// gateway file APIs), so no backend changes are required:
//
//   run/logs/scienceflow_time_trace.csv   one row per LLM/tool span
//   run/.logs/agent_provider_calls.jsonl  one record per provider call
//   run/.logs/agent_runtime_events.jsonl  structured agent event stream
//   run/.logs/tool_outputs/index.txt      tool output compression ledger
//   run/.agent_memory/productivity.json   write/edit productivity counters
//   run/logs/lhr_estra_events.jsonl       context-limit / compaction events (heavy)
//
// The parsers are defensive: unknown shapes are skipped and every aggregate is
// optional so a partially written file never breaks the monitor.

// ── time_trace.csv ─────────────────────────────────────────────────────────

export interface TraceRow {
  timestamp: string;
  nodeId: string;
  category: string;
  operation: string;
  durationHrs: number;
  tokensInput: number;
  tokensOutput: number;
  tokensCached: number;
  cacheRate: number | null;
  costUsd: number | null;
  ttftSec: number | null;
  tpotMs: number | null;
  failovers: number;
  status: string;
  model: string;
  llmRole: string;
  round: number | null;
  tool: string;
}

export interface TimeTraceSummary {
  calls: number;
  tokensIn: number;
  tokensOut: number;
  tokensCached: number;
  cacheRate: number | null;
  costUsd: number | null;
  costKnownCalls: number;
  avgTtftSec: number | null;
  maxTtftSec: number | null;
  avgTpotMs: number | null;
  maxTpotMs: number | null;
  failovers: number;
  errors: number;
  rounds: number;
  models: string[];
  roles: Record<string, number>;
}

function splitCsvLine(line: string): string[] {
  const out: string[] = [];
  let cur = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (inQuotes) {
      if (c === '"') {
        if (line[i + 1] === '"') {
          cur += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        cur += c;
      }
    } else if (c === '"') {
      inQuotes = true;
    } else if (c === ',') {
      out.push(cur);
      cur = '';
    } else {
      cur += c;
    }
  }
  out.push(cur);
  return out;
}

function parseDetail(detail: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const part of detail.split(';')) {
    const idx = part.indexOf('=');
    if (idx <= 0) continue;
    out[part.slice(0, idx).trim()] = part.slice(idx + 1).trim();
  }
  return out;
}

function num(v: string | undefined): number | null {
  if (v === undefined) return null;
  const t = v.trim();
  if (!t) return null;
  const n = Number(t);
  return Number.isFinite(n) ? n : null;
}

function int(v: string | undefined): number {
  const n = num(v);
  return n === null ? 0 : Math.round(n);
}

export function parseTimeTrace(text: string): { rows: TraceRow[]; summary: TimeTraceSummary } {
  const lines = text.split(/\r?\n/).filter((l) => l.trim());
  const rows: TraceRow[] = [];
  if (lines.length < 2) return { rows, summary: emptyTraceSummary() };

  const headers = splitCsvLine(lines[0]).map((h) => h.trim());
  const col = new Map(headers.map((h, i) => [h, i]));
  const cell = (fields: string[], name: string): string => {
    const i = col.get(name);
    return i === undefined ? '' : (fields[i] ?? '');
  };

  for (const line of lines.slice(1)) {
    const fields = splitCsvLine(line);
    const detail = parseDetail(cell(fields, 'detail'));
    rows.push({
      timestamp: cell(fields, 'timestamp'),
      nodeId: cell(fields, 'node_id'),
      category: cell(fields, 'category'),
      operation: cell(fields, 'operation'),
      durationHrs: num(cell(fields, 'duration_hrs')) ?? 0,
      tokensInput: int(cell(fields, 'tokens_input')),
      tokensOutput: int(cell(fields, 'tokens_output')),
      tokensCached: int(cell(fields, 'tokens_cached')),
      cacheRate: num(cell(fields, 'token_cached_rate')),
      costUsd: num(cell(fields, 'llm_cost_usd')),
      ttftSec: num(cell(fields, 'ttft_sec')),
      tpotMs: num(cell(fields, 'tpot_ms')),
      failovers: int(cell(fields, 'failover_count')),
      status: cell(fields, 'status'),
      model: detail['model'] || '',
      llmRole: detail['llm_role'] || '',
      round: num(detail['round']),
      tool: detail['tool'] || '',
    });
  }

  const llmRows = rows.filter((r) => r.category === 'llm_api' || r.tokensInput > 0);
  return { rows, summary: summarizeTraceRows(llmRows) };
}

// Aggregate trace rows into monitor summary values. Accepts rows from one or
// several trace files (heavy mode writes one per worker).
export function summarizeTraceRows(rows: TraceRow[]): TimeTraceSummary {
  const summary = emptyTraceSummary();
  summary.calls = rows.length;
  let ttftSum = 0;
  let ttftCount = 0;
  let tpotSum = 0;
  let tpotCount = 0;
  const modelSet = new Set<string>();
  const rounds = new Set<number>();
  for (const r of rows) {
    summary.tokensIn += r.tokensInput;
    summary.tokensOut += r.tokensOutput;
    summary.tokensCached += r.tokensCached;
    if (r.costUsd !== null) {
      summary.costUsd = (summary.costUsd ?? 0) + r.costUsd;
      summary.costKnownCalls += 1;
    }
    if (r.ttftSec !== null) {
      ttftSum += r.ttftSec;
      ttftCount += 1;
      summary.maxTtftSec = Math.max(summary.maxTtftSec ?? 0, r.ttftSec);
    }
    if (r.tpotMs !== null) {
      tpotSum += r.tpotMs;
      tpotCount += 1;
      summary.maxTpotMs = Math.max(summary.maxTpotMs ?? 0, r.tpotMs);
    }
    summary.failovers += r.failovers;
    if (r.status && r.status !== 'ok') summary.errors += 1;
    if (r.model) modelSet.add(r.model);
    if (r.llmRole) summary.roles[r.llmRole] = (summary.roles[r.llmRole] || 0) + 1;
    if (r.round !== null) rounds.add(r.round);
  }
  summary.models = Array.from(modelSet);
  summary.rounds = rounds.size;
  summary.cacheRate = summary.tokensIn > 0 ? summary.tokensCached / summary.tokensIn : null;
  summary.avgTtftSec = ttftCount > 0 ? ttftSum / ttftCount : null;
  summary.avgTpotMs = tpotCount > 0 ? tpotSum / tpotCount : null;
  return summary;
}

function emptyTraceSummary(): TimeTraceSummary {
  return {
    calls: 0,
    tokensIn: 0,
    tokensOut: 0,
    tokensCached: 0,
    cacheRate: null,
    costUsd: null,
    costKnownCalls: 0,
    avgTtftSec: null,
    maxTtftSec: null,
    avgTpotMs: null,
    maxTpotMs: null,
    failovers: 0,
    errors: 0,
    rounds: 0,
    models: [],
    roles: {},
  };
}

// ── agent_provider_calls.jsonl ─────────────────────────────────────────────

export interface ProviderCall {
  timestamp: number;
  callSeq: number;
  model: string;
  llmRole: string;
  attempt: number;
  elapsedSec: number;
  ttftSec: number | null;
  tpotSec: number | null;
  tokensInput: number;
  tokensOutput: number;
  tokensCached: number;
  cacheRate: number | null;
  finishReason: string;
  callKind: string;
  turnKind: string;
  providerMessageCount: number | null;
  compactionGeneration: number | null;
}

export interface ProviderCallsSummary {
  calls: number;
  retries: number;
  cacheRate: number | null;
  avgTtftSec: number | null;
  maxTtftSec: number | null;
  avgTpotSec: number | null;
  maxTpotSec: number | null;
  finishReasons: Record<string, number>;
  tokensIn: number;
  tokensOut: number;
  lastContext: {
    model: string;
    messageCount: number | null;
    compactionGeneration: number | null;
    tokensInput: number;
  } | null;
}

function tailLines(text: string, maxLines: number): string[] {
  if (!text) return [];
  // Bound the work on pathologically large files: keep the newest ~4MB.
  const MAX_CHARS = 4 * 1024 * 1024;
  const slice = text.length > MAX_CHARS ? text.slice(text.length - MAX_CHARS) : text;
  const lines = slice.split('\n');
  if (lines.length <= maxLines) return lines;
  return lines.slice(lines.length - maxLines);
}

export function parseProviderCalls(
  text: string,
  maxLines = 400,
): { calls: ProviderCall[]; summary: ProviderCallsSummary } {
  const calls: ProviderCall[] = [];
  for (const line of tailLines(text, maxLines)) {
    const t = line.trim();
    if (!t.startsWith('{')) continue;
    try {
      const o = JSON.parse(t) as Record<string, unknown>;
      calls.push({
        timestamp: Number(o.timestamp) || 0,
        callSeq: Number(o.call_seq) || 0,
        model: String(o.model || ''),
        llmRole: o.llm_role == null ? '' : String(o.llm_role),
        attempt: Number(o.attempt) || 1,
        elapsedSec: Number(o.elapsed_sec) || 0,
        ttftSec: o.ttft_sec == null ? null : Number(o.ttft_sec),
        tpotSec: o.tpot_sec == null ? null : Number(o.tpot_sec),
        tokensInput: Number(o.tokens_input) || 0,
        tokensOutput: Number(o.tokens_output) || 0,
        tokensCached: Number(o.tokens_cached) || 0,
        cacheRate: o.cache_rate == null ? null : Number(o.cache_rate),
        finishReason: String(o.finish_reason || ''),
        callKind: String(o.call_kind || ''),
        turnKind: String(o.turn_kind || ''),
        providerMessageCount:
          o.provider_message_count == null ? null : Number(o.provider_message_count),
        compactionGeneration:
          o.compaction_generation == null ? null : Number(o.compaction_generation),
      });
    } catch {
      /* skip malformed line */
    }
  }

  const summary: ProviderCallsSummary = {
    calls: calls.length,
    retries: 0,
    cacheRate: null,
    avgTtftSec: null,
    maxTtftSec: null,
    avgTpotSec: null,
    maxTpotSec: null,
    finishReasons: {},
    tokensIn: 0,
    tokensOut: 0,
    lastContext: null,
  };
  let ttftSum = 0;
  let ttftCount = 0;
  let tpotSum = 0;
  let tpotCount = 0;
  for (const c of calls) {
    if (c.attempt > 1) summary.retries += 1;
    summary.tokensIn += c.tokensInput;
    summary.tokensOut += c.tokensOutput;
    if (c.ttftSec !== null && Number.isFinite(c.ttftSec)) {
      ttftSum += c.ttftSec;
      ttftCount += 1;
      summary.maxTtftSec = Math.max(summary.maxTtftSec ?? 0, c.ttftSec);
    }
    if (c.tpotSec !== null && Number.isFinite(c.tpotSec)) {
      tpotSum += c.tpotSec;
      tpotCount += 1;
      summary.maxTpotSec = Math.max(summary.maxTpotSec ?? 0, c.tpotSec);
    }
    if (c.finishReason) {
      summary.finishReasons[c.finishReason] = (summary.finishReasons[c.finishReason] || 0) + 1;
    }
  }
  summary.avgTtftSec = ttftCount > 0 ? ttftSum / ttftCount : null;
  summary.avgTpotSec = tpotCount > 0 ? tpotSum / tpotCount : null;
  const cachedTotal = calls.reduce((n, c) => n + c.tokensCached, 0);
  summary.cacheRate = summary.tokensIn > 0 ? cachedTotal / summary.tokensIn : null;
  const last = calls[calls.length - 1];
  if (last) {
    summary.lastContext = {
      model: last.model,
      messageCount: last.providerMessageCount,
      compactionGeneration: last.compactionGeneration,
      tokensInput: last.tokensInput,
    };
  }
  return { calls, summary };
}

// ── agent_runtime_events.jsonl ─────────────────────────────────────────────

export interface RuntimeEventFailure {
  name: string;
  preview: string;
  at: string;
}

export interface RuntimeEventsSummary {
  counts: Record<string, number>;
  toolStarted: number;
  toolCompleted: number;
  toolFailed: number;
  llmRequested: number;
  llmCompleted: number;
  turnsStarted: number;
  turnsCompleted: number;
  memoryAppends: number;
  messageDeltas: number;
  tokensIn: number;
  tokensOut: number;
  tokensCached: number;
  toolFailures: RuntimeEventFailure[];
  lastEventAt: string;
  lastEventType: string;
  startedAt: string | null;
  completedAt: string | null;
}

export function parseRuntimeEvents(text: string, maxLines = 2000): RuntimeEventsSummary {
  const summary: RuntimeEventsSummary = {
    counts: {},
    toolStarted: 0,
    toolCompleted: 0,
    toolFailed: 0,
    llmRequested: 0,
    llmCompleted: 0,
    turnsStarted: 0,
    turnsCompleted: 0,
    memoryAppends: 0,
    messageDeltas: 0,
    tokensIn: 0,
    tokensOut: 0,
    tokensCached: 0,
    toolFailures: [],
    lastEventAt: '',
    lastEventType: '',
    startedAt: null,
    completedAt: null,
  };
  for (const line of tailLines(text, maxLines)) {
    const t = line.trim();
    if (!t.startsWith('{')) continue;
    try {
      const o = JSON.parse(t) as Record<string, unknown>;
      const type = String(o.type || '');
      if (!type) continue;
      const at = String(o.timestamp || '');
      summary.counts[type] = (summary.counts[type] || 0) + 1;
      summary.lastEventAt = at;
      summary.lastEventType = type;
      switch (type) {
        case 'session.started':
          summary.startedAt = at;
          break;
        case 'session.completed':
          summary.completedAt = at;
          break;
        case 'tool.started':
          summary.toolStarted += 1;
          break;
        case 'tool.completed':
          summary.toolCompleted += 1;
          break;
        case 'tool.failed': {
          summary.toolFailed += 1;
          const payload = (o.payload || {}) as Record<string, unknown>;
          if (summary.toolFailures.length < 8) {
            summary.toolFailures.push({
              name: String(payload.name || 'tool'),
              preview: String(payload.preview || '').slice(0, 180),
              at,
            });
          }
          break;
        }
        case 'llm.requested':
          summary.llmRequested += 1;
          break;
        case 'llm.completed': {
          summary.llmCompleted += 1;
          const payload = (o.payload || {}) as Record<string, unknown>;
          summary.tokensIn += Number(payload.input_tokens) || 0;
          summary.tokensOut += Number(payload.output_tokens) || 0;
          summary.tokensCached += Number(payload.cached_input_tokens) || 0;
          break;
        }
        case 'agent.turn.started':
          summary.turnsStarted += 1;
          break;
        case 'agent.turn.completed':
          summary.turnsCompleted += 1;
          break;
        case 'memory.appended':
          summary.memoryAppends += 1;
          break;
        case 'message.delta':
          summary.messageDeltas += 1;
          break;
        default:
          break;
      }
    } catch {
      /* skip malformed line */
    }
  }
  return summary;
}

// ── tool_outputs/index.txt ─────────────────────────────────────────────────

export interface ToolIndexSummary {
  calls: number;
  rawChars: number;
  compressedChars: number;
  reductionPct: number | null;
  reducers: Record<string, number>;
}

export function parseToolIndex(text: string): ToolIndexSummary {
  const summary: ToolIndexSummary = {
    calls: 0,
    rawChars: 0,
    compressedChars: 0,
    reductionPct: null,
    reducers: {},
  };
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim()) continue;
    summary.calls += 1;
    const get = (key: string): string => {
      const m = line.match(new RegExp(`(?:^|\\s)${key}=([^\\s]+)`));
      return m ? m[1] : '';
    };
    summary.rawChars += Number(get('raw_chars')) || 0;
    summary.compressedChars += Number(get('compressed_chars')) || 0;
    const reducer = get('reducer') || 'none';
    summary.reducers[reducer] = (summary.reducers[reducer] || 0) + 1;
  }
  if (summary.rawChars > 0) {
    summary.reductionPct = Math.max(
      0,
      Math.min(100, ((summary.rawChars - summary.compressedChars) / summary.rawChars) * 100),
    );
  }
  return summary;
}

// ── productivity.json ──────────────────────────────────────────────────────

export interface ProductivitySummary {
  writeCount: number;
  editCount: number;
  writeSuccessCount: number;
  editSuccessCount: number;
  hadMetric: boolean;
}

export function parseProductivity(text: string): ProductivitySummary | null {
  try {
    const o = JSON.parse(text) as Record<string, unknown>;
    return {
      writeCount: Number(o.write_count) || 0,
      editCount: Number(o.edit_count) || 0,
      writeSuccessCount: Number(o.write_success_count) || 0,
      editSuccessCount: Number(o.edit_success_count) || 0,
      hadMetric: o.had_metric === true,
    };
  } catch {
    return null;
  }
}

// ── lhr_estra_events.jsonl (heavy mode context pressure) ──────────────────

export interface ContextEventsSummary {
  checks: number;
  triggers: number;
  lastTriggerAt: string;
  lastOmitted: number | null;
  lastGeneration: string;
}

export function parseContextEvents(text: string, maxLines = 2000): ContextEventsSummary {
  const summary: ContextEventsSummary = {
    checks: 0,
    triggers: 0,
    lastTriggerAt: '',
    lastOmitted: null,
    lastGeneration: '',
  };
  for (const line of tailLines(text, maxLines)) {
    const t = line.trim();
    if (!t.startsWith('{')) continue;
    try {
      const o = JSON.parse(t) as Record<string, unknown>;
      const event = String(o.event || '');
      if (event === 'context_limit_estra_check') {
        summary.checks += 1;
        const omitted = Number(o.omitted_before);
        if (Number.isFinite(omitted)) summary.lastOmitted = omitted;
      } else if (event === 'cache_hygiene_compact_triggered') {
        summary.triggers += 1;
      } else {
        continue;
      }
      const gen = String(o.context_generation || '');
      if (gen) summary.lastGeneration = gen;
      summary.lastTriggerAt = String(o.timestamp || '') || summary.lastTriggerAt;
    } catch {
      /* skip malformed line */
    }
  }
  return summary;
}
