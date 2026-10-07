import { useEffect, useRef, useState } from 'react';
import { useGatewayStore } from '../store/useGatewayStore';
import { workspaceReadText } from '../api/gateway';
import type { FileNode } from '../types';
import {
  parseTimeTrace,
  summarizeTraceRows,
  parseProviderCalls,
  parseRuntimeEvents,
  parseToolIndex,
  parseProductivity,
  parseContextEvents,
  type TraceRow,
  type TimeTraceSummary,
  type ProviderCall,
  type ProviderCallsSummary,
  type RuntimeEventsSummary,
  type ToolIndexSummary,
  type ProductivitySummary,
  type ContextEventsSummary,
} from '../utils/monitorSources';

export interface MonitorSourcesState {
  trace: TimeTraceSummary | null;
  traceRows: TraceRow[];
  provider: ProviderCallsSummary | null;
  providerCalls: ProviderCall[];
  events: RuntimeEventsSummary | null;
  toolIndex: ToolIndexSummary | null;
  productivity: ProductivitySummary | null;
  contextEvents: ContextEventsSummary | null;
  lastUpdated: number;
}

const INITIAL: MonitorSourcesState = {
  trace: null,
  traceRows: [],
  provider: null,
  providerCalls: [],
  events: null,
  toolIndex: null,
  productivity: null,
  contextEvents: null,
  lastUpdated: 0,
};

interface SourceSpec {
  key: keyof Omit<MonitorSourcesState, 'lastUpdated' | 'traceRows' | 'providerCalls'>;
  suffix: string;
  intervalMs: number;
  multi?: boolean;
  maxFiles?: number;
}

// Workspace-relative suffixes; matched against the gateway file tree so the
// session directory layout (run/ prefix) does not have to be hardcoded.
const SPECS: SourceSpec[] = [
  { key: 'trace', suffix: 'scienceflow_time_trace.csv', intervalMs: 5000, multi: true, maxFiles: 6 },
  { key: 'provider', suffix: 'agent_provider_calls.jsonl', intervalMs: 8000 },
  { key: 'events', suffix: 'agent_runtime_events.jsonl', intervalMs: 12000 },
  { key: 'toolIndex', suffix: 'tool_outputs/index.txt', intervalMs: 15000 },
  { key: 'productivity', suffix: 'productivity.json', intervalMs: 15000 },
  { key: 'contextEvents', suffix: 'lhr_estra_events.jsonl', intervalMs: 15000 },
];

function matchNodes(tree: FileNode[], suffix: string, maxFiles: number): FileNode[] {
  const matches = tree.filter((n) => n.type === 'file' && n.path.endsWith(suffix));
  matches.sort((a, b) => b.mtime - a.mtime);
  return matches.slice(0, maxFiles);
}

// Polls the per-run telemetry files (time trace, provider calls, runtime
// events, compression ledger, productivity, context events) and keeps parsed
// summaries in state. Files are only re-downloaded when their size/mtime from
// the gateway file tree changed, so an idle session costs nothing.
export function useMonitorSources(enabled: boolean): MonitorSourcesState {
  const [state, setState] = useState<MonitorSourcesState>(INITIAL);
  const cacheRef = useRef(new Map<string, string>());
  const lastFetchRef = useRef(new Map<string, number>());
  const busyRef = useRef(false);

  useEffect(() => {
    if (!enabled) {
      cacheRef.current.clear();
      lastFetchRef.current.clear();
      setState(INITIAL);
      return;
    }
    let cancelled = false;

    const poll = async () => {
      if (cancelled || busyRef.current) return;
      const { token, sessionId, fileTree } = useGatewayStore.getState();
      if (!token || !sessionId) return;
      busyRef.current = true;
      const now = Date.now();
      const next: Partial<MonitorSourcesState> = {};
      try {
        for (const spec of SPECS) {
          const last = lastFetchRef.current.get(spec.key) || 0;
          if (now - last < spec.intervalMs) continue;
          const nodes = matchNodes(fileTree, spec.suffix, spec.multi ? spec.maxFiles || 6 : 1);
          if (nodes.length === 0) continue;
          const meta = nodes.map((n) => `${n.path}:${n.size}:${n.mtime}`).join('|');
          if (cacheRef.current.get(spec.key) === meta) {
            lastFetchRef.current.set(spec.key, now);
            continue;
          }

          if (spec.key === 'trace') {
            const rows: TraceRow[] = [];
            for (const node of nodes) {
              try {
                const text = await workspaceReadText(token, sessionId, node.path);
                rows.push(...parseTimeTrace(text).rows);
              } catch {
                /* skip unreadable trace file */
              }
            }
            rows.sort((a, b) => (a.timestamp < b.timestamp ? -1 : 1));
            next.traceRows = rows;
            next.trace = summarizeTraceRows(
              rows.filter((r) => r.category === 'llm_api' || r.tokensInput > 0),
            );
          } else {
            const node = nodes[0];
            let text = '';
            try {
              text = await workspaceReadText(token, sessionId, node.path);
            } catch {
              continue;
            }
            switch (spec.key) {
              case 'provider': {
                const parsed = parseProviderCalls(text);
                next.provider = parsed.summary;
                next.providerCalls = parsed.calls;
                break;
              }
              case 'events':
                next.events = parseRuntimeEvents(text);
                break;
              case 'toolIndex':
                next.toolIndex = parseToolIndex(text);
                break;
              case 'productivity':
                next.productivity = parseProductivity(text);
                break;
              case 'contextEvents':
                next.contextEvents = parseContextEvents(text);
                break;
              default:
                break;
            }
          }
          cacheRef.current.set(spec.key, meta);
          lastFetchRef.current.set(spec.key, now);
        }
        if (!cancelled && Object.keys(next).length > 0) {
          setState((prev) => ({ ...prev, ...next, lastUpdated: Date.now() }));
        }
      } finally {
        busyRef.current = false;
      }
    };

    void poll();
    const timer = window.setInterval(() => void poll(), 3000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [enabled]);

  return state;
}
