import { useRef, useEffect, useMemo } from 'react';
import { useGatewayStore } from '../store/useGatewayStore';
import { formatToolArgPreview } from '../utils/agentLogParser';
import clsx from 'clsx';
import { Terminal, FileText, Pencil, FileEdit, Search, FolderOpen, List, Code, Wrench } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';

const TOOL_ICONS: Record<string, LucideIcon> = {
  bash: Terminal,
  read: FileText,
  write: Pencil,
  edit: FileEdit,
  grep: Search,
  glob: FolderOpen,
  ls: List,
  python: Code,
  python3: Code,
};

export function AgentLineageView() {
  const { parsedLog, status } = useGatewayStore();
  const scrollRef = useRef<HTMLDivElement>(null);
  const stickRef = useRef(true);

  useEffect(() => {
    if (scrollRef.current && stickRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [parsedLog]);

  const toolCalls = parsedLog.toolCalls;
  const lastRun = useMemo(
    () => (parsedLog.runs.length > 0 ? parsedLog.runs[parsedLog.runs.length - 1] : null),
    [parsedLog],
  );

  const grouped = useMemo(() => groupByTool(toolCalls), [toolCalls]);

  if (toolCalls.length === 0) {
    return (
      <div className="dim" style={{ display: 'grid', placeItems: 'center', height: '100%', textAlign: 'center' }}>
        <div>
          <p style={{ margin: 0, fontSize: 14 }}>Agent Lineage</p>
          <p style={{ margin: '8px 0 0', fontSize: 12 }}>
            {status === 'idle' ? '连接日志网关后，智能体的 step 演进将在此呈现。' : '等待工具调用…'}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="lineage-view" style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0 }}>
      {/* Stats bar */}
      <div style={{ display: 'flex', gap: 16, padding: '8px 14px', borderBottom: '1px solid var(--line)', flexShrink: 0, flexWrap: 'wrap' }}>
        <Stat label="Steps" value={String(toolCalls.length)} />
        <Stat label="Tools" value={String(grouped.length)} />
        {lastRun?.taskId && <Stat label="Task" value={lastRun.taskId.replace(/^task-/, '').slice(-12)} />}
        {lastRun?.mode && <Stat label="Mode" value={lastRun.mode} />}
      </div>

      {/* Step timeline */}
      <div ref={scrollRef} onScroll={(e) => {
        const el = e.currentTarget;
        stickRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
      }} style={{ flex: 1, minHeight: 0, overflow: 'auto', padding: '14px' }}>
        <div className="lineage-timeline" style={{ position: 'relative', paddingLeft: 24 }}>
          {/* Vertical line */}
          <div style={{
            position: 'absolute',
            left: 8,
            top: 0,
            bottom: 0,
            width: 2,
            background: 'var(--line)',
          }} />

          {toolCalls.map((tc, i) => {
            const IconComp = TOOL_ICONS[tc.tool] || Wrench;
            return (
              <div key={tc.index} className="lineage-node" style={{ position: 'relative', marginBottom: 12 }}>
                {/* Dot */}
                <div style={{
                  position: 'absolute',
                  left: -22,
                  top: 4,
                  width: 18,
                  height: 18,
                  borderRadius: '50%',
                  display: 'grid',
                  placeItems: 'center',
                  fontSize: 10,
                  background: 'var(--panel)',
                  border: '2px solid var(--accent, #5cc8ff)',
                  color: 'var(--cyan, #5cc8ff)',
                  fontFamily: 'var(--mono)',
                  fontWeight: 700,
                  zIndex: 1,
                }}>
                  {i + 1}
                </div>

                <div className="lineage-node-body" style={{
                  padding: '6px 12px',
                  borderRadius: 6,
                  background: 'rgba(255,255,255,0.02)',
                  border: '1px solid var(--line)',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                    <IconComp size={13} />
                    <strong style={{ fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--text)' }}>{tc.tool}</strong>
                    <span className="dim" style={{ fontSize: 10 }}>step {i + 1}</span>
                    <span className="dim" style={{ fontSize: 10, fontFamily: 'var(--mono)' }}>R{tc.runIndex + 1}</span>
                  </div>
                  <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--soft)', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                    {formatToolArgPreview(tc.tool, tc.args)}
                  </div>
                  {tc.thought && (
                    <div style={{ marginTop: 4, fontSize: 11, color: 'var(--muted)', lineHeight: 1.4, fontStyle: 'italic' }}>
                      {tc.thought.length > 120 ? tc.thought.slice(0, 120) + '…' : tc.thought}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Tool usage summary */}
      {grouped.length > 0 && (
        <div style={{ padding: '8px 14px', borderTop: '1px solid var(--line)', flexShrink: 0 }}>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {grouped.map(({ tool, count }) => {
              const IconComp = TOOL_ICONS[tool] || Wrench;
              return (
                <span key={tool} className="pill" style={{ fontSize: 10, display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                  <IconComp size={11} /> {tool} ×{count}
                </span>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: 'green' | 'red' | 'yellow' }) {
  const color = accent === 'green' ? 'var(--green)' : accent === 'red' ? 'var(--red)' : accent === 'yellow' ? 'var(--yellow)' : 'var(--text)';
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <span className="dim" style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: 0.5 }}>{label}</span>
      <span style={{ fontSize: 13, fontFamily: 'var(--mono)', fontWeight: 700, color }}>{value}</span>
    </div>
  );
}

interface ToolGroup {
  tool: string;
  count: number;
}

function groupByTool(toolCalls: { tool: string }[]): ToolGroup[] {
  const map = new Map<string, number>();
  for (const tc of toolCalls) {
    map.set(tc.tool, (map.get(tc.tool) || 0) + 1);
  }
  return Array.from(map.entries())
    .map(([tool, count]) => ({ tool, count }))
    .sort((a, b) => b.count - a.count);
}
