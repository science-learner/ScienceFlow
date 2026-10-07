// Parses the ScienceFlow agent transcript (installed CLI, interaction.log +
// RAW.log mixed) into per-run structured models.
//
// The gateway streams two sources over SSE, distinguished only by content:
//
// 1. interaction.log — the CLI's full interaction record. The gateway strips
//    ANSI escapes and the loguru prefix from LIVE lines, but the backfill
//    snapshot reads the file from disk, so lines may still carry
//    `2026-09-28 05:39:59 | INFO     | ` prefixes and color escapes. Line
//    shapes (after normalization):
//
//      [draft|?] [user] 你好
//      [draft|?] [repl-run 1] ====== Agent Iteration 1/500 =======
//      [draft|?] [tool-calls-count] round=1 n=1 first=bash
//      [draft|?] [thought] List workspace contents
//      [draft|?] [tool-call] bash {"bash_kind": "inspect", "command": "ls"}
//      [draft|?] [tool-result] bash exit=ok [exit=0, 0.0s]
//      total 0                       ← untagged tool output lines
//      ...[truncated: 2 more lines, 107 chars omitted; ...]
//      [draft|?] [assistant] Final markdown answer
//      **Summary:**                  ← untagged answer continuation
//
// 2. RAW.log — stdout/stderr mirror. Carries the per-invocation header,
//    REPL banner, and the manual-stop marker:
//
//      === [2026-09-28T05:39:56Z] task=task-123 mode=lite query="你好" ===
//      === cmd: python -m scienceflow.cli repl ... ===
//      > 手动终止输出 · 2026-09-28 05:40:50       ← UTC, like all CLI stamps
//
// Both live and backfilled content flow through the same normalizer below.
// The CLI writes every timestamp in UTC without a zone suffix; see toUtcIso.

export type AgentStepKind = 'thought' | 'tool' | 'iteration' | 'notice';

export interface AgentToolStep {
  kind: 'tool';
  tool: string;
  args: Record<string, unknown> | null;
  argsRaw: string;
  // [thought] text seen immediately before this call (same round).
  thought: string;
  // Result status: null while the call is still running.
  ok: boolean | null;
  resultDetail: string;
  resultAt: string;
  // Tool stdout/stderr lines that followed the result (may contain
  // "...[truncated: ...]" markers emitted by the CLI).
  output: string[];
  round: number;
  at: string;
}

export interface AgentThoughtStep {
  kind: 'thought';
  text: string;
  at: string;
}

export interface AgentIterationStep {
  kind: 'iteration';
  run: number;
  current: number;
  max: number;
  at: string;
}

export interface AgentNoticeStep {
  kind: 'notice';
  text: string;
  at: string;
}

export type ParsedAgentStep =
  | AgentToolStep
  | AgentThoughtStep
  | AgentIterationStep
  | AgentNoticeStep;

// One agent invocation ("run"): one [user] query plus everything the agent
// did until the next query / RAW.log invocation header.
export interface ParsedAgentRun {
  index: number;
  // Gateway task id from the RAW.log invocation header (may be empty when
  // only interaction.log content is present).
  taskId: string;
  mode: string;
  query: string;
  // True once a [user] line confirmed the query (RAW-header-only runs may
  // have a query but never started).
  queryConfirmed: boolean;
  startedAt: string;
  steps: ParsedAgentStep[];
  answer: string;
  answerAt: string;
  stopMarker: string;
  iteration: { current: number; max: number } | null;
  round: number;
}

export interface ParsedAgentLog {
  format: 'v2';
  runs: ParsedAgentRun[];
  // Flat tool calls across all runs (for the lineage view).
  toolCalls: ToolCall[];
  raw: string;
}

export interface ToolCall {
  index: number;
  tool: string;
  args: Record<string, unknown>;
  thought: string;
  runIndex: number;
}

export interface ParsedChatLogMessage {
  message_id: string;
  role: 'user' | 'assistant' | 'platform';
  content: string;
  created_at: string;
  // Index into ParsedAgentLog.runs for run-anchored rendering.
  runIndex: number;
}

// ── Line normalization ─────────────────────────────────────────────────────

const ANSI_RE = /\x1b\[[0-9;]*[A-Za-z]/g;
// `2026-09-28 05:39:59 | INFO     | ` — loguru's text prefix (backfill only).
const LOGURU_RE = /^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| [A-Z]+\s*\| /;
// Channel tag like `[draft|?]` emitted by the CLI transcript writer.
const CHANNEL_TAG_RE = /^\[[A-Za-z_][\w-]*\|[^\]]*\]\s*/;

// `=== [RFC3339] task=task-123 mode=lite query="hi there" ===`
const RAW_HEADER_RE =
  /^===\s*\[([^\]]+)\]\s*task=(\S+)(?:\s+mode=(\S+))?(?:\s+query=(?:"((?:\\.|[^"])*)"|(\S+)))?\s*===/;
const ITERATION_RE = /^\[repl-run (\d+)\]\s*=+\s*Agent Iteration (\d+)\/(\d+)\s*=+$/;
const TOOL_CALLS_COUNT_RE = /^\[tool-calls-count\]\s*round=(\d+)\s+n=(\d+)\s+first=(\S+)/;
const TOOL_CALL_RE = /^\[tool-call\]\s+(\S+)\s*([\s\S]*)$/;
const TOOL_RESULT_RE = /^\[tool-result\]\s+(\S+)\s+exit=(ok|err)\s*([\s\S]*)$/;
const DURATION_RE = /\[?(\d+(?:\.\d+)?)s\]?/;
const STOP_MARKER_RE = /^>\s*手动终止/;
const STOP_TIME_RE = /(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})/;
const OUTPUT_TRUNC_RE = /^\.\.\.\[truncated:/;

// Tags that never carry user-visible value.
const NOISE_TAGS = new Set(['tool-calls', 'embedded-full-run', 'repl-auto']);

// The CLI logs every timestamp in UTC but without a zone marker
// (`2026-09-28 05:39:59`), and JS parses that shape as *local* time — a
// 8-hour skew on CST machines that scrambles message ordering once history
// is re-sorted. Normalize to RFC3339 UTC here; RAW.log headers already
// carry an explicit `Z`.
function toUtcIso(time: string): string {
  if (!time) return '';
  if (time.includes('T')) return time;
  return time.replace(' ', 'T') + 'Z';
}

function normalizeLine(line: string): { text: string; time: string } {
  let text = line.replace(ANSI_RE, '');
  let time = '';
  const m = text.match(LOGURU_RE);
  if (m) {
    time = toUtcIso(m[1]);
    text = text.slice(m[0].length);
  }
  text = text.replace(CHANNEL_TAG_RE, '');
  return { text, time };
}

function tryParseJSON(text: string): Record<string, unknown> | null {
  const trimmed = text.trim();
  if (!trimmed.startsWith('{') || !trimmed.endsWith('}')) return null;
  try {
    const parsed = JSON.parse(trimmed) as unknown;
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : null;
  } catch {
    return null;
  }
}

function unquote(s: string): string {
  if (
    (s.startsWith('"') && s.endsWith('"')) ||
    (s.startsWith("'") && s.endsWith("'"))
  ) {
    return s.slice(1, -1).replace(/\\([\\"'])/g, '$1');
  }
  return s;
}

// ── Parser ─────────────────────────────────────────────────────────────────

export function parseAgentLog(raw: string): ParsedAgentLog {
  const runs: ParsedAgentRun[] = [];
  const toolCalls: ToolCall[] = [];
  let globalToolIndex = 0;

  const lastRun = (): ParsedAgentRun | null =>
    runs.length > 0 ? runs[runs.length - 1] : null;

  const newRun = (time: string): ParsedAgentRun => {
    const run: ParsedAgentRun = {
      index: runs.length,
      taskId: '',
      mode: '',
      query: '',
      queryConfirmed: false,
      startedAt: time,
      steps: [],
      answer: '',
      answerAt: '',
      stopMarker: '',
      iteration: null,
      round: 0,
    };
    runs.push(run);
    return run;
  };

  let pendingThought = '';
  let pendingThoughtAt = '';
  // Most recent tool step, receives untagged output lines.
  let lastTool: AgentToolStep | null = null;
  // FIFO of tool steps awaiting a [tool-result] (multi-call rounds).
  let pendingCalls: AgentToolStep[] = [];
  // True after [assistant] — untagged lines continue the answer.
  let inAnswer = false;
  // True while untagged lines are streaming into lastTool.output. Tool
  // outputs often `cat`/`tail` the interaction log itself, re-injecting
  // copies of earlier tagged lines ([user] ..., [tool-call] ...); those
  // echoes must never be dispatched as real events. A tagged line is an
  // echo when it arrives mid-output and its text duplicates an already
  // dispatched tag — or when it is a [user] line, which can never
  // legitimately interrupt tool output.
  let outputActive = false;
  const seenTagged = new Set<string>();

  const rememberTag = (line: string) => {
    seenTagged.add(line);
    if (seenTagged.size > 1000) seenTagged.clear();
    outputActive = false;
  };

  const flushPendingThought = (run: ParsedAgentRun) => {
    if (!pendingThought.trim()) return;
    run.steps.push({ kind: 'thought', text: pendingThought.trim(), at: pendingThoughtAt });
    pendingThought = '';
    pendingThoughtAt = '';
  };

  const lines = raw.split('\n');
  for (const rawLine of lines) {
    const { text: line, time } = normalizeLine(rawLine);

    // Blank lines: meaningful only inside the answer (markdown paragraphs).
    if (!line.trim()) {
      if (inAnswer) {
        const run = lastRun();
        if (run) run.answer += '\n';
      }
      continue;
    }

    // ── RAW.log lines ──
    if (line.startsWith('===')) {
      const hm = line.match(RAW_HEADER_RE);
      if (!hm) continue; // `=== cmd: ... ===` and friends
      const [, ts, taskId, mode, quotedQuery, bareQuery] = hm;
      const headerQuery = quotedQuery !== undefined ? unquote(quotedQuery) : bareQuery || '';
      flushPendingThoughtAtBoundary();
      rememberTag(line);
      // A new invocation header resets any mid-run streaming state.
      lastTool = null;
      pendingCalls = [];
      inAnswer = false;
      // Attach to an earlier run that matches the query but has no task id
      // (backfill can deliver interaction.log lines before RAW.log headers).
      const match = [...runs]
        .reverse()
        .find((r) => !r.taskId && r.query === headerQuery);
      if (match) {
        match.taskId = taskId;
        match.mode = mode || match.mode;
        if (!match.startedAt) match.startedAt = ts;
        continue;
      }
      const run = newRun(ts);
      run.taskId = taskId;
      run.mode = mode || '';
      run.query = headerQuery;
      continue;
    }

    if (STOP_MARKER_RE.test(line)) {
      if (outputActive && lastTool) {
        // Echo of an earlier stop marker inside tool output.
        lastTool.output.push(line);
        continue;
      }
      const run = lastRun();
      if (run) run.stopMarker = line.replace(/^>\s*/, '').trim();
      inAnswer = false;
      continue;
    }

    // ── Tagged interaction lines ──
    if (line.startsWith('[')) {
      // Noise tags never carry value (and their echoes appear inside tool
      // output when the transcript is cat-ed), so drop them unconditionally.
      const noiseMatch = line.match(/^\[([\w-]+)\]/);
      if (noiseMatch && NOISE_TAGS.has(noiseMatch[1])) continue;

      // Echo guard: tagged-looking lines inside tool streaming are copies of
      // earlier log content (cat/tail of the transcript), never real events.
      const tagEcho =
        lastTool !== null && !inAnswer && outputActive && seenTagged.has(line);

      // [tool-result] lines routinely repeat verbatim (identical exit status
      // and duration text), so a duplicate is only an echo when no call is
      // awaiting a result; otherwise it is the (possibly repeated) real
      // result and must complete the pending call.
      let m = line.match(TOOL_RESULT_RE);
      if (m) {
        const hasPending = pendingCalls.length > 0;
        const resultEcho =
          !hasPending && lastTool !== null && seenTagged.has(line);
        if (resultEcho) {
          lastTool!.output.push(line);
          outputActive = true;
          continue;
        }
        const run = lastRun() || newRun(time);
        const [, tool, exit, detail] = m;
        // Pair with the oldest pending call of the same tool; if none (e.g.
        // the call line was truncated away), synthesize a step.
        const idx = pendingCalls.findIndex((c) => c.tool === tool);
        let step: AgentToolStep | undefined;
        if (idx >= 0) {
          step = pendingCalls.splice(idx, 1)[0];
        } else {
          step = {
            kind: 'tool',
            tool,
            args: null,
            argsRaw: '',
            thought: '',
            ok: exit === 'ok',
            resultDetail: detail.trim(),
            resultAt: time,
            output: [],
            round: run.round,
            at: time,
          };
          run.steps.push(step);
        }
        step.ok = exit === 'ok';
        step.resultDetail = detail.trim();
        step.resultAt = time;
        lastTool = step;
        inAnswer = false;
        rememberTag(line);
        continue;
      }

      // A real [user] line can only arrive at stream start, after a RAW
      // invocation header, or after the previous run's [assistant] — never
      // while tool output is mid-flight. The only exception is a call still
      // awaiting its result (killed mid-flight), after which a genuine new
      // user line may follow directly.
      if (lastTool && !inAnswer) {
        const userEcho = line.startsWith('[user]') && lastTool.ok !== null;
        if (userEcho || tagEcho) {
          lastTool.output.push(line);
          outputActive = true;
          continue;
        }
      }

      m = line.match(ITERATION_RE);
      if (m) {
        const run = lastRun() || newRun(time);
        const it = { run: Number(m[1]), current: Number(m[2]), max: Number(m[3]) };
        run.iteration = { current: it.current, max: it.max };
        run.steps.push({ kind: 'iteration', ...it, at: time });
        rememberTag(line);
        continue;
      }

      m = line.match(TOOL_CALLS_COUNT_RE);
      if (m) {
        const run = lastRun() || newRun(time);
        run.round = Number(m[1]);
        rememberTag(line);
        continue;
      }

      m = line.match(TOOL_CALL_RE);
      if (m) {
        const run = lastRun() || newRun(time);
        const tool = m[1];
        const argsRaw = m[2].trim();
        const step: AgentToolStep = {
          kind: 'tool',
          tool,
          args: tryParseJSON(argsRaw),
          argsRaw,
          thought: pendingThought,
          ok: null,
          resultDetail: '',
          resultAt: '',
          output: [],
          round: run.round,
          at: time,
        };
        pendingThought = '';
        pendingThoughtAt = '';
        run.steps.push(step);
        lastTool = step;
        pendingCalls.push(step);
        inAnswer = false;
        rememberTag(line);
        continue;
      }

      m = line.match(/^\[thought\]\s*([\s\S]*)$/);
      if (m) {
        const run = lastRun() || newRun(time);
        flushPendingThought(run);
        pendingThought = m[1].trim();
        pendingThoughtAt = time;
        rememberTag(line);
        continue;
      }

      m = line.match(/^\[assistant\]\s*([\s\S]*)$/);
      if (m) {
        const run = lastRun() || newRun(time);
        flushPendingThought(run);
        if (run.answer) run.answer += '\n';
        run.answer += m[1];
        run.answerAt = time;
        inAnswer = true;
        lastTool = null;
        pendingCalls = [];
        rememberTag(line);
        continue;
      }

      m = line.match(/^\[user\]\s*([\s\S]*)$/);
      if (m) {
        flushPendingThoughtAtBoundary();
        const query = m[1].trim();
        const run = lastRun();
        if (run && !run.queryConfirmed && run.query === query) {
          // RAW.log header already created this run (live arrival order).
          run.queryConfirmed = true;
        } else {
          const next = newRun(time);
          next.query = query;
          next.queryConfirmed = true;
        }
        inAnswer = false;
        lastTool = null;
        pendingCalls = [];
        rememberTag(line);
        continue;
      }

      const tagMatch = line.match(/^\[([\w-]+)\]\s*([\s\S]*)$/);
      if (tagMatch) {
        if (NOISE_TAGS.has(tagMatch[1])) {
          rememberTag(line);
          continue;
        }
        const run = lastRun() || newRun(time);
        run.steps.push({ kind: 'notice', text: tagMatch[2].trim(), at: time });
        rememberTag(line);
        continue;
      }
    }

    // ── Untagged continuation lines ──
    if (line === '--- Agent Output ---') continue;
    if (inAnswer) {
      const run = lastRun();
      if (run) run.answer += '\n' + line;
      continue;
    }
    if (lastTool) {
      // Skip leading blank padding in tool output.
      if (!line.trim() && lastTool.output.length === 0) continue;
      lastTool.output.push(line);
      outputActive = true;
      continue;
    }
    // Anything else (REPL banner, stray stdout) has no display value.
  }

  function flushPendingThoughtAtBoundary() {
    const run = lastRun();
    if (run) flushPendingThought(run);
  }

  // Collect flat tool calls.
  for (const run of runs) {
    for (const step of run.steps) {
      if (step.kind === 'tool') {
        toolCalls.push({
          index: globalToolIndex++,
          tool: step.tool,
          args: step.args || {},
          thought: step.thought,
          runIndex: run.index,
        });
      }
    }
  }

  return { format: 'v2', runs, toolCalls, raw };
}

// ── Chat message extraction ────────────────────────────────────────────────

// Rebuilds the chat transcript from parsed runs: user query, assistant
// answer and stop markers, in log order.
export function parseHistoricalChatMessages(
  runs: ParsedAgentRun[],
): ParsedChatLogMessage[] {
  const messages: ParsedChatLogMessage[] = [];
  for (const run of runs) {
    if (!run.queryConfirmed && !run.steps.length && !run.answer && !run.stopMarker) {
      continue;
    }
    if (run.query) {
      messages.push({
        message_id: `history-user-${run.index}`,
        role: 'user',
        content: run.query,
        created_at: run.startedAt || '',
        runIndex: run.index,
      });
    }
    if (run.answer.trim()) {
      messages.push({
        message_id: `history-assistant-${run.index}`,
        role: 'assistant',
        content: run.answer.trim(),
        created_at: run.answerAt || run.startedAt || '',
        runIndex: run.index,
      });
    }
    if (run.stopMarker) {
      const timeText = run.stopMarker.match(STOP_TIME_RE)?.[1] || '';
      const stopIso = toUtcIso(timeText);
      // The stop marker only carries second precision; when a run is killed
      // within the second it started, clamp to startedAt so the chip sorts
      // after its question instead of above it.
      let created = stopIso;
      const stopMs = stopIso ? Date.parse(stopIso) : NaN;
      const startMs = run.startedAt ? Date.parse(run.startedAt) : NaN;
      if (!Number.isNaN(stopMs) && !Number.isNaN(startMs) && stopMs < startMs) {
        created = run.startedAt;
      }
      messages.push({
        message_id: `history-stop-${run.index}`,
        role: 'platform',
        content: run.stopMarker,
        created_at: created,
        runIndex: run.index,
      });
    }
  }
  return messages;
}

// ── Display helpers ────────────────────────────────────────────────────────

export function isOutputTruncationMarker(line: string): boolean {
  return OUTPUT_TRUNC_RE.test(line);
}

export function formatToolArgPreview(
  tool: string,
  args: Record<string, unknown> | null,
): string {
  if (!args) return '';
  if (tool === 'bash' && typeof args['command'] === 'string') {
    const kind = typeof args['bash_kind'] === 'string' ? String(args['bash_kind']) : '';
    const prefix = kind && kind !== 'other' ? `[${kind}] ` : '';
    return `${prefix}$ ${args['command']}`;
  }
  if (tool === 'read' || tool === 'write' || tool === 'edit') {
    const path = typeof args['path'] === 'string' ? args['path'] : '';
    if (!path) return JSON.stringify(args).slice(0, 300);
    const offset = Number(args['offset'] || 0);
    const limit = Number(args['limit'] || 0);
    const range = offset || limit ? ` (lines ${offset || 1}-${offset + limit || '…'})` : '';
    return `${path}${range}`;
  }
  if ((tool === 'ls' || tool === 'glob') && args['path']) {
    return String(args['path']);
  }
  if (tool === 'grep' && args['pattern']) {
    return `grep "${args['pattern']}"`;
  }
  const json = JSON.stringify(args, null, 2);
  return json.length > 300 ? json.slice(0, 300) + '…' : json;
}

// Human duration between two parseable timestamps ('' when unknown).
export function runDuration(run: ParsedAgentRun): string {
  const start = run.startedAt ? Date.parse(run.startedAt) : NaN;
  const end = run.answerAt ? Date.parse(run.answerAt) : NaN;
  if (Number.isNaN(start) || Number.isNaN(end) || end <= start) return '';
  const sec = Math.round((end - start) / 1000);
  if (sec < 60) return `${sec}s`;
  const min = Math.floor(sec / 60);
  return `${min}m ${sec % 60}s`;
}
