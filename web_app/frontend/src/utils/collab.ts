import type {
  ScienceFlowState,
  AgentMapModel,
  AgentMapAgent,
  AgentMapWorker,
  AgentMapIssue,
  AgentMapBoard,
  AgentMapInboxItem,
  AgentMapEvent,
  ChatMessage,
  ChatToolEvent,
  WorkerSummary,
} from '../types';

interface ChatContext {
  busy: boolean;
  warmup: boolean;
  runState: string;
  queueDepth: number;
  routeMode: string;
  progressLabel: string;
  toolEvents: ChatToolEvent[];
  assistantPreview: string;
  lastEventAt: string;
  injectionState: string;
  settingsReady: boolean;
  elapsed: string;
  messages: ChatMessage[];
  agentPositionId: string;
}

export function buildAgentMapModel(state: ScienceFlowState | null, chat: ChatContext): AgentMapModel {
  const task = state?.task;
  const hasTask = Boolean(task?.task_root);
  const recentEvents = replRecentEvents(chat, hasTask);
  const issue = replIssue(chat, state, hasTask, recentEvents);
  const statusClass = replStatusClass(chat, hasTask, issue);
  const actorState = replActorState(chat, statusClass, issue);
  const thoughtLines = replThoughtLines(chat, recentEvents, issue, hasTask);
  const progress = replProgress(chat, statusClass, hasTask);

  const actor: AgentMapAgent = {
    id: 'repl-agent',
    worker_id: 'Agent',
    label: 'Agent',
    kind: 'repl',
    pointId: chat.agentPositionId || 'coordinator',
    color: '#62d884',
    statusClass,
    actorState,
    issue,
    progress,
    thought: thoughtLines.join('; '),
    thoughtLines,
    bestCurve: { values: [progress > 0 ? Math.max(8, progress - 18) : 0, progress], best: null, gain: null, hasMetric: false, lowerIsBetter: true },
    spark: replSparkline(progress),
    last_event_at: chat.lastEventAt || '',
    stage_count: recentEvents.length,
    current_stage: thoughtLines[0] || '',
    sendContext: replSendContext(chat, issue, hasTask),
    route: chat.routeMode || 'chat',
    runState: chat.runState || 'idle',
  };

  const workerSummary: WorkerSummary = {
    source: hasTask ? 'repl' : 'none',
    total: 0,
    active: statusClass === 'active' ? 1 : 0,
    completed: statusClass === 'completed' ? 1 : 0,
    failed: statusClass === 'failed' ? 1 : 0,
    idle: statusClass === 'idle' ? 1 : 0,
    event_count: recentEvents.length,
    generated_at: chat.lastEventAt || '',
    global_best_metric: null,
    global_best_stage_id: '',
    global_best_worker_id: '',
  };

  const board: AgentMapBoard = replBoard(actor, chat, hasTask);

  return {
    mode: hasTask ? 'repl' : 'empty',
    agents: [actor],
    visibleAgents: [actor],
    workers: [],
    workerSummary,
    selectedAgent: actor,
    elapsed: chat.elapsed || '--',
    stats: { total: 1, active: statusClass === 'active' ? 1 : 0, completed: statusClass === 'completed' ? 1 : 0, failed: statusClass === 'failed' ? 1 : 0 },
    board,
    inboxItems: replInboxItems(actor, chat, issue, hasTask),
  };
}

function replStatusClass(chat: ChatContext, hasTask: boolean, issue: AgentMapIssue | null): string {
  if (!hasTask) return 'idle';
  if (issue || ['failed', 'cancelled'].includes(chat.runState)) return 'failed';
  if (chat.warmup || chat.busy || ['running', 'finalizing', 'cancelling'].includes(chat.runState)) return 'active';
  if (Number(chat.queueDepth || 0) > 0) return 'active';
  if (chat.runState === 'completed') return 'completed';
  return 'idle';
}

function replActorState(chat: ChatContext, statusClass: string, issue: AgentMapIssue | null): string {
  if (issue) return 'blocked';
  if (chat.runState === 'cancelling') return 'blocked';
  if (statusClass === 'failed') return 'failed';
  if (statusClass === 'completed') return 'done';
  if (chat.warmup || Number(chat.queueDepth || 0) > 0) return 'working';
  if (chat.busy || ['running', 'finalizing'].includes(chat.runState)) {
    return hasRunningTool(chat.toolEvents) ? 'working' : 'talking';
  }
  return 'idle';
}

function hasRunningTool(events: ChatToolEvent[]): boolean {
  return events.some((e) => e.type === 'tool_started' || e.status === 'running');
}

function replProgress(chat: ChatContext, statusClass: string, hasTask: boolean): number {
  if (!hasTask) return 0;
  if (statusClass === 'failed') return 18;
  if (statusClass === 'completed') return 100;
  if (chat.warmup) return 28;
  if (Number(chat.queueDepth || 0) > 0) return 36;
  if (chat.busy || chat.runState === 'running') return hasRunningTool(chat.toolEvents) ? 72 : 58;
  if (chat.runState === 'finalizing') return 88;
  return 0;
}

function replRecentEvents(chat: ChatContext, hasTask: boolean): AgentMapEvent[] {
  const events: AgentMapEvent[] = [];
  if (!hasTask) { events.push({ type: 'setup', message: 'create a task first', status: 'idle', created_at: '' }); return events; }
  if (chat.settingsReady === false) events.push({ type: 'settings', message: 'model settings need API key', status: 'blocked', created_at: '' });
  if (chat.warmup) events.push({ type: 'warmup', message: 'agent initializing', status: 'running', created_at: '' });
  if (chat.queueDepth > 0) events.push({ type: 'queue', message: `queued ${chat.queueDepth}`, status: 'queued', created_at: '' });
  if (chat.progressLabel) events.push({ type: 'run', message: chat.progressLabel, status: chat.runState || 'running', created_at: '' });
  for (const e of (chat.toolEvents || []).slice(-8)) {
    const msg = toolEventMessage(e);
    if (msg) events.push({ type: e.type || 'tool', message: msg, status: e.status || '', created_at: e.created_at || '' });
  }
  if (chat.assistantPreview) events.push({ type: 'assistant', message: chat.assistantPreview, status: 'streaming', created_at: '' });
  if (!events.length) events.push({ type: 'ready', message: chat.routeMode === 'agent' ? 'ready for agent work' : 'ready for chat', status: 'idle', created_at: '' });
  return events.slice(-8);
}

function replIssue(chat: ChatContext, state: ScienceFlowState | null, hasTask: boolean, recentEvents: AgentMapEvent[]): AgentMapIssue | null {
  if (!hasTask) return null;
  const failedTool = [...(chat.toolEvents || [])].reverse().find((e) => {
    const text = `${e.type || ''} ${e.status || ''} ${e.error || ''} ${e.message || ''}`.toLowerCase();
    return /(fail|error|exception|timeout|cancel)/.test(text);
  });
  if (failedTool) return { kind: 'tool', severity: 'issue', title: 'Tool needs attention', body: failedTool.error || failedTool.message || toolEventMessage(failedTool) || 'A tool step failed.', worker_id: 'Agent' };
  if (chat.injectionState === 'rejected') return { kind: 'chat', severity: 'issue', title: 'Message rejected', body: 'The running agent rejected the latest injected message.', worker_id: 'Agent' };
  if (chat.runState === 'failed') return { kind: 'chat', severity: 'issue', title: 'Agent run failed', body: chat.progressLabel || 'The current Chat Agent run ended with an error.', worker_id: 'Agent' };
  if (chat.runState === 'cancelled') return { kind: 'chat', severity: 'issue', title: 'Agent run cancelled', body: 'The latest Chat Agent run was stopped.', worker_id: 'Agent' };
  if (chat.settingsReady === false) return { kind: 'settings', severity: 'issue', title: 'Configure model', body: 'Choose a model before running the agent.', worker_id: 'Agent' };
  const warning = firstWarning(state);
  if (warning) return { kind: 'state', severity: 'issue', title: 'Workspace warning', body: warning, worker_id: 'Agent' };
  const issueEvent = [...recentEvents].reverse().find((e) => /(fail|error|blocked|reject|exception|traceback)/.test(`${e.type || ''} ${e.status || ''} ${e.message || ''}`.toLowerCase()));
  if (issueEvent) return { kind: 'event', severity: 'issue', title: 'Agent needs attention', body: issueEvent.message || 'The latest Agent event needs attention.', worker_id: 'Agent' };
  return null;
}

function firstWarning(state: ScienceFlowState | null): string {
  const warnings = state?.warnings || [];
  const w = warnings.find((item) => /error|fail|outside allowed roots/i.test(String(item || '')));
  return w ? cleanText(w) : '';
}

function replThoughtLines(chat: ChatContext, recentEvents: AgentMapEvent[], issue: AgentMapIssue | null, hasTask: boolean): string[] {
  if (!hasTask) return ['create a task first'];
  if (issue) return [issue.body || issue.title || 'agent needs attention'];
  const lines: string[] = [];
  if (chat.warmup) lines.push('agent initializing');
  if (chat.routeMode === 'agent') lines.push('agent route active');
  if (chat.busy || chat.runState === 'running') lines.push(chat.progressLabel || 'working on request');
  if (chat.runState === 'finalizing') lines.push('finalizing answer');
  if (chat.queueDepth > 0) lines.push(`queue depth ${chat.queueDepth}`);
  for (const e of (recentEvents || []).slice(-4)) {
    if (e.message && !lines.includes(e.message)) lines.push(e.message);
  }
  if (!lines.length) lines.push(chat.routeMode === 'agent' ? 'ready for agent work' : 'ready for chat');
  return lines.map(cleanText).filter(Boolean).slice(-4);
}

function replSendContext(chat: ChatContext, issue: AgentMapIssue | null, hasTask: boolean): string {
  if (!hasTask) return 'Create a task first before sending to Agent.';
  if (issue) return `Context: ${issue.title || 'Agent issue'} - ${cleanText(issue.body || '')}`;
  const latest = latestToolEvent(chat.toolEvents);
  if (latest) return `Context: latest Agent step - ${toolEventMessage(latest) || toolEventShort(latest)}`;
  if (chat.progressLabel) return `Context: Agent is ${chat.progressLabel}`;
  if (chat.assistantPreview) return `Context: latest visible Agent output - ${cleanText(chat.assistantPreview)}`;
  return `Target Agent - ${chat.routeMode === 'agent' ? '@agent route' : '@chat route'}`;
}

function replInboxItems(actor: AgentMapAgent, chat: ChatContext, issue: AgentMapIssue | null, hasTask: boolean): AgentMapInboxItem[] {
  if (!hasTask) return [{ id: 'empty-create-task', kind: 'empty', worker_id: 'Agent', title: 'Create a task first', body: 'Create or select an active workspace.', meta: 'idle' }];
  if (issue) return [{ id: `issue-${issue.kind || 'agent'}`, kind: 'issue', worker_id: 'Agent', title: issue.title || 'Agent needs attention', body: issue.body || actor.thought || 'Open Send to respond.', meta: 'Agent' }];
  return [{ id: 'summary-repl-agent', kind: 'summary', worker_id: 'Agent', title: chat.busy ? 'Agent is working' : 'Agent ready', body: actor.thoughtLines?.[0] || actor.thought || 'ready', meta: chat.routeMode === 'agent' ? '@agent' : '@chat' }];
}

function replBoard(actor: AgentMapAgent, chat: ChatContext, hasTask: boolean): AgentMapBoard {
  const currentTool = latestToolEvent(chat.toolEvents);
  return {
    title: '',
    rows: (actor.thoughtLines?.length ? actor.thoughtLines : [actor.thought || 'ready']).slice(-4),
    stats: [
      ['Route', chat.routeMode === 'agent' ? '@agent' : '@chat'],
      ['Run', hasTask ? (chat.runState || 'idle') : 'no task'],
      ['Queue', String(chat.queueDepth || 0)],
      ['Tool', currentTool ? toolEventShort(currentTool) : 'none'],
      ['Mode', chat.busy ? 'live' : 'ready'],
      ['Elapsed', chat.elapsed || '--'],
    ],
  };
}

function latestToolEvent(events: ChatToolEvent[]): ChatToolEvent | null {
  return [...events].reverse().find((e) => /tool_|artifact_created|report_updated/.test(String(e?.type || ''))) || null;
}

function toolEventMessage(e: ChatToolEvent): string {
  if (!e) return '';
  if (e.error) return cleanText(e.error);
  if (e.message) return cleanText(e.message);
  if (e.summary) return cleanText(e.summary);
  if (e.type === 'artifact_created') return `artifact ${e.title || e.path || 'created'}`;
  if (e.type === 'report_updated') return 'report updated';
  const name = toolName(e);
  if (e.type === 'tool_finished') return `${name} ${e.status === 'ok' ? 'done' : e.status || 'finished'}`;
  if (e.type === 'tool_started') return `${name} running`;
  return cleanText(e.type || 'event');
}

function toolEventShort(e: ChatToolEvent): string {
  if (!e) return '';
  const name = toolName(e);
  if (e.status) return `${name}:${e.status}`;
  if (e.type === 'artifact_created') return 'artifact';
  if (e.type === 'report_updated') return 'report';
  return name;
}

function toolName(e: ChatToolEvent): string {
  const raw = String(e?.name || (e as unknown as Record<string, unknown>)?.tool || e?.type || 'tool').toLowerCase();
  if (/bash|shell|terminal|python/.test(raw)) return 'compute';
  if (/read|list|grep|find|glob|ls/.test(raw)) return 'inspect';
  if (/write|edit|patch/.test(raw)) return 'update';
  if (/report/.test(raw)) return 'report';
  if (/artifact/.test(raw)) return 'artifact';
  return 'tool';
}

function replSparkline(progress: number): number[] {
  const safe = Math.max(0, Math.min(100, Number(progress) || 0));
  return [0, 10, 16, 28, 40, 52, 64, 72, 82, 90, safe].map((v, i) => {
    if (safe <= 0) return i % 2 ? 5 : 2;
    return Math.max(2, Math.min(100, v ? Math.min(v, safe + i * 2) : Math.max(4, safe * 0.12)));
  });
}

function cleanText(value: string): string {
  return String(value || '')
    .replace(/\[LLM memory view:[^\]]*\]\s*/g, '')
    .replace(/<\|tool_calls\|>[\s\S]*?(?:<\|[\/]tool_calls\|>|$)/g, '')
    .replace(/<｜｜DSML｜｜tool_calls>[\s\S]*?<[\/]｜｜DSML｜｜tool_calls>/g, '')
    .replace(/\s+/g, ' ')
    .replace(/[<>]/g, '')
    .trim()
    .slice(0, 180);
}
