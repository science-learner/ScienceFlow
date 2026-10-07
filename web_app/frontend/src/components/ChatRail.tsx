import { useState, useRef, useEffect, useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Upload, Tooltip, Dropdown } from 'antd';
import type { UploadProps } from 'antd';
import { useAppStore } from '../store/useAppStore';
import { useGatewayStore } from '../store/useGatewayStore';
import { gatewayInvokeAgent, gatewayStopAgent, workspaceUploadFiles } from '../api/gateway';
import { useT } from '../i18n/useT';
import * as api from '../api/client';
import { debug } from '../utils/debug';
import {
  formatToolArgPreview,
  isOutputTruncationMarker,
  runDuration,
  type ParsedAgentRun,
  type ParsedAgentStep,
} from '../utils/agentLogParser';
import { sanitizeReportMarkdown } from '../utils/helpers';
import clsx from 'clsx';
import {
  Upload as UploadIcon, FolderUp, Mic, ArrowUp, Square, Terminal, FileText,
  Pencil, FileEdit, Search, FolderOpen, List, Code, Wrench, ChevronRight,
  Check, X, Loader2, Brain, Copy, Sparkles,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import type { ChatMessage } from '../types';

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

const TOOL_COLORS: Record<string, string> = {
  bash: '#62d884',
  read: '#5cc8ff',
  write: '#f2c86b',
  edit: '#ff9f6e',
  grep: '#e99cff',
  glob: '#5cc8ff',
  ls: '#aab7c6',
  python: '#62d884',
  python3: '#62d884',
};

export function ChatRail() {
  const {
    chatMessages, chatSessionId, sessionsPanelOpen,
    setSessionsPanelOpen, addChatMessage, setChatBusy,
    setChatRunState, chatBusy, chatRunState, chatRouteMode,
    setChatRouteMode, chatSendInFlight,
    currentState,
  } = useAppStore();
  const { sessionList, fetchSessionList, switchSession, createSession, sessionId: gwSessionId } = useGatewayStore();
  const t = useT();

  const [input, setInput] = useState('');
  const [datasetStatus, setDatasetStatus] = useState('');
  const [datasetUploadProgress, setDatasetUploadProgress] = useState(false);
  const [datasetUploadLabel, setDatasetUploadLabel] = useState(t.chatRail.preparingUpload);
  const [datasetUploadSpeed, setDatasetUploadSpeed] = useState('--/s');
  const [datasetUploadPercent, setDatasetUploadPercent] = useState('0%');
  const [datasetUploadBarWidth, setDatasetUploadBarWidth] = useState(0);
  const [taskMode, setTaskMode] = useState<'lite' | 'heavy'>('lite');
  const [taskSetupOpen, setTaskSetupOpen] = useState(false);
  const [voiceState, setVoiceState] = useState<'idle' | 'starting' | 'listening' | 'error' | 'unsupported'>('idle');
  const [switchBusy, setSwitchBusy] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  // Sticky-bottom: only auto-scroll while the user is already near the bottom.
  const chatStickRef = useRef(true);
  const folderInputRef = useRef<HTMLInputElement>(null);

  // Steps appended to the live run grow the transcript without touching the
  // message list; follow them too so the newest trace stays visible.
  const liveStepCount = useGatewayStore((s) =>
    s.parsedLog.runs.reduce((n, r) => n + r.steps.length, 0),
  );

  const taskId = currentState?.task?.task_id || '';
  const datasetName = currentState?.task?.dataset_name || '';

  useEffect(() => {
    if (!chatStickRef.current) return;
    const container = messagesEndRef.current?.closest('.rail-messages');
    if (container) container.scrollTop = container.scrollHeight;
  }, [chatMessages, liveStepCount]);

  useEffect(() => {
    if (datasetName) setDatasetStatus(datasetName);
    else setDatasetStatus(t.chatRail.noDataset);
  }, [datasetName]);

  // Load gateway session list when the panel opens.
  useEffect(() => {
    if (sessionsPanelOpen) fetchSessionList();
  }, [sessionsPanelOpen, fetchSessionList]);

  const handleSend = async () => {
    const text = input.trim();
    debug.log("ChatRail", "handleSend text=", text.slice(0, 60), "chatSessionId=", chatSessionId, "chatBusy=", chatBusy, "chatSendInFlight=", chatSendInFlight);
    if (!text || chatBusy || chatSendInFlight) {
      debug.log("ChatRail", "handleSend blocked: !text=", !text, "chatBusy=", chatBusy, "chatSendInFlight=", chatSendInFlight);
      return;
    }
    let targetSessionId = chatSessionId;
    if (!targetSessionId) {
      const session = await createSession();
      if (!session) {
        debug.log("ChatRail", "failed to create gateway session");
        return;
      }
      targetSessionId = session.session_id;
      useAppStore.getState().setChatSessionId(targetSessionId);
      debug.log("ChatRail", "created gateway session before invoke", targetSessionId);
    }
    setInput('');

    const idempotencyKey = `msg-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;

    const userMsg: ChatMessage = {
      message_id: idempotencyKey,
      role: 'user',
      content: text,
      created_at: new Date().toISOString(),
      route: chatRouteMode,
    };
    addChatMessage(userMsg);

    useAppStore.setState({ chatSendInFlight: true, chatBusy: true, chatRunState: 'running' });
    try {
      debug.log("ChatRail", "invoking agent on gateway session", targetSessionId, "mode=", taskMode);
      const { token } = useGatewayStore.getState();
      const task = await gatewayInvokeAgent(token, targetSessionId, text, taskMode);
      debug.log("ChatRail", "invoke response:", task);
      addChatMessage({
        message_id: `task-${task.id}`,
        role: 'platform',
        content: `Task started: ${task.id} (status: ${task.status})`,
        created_at: new Date().toISOString(),
      });
    } catch (e: unknown) {
      debug.error("ChatRail", "invoke failed:", e);
      addChatMessage({
        message_id: `err-${Date.now()}`,
        role: 'platform',
        content: `Failed: ${(e as Error).message || t.chatRail.unknownError}`,
        created_at: new Date().toISOString(),
      });
      setChatBusy(false);
      setChatRunState('idle');
    } finally {
      useAppStore.setState({ chatSendInFlight: false });
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleCancel = async () => {
    const { token, sessionId: gatewaySessionId } = useGatewayStore.getState();
    const sessionId = gatewaySessionId || chatSessionId;
    debug.log('ChatRail', 'handleCancel clicked', { sessionId, hasToken: !!token, chatBusy, chatRunState });
    if (!token || !sessionId) {
      debug.warn('ChatRail', 'cannot stop agent: missing gateway token or session id');
      return;
    }

    setChatRunState('cancelling');
    try {
      const task = await gatewayStopAgent(token, sessionId);
      debug.log('ChatRail', 'stop response:', task);
      setChatBusy(false);
      setChatRunState(task.status === 'done' ? 'completed' : 'cancelled');
      await fetchSessionList();
    } catch (e: unknown) {
      debug.error('ChatRail', 'stop agent failed:', e);
      setChatBusy(true);
      setChatRunState('running');
    }
  };

  const handleRefreshSessions = async () => {
    if (switchBusy) return;
    setSwitchBusy(true);
    try {
      await fetchSessionList();
    } finally {
      setSwitchBusy(false);
    }
  };

  const handleSwitchSession = async (sessionId: string) => {
    if (switchBusy || sessionId === gwSessionId) return;
    setSwitchBusy(true);
    try {
      const session = await switchSession(sessionId);
      if (session) {
        useAppStore.getState().setChatMessages([]);
        useAppStore.getState().setChatSessionId(sessionId);
        useAppStore.getState().clearTimeline();
        setSessionsPanelOpen(false);
      }
    } finally {
      setSwitchBusy(false);
    }
  };

  const handleTaskModeSelect = (mode: 'lite' | 'heavy') => {
    setTaskMode(mode);
  };

  const handleUploadRequest: UploadProps['customRequest'] = async (options) => {
    const { file, onSuccess, onError } = options;
    const { token, sessionId } = useGatewayStore.getState();
    if (!token || !sessionId) {
      onError?.(new Error('No gateway session'));
      return;
    }
    setDatasetUploadProgress(true);
    setDatasetUploadLabel(t.chatRail.uploading);
    try {
      const result = await workspaceUploadFiles(token, sessionId, '', [file as File]);
      setDatasetStatus(`${result.uploaded?.length || 1} file(s) uploaded`);
      setDatasetUploadLabel(t.chatRail.complete);
      setDatasetUploadPercent('100%');
      setDatasetUploadBarWidth(100);
      onSuccess?.(result);
    } catch (e: unknown) {
      setDatasetUploadLabel(`Error: ${(e as Error).message}`);
      onError?.(e as Error);
    } finally {
      setTimeout(() => setDatasetUploadProgress(false), 2000);
    }
  };

  const handleFolderChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;
    const { token, sessionId } = useGatewayStore.getState();
    if (!token || !sessionId) return;
    setDatasetUploadProgress(true);
    setDatasetUploadLabel(t.chatRail.uploadingFolder);
    try {
      const result = await workspaceUploadFiles(token, sessionId, '', files);
      setDatasetStatus(`${result.uploaded?.length || files.length} file(s) uploaded`);
      setDatasetUploadLabel(t.chatRail.complete);
      setDatasetUploadPercent('100%');
      setDatasetUploadBarWidth(100);
    } catch (e: unknown) {
      setDatasetUploadLabel(`Error: ${(e as Error).message}`);
    } finally {
      setTimeout(() => setDatasetUploadProgress(false), 2000);
    }
  };

  const handleVoiceToggle = () => {
    setVoiceState((s) => s === 'idle' ? 'starting' : 'idle');
    if (voiceState === 'idle') setTimeout(() => setVoiceState('listening'), 500);
  };

  const isCancelling = chatRunState === 'cancelling';
  const isAgentActive = chatBusy || chatRunState === 'running' || isCancelling;
  const actionState = isAgentActive ? 'stop' : 'send';

  return (
    <aside className="card chat-rail">
      <div className="card-head">
        <span className="card-title">{t.chatRail.cockpit}</span>
        <div className="chat-head-actions">
          <span className={clsx('chat-status-badge', chatBusy && 'busy')}>
            <span className={clsx('state-dot', chatBusy ? 'pulse' : 'idle')} />
            {chatRunState === 'running' ? t.chatRail.running : chatRunState === 'cancelling' ? t.chatRail.stopping : chatRunState === 'cancelled' ? t.chatRail.stopped : chatRunState === 'completed' ? t.chatRail.done : t.common.idle}
          </span>
          <button className="btn" id="frontToggleSessions" onClick={() => setSessionsPanelOpen(!sessionsPanelOpen)}>
            {t.chatRail.sessions}
          </button>
        </div>
      </div>
      <div className="card-body">
        <div className={clsx('session-history-panel', sessionsPanelOpen && 'active')} id="frontSessionPanel">
          <div className="session-history-head">
            <span>{t.chatRail.chatSessions}</span>
            <div className="session-actions">
              <button className="btn" onClick={handleRefreshSessions}>{t.chatRail.refresh}</button>
            </div>
          </div>
          <div className="session-list" id="frontSessionList">
            {sessionList.length === 0 && (
              <div className="dim" style={{ padding: '10px 8px', fontSize: 11 }}>No sessions</div>
            )}
            {sessionList.map((s) => {
              const active = s.session_id === gwSessionId;
              const agentStatus = s.agent?.status || 'idle';
              const isRunning = agentStatus === 'active';
              return (
                <button
                  key={s.session_id}
                  disabled={switchBusy}
                  className={clsx('session-card', active && 'active')}
                  onClick={() => handleSwitchSession(s.session_id)}
                >
                  <div className="session-card-head">
                    <span className="session-card-id" style={{ fontFamily: 'var(--mono)', fontSize: 11, color: active ? 'var(--accent)' : 'var(--muted)' }}>
                      {s.session_id.slice(0, 20)}...
                    </span>
                    <span className={clsx('session-agent-badge', isRunning && 'running')}>
                      {isRunning ? '● running' : 'idle'}
                    </span>
                  </div>
                  <div className="session-card-meta" style={{ fontSize: 11, color: 'var(--soft)', marginTop: 3 }}>
                    {s.sources?.length ?? 0} sources
                    {s.last_active && (
                      <span className="dim" style={{ marginLeft: 8 }}>
                        {formatLastActive(s.last_active)}
                      </span>
                    )}
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        <div className="agent-cockpit" data-agent-cockpit>
          <section className="agent-cockpit-panel agent-cockpit-transcript" data-agent-transcript>
            <div className="messages rail-messages" id="frontMessages" onScroll={(e) => {
              const el = e.currentTarget;
              chatStickRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
            }}>
              <ChatTranscript busy={chatBusy} />
              {chatBusy && chatMessages.length === 0 && (
                <div className="agent-thinking">
                  <div className="typing-dots"><i /><i /><i /></div>
                  <span style={{ color: 'var(--muted)', fontSize: 13, fontFamily: 'var(--mono)' }}>Agent is thinking...</span>
                </div>
              )}
              {!chatBusy && chatMessages.length === 0 && <AgentEmptyState />}
              <div ref={messagesEndRef} />
            </div>
          </section>

          <section className={clsx('agent-cockpit-send', 'is-expanded', taskMode === 'heavy' && 'mode-heavy', chatBusy && 'agent-running')} data-agent-cockpit-send>
            <button className="agent-cockpit-send-toggle" type="button">
              <strong>{t.chatRail.send}</strong>
              <span>{chatSessionId ? chatSessionId.slice(0, 12) + '...' : t.chatRail.collapsed}</span>
            </button>
            <div className={clsx('front-composer', taskMode === 'heavy' && 'mode-heavy')}>
              <div className="composer-input-wrap">
                <textarea id="frontChatInput" enterKeyHint="send" placeholder={t.chatRail.inputPlaceholder}
                  value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={handleKeyDown} disabled={chatBusy} />
              </div>
              <div className="composer-tools">
                <div className={clsx('task-setup-popover', (taskSetupOpen || datasetUploadProgress) && 'active')} hidden={!(taskSetupOpen || datasetUploadProgress)}>
                  <div className="task-setup-head">
                    <strong>{t.chatRail.taskSetup}</strong>
                    <span>{taskId ? t.chatRail.ready : t.chatRail.noTask}</span>
                    <button className="task-setup-close" type="button" onClick={() => setTaskSetupOpen(false)}>&times;</button>
                  </div>
                  <div className="task-setup-empty" hidden={!!taskId}>{t.chatRail.createTaskFirst}</div>
                  <div className="task-setup-controls" hidden={!taskId}>
                    <div className="dataset-upload-progress" hidden={!datasetUploadProgress}>
                      <div className="dataset-upload-progress-head"><span>{datasetUploadLabel}</span><span className="dataset-upload-progress-stats"><span>{datasetUploadSpeed}</span><span>{datasetUploadPercent}</span></span></div>
                      <div className="dataset-upload-progress-track"><i style={{ width: `${datasetUploadBarWidth}%` }}></i></div>
                    </div>
                    <div className="dataset-status-row">
                      <div className="dataset-status">{datasetStatus}</div>
                    </div>
                  </div>
                </div>

                <Dropdown
                  trigger={['click']}
                  menu={{
                    items: [
                      {
                        key: 'file',
                        label: (
                          <Upload customRequest={handleUploadRequest} showUploadList={false} multiple>
                            <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                              <UploadIcon size={14} />
                              {t.chatRail.uploadFile}
                            </span>
                          </Upload>
                        ),
                      },
                      {
                        key: 'folder',
                        icon: (
                          <FolderUp size={14} />
                        ),
                        label: t.chatRail.uploadFolder,
                        onClick: () => folderInputRef.current?.click(),
                      },
                    ],
                  }}
                >
                  <button className="composer-add" type="button">+</button>
                </Dropdown>
                <input ref={folderInputRef} type="file" {...({ webkitdirectory: '' } as React.InputHTMLAttributes<HTMLInputElement>)} multiple hidden onChange={handleFolderChange} />

                <Tooltip title={taskMode === 'lite' ? 'Lite mode — click for Heavy' : 'Heavy mode — click for Lite'}>
                  <button
                    className={clsx('mode-toggle', taskMode === 'heavy' && 'heavy')}
                    type="button"
                    onClick={() => handleTaskModeSelect(taskMode === 'lite' ? 'heavy' : 'lite')}
                  >
                    <span className="mode-toggle-track">
                      <span className="mode-toggle-thumb" />
                    </span>
                    <span className="mode-toggle-label">{taskMode}</span>
                  </button>
                </Tooltip>

                <div className="composer-actions">
                  {false && (
                    <button className={clsx('btn chat-action-button chat-voice-button', voiceState !== 'idle' && `is-${voiceState}`)} onClick={handleVoiceToggle}>
                      <span className="chat-action-icon voice">
                        <Mic size={18} />
                      </span>
                    </button>
                  )}
                  <button
                    className={clsx('btn primary chat-action-button', actionState === 'stop' && 'danger')}
                    data-action-state={actionState === 'stop' ? 'stop' : 'send'}
                    onClick={actionState === 'stop' ? handleCancel : handleSend}
                    disabled={isCancelling || (actionState === 'send' && (!input.trim() || chatSendInFlight))}>
                    {actionState === 'send' ? (
                      <span className="chat-action-icon send"><ArrowUp size={18} strokeWidth={2.5} /></span>
                    ) : (
                      <span className="chat-action-icon stop"><Square size={16} fill="currentColor" /></span>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </section>
        </div>
      </div>
    </aside>
  );
}

function formatLastActive(rfc3339: string): string {
  try {
    const d = new Date(rfc3339);
    const now = Date.now();
    const diff = now - d.getTime();
    const sec = Math.floor(diff / 1000);
    if (sec < 0) return 'just now';
    if (sec < 60) return `${sec}s ago`;
    const min = Math.floor(sec / 60);
    if (min < 60) return `${min}m ago`;
    const hr = Math.floor(min / 60);
    if (hr < 24) return `${hr}h ago`;
    const day = Math.floor(hr / 24);
    return `${day}d ago`;
  } catch {
    return '';
  }
}

// ── Transcript ─────────────────────────────────────────────────────────────

// Renders the merged chat message list, interleaving each run's execution
// timeline (parsed from the agent log) right below its user message:
//
//   [user bubble]
//   [run activity card — thoughts / tool calls / iterations]
//   [assistant answer]
//   [system notices]
function ChatTranscript({ busy }: { busy: boolean }) {
  const chatMessages = useAppStore((s) => s.chatMessages);
  const runs = useGatewayStore((s) => s.parsedLog.runs);

  return useMemo(() => {
    const claimed = new Set<number>();
    for (const m of chatMessages) {
      if (m.role === 'user' && m.message_id.startsWith('history-') && m.runIndex != null) {
        claimed.add(m.runIndex);
      }
    }
    const latestRunIndex = runs.length > 0 ? runs[runs.length - 1].index : -1;

    const items: React.ReactNode[] = [];
    chatMessages.forEach((msg, i) => {
      const isLast = i === chatMessages.length - 1;

      if (msg.role === 'user') {
        let runIndex = msg.runIndex != null && runs[msg.runIndex] ? msg.runIndex : undefined;
        if (runIndex == null) {
          // Optimistic message: claim the first unclaimed run carrying the
          // same query text (once its [user] line has been parsed). The
          // backend embeds the route mode into the logged query.
          const stripMode = (s: string) => s.replace(/^\[mode=\S+\]\s*/, '');
          const found = runs.find(
            (r) => !claimed.has(r.index) && stripMode(r.query) === msg.content,
          );
          if (found) runIndex = found.index;
        }
        if (runIndex != null) claimed.add(runIndex);

        items.push(<UserMessage key={msg.message_id} message={msg} />);
        const run = runIndex != null ? runs[runIndex] : undefined;
        if (run) {
          items.push(
            <RunActivityCard
              key={`run-${run.index}`}
              run={run}
              live={run.index === latestRunIndex}
            />,
          );
        } else if (isLast && busy) {
          items.push(<RunPendingCard key={`pending-${msg.message_id}`} />);
        }
        return;
      }

      if (msg.role === 'assistant') {
        items.push(
          <AssistantMessage key={msg.message_id} message={msg} streaming={busy && isLast} />,
        );
        return;
      }

      // platform
      if (msg.message_id.startsWith('history-stop-')) {
        items.push(
          <div key={msg.message_id} className="chat-system-marker">
            {msg.content}
          </div>,
        );
        return;
      }
      // "Task started: <task-id>" notices become redundant once the run's
      // timeline (which shows the task id) has been parsed from the log.
      const taskMatch = msg.content.match(/^Task started:\s*(\S+)/);
      if (taskMatch && runs.some((r) => r.taskId === taskMatch[1])) return;
      items.push(<SystemNotice key={msg.message_id} message={msg} />);
    });

    return <>{items}</>;
  }, [chatMessages, runs, busy]);
}

// ── Run activity card ──────────────────────────────────────────────────────

type RunStatus = 'running' | 'done' | 'stopped' | 'idle';

function runStatus(run: ParsedAgentRun, live: boolean): RunStatus {
  if (run.stopMarker) return 'stopped';
  if (run.answer.trim()) return 'done';
  if (live) return 'running';
  return 'idle';
}

function RunActivityCard({ run, live }: { run: ParsedAgentRun; live: boolean }) {
  const t = useT();
  const status = runStatus(run, live);
  // Latest run starts expanded; older runs start collapsed. When a new run
  // begins, the previously-live card auto-collapses (user can re-expand).
  const [open, setOpen] = useState(live);
  const prevLive = useRef(live);
  useEffect(() => {
    if (prevLive.current && !live) setOpen(false);
    prevLive.current = live;
  }, [live]);

  // The timeline body is capped by max-height, so while the run is live the
  // newest steps would fall below the fold. Follow the bottom (like the chat
  // transcript) until the user scrolls up inside the timeline.
  const timelineRef = useRef<HTMLDivElement>(null);
  const timelineStickRef = useRef(true);
  useEffect(() => {
    if (!open) return;
    const el = timelineRef.current;
    if (!el) return;
    if (live && timelineStickRef.current) {
      el.scrollTop = el.scrollHeight;
    } else if (!live) {
      // Freshly expanded history card: start at the top of the timeline.
      el.scrollTop = 0;
    }
  }, [open, live, run.steps.length]);

  const handleTimelineScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const el = e.currentTarget;
    timelineStickRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
  };

  const toolCount = run.steps.reduce((n, s) => (s.kind === 'tool' ? n + 1 : n), 0);
  const duration = runDuration(run);

  const statusLabel =
    status === 'running' ? t.chatRail.running
      : status === 'done' ? t.chatRail.done
        : status === 'stopped' ? t.chatRail.stopped
          : t.common.idle;

  return (
    <div className={clsx('run-activity', `is-${status}`, !open && 'is-closed')}>
      <button className="run-activity-head" type="button" onClick={() => setOpen(!open)} aria-expanded={open}>
        <ChevronRight size={12} className={clsx('run-activity-chevron', open && 'expanded')} />
        <span className="run-chip">{t.chatRail.runLabel} {run.index + 1}</span>
        {run.taskId && <span className="run-task-id" title={run.taskId}>{run.taskId.replace(/^task-/, '').slice(-8)}</span>}
        <span className="run-stats">
          {toolCount > 0 && <span>{toolCount} {t.chatRail.toolCallsUnit}</span>}
          {run.iteration && run.iteration.max > 0 && (
            <span>{t.chatRail.iterationsShort} {run.iteration.current}/{run.iteration.max}</span>
          )}
          {duration && <span>{duration}</span>}
        </span>
        <span className={clsx('run-status-pill', `is-${status}`)}>
          {status === 'running' && <span className="state-dot pulse" />}
          {statusLabel}
        </span>
      </button>
      {open && (
        <div className="run-timeline" ref={timelineRef} onScroll={handleTimelineScroll}>
          {run.steps.map((step, i) => (
            <RunStep key={i} step={step} live={live && i === run.steps.length - 1} />
          ))}
          {status === 'running' && (
            <div className="run-step is-waiting">
              <span className="run-step-node"><Loader2 size={11} className="run-spin" /></span>
              <span className="run-waiting-text">{t.chatRail.agentWorking}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function RunStep({ step, live }: { step: ParsedAgentStep; live: boolean }) {
  const t = useT();

  if (step.kind === 'iteration') {
    return (
      <div className="run-iter">
        <span>{t.chatRail.iterationsShort} {step.current}/{step.max}</span>
      </div>
    );
  }

  if (step.kind === 'notice') {
    return (
      <div className="run-step is-notice">
        <span className="run-step-node"><Wrench size={11} /></span>
        <span className="run-notice-text">{step.text}</span>
      </div>
    );
  }

  if (step.kind === 'thought') {
    return (
      <div className="run-step is-thought">
        <span className="run-step-node"><Brain size={11} /></span>
        <div className="run-step-main">
          <div className="run-step-head">
            <span className="run-step-label">{t.chatRail.reasoning}</span>
            {live && <span className="state-dot pulse" />}
          </div>
          <p className="run-thought-text">{step.text}</p>
        </div>
      </div>
    );
  }

  // tool step (all other kinds returned above)
  const IconComp = TOOL_ICONS[step.tool] || Wrench;
  const color = TOOL_COLORS[step.tool] || '#aab7c6';
  const status = step.ok === null ? 'running' : step.ok ? 'ok' : 'err';
  const args = formatToolArgPreview(step.tool, step.args) || step.argsRaw;
  const duration = step.resultDetail.match(/\[(\d+(?:\.\d+)?)s/)?.[1];
  const truncCount = step.output.filter(isOutputTruncationMarker).length;

  return (
    <div className={clsx('run-step', 'is-tool', `is-${status}`)}>
      <span className="run-step-node" style={{ color, borderColor: color }}>
        <IconComp size={11} />
      </span>
      <div className="run-step-main">
        <div className="run-step-head">
          <span className="run-tool-name" style={{ color }}>{step.tool}</span>
          {args && <code className="run-tool-args" title={args}>{args}</code>}
          <span className={clsx('run-tool-status', `is-${status}`)}>
            {status === 'running' && <Loader2 size={11} className="run-spin" />}
            {status === 'ok' && <Check size={11} />}
            {status === 'err' && <X size={11} />}
            {duration && <em>{duration}s</em>}
          </span>
        </div>
        {step.thought && <p className="run-thought-inline">{step.thought}</p>}
        {step.ok === false && step.resultDetail && (
          <p className="run-tool-error">{step.resultDetail}</p>
        )}
        {step.output.length > 0 && (
          <details className="run-step-output">
            <summary>
              {t.chatRail.output}
              <em>{step.output.length}</em>
              {truncCount > 0 && <span className="run-output-trunc-badge">truncated</span>}
            </summary>
            <pre className="run-output-pre">
              {step.output.map((line, i) => (
                <span key={i} className={clsx(isOutputTruncationMarker(line) && 'run-output-trunc')}>
                  {line + '\n'}
                </span>
              ))}
            </pre>
          </details>
        )}
      </div>
    </div>
  );
}

function RunPendingCard() {
  const t = useT();
  return (
    <div className="run-activity is-running">
      <div className="run-activity-head">
        <span className="run-chip">{t.chatRail.runLabel}</span>
        <span className="run-status-pill is-running">
          <span className="state-dot pulse" />
          {t.chatRail.agentStarting}
        </span>
      </div>
    </div>
  );
}

function AgentEmptyState() {
  const t = useT();
  const status = useGatewayStore((s) => s.status);
  const connected = status === 'connected';
  return (
    <div className="agent-empty-state">
      <span className={clsx('agent-empty-icon', connected && 'is-ready')}>
        <Sparkles size={18} />
      </span>
      <strong>{t.chatRail.emptyTitle}</strong>
      <p>{connected ? t.chatRail.emptyHint : t.chatRail.emptyOfflineHint}</p>
    </div>
  );
}

// ── Message bubbles ────────────────────────────────────────────────────────

function messageTime(created_at: string): string {
  if (!created_at) return '';
  const d = new Date(created_at);
  if (isNaN(d.getTime())) return '';
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function UserMessage({ message }: { message: ChatMessage }) {
  const t = useT();
  const time = messageTime(message.created_at);
  return (
    <div className="message user">
      <div className="message-meta">
        <strong>{t.chatRail.you}</strong>
        {time && <span className="msg-time">{time}</span>}
      </div>
      <div className="message-body">
        <div className="user-text">{message.content}</div>
      </div>
    </div>
  );
}

function AssistantMessage({ message, streaming }: { message: ChatMessage; streaming: boolean }) {
  const t = useT();
  const [copied, setCopied] = useState(false);
  const time = messageTime(message.created_at);

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }).catch(() => {});
  };

  return (
    <div className="message assistant">
      <div className="message-meta">
        <strong>{t.chatRail.agent}</strong>
        {time && <span className="msg-time">{time}</span>}
        <button
          className={clsx('msg-copy', copied && 'ok')}
          type="button"
          onClick={handleCopy}
          title={copied ? t.chatRail.copied : t.chatRail.copyAnswer}
          aria-label={t.chatRail.copyAnswer}
        >
          {copied ? <Check size={12} strokeWidth={2.5} /> : <Copy size={12} />}
        </button>
      </div>
      <div className="message-body">
        <div className="chat-markdown">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {sanitizeReportMarkdown(message.content)}
          </ReactMarkdown>
          {streaming && <span className="stream-cursor" />}
        </div>
        {message.decision && <DecisionCardView decision={message.decision} />}
      </div>
    </div>
  );
}

function SystemNotice({ message }: { message: ChatMessage }) {
  const isError = /^Failed:/i.test(message.content);
  const taskMatch = message.content.match(/^Task started:\s*(\S+)/);
  return (
    <div className={clsx('chat-system-chip', isError && 'is-error')}>
      {taskMatch ? (
        <>
          <span className="chat-system-chip-label">task</span>
          <code>{taskMatch[1].replace(/^task-/, '').slice(-12)}</code>
        </>
      ) : (
        <span>{message.content}</span>
      )}
    </div>
  );
}

function DecisionCardView({ decision }: { decision: NonNullable<ChatMessage['decision']> }) {
  const t = useT();
  const handleResolve = async (choice: string) => {
    try { await api.resolveDecision(decision.decision_id, choice); } catch {}
  };
  return (
    <div className={clsx('message decision', decision.resolved && 'resolved', decision.expired && 'expired')} style={{ marginTop: 6 }}>
      <div className="decision-body">
        <strong>{decision.title}</strong>
        <p>{decision.body}</p>
        <div className="decision-meta">{decision.resolved ? t.chatRail.resolved : decision.expired ? t.chatRail.expired : t.chatRail.pendingDecision}</div>
        {!decision.resolved && !decision.expired && (
          <div className="decision-options">
            {decision.options.map((opt) => <button key={opt.key} className="btn" onClick={() => handleResolve(opt.key)}>{opt.label}</button>)}
          </div>
        )}
      </div>
    </div>
  );
}
