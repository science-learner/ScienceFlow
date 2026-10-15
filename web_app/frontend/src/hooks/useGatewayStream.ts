import { useEffect } from 'react';
import { Events } from '@wailsio/runtime';
import { useGatewayStore } from '../store/useGatewayStore';
import { useAppStore } from '../store/useAppStore';
import {
  formatGatewayLogEvent,
  gatewayAgentStatus,
  gatewayStartStream,
  isGatewaySessionGone,
  type GatewayLogEvent,
  type GatewayStatusEvent,
  type GatewayBackfillEvent,
  type GatewayBackfillDoneEvent,
} from '../api/gateway';
import { parseHistoricalChatMessages } from '../utils/agentLogParser';

// Bridges the gateway_server SSE stream (forwarded by the Go layer as Wails
// events) into the gateway store. Mount once near the app root.
//
// Each gateway "log" event carries a `message` field with one line of the raw
// agent log. We feed the raw line into the store's rawBuffer (for the agent
// log parser → chat transcript/lineage views) and also push a formatted display
// line into `lines` (for the plain logs panel).
//
// On session switch the gateway replays cached history via "backfill" /
// "backfill-done" events before resuming live tailing.
export function useGatewayStream() {
  useEffect(() => {
    const isCurrentSessionEvent = (eventSessionId?: string) => {
      const currentSessionId = useGatewayStore.getState().sessionId;
      return !eventSessionId || !currentSessionId || eventSessionId === currentSessionId;
    };

    // Rebuild the chat message list from the gateway log (parsed agent runs:
    // user queries + assistant answers + stop markers, in log order) merged
    // with optimistic local messages (just-typed input, task/error notices)
    // that are not in the log yet.
    let lastChatSyncAt = 0;
    const syncChatMessages = (force = false) => {
      const now = Date.now();
      if (!force && now - lastChatSyncAt < 400) return;
      lastChatSyncAt = now;
      const gw = useGatewayStore.getState();
      const parsed = parseHistoricalChatMessages(gw.parsedLog.runs);
      // The backend embeds the route mode into the logged query
      // ("[mode=lite] ..."); strip it so history matches what the user typed.
      const stripMode = (s: string) => s.replace(/^\[mode=\S+\]\s*/, '');
      for (const m of parsed) {
        if (m.role === 'user') m.content = stripMode(m.content);
      }
      const parsedUserContents = new Set(
        parsed.filter((m) => m.role === 'user').map((m) => m.content),
      );
      const optimistic = useAppStore
        .getState()
        .chatMessages.filter(
          (m) =>
            !m.message_id.startsWith('history-') &&
            !(m.role === 'user' && parsedUserContents.has(m.content)),
        );
      // Carry the true local send timestamp from optimistic user messages
      // onto their parsed-history counterparts, so locally-added messages
      // (e.g. "Task started") sort below the question instead of above it.
      const pendingMeta = new Map<string, string[]>();
      for (const m of useAppStore.getState().chatMessages) {
        if (m.role === 'user') {
          const list = pendingMeta.get(m.content) || [];
          list.push(m.created_at);
          pendingMeta.set(m.content, list);
        }
      }
      for (const m of parsed) {
        if (m.role !== 'user') continue;
        const list = pendingMeta.get(m.content);
        if (list && list.length > 0) {
          m.created_at = list.shift()!;
        }
      }
      // Sort by parsed epoch time, not string compare: created_at mixes UTC
      // RFC3339 (RAW.log headers), local ISO (optimistic sends) and gateway
      // event timestamps. '' sorts first (backfill-only history), which is
      // correct because backfilled content is always older than live sends.
      // Array.sort is stable so equal timestamps keep parse order.
      const ts = (v?: string) => {
        if (!v) return 0;
        const t = Date.parse(v);
        return Number.isNaN(t) ? 0 : t;
      };
      const merged = [...parsed, ...optimistic].sort((a, b) => ts(a.created_at) - ts(b.created_at));
      useAppStore.getState().setChatMessages(merged);
    };

    const offLog = Events.On('gateway-log', (ev) => {
      const e = ev.data as GatewayLogEvent;
      if (!isCurrentSessionEvent(e.session_id)) return;
      const store = useGatewayStore.getState();
      const rawLine = e.message || '';
      store.appendLine(formatGatewayLogEvent(e), rawLine);
      syncChatMessages();
    });

    const offBackfill = Events.On('gateway-backfill', (ev) => {
      const e = ev.data as GatewayBackfillEvent;
      if (!isCurrentSessionEvent(e.session_id)) return;
      useGatewayStore.getState().appendBackfill(e.source, e.path, e.content || '', e.truncated);
    });

    const offBackfillDone = Events.On('gateway-backfill-done', (ev) => {
      const e = ev.data as GatewayBackfillDoneEvent;
      if (!isCurrentSessionEvent(e.session_id)) return;
      const gatewayStore = useGatewayStore.getState();
      gatewayStore.finishBackfill(e.count || 0);
      syncChatMessages(true);
    });

    const offStatus = Events.On('gateway-status', (ev) => {
      const e = ev.data as GatewayStatusEvent;
      // Server mode shares the Go process across browser clients: only apply
      // the connection state of our own session.
      if (!isCurrentSessionEvent(e.session_id)) return;
      useGatewayStore.getState().setStatus(e.status, e.error);
    });

    // Terminal task failures (agent process failed to start, crashed, timed
    // out or was killed) would otherwise be invisible in the chat: the agent
    // never produced a transcript, so the user just sees "no reply". Surface
    // the task error once per task as a platform message.
    const notifiedTerminal = new Set<string>();
    const notifyTaskTerminal = (task: { id?: string; error?: string } | null | undefined, kind: 'failed' | 'cancelled') => {
      if (!task || !task.id || notifiedTerminal.has(task.id)) return;
      notifiedTerminal.add(task.id);
      const detail = (task.error || '').trim();
      const label = kind === 'failed' ? 'Task failed' : 'Task cancelled';
      useAppStore.getState().addChatMessage({
        message_id: `task-${kind}-${task.id}`,
        role: 'platform',
        content: detail ? `${label}: ${detail}` : label,
        created_at: new Date().toISOString(),
      });
    };

    // Re-assert cadence: every ~10s when healthy, every ~3s while not
    // connected so a stalled/evicted stream recovers quickly.
    let lastStreamEnsureAt = 0;
    const ensureStreamSubscription = (sessionId: string) => {
      const now = Date.now();
      const connected = useGatewayStore.getState().status === 'connected';
      const interval = connected ? 10000 : 3000;
      if (now - lastStreamEnsureAt < interval) return;
      lastStreamEnsureAt = now;
      gatewayStartStream(sessionId);
    };

    const statusTimer = window.setInterval(async () => {
      const { token, sessionId } = useGatewayStore.getState();
      if (!token || !sessionId) return;
      // Re-assert our gateway SSE subscription (idempotent server-side). This
      // self-heals two server-mode cases: another browser client stopped all
      // streams (logout / session switch), or the stream silently stalled.
      // Cheap when already subscribed; more eager while disconnected.
      ensureStreamSubscription(sessionId);
      // While a stop request is in flight, don't let the poll overwrite the
      // "cancelling" state (the backend may still report "running" briefly).
      if (useAppStore.getState().chatRunState === 'cancelling') return;
      try {
        const status = await gatewayAgentStatus(token, sessionId);
        const app = useAppStore.getState();
        const taskStatus = status.task?.status || status.status;
        if (taskStatus === 'running' || taskStatus === 'queued') {
          app.setChatBusy(true);
          app.setChatRunState('running');
        } else if (taskStatus === 'done' || taskStatus === 'completed') {
          app.setChatBusy(false);
          app.setChatRunState('completed');
          syncChatMessages(true);
        } else if (taskStatus === 'failed' || taskStatus === 'error') {
          app.setChatBusy(false);
          app.setChatRunState('failed');
          notifyTaskTerminal(status.task, 'failed');
          syncChatMessages(true);
        } else if (taskStatus === 'idle') {
          app.setChatBusy(false);
          app.setChatRunState('idle');
          syncChatMessages(true);
        } else if (taskStatus === 'killed' || taskStatus === 'cancelled') {
          app.setChatBusy(false);
          app.setChatRunState('cancelled');
          notifyTaskTerminal(status.task, 'cancelled');
          syncChatMessages(true);
        }
      } catch (e) {
        // The SSE stream remains the primary live status channel. A stale
        // session id (typically after a gateway restart) surfaces as 404
        // "session not found": rebuild the session so the chat, report and
        // monitor panels recover instead of spamming failed calls.
        if (isGatewaySessionGone(e)) {
          const newId = await useGatewayStore.getState().recoverSession();
          if (newId) useAppStore.getState().setChatSessionId(newId);
        }
      }
    }, 1500);

    return () => {
      offLog();
      offBackfill();
      offBackfillDone();
      offStatus();
      window.clearInterval(statusTimer);
    };
  }, []);
}
