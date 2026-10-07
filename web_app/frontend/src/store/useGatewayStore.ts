import { create } from 'zustand';
import {
  gatewayLogin,
  gatewayListSources,
  gatewayCreateSession,
  gatewayUpdateSessionSources,
  gatewayDeleteSession,
  gatewayListSessions,
  gatewayActivateSession,
  gatewayStartStream,
  gatewayStopStream,
  gatewayFetchFiles,
  gatewayTestConnection,
  type GatewayHealth,
  type GatewaySourceInfo,
  type GatewaySession,
} from '../api/gateway';
import { parseAgentLog, type ParsedAgentLog } from '../utils/agentLogParser';
import type { FileNode, FileTreeNode } from '../types';

export type GatewayStatus = 'idle' | 'connecting' | 'connected' | 'disconnected' | 'unauthorized' | 'error';

const MAX_LINES = 2000;
const MAX_RAW = 256 * 1024;
const SWITCH_SESSION_TIMEOUT_MS = 10000;

function withTimeout<T>(promise: Promise<T>, timeoutMs: number, message: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(() => reject(new Error(message)), timeoutMs);
    promise.then(
      (value) => {
        window.clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        window.clearTimeout(timer);
        reject(error);
      },
    );
  });
}

interface GatewayState {
  status: GatewayStatus;
  lastError: string;
  user: string;
  userName: string;
  password: string;
  token: string;
  sessionId: string;
  sources: GatewaySourceInfo[];
  subscribedSources: string[];
  sessionList: GatewaySession[];
  lines: string[];
  rawBuffer: string;
  parsedLog: ParsedAgentLog;

  // Backfill (session-switch historical log replay)
  backfilling: boolean;
  backfillCount: number;

  // File tree sync
  fileTreeRoot: string;
  fileTree: FileNode[];
  fileNestedTree: FileTreeNode[];
  fileTreeVersion: number;

  connect: (userName: string, password: string) => Promise<void>;
  disconnect: () => Promise<void>;
  testConnection: () => Promise<GatewayHealth>;
  setSubscribedSources: (sources: string[]) => Promise<void>;
  fetchSessionList: () => Promise<void>;
  nameSessionFromQuery: (sessionId: string, query: string) => void;
  prepareNewSession: () => void;
  createSession: () => Promise<GatewaySession | null>;
  switchSession: (sessionId: string) => Promise<GatewaySession | null>;
  deleteSession: (sessionId: string) => Promise<GatewaySession | null>;
  appendLine: (displayLine: string, rawLine?: string) => void;
  appendBackfill: (src: string, path: string, content: string, truncated: boolean) => void;
  finishBackfill: (count: number) => void;
  clearLines: () => void;
  setStatus: (status: GatewayStatus, error?: string) => void;
  getParsedLog: () => ParsedAgentLog;
  applyFileTree: (root: string, tree: FileNode[]) => void;
  applyFileChanges: (added: FileNode[], removed: FileNode[], modified: FileNode[]) => void;
  refetchFileTree: () => Promise<void>;
  clearFileTree: () => void;
}

const EMPTY_PARSED_LOG = { format: 'v2' as const, runs: [], toolCalls: [], raw: '' };

export const useGatewayStore = create<GatewayState>((set, get) => ({
  status: 'idle',
  lastError: '',
  user: '',
  userName: '',
  password: '',
  token: '',
  sessionId: '',
  sources: [],
  subscribedSources: [],
  sessionList: [],
  lines: [],
  rawBuffer: '',
  parsedLog: EMPTY_PARSED_LOG,
  backfilling: false,
  backfillCount: 0,
  fileTreeRoot: '',
  fileTree: [],
  fileNestedTree: [],
  fileTreeVersion: 0,

  connect: async (userName, password) => {
    set({ status: 'connecting', lastError: '', userName, password, lines: [], rawBuffer: '', parsedLog: EMPTY_PARSED_LOG });
    try {
      localStorage.setItem('scienceflow.gwUser', userName);
      localStorage.setItem('scienceflow.gwPass', password);
      const login = await gatewayLogin(userName, password);
      const sources = await gatewayListSources(login.token);
      const allNames = sources.map((s) => s.name);
      const listed = await gatewayListSessions(login.token);
      const existing = (listed.sessions || [])
        .filter((item) => item.session_id)
        .sort((a, b) => (b.last_active || '').localeCompare(a.last_active || ''))[0];
      const session = existing
        ? await gatewayActivateSession(login.token, existing.session_id)
        : await gatewayCreateSession(login.token, allNames);
      gatewayStartStream(session.session_id);
      set({
        status: 'connecting',
        user: login.user,
        token: login.token,
        sources,
        subscribedSources: session.sources || allNames,
        sessionId: session.session_id,
        sessionList: existing ? listed.sessions || [] : [session],
        lastError: '',
      });
    } catch (e) {
      set({ status: 'error', lastError: e instanceof Error ? e.message : String(e) });
    }
  },

  disconnect: async () => {
    const { token, sessionId } = get();
    gatewayStopStream();
    if (token && sessionId) {
      try {
        await gatewayDeleteSession(token, sessionId);
      } catch {
        /* ignore */
      }
    }
    set({
      status: 'idle',
      lastError: '',
      user: '',
      token: '',
      sessionId: '',
      sources: [],
      subscribedSources: [],
    });
  },

  // Probes the gateway server (GET /healthz) and mirrors the result into the
  // shared status so the topbar indicator updates from the settings dialog.
  testConnection: async () => {
    set({ status: 'connecting' });
    try {
      const health = await gatewayTestConnection();
      set({ status: 'connected', lastError: '' });
      return health;
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      set({ status: 'error', lastError: message });
      throw e;
    }
  },

  setSubscribedSources: async (sources) => {
    const { token, sessionId } = get();
    if (!token || !sessionId) return;
    try {
      const session = await gatewayUpdateSessionSources(token, sessionId, sources);
      set({ subscribedSources: session.sources || sources });
    } catch (e) {
      set({ lastError: e instanceof Error ? e.message : String(e) });
    }
  },

  fetchSessionList: async () => {
    const { token } = get();
    if (!token) return;
    try {
      const res = await gatewayListSessions(token);
      set({ sessionList: res.sessions || [] });
    } catch { /* silent */ }
  },

  // 首次提问即为会话命名（镜像 lgw 的 MaybeNameFromQuery 规则），
  // invoke 成功后就地更新列表，不必等下次 fetchSessionList。
  nameSessionFromQuery: (sessionId, query) => {
    const line = query.trim().split(/\r?\n/)[0].replace(/\s+/g, ' ').trim();
    if (!line) return;
    set((state) => ({
      sessionList: state.sessionList.map((item) =>
        item.session_id === sessionId && !item.name ? { ...item, name: line } : item,
      ),
    }));
  },

  prepareNewSession: () => {
    gatewayStopStream();
    set({
      sessionId: '',
      subscribedSources: [],
      lines: [],
      rawBuffer: '',
      parsedLog: EMPTY_PARSED_LOG,
      fileTreeRoot: '',
      fileTree: [],
      fileNestedTree: [],
      fileTreeVersion: 0,
      backfilling: false,
      backfillCount: 0,
    });
  },

  createSession: async () => {
    const { token, sources } = get();
    if (!token) return null;
    try {
      const session = await gatewayCreateSession(token, sources.map((s) => s.name));
      gatewayStopStream();
      gatewayStartStream(session.session_id);
      set((state) => ({
        sessionId: session.session_id,
        subscribedSources: session.sources || sources.map((s) => s.name),
        sessionList: [session, ...state.sessionList.filter((item) => item.session_id !== session.session_id)],
        lines: [],
        rawBuffer: '',
        parsedLog: EMPTY_PARSED_LOG,
        fileTreeRoot: '',
        fileTree: [],
        fileNestedTree: [],
        fileTreeVersion: 0,
        backfilling: false,
        backfillCount: 0,
      }));
      return session;
    } catch (e) {
      set({ lastError: e instanceof Error ? e.message : String(e) });
      return null;
    }
  },

  deleteSession: async (id) => {
    const { token, sessionId: current, sessionList } = get();
    if (!token || !id) return null;
    try {
      await gatewayDeleteSession(token, id);
      const remaining = sessionList.filter((item) => item.session_id !== id);
      set({ sessionList: remaining });
      if (id === current) {
        const next = remaining[0];
        if (next) {
          return await get().switchSession(next.session_id);
        }
        get().prepareNewSession();
      }
      return null;
    } catch (e) {
      set({ lastError: e instanceof Error ? e.message : String(e) });
      return null;
    }
  },

  switchSession: async (id) => {
    const { token, sessionId: current } = get();
    if (!token || !id) return null;
    if (id === current) return null;
    try {
      const session = await withTimeout(
        gatewayActivateSession(token, id),
        SWITCH_SESSION_TIMEOUT_MS,
        'Session switch timed out',
      );
      // Stop current stream, rebuild SSE for the new session.
      gatewayStopStream();
      gatewayStartStream(id);
      set({
        sessionId: id,
        subscribedSources: session.sources || [],
        lines: [],
        rawBuffer: '',
        parsedLog: EMPTY_PARSED_LOG,
        fileTreeRoot: '',
        fileTree: [],
        fileNestedTree: [],
        fileTreeVersion: 0,
        backfilling: true,
        backfillCount: 0,
      });
      return session;
    } catch (e) {
      set({ lastError: e instanceof Error ? e.message : String(e) });
      return null;
    }
  },

  appendLine: (displayLine: string, rawLine?: string) => {
    set((s) => {
      const next = [...s.lines, displayLine];
      const trimmed = next.length > MAX_LINES ? next.slice(-MAX_LINES) : next;
      const raw = rawLine !== undefined ? s.rawBuffer + rawLine + '\n' : s.rawBuffer;
      const trimmedRaw = raw.length > MAX_RAW ? raw.slice(-MAX_RAW) : raw;
      return {
        lines: trimmed,
        rawBuffer: trimmedRaw,
        parsedLog: parseAgentLog(trimmedRaw),
      };
    });
  },

  appendBackfill: (src: string, path: string, content: string, truncated: boolean) => {
    set((s) => {
      // A backfill replay carries the session's full snapshot. If it arrives
      // while the buffer already holds live-tailed content (e.g. after an SSE
      // reconnect), reset first — otherwise everything would be duplicated.
      const fresh = !s.backfilling;
      const baseBuffer = fresh ? '' : s.rawBuffer;
      const baseLines = fresh ? [] : s.lines;
      // Split the snapshotted content into display lines and raw feed.
      const header = `── ${src} · ${path}${truncated ? ' (truncated)' : ''} ──`;
      const contentLines = content ? content.split('\n') : [];
      const display = [header, ...contentLines];
      const next = [...baseLines, ...display];
      const trimmed = next.length > MAX_LINES ? next.slice(-MAX_LINES) : next;
      const raw = baseBuffer.length === 0 ? content : baseBuffer + '\n' + content;
      const trimmedRaw = raw.length > MAX_RAW ? raw.slice(-MAX_RAW) : raw;
      return {
        lines: trimmed,
        rawBuffer: trimmedRaw,
        parsedLog: parseAgentLog(trimmedRaw),
        backfilling: true,
      };
    });
  },

  finishBackfill: (count: number) => {
    set({ backfilling: false, backfillCount: count });
  },

  clearLines: () => set({ lines: [], rawBuffer: '', parsedLog: EMPTY_PARSED_LOG, backfilling: false, backfillCount: 0 }),

  setStatus: (status, error) => {
    set({ status, ...(error !== undefined ? { lastError: error } : {}) });
  },

  getParsedLog: () => get().parsedLog,

  applyFileTree: (root, tree) => {
    set((s) => ({
      fileTreeRoot: root,
      fileTree: tree,
      fileNestedTree: buildNestedTree(tree),
      fileTreeVersion: s.fileTreeVersion + 1,
    }));
  },

  applyFileChanges: (added, removed, modified) => {
    set((s) => {
      const map = new Map<string, FileNode>();
      for (const n of s.fileTree) map.set(n.path, n);

      for (const n of added) map.set(n.path, n);
      for (const n of modified) map.set(n.path, n);
      for (const n of removed) {
        // Remove the node and any descendants (paths under it)
        const prefix = n.path + '/';
        for (const key of map.keys()) {
          if (key === n.path || key.startsWith(prefix)) {
            map.delete(key);
          }
        }
      }

      const tree = Array.from(map.values()).sort((a, b) => a.path.localeCompare(b.path));
      return {
        fileTree: tree,
        fileNestedTree: buildNestedTree(tree),
        fileTreeVersion: s.fileTreeVersion + 1,
      };
    });
  },

  refetchFileTree: async () => {
    const { token, sessionId } = get();
    if (!token || !sessionId) return;
    try {
      const res = await gatewayFetchFiles(token, sessionId);
      get().applyFileTree(res.root || '', res.tree || []);
    } catch { /* silent */ }
  },

  clearFileTree: () => set({ fileTreeRoot: '', fileTree: [], fileNestedTree: [], fileTreeVersion: 0 }),
}));

// 调试句柄：控制台可用 __gws.getState() 检查 rawBuffer / parsedLog。
if (typeof window !== 'undefined') {
  (window as unknown as Record<string, unknown>).__gws = useGatewayStore;
}

// Build a nested tree from a flat list of FileNode (relative slash-delimited paths).
// Dir nodes are inferred from path prefixes even if not explicitly in the list.
function buildNestedTree(flat: FileNode[]): FileTreeNode[] {
  const nodeMap = new Map<string, FileTreeNode>();

  // First pass: create nodes for all explicit entries.
  for (const n of flat) {
    nodeMap.set(n.path, {
      name: n.name,
      path: n.path,
      type: n.type,
      size: n.size,
      mtime: n.mtime,
      children: n.type === 'dir' ? [] : undefined,
    });
  }

  // Second pass: ensure parent dir nodes exist for all paths.
  for (const n of flat) {
    const parts = n.path.split('/');
    let current = '';
    for (let i = 0; i < parts.length - 1; i++) {
      current = current ? current + '/' + parts[i] : parts[i];
      if (!nodeMap.has(current)) {
        nodeMap.set(current, {
          name: parts[i],
          path: current,
          type: 'dir',
          size: 0,
          mtime: 0,
          children: [],
        });
      }
    }
  }

  // Third pass: link children to parents.
  const roots: FileTreeNode[] = [];
  for (const node of nodeMap.values()) {
    const parts = node.path.split('/');
    if (parts.length === 1) {
      roots.push(node);
    } else {
      const parentPath = parts.slice(0, -1).join('/');
      const parent = nodeMap.get(parentPath);
      if (parent) {
        if (!parent.children) parent.children = [];
        parent.children.push(node);
      } else {
        roots.push(node);
      }
    }
  }

  // Sort: dirs first, then files, alphabetically.
  const sortNodes = (nodes: FileTreeNode[]) => {
    nodes.sort((a, b) => {
      if (a.type === 'dir' && b.type !== 'dir') return -1;
      if (a.type !== 'dir' && b.type === 'dir') return 1;
      return a.name.localeCompare(b.name);
    });
    for (const n of nodes) {
      if (n.children) sortNodes(n.children);
    }
  };
  sortNodes(roots);

  return roots;
}