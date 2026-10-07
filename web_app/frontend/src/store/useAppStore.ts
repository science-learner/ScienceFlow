import { create } from 'zustand';
import type {
  ScienceFlowState,
  View,
  FrontTab,
  L1Tab,
  Theme,
  FontSize,
  L1Scope,
  ChatRunState,
  ChatRouteMode,
  ChatMessage,
  ChatToolEvent,
} from '../types';
import type { Lang } from '../i18n/translations';
import * as api from '../api/client';
import { gatewayFetchFiles, workspaceDownloadFile } from '../api/gateway';
import { useGatewayStore } from './useGatewayStore';

type ReportListItem = {
  path: string;
  filename: string;
  relative_path: string;
  title: string;
  created_at: string;
  session_id?: string;
  session_label?: string;
  task_root?: string;
  report_key: string;
  source?: 'api' | 'gateway';
};

interface AppState {
  // Theme & layout
  theme: Theme;
  fontSize: FontSize;
  // Agent Map background image file name from <program dir>/themes/backgrounds
  // ('' = bundled default background).
  agentMapBackground: string;
  language: Lang;
  setTheme: (theme: Theme) => void;
  setFontSize: (size: FontSize) => void;
  setAgentMapBackground: (name: string) => void;
  setLanguage: (lang: Lang) => void;

  // View
  currentView: View;
  frontTab: FrontTab;
  l1Tab: L1Tab;
  setView: (view: View) => void;
  setFrontTab: (tab: FrontTab) => void;
  setL1Tab: (tab: L1Tab) => void;

  // Scope
  l1Scope: L1Scope;
  selectedRunIndex: number;
  selectedNodeIndex: number;
  setL1Scope: (scope: L1Scope) => void;
  selectRun: (index: number) => void;
  selectNode: (index: number) => void;

  // State
  currentState: ScienceFlowState | null;
  stateSignature: string;
  moduleEtags: Map<string, string>;
  transportMetrics: Record<string, unknown> | null;
  stateStablePolls: number;

  // Actions
  refreshState: (options?: { force?: boolean; compact?: boolean }) => Promise<void>;
  applyStatePatch: (patch: { modules?: Record<string, unknown>; signature?: string; module_etags?: Record<string, string> }) => void;

  // Chat
  chatSessionId: string;
  chatMessages: ChatMessage[];
  chatBusy: boolean;
  chatQueueDepth: number;
  chatSendInFlight: boolean;
  chatWarmup: boolean;
  chatRunState: ChatRunState;
  chatRouteMode: ChatRouteMode;
  chatProgressLabel: string;
  toolEvents: ChatToolEvent[];
  sessionsPanelOpen: boolean;
  agentPositionId: string;

  clearTimeline: () => void;

  setChatSessionId: (id: string) => void;
  setChatMessages: (messages: ChatMessage[]) => void;
  addChatMessage: (message: ChatMessage) => void;
  setChatBusy: (busy: boolean) => void;
  setChatRunState: (state: ChatRunState) => void;
  setChatRouteMode: (mode: ChatRouteMode) => void;
  setSessionsPanelOpen: (open: boolean) => void;

  // Settings
  settingsPanelOpen: boolean;
  statePanelOpen: boolean;

  setSettingsPanelOpen: (open: boolean) => void;
  setStatePanelOpen: (open: boolean) => void;

  // Batch panel
  batchPanelCollapsed: boolean;
  setBatchPanelCollapsed: (collapsed: boolean) => void;

  // Monitor panel
  monitorCollapsed: boolean;
  setMonitorCollapsed: (collapsed: boolean) => void;

  // Workspace
  selectedWorkspacePath: string;
  workspaceTreeCollapsed: boolean;

  setSelectedWorkspacePath: (path: string) => void;
  setWorkspaceTreeCollapsed: (collapsed: boolean) => void;

  // Reports (recursively scan task_root for .md files)
  reportList: ReportListItem[];
  selectedReportPath: string;
  reportContent: string;
  fetchReports: () => Promise<void>;
  fetchReportContent: (path: string) => Promise<void>;
  clearReportState: () => void;

  // Agent map
  selectedAgentWorkerId: string;
  selectAgentMapWorker: (workerId: string) => void;

  // Caches
  workspaceFileTextCache: Map<string, { content: string; contentType: string; encoding: string }>;

  // Derived
  activeTaskRoot: () => string;
}

// Reads the persisted Agent Map background. Legacy built-in values
// ('default' / 'lab') migrate to '' (bundled default); anything else is a
// file name from <program dir>/themes/backgrounds.
function storedAgentMapBackground(): string {
  const value = localStorage.getItem('scienceflow.agentMapBackground') || '';
  return value === 'default' || value === 'lab' ? '' : value;
}

export const useAppStore = create<AppState>((set, get) => ({
  // Theme & layout
  theme: (localStorage.getItem('scienceflow.theme') as Theme) || 'scienceflow-dark',
  fontSize: (localStorage.getItem('scienceflow.fontSize') as FontSize) || 'default',
  agentMapBackground: storedAgentMapBackground(),
  language: (localStorage.getItem('scienceflow.language') as Lang) || 'zh-CN',

  setTheme: (theme) => {
    localStorage.setItem('scienceflow.theme', theme);
    document.documentElement.dataset.theme = theme;
    set({ theme });
  },
  setFontSize: (size) => {
    localStorage.setItem('scienceflow.fontSize', size);
    document.documentElement.dataset.fontSize = size;
    set({ fontSize: size });
  },
  setAgentMapBackground: (name) => {
    localStorage.setItem('scienceflow.agentMapBackground', name);
    set({ agentMapBackground: name });
  },
  setLanguage: (lang) => {
    localStorage.setItem('scienceflow.language', lang);
    set({ language: lang });
  },

  // View
  currentView: 'l0',
  frontTab: 'agent-map',
  l1Tab: 'workspace',
  setView: (view) => set({ currentView: view }),
  setFrontTab: (tab) => set({ frontTab: tab }),
  setL1Tab: (tab) => set({ l1Tab: tab }),

  // Scope
  l1Scope: 'task',
  selectedRunIndex: 0,
  selectedNodeIndex: 0,
  setL1Scope: (scope) => set({ l1Scope: scope }),
  selectRun: (index) => set({ selectedRunIndex: Math.max(0, index) }),
  selectNode: (index) => set({ selectedNodeIndex: Math.max(0, index) }),

  // State
  currentState: null,
  stateSignature: '',
  moduleEtags: new Map(),
  transportMetrics: null,
  stateStablePolls: 0,

  refreshState: async (options = {}) => {
    const { force, compact } = options;
    const taskRoot = get().activeTaskRoot();
    const usePatch = Boolean(get().currentState) && !force;

    if (usePatch) {
      const stateSig = get().stateSignature;
      const etags: Record<string, string> = {};
      get().moduleEtags.forEach((v, k) => { etags[k] = v; });
      try {
        const payload = await api.fetchStatePatch({ stateSig, taskRoot: taskRoot || undefined });
        if ('state_patch' in payload && payload.state_patch) {
          get().applyStatePatch(payload.state_patch);
          set({ stateStablePolls: get().stateStablePolls + (payload.state_patch.not_modified ? 1 : 0) });
        }
      } catch { /* silent */ }
    } else {
      try {
        const payload = await api.fetchState({
          compact: compact !== false,
          taskRoot: taskRoot || undefined,
        });
        if ('state' in payload && payload.state) {
          set({ currentState: payload.state, stateStablePolls: 0 });
          if (payload.state_signature) set({ stateSignature: payload.state_signature });
          if (payload.module_etags) {
            const m = new Map<string, string>();
            Object.entries(payload.module_etags).forEach(([k, v]) => m.set(k, v));
            set({ moduleEtags: m });
          }
        }
      } catch { /* silent */ }
    }
  },

  applyStatePatch: (patch) => {
    if (!patch.modules || Object.keys(patch.modules).length === 0) return;
    const state = get().currentState;
    if (!state) return;
    const updated = { ...state };
    for (const [key, value] of Object.entries(patch.modules)) {
      (updated as Record<string, unknown>)[key] = value;
    }
    set({ currentState: updated });
    if (patch.signature) set({ stateSignature: patch.signature });
  },

  // Chat
  chatSessionId: '',
  chatMessages: [],
  chatBusy: false,
  chatQueueDepth: 0,
  chatSendInFlight: false,
  chatWarmup: false,
  chatRunState: 'idle',
  chatRouteMode: 'chat',
  chatProgressLabel: '',
  toolEvents: [],
  sessionsPanelOpen: false,
  agentPositionId: 'coordinator',

  setChatSessionId: (id) => set({ chatSessionId: id, ...(id ? {} : { selectedWorkspacePath: '' }) }),
  setChatMessages: (messages) => set({ chatMessages: messages }),
  addChatMessage: (message) => set((s) => ({ chatMessages: [...s.chatMessages, message] })),
  setChatBusy: (busy) => set({ chatBusy: busy }),
  setChatRunState: (state) => set({ chatRunState: state }),
  setChatRouteMode: (mode) => set({ chatRouteMode: mode }),
  setSessionsPanelOpen: (open) => set({ sessionsPanelOpen: open }),

  // Settings
  settingsPanelOpen: false,
  statePanelOpen: false,

  setSettingsPanelOpen: (open) => set({ settingsPanelOpen: open }),
  setStatePanelOpen: (open) => set({ statePanelOpen: open }),

  // Batch panel
  batchPanelCollapsed: true,
  setBatchPanelCollapsed: (collapsed) => set({ batchPanelCollapsed: collapsed }),

  // Monitor panel
  monitorCollapsed: false,
  setMonitorCollapsed: (collapsed) => set({ monitorCollapsed: collapsed }),

  // Workspace
  selectedWorkspacePath: '',
  workspaceTreeCollapsed: true,

  setSelectedWorkspacePath: (path) => set({ selectedWorkspacePath: path }),
  setWorkspaceTreeCollapsed: (collapsed) => set({ workspaceTreeCollapsed: collapsed }),

  // Reports (current session workspace Markdown files)
  reportList: [] as ReportListItem[],
  selectedReportPath: '',
  reportContent: '',
  fetchReports: async () => {
    try {
      const gatewayState = useGatewayStore.getState();
      const { token } = gatewayState;
      const currentSessionId = gatewayState.sessionId || get().chatSessionId;

      if (!token || !currentSessionId) {
        const result = await api.fetchReports(get().activeTaskRoot() || undefined, get().chatSessionId || undefined).catch(() => ({ reports: [] }));
        const list = (result.reports || []).map((report) => ({
          ...report,
          report_key: `api::${report.path}`,
          source: 'api' as const,
        }));
        set({ reportList: list });
        const selected = list.find((report) => report.report_key === get().selectedReportPath) || list[0];
        if (selected) {
          set({ selectedReportPath: selected.report_key });
          const c = await api.fetchReportContent(selected.path, get().activeTaskRoot() || undefined, get().chatSessionId || undefined);
          set({ reportContent: c.content || '' });
        } else {
          set({ selectedReportPath: '', reportContent: '' });
        }
        return;
      }

      // List every Markdown file in the current session's workspace as its own
      // report entry; the dropdown shows the plain filename with extension.
      const files = await gatewayFetchFiles(token, currentSessionId);
      const mdFiles = (files.tree || [])
        .filter((file) => file.type === 'file' && file.path.toLowerCase().endsWith('.md'))
        .sort((a, b) => a.path.localeCompare(b.path));

      const list: ReportListItem[] = mdFiles.map((file) => {
        const filename = file.path.split('/').pop() || file.path;
        return {
          path: file.path,
          filename,
          relative_path: file.path,
          title: filename,
          created_at: '',
          session_id: currentSessionId,
          session_label: '',
          task_root: files.root || '',
          report_key: `gw::md::${file.path}`,
          source: 'gateway' as const,
        };
      });

      const selected = list.find((report) => report.report_key === get().selectedReportPath) || list[0];
      if (!selected) {
        set({ reportList: [], selectedReportPath: '', reportContent: '' });
        return;
      }

      set({ reportList: list, selectedReportPath: selected.report_key });
      const content = await downloadGatewayFileText(token, currentSessionId, selected.path);
      set({ reportContent: content });
    } catch (e) {
      console.error('[KeyReport] fetchReports failed:', e);
    }
  },
  fetchReportContent: async (path) => {
    try {
      const selected = get().reportList.find((report) => report.report_key === path || report.path === path);
      if (!selected) return;
      if (selected.source === 'gateway' && selected.session_id) {
        const { token } = useGatewayStore.getState();
        if (!token) return;
        const content = await downloadGatewayFileText(token, selected.session_id, selected.path);
        set({ reportContent: content, selectedReportPath: selected.report_key });
        return;
      }
      const c = await api.fetchReportContent(selected.path, selected.task_root || undefined, selected.session_id || undefined);
      set({ reportContent: c.content || '', selectedReportPath: selected.report_key });
    } catch (e) {
      console.error('[KeyReport] fetchReportContent failed:', e);
    }
  },
  clearReportState: () => set({ reportList: [], selectedReportPath: '', reportContent: '' }),

  // Agent map
  selectedAgentWorkerId: '',
  selectAgentMapWorker: (workerId) => set({ selectedAgentWorkerId: workerId }),

  clearTimeline: () => set({ toolEvents: [] }),

  // Caches
  workspaceFileTextCache: new Map(),

  // Derived
  activeTaskRoot: () => {
    const state = get().currentState;
    return state?.task?.task_root || '';
  },
}));

// 调试句柄：控制台可用 __app.getState() 检查 chatMessages 等应用状态。
if (typeof window !== 'undefined') {
  (window as unknown as Record<string, unknown>).__app = useAppStore;
}

// Download a single workspace file from the gateway session and return its text.
async function downloadGatewayFileText(token: string, sessionId: string, path: string): Promise<string> {
  const blob = await workspaceDownloadFile(token, sessionId, path);
  return blob.text();
}
