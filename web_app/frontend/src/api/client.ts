import type {
  ScienceFlowState,
  StatePatch,
  AuthUser,
} from '../types';

import { ListThemeBackgrounds, Proxy, ReadThemeBackground } from '../../bindings/scienceflow/app';

import { debug } from '../utils/debug';

// ── Client id ──

function getClientId(): string {
  let id = localStorage.getItem('scienceflow.clientId');
  if (!id) {
    id = `web-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
    localStorage.setItem('scienceflow.clientId', id);
  }
  return id;
}

// ── Auth token ──

const AUTH_KEY = 'scienceflow.auth';
const AUTH_EXPIRE_DAYS = 7;

interface StoredAuth {
  token: string;
  user: AuthUser;
  expires_at: number;
}

function _getAuthToken(): string | null {
  try {
    const raw = localStorage.getItem(AUTH_KEY);
    if (!raw) return null;
    const stored: StoredAuth = JSON.parse(raw);
    if (Date.now() > stored.expires_at) {
      localStorage.removeItem(AUTH_KEY);
      return null;
    }
    return stored.token;
  } catch {
    return null;
  }
}

export function saveAuth(token: string, user: AuthUser): void {
  const stored: StoredAuth = {
    token,
    user,
    expires_at: Date.now() + AUTH_EXPIRE_DAYS * 86400 * 1000,
  };
  localStorage.setItem(AUTH_KEY, JSON.stringify(stored));
}

export function loadAuth(): { token: string; user: AuthUser } | null {
  try {
    const raw = localStorage.getItem(AUTH_KEY);
    if (!raw) return null;
    const stored: StoredAuth = JSON.parse(raw);
    if (Date.now() > stored.expires_at) {
      localStorage.removeItem(AUTH_KEY);
      return null;
    }
    return { token: stored.token, user: stored.user };
  } catch {
    return null;
  }
}

export function clearAuth(): void {
  localStorage.removeItem(AUTH_KEY);
}

// ── Generic JSON/text proxy request ──

interface ProxyResult {
  notModified?: boolean;
  etag?: string;
  [key: string]: unknown;
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T & ProxyResult> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'X-Scienceflow-Client-Id': getClientId(),
    ...((options.headers as Record<string, string>) || {}),
  };

  const token = _getAuthToken();
  if (token && (path.startsWith('/api/') || path.startsWith('/api'))) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const method = (options.method || 'GET').toUpperCase();
  const body =
    options.body === undefined || options.body === null
      ? ''
      : typeof options.body === 'string'
        ? options.body
        : JSON.stringify(options.body);

  const res = await Proxy({ method, path, body, headers });

  if (res.notModified || res.status === 304) {
    return { notModified: true, etag: res.etag || '' } as unknown as T & ProxyResult;
  }

  if (res.status < 200 || res.status >= 300) {
    const bodyText = res.body || '';
    const message = bodyText
      ? (() => {
          try {
            return JSON.parse(bodyText).error || bodyText;
          } catch {
            return bodyText;
          }
        })()
      : '';
    throw new Error(message || `HTTP ${res.status}`);
  }

  const data = res.body ? JSON.parse(res.body) : {};
  data.etag = res.etag || '';
  if (res.transport) {
    data._transport_client = res.transport;
  }
  return data as T & ProxyResult;
}

function jsonParams(params: Record<string, string | number | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== '') search.set(key, String(value));
  }
  const qs = search.toString();
  return qs ? `?${qs}` : '';
}

// ── State ──
export async function fetchState(options: {
  compact?: boolean;
  force?: boolean;
  detailNodeId?: string;
  budget?: Record<string, number>;
  etags?: Record<string, string>;
  stateSig?: string;
  taskRoot?: string;
} = {}): Promise<{ state: ScienceFlowState; state_signature?: string; module_etags?: Record<string, string> }> {
  const params: Record<string, string | number> = {};
  if (options.compact !== false) params.compact = '1';
  if (options.detailNodeId) params.detail_node_id = options.detailNodeId;
  if (options.budget) {
    for (const [key, value] of Object.entries(options.budget)) {
      params[key] = value;
    }
  }
  if (options.taskRoot) params.task_root = options.taskRoot;
  const qs = jsonParams(params);
  return request<{
    state: ScienceFlowState;
    state_signature?: string;
    module_etags?: Record<string, string>;
  }>(`/api/state${qs}`);
}

export async function fetchStatePatch(options: {
  compact?: boolean;
  detailNodeId?: string;
  budget?: Record<string, number>;
  etags?: Record<string, string>;
  stateSig?: string;
  taskRoot?: string;
} = {}): Promise<{ state_patch?: StatePatch }> {
  const params: Record<string, string | number> = {};
  params.compact = '1';
  if (options.detailNodeId) params.detail_node_id = options.detailNodeId;
  if (options.budget) {
    for (const [key, value] of Object.entries(options.budget)) {
      params[key] = value;
    }
  }
  if (options.taskRoot) params.task_root = options.taskRoot;
  if (options.stateSig) params.state_sig = options.stateSig;
  const qs = jsonParams(params);
  return request<{ state_patch?: StatePatch }>(`/api/state/patch${qs}`);
}

// ── Workspace ──
export async function fetchWorkspaceFileContent(path: string, taskRoot?: string, sessionId?: string): Promise<{ content: string; content_type: string; encoding?: string }> {
  const params = new URLSearchParams({ path });
  if (taskRoot) params.set('task_root', taskRoot);
  if (sessionId) params.set('session_id', sessionId);
  return request(`/api/workspace/file/content?${params.toString()}`);
}

// ── Decisions ──
export async function resolveDecision(decisionId: string, choice: string): Promise<{ ok: boolean }> {
  return request(`/api/decisions/${encodeURIComponent(decisionId)}`, {
    method: 'POST',
    body: JSON.stringify({ choice }),
  });
}

// ── Reports ──
export async function fetchReports(taskRoot?: string, sessionId?: string): Promise<{ reports: { path: string; filename: string; relative_path: string; title: string; created_at: string }[] }> {
  const params = new URLSearchParams();
  if (taskRoot) params.set('task_root', taskRoot);
  if (sessionId) params.set('session_id', sessionId);
  const qs = params.toString();
  return request(`/api/reports${qs ? '?' + qs : ''}`);
}

export async function fetchReportContent(reportPath: string, taskRoot?: string, sessionId?: string): Promise<{ content: string; content_type?: string }> {
  const params = new URLSearchParams({ path: reportPath });
  if (taskRoot) params.set('task_root', taskRoot);
  if (sessionId) params.set('session_id', sessionId);
  return request(`/api/reports/content?${params.toString()}`);
}

// ── Auth (gateway_server /login) ──

export async function validateAuth(): Promise<boolean> {
  const token = _getAuthToken();
  if (!token) return false;
  try {
    const res = await Proxy({
      method: 'GET',
      path: '/sources',
      headers: { Authorization: `Bearer ${token}` },
    });
    return res.status >= 200 && res.status < 300;
  } catch {
    return false;
  }
}

export async function login(username: string, password: string): Promise<{ token: string; user: AuthUser }> {
  const res = await Proxy({
    method: 'POST',
    path: '/login',
    body: JSON.stringify({ user_name: username, password }),
    headers: { 'Content-Type': 'application/json' },
  });
  if (res.status < 200 || res.status >= 300) {
    const bodyText = res.body || '';
    const message = bodyText
      ? (() => {
          try {
            return JSON.parse(bodyText).detail || bodyText;
          } catch {
            return bodyText;
          }
        })()
      : 'Login failed';
    throw new Error(message);
  }
  const data = JSON.parse(res.body || '{}');
  const user: AuthUser = {
    username: data.user || username,
    display_name: data.user || username,
    created_at: data.expires_at || '',
  };
  saveAuth(data.token, user);
  return { token: data.token, user };
}

// ── Theme backgrounds (<program dir>/themes/backgrounds) ──

export async function fetchThemeBackgrounds(): Promise<string[]> {
  try {
    return await ListThemeBackgrounds();
  } catch {
    return [];
  }
}

export async function fetchThemeBackgroundData(name: string): Promise<string> {
  if (!name) return '';
  try {
    return await ReadThemeBackground(name);
  } catch {
    return '';
  }
}