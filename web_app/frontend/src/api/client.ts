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
    throw new Error((message || `HTTP ${res.status}`).slice(0, 300));
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

export type AuthValidation = 'valid' | 'invalid' | 'unreachable';

// validateAuth re-checks the cached token against the gateway. Distinguishes
// a rejected token ('invalid' → the session must be dropped) from a gateway
// that is merely unreachable ('unreachable' → keep the 7-day cache entry; a
// transient outage or a wrong address must not burn the session).
export async function validateAuth(): Promise<AuthValidation> {
  const token = _getAuthToken();
  if (!token) return 'invalid';
  let res;
  try {
    res = await Proxy({
      method: 'GET',
      path: '/sources',
      headers: { Authorization: `Bearer ${token}` },
    });
  } catch {
    return 'unreachable';
  }
  if (res.status === 401 || res.status === 403) return 'invalid';
  if (res.status >= 500) return 'unreachable';
  if (res.status < 200 || res.status >= 300) return 'invalid';
  // A valid gateway answers with {"sources": [...]} (older builds: a bare
  // array). HTML bodies (SPA fallback, reverse proxy page) mean the address
  // is not the log gateway.
  try {
    const data = JSON.parse(res.body || '{}');
    if (Array.isArray(data)) return 'valid';
    if (data && typeof data === 'object' && Array.isArray((data as { sources?: unknown }).sources)) {
      return 'valid';
    }
  } catch {
    /* not JSON */
  }
  return 'invalid';
}

// InvalidCredentialsError marks a login rejection (HTTP 401/403). The UI maps
// it to a single unified message instead of echoing the gateway text.
export class InvalidCredentialsError extends Error {
  constructor() {
    super('invalid credentials');
    this.name = 'InvalidCredentialsError';
  }
}

export async function login(username: string, password: string): Promise<{ token: string; user: AuthUser }> {
  const res = await Proxy({
    method: 'POST',
    path: '/login',
    body: JSON.stringify({ user_name: username, password }),
    headers: { 'Content-Type': 'application/json' },
  });
  if (res.status === 401 || res.status === 403) {
    throw new InvalidCredentialsError();
  }
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
    throw new Error(message.slice(0, 300));
  }
  let data: { token?: string; expires_at?: string; user?: string; role?: string };
  try {
    data = JSON.parse(res.body || '{}');
  } catch {
    throw new Error(
      `Unexpected non-JSON response (HTTP ${res.status}): ${(res.body || '(empty)').slice(0, 200)}`,
    );
  }
  if (!data.token) {
    throw new Error(`Login response is missing a token: ${(res.body || '').slice(0, 200)}`);
  }
  const user: AuthUser = {
    username: data.user || username,
    display_name: data.user || username,
    created_at: data.expires_at || '',
    role: data.role || 'user',
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