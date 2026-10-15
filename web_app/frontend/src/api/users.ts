import { Proxy } from '../../bindings/scienceflow/app';
import { loadAuth } from './client';

// Gateway account-management API (super admin only), see
// doc/stage2/user_admin.md: GET/POST /users, PUT/DELETE /users/{name}.

export interface GatewayUserInfo {
  name: string;
  role: string;
  admin: boolean;
  sources: string[] | null;
}

export interface CreateUserPayload {
  name: string;
  password: string;
  sources?: string[] | null;
  role?: string;
}

export interface UpdateUserPayload {
  password?: string;
  sources?: string[] | null;
  role?: string;
}

function authToken(): string {
  const auth = loadAuth();
  if (!auth?.token) throw new Error('not authenticated');
  return auth.token;
}

function errorDetail(body: string): string {
  const text = (body || '').trim();
  if (!text) return '';
  try {
    const parsed = JSON.parse(text);
    return parsed.error || parsed.detail || text;
  } catch {
    return text;
  }
}

async function usersRequest<T>(
  method: string,
  path: string,
  body?: unknown,
): Promise<T> {
  const res = await Proxy({
    method,
    path,
    body: body === undefined ? '' : JSON.stringify(body),
    headers: {
      Authorization: `Bearer ${authToken()}`,
      'Content-Type': 'application/json',
    },
  });
  if (res.status === 401) {
    throw new Error('用户不存在或密码错误');
  }
  if (res.status === 403) {
    throw new Error(errorDetail(res.body) || '需要超级管理员权限');
  }
  if (res.status < 200 || res.status >= 300) {
    throw new Error(errorDetail(res.body) || `HTTP ${res.status}`);
  }
  if (res.status === 204 || !res.body) return {} as T;
  return JSON.parse(res.body) as T;
}

export async function listUsers(): Promise<GatewayUserInfo[]> {
  const data = await usersRequest<{ users: GatewayUserInfo[] }>('GET', '/users');
  return data.users || [];
}

export async function createUser(payload: CreateUserPayload): Promise<GatewayUserInfo> {
  const data = await usersRequest<{ user: GatewayUserInfo }>('POST', '/users', {
    name: payload.name,
    password: payload.password,
    sources: payload.sources ?? null,
    role: payload.role || 'user',
  });
  return data.user;
}

export async function updateUser(name: string, payload: UpdateUserPayload): Promise<GatewayUserInfo> {
  const data = await usersRequest<{ user: GatewayUserInfo }>(
    'PUT',
    `/users/${encodeURIComponent(name)}`,
    payload,
  );
  return data.user;
}

export async function deleteUser(name: string): Promise<void> {
  await usersRequest('DELETE', `/users/${encodeURIComponent(name)}`);
}
