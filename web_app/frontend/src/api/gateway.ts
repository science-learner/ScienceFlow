import {
  GatewayLogin,
  GatewayListSources,
  GatewayCreateSession,
  GatewayGetSession,
  GatewayUpdateSessionSources,
  GatewayDeleteSession,
  GatewayListSessions,
  GatewayActivateSession,
  GatewayReadSnapshot,
  GatewayStartStream,
  GatewayStopStream,
  GatewayInvokeAgent,
  GatewayAgentStatus,
  GatewayStopAgent,
  GatewayTestConnection,
  GatewayFetchFiles,
  GatewayFetchMonitor,
  GatewayListModels,
  GatewayCreateModel,
  GatewayActivateModel,
  GatewayGetModelStages,
  GatewaySetModelStages,
  GatewayUpdateModel,
  GatewayDeleteModel,
  WorkspaceDownload,
  WorkspaceUploadFiles,
  SetGatewayURL,
  GetSystemStats,
} from '../../bindings/scienceflow/app';
import * as wailsModels from '../../bindings/scienceflow/models';
import type { FileTreeResponse, FileNode } from '../types';

export interface FilePayloadResult {
  filename: string;
  contentType: string;
  data: string;
}

export interface WorkspaceUploadResult {
  root: string;
  uploaded: { path: string; size: number }[];
}

// TypeScript mirrors of the gateway_server contract (see
// log_gateway_server/docs/gateway_server_dev.md).

export interface GatewayLoginResult {
  token: string;
  expires_at: string;
  user: string;
  sources: string[];
}

export interface GatewaySourceInfo {
  name: string;
  files: string[];
}

export interface GatewaySession {
  session_id: string;
  name?: string; // 空表示未命名（还没发过提问），首次提问后为提问首行
  user?: string;
  sources: string[];
  last_active?: string;
  last_active_nanos?: number;
  agent?: {
    status: string;
    task?: GatewayTaskSnapshot | null;
  } | null;
}

export interface GatewaySessionsResponse {
  user: string;
  sessions: GatewaySession[];
}

export interface GatewaySnapshot {
  path?: string;
  size?: number;
  returned?: number;
  truncated?: boolean;
  content?: string;
  source?: string;
  files?: string[];
}

export interface GatewayLogEvent {
  session_id?: string;
  input: string;
  file: string;
  path: string;
  message?: string;
  data?: string;
  encoding?: string;
  mode?: string;
  timestamp: string;
  offset: number;
}

export interface GatewayStatusEvent {
  status: 'connecting' | 'connected' | 'disconnected' | 'unauthorized';
  error?: string;
}

export interface GatewayBackfillEvent {
  session_id?: string;
  source: string;
  path: string;
  size: number;
  returned: number;
  truncated: boolean;
  content: string;
}

export interface GatewayBackfillDoneEvent {
  session_id?: string;
  count: number;
}

export function gatewayLogin(userName: string, password: string): Promise<GatewayLoginResult> {
  return GatewayLogin({ user_name: userName, password });
}

// gatewayTestConnection probes the gateway server itself (GET /healthz); it
// does not touch the ScienceFlow backend behind the gateway.
export type GatewayHealth = wailsModels.GatewayHealth;

export function gatewayTestConnection(): Promise<GatewayHealth> {
  return GatewayTestConnection();
}

export function gatewayListSources(token: string): Promise<GatewaySourceInfo[]> {
  return GatewayListSources(token);
}

export function gatewayCreateSession(token: string, sources: string[]): Promise<GatewaySession> {
  return GatewayCreateSession(token, sources);
}

export function gatewayGetSession(token: string, sessionId: string): Promise<GatewaySession> {
  return GatewayGetSession(token, sessionId);
}

export function gatewayUpdateSessionSources(token: string, sessionId: string, sources: string[]): Promise<GatewaySession> {
  return GatewayUpdateSessionSources(token, sessionId, sources);
}

export function gatewayDeleteSession(token: string, sessionId: string): Promise<void> {
  return GatewayDeleteSession(token, sessionId);
}

export function gatewayListSessions(token: string): Promise<GatewaySessionsResponse> {
  return GatewayListSessions(token) as Promise<GatewaySessionsResponse>;
}

export function gatewayActivateSession(token: string, sessionId: string): Promise<GatewaySession> {
  return GatewayActivateSession(token, sessionId) as Promise<GatewaySession>;
}

export function gatewayReadSnapshot(token: string, opts: { path?: string; source?: string }): Promise<GatewaySnapshot> {
  return GatewayReadSnapshot(token, opts.path || '', opts.source || '');
}

export function gatewayStartStream(sessionId: string): void {
  GatewayStartStream(sessionId);
}

export function gatewayStopStream(): void {
  GatewayStopStream();
}

// ── Agent invoke / status / stop ──

export type GatewayTaskSnapshot = wailsModels.GatewayTaskSnapshot;
export type GatewayQueueStats = wailsModels.GatewayQueueStats;
export type GatewayAgentStatusResult = wailsModels.GatewayAgentStatus;

export function gatewayInvokeAgent(token: string, sessionId: string, query: string, mode?: 'lite' | 'heavy'): Promise<GatewayTaskSnapshot> {
  return GatewayInvokeAgent(token, sessionId, { query, mode: mode || '' });
}

export function gatewayAgentStatus(token: string, sessionId: string): Promise<GatewayAgentStatusResult> {
  return GatewayAgentStatus(token, sessionId);
}

export function gatewayStopAgent(token: string, sessionId: string): Promise<GatewayTaskSnapshot> {
  return GatewayStopAgent(token, sessionId);
}

// ── File tree sync ──

export function gatewayFetchFiles(token: string, sessionId: string): Promise<FileTreeResponse> {
  return GatewayFetchFiles(token, sessionId) as Promise<FileTreeResponse>;
}

export type GatewayMonitorMetrics = wailsModels.GatewayMonitorMetrics;
export type GatewayModelInfo = wailsModels.GatewayModelInfo;

export function gatewayListModels(token: string): Promise<GatewayModelInfo[]> {
  return GatewayListModels(token) as Promise<GatewayModelInfo[]>;
}

export function gatewayCreateModel(
  token: string,
  req: { model_name: string; api_key: string; api_url: string },
): Promise<GatewayModelInfo> {
  return GatewayCreateModel(token, req as wailsModels.GatewayModelCreateRequest) as Promise<GatewayModelInfo>;
}

export function gatewayUpdateModel(
  token: string,
  modelId: string,
  req: { model_name: string; api_key: string; api_url: string },
): Promise<GatewayModelInfo> {
  return GatewayUpdateModel(token, modelId, req as wailsModels.GatewayModelUpdateRequest) as Promise<GatewayModelInfo>;
}

export function gatewayDeleteModel(token: string, modelId: string): Promise<void> {
  return GatewayDeleteModel(token, modelId);
}

export function gatewayActivateModel(token: string, modelId: string, sessionId: string): Promise<GatewayModelInfo> {
  return GatewayActivateModel(token, modelId, sessionId) as Promise<GatewayModelInfo>;
}

// Stage model selection ("coder" / "feedbacker"), resolved per session.
// An empty id means the stage follows the session's main model.
export interface GatewayModelStages {
  coder_model_id: string;
  feedback_model_id: string;
}

export function gatewayGetModelStages(token: string, sessionId: string): Promise<GatewayModelStages> {
  return GatewayGetModelStages(token, sessionId) as Promise<GatewayModelStages>;
}

export function gatewaySetModelStages(
  token: string,
  sessionId: string,
  coderModelId: string,
  feedbackModelId: string,
): Promise<GatewayModelStages> {
  return GatewaySetModelStages(token, sessionId, coderModelId, feedbackModelId) as Promise<GatewayModelStages>;
}

export function gatewayFetchMonitor(token: string, sessionId: string): Promise<GatewayMonitorMetrics> {
  return GatewayFetchMonitor(token, sessionId) as Promise<GatewayMonitorMetrics>;
}

// SystemStats mirrors the desktop app's local-machine resource snapshot
// (CPU load, memory and disk usage of the machine running ScienceFlow).
export interface SystemStats {
  cpu_percent: number;
  memory_used_bytes: number;
  memory_total_bytes: number;
  memory_percent: number;
  disk_used_bytes: number;
  disk_total_bytes: number;
  disk_percent: number;
  disk_path: string;
}

// getSystemStats fetches local-machine stats via the Wails bridge. Returns
// null when the bridge is unavailable (e.g. plain vite dev in a browser).
export async function getSystemStats(): Promise<SystemStats | null> {
  const bridge = (window as any)?.go?.main?.App?.GetSystemStats;
  if (typeof bridge !== 'function') return null;
  try {
    return (await GetSystemStats()) as SystemStats;
  } catch {
    return null;
  }
}

export type { FileNode };

// ── Workspace file/directory download/upload ──

function base64ToBlob(data: string, contentType: string): Blob {
  const bin = atob(data);
  const len = bin.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type: contentType || 'application/octet-stream' });
}

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      const base64 = result.split(',')[1] || result;
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

export async function workspaceDownloadFile(token: string, sessionId: string, path: string): Promise<Blob> {
  const payload = await WorkspaceDownload(token, sessionId, path, false);
  return base64ToBlob(payload.data, payload.contentType);
}

// workspaceReadText downloads a workspace file and decodes it as UTF-8 text.
// Used by the monitor to parse telemetry files (.logs/*.jsonl, *.csv).
export async function workspaceReadText(token: string, sessionId: string, path: string): Promise<string> {
  const blob = await workspaceDownloadFile(token, sessionId, path);
  return blob.text();
}

export async function workspaceDownloadDir(token: string, sessionId: string, path: string): Promise<Blob> {
  const payload = await WorkspaceDownload(token, sessionId, path, true);
  return base64ToBlob(payload.data, payload.contentType);
}

export async function workspaceUploadFiles(
  token: string,
  sessionId: string,
  targetDir: string,
  files: File[],
): Promise<WorkspaceUploadResult> {
  const uploadedFiles: { filename: string; data: string }[] = [];
  for (const f of files) {
    const data = await fileToBase64(f);
    uploadedFiles.push({ filename: (f as any).webkitRelativePath || f.name, data });
  }
  return WorkspaceUploadFiles(token, sessionId, targetDir, uploadedFiles) as Promise<WorkspaceUploadResult>;
}

export function setGatewayURL(url: string): void {
  void SetGatewayURL(url || '');
}

// Auth failures are signalled by the gateway with HTTP 401 and the fixed body
// "用户不存在或密码错误"; the Go layer surfaces that exact message as the error
// text, so we can detect it reliably.
export function isGatewayAuthError(err: unknown): boolean {
  const msg = err instanceof Error ? err.message : String(err);
  return msg.includes('用户不存在或密码错误');
}

// Decode a gateway log event into a single display line.
export function formatGatewayLogEvent(e: GatewayLogEvent): string {
  const ts = e.timestamp ? e.timestamp.slice(11, 23) : '';
  if (e.encoding === 'base64') {
    return `[${ts}] ${e.file} (${e.mode}) [base64 ${(e.data || '').length} chars]`;
  }
  return `[${ts}] ${e.file} (${e.mode}) ${e.message || ''}`;
}