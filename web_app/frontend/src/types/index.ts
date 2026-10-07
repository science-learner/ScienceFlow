export interface ScienceFlowState {
  schema_version?: string;
  generated_at?: string;
  attached?: boolean;
  task: TaskInfo;
  summary: Summary;
  runs: Run[];
  nodes: NodeInfo[];
  timeline: TimelineEvent[];
  report: ReportInfo | null;
  artifacts: Artifact[];
  monitor: MonitorData;
  commands: Command[];
  jobs: Job[];
  promote: PromoteInfo | null;
  warnings: string[];
  transport?: TransportInfo;
}

export interface TaskInfo {
  task_id: string;
  task_root: string;
  mode: 'lite' | 'heavy' | '';
  dataset_path: string;
  dataset_name: string;
  created_at: string;
  updated_at: string;
}

export interface Summary {
  status: string;
  nodes: number;
  runs: number;
  best_metric: number | null;
  elapsed: string;
}

export interface Run {
  run_id: string;
  status: string;
  best_metric: number | null;
  elapsed: string;
  created_at: string;
}

export interface NodeInfo {
  node_id: string;
  parent_node_id: string;
  reference_node_ids: string[];
  status: string;
  best_metric: number | null;
  elapsed: string;
  stage_count: number;
  readme: string;
  solution: string;
  submission: string;
  metrics: Record<string, number | string>;
  created_at: string;
  updated_at: string;
}

export interface TimelineEvent {
  type: string;
  message: string;
  time: string;
  node_id: string;
}

export interface ReportInfo {
  content?: string;
  path?: string;
  title?: string;
  report_id?: string;
  updated_at?: string;
}

export interface Artifact {
  artifact_id: string;
  title: string;
  path: string;
  node_id: string;
  created_at: string;
  content_type: string;
}

export interface MonitorData {
  runtime: RuntimeInfo;
  cost: CostInfo;
  scheduler: SchedulerInfo;
  resources: ResourcesInfo;
  workers: WorkerInfo[];
  worker_summary: WorkerSummary;
  alerts: Alert[];
  performance: PerformanceInfo;
}

export interface RuntimeInfo {
  status: string;
  started_at: string;
  elapsed: string;
}

export interface CostInfo {
  total: number;
  currency: string;
  breakdown: Record<string, number>;
}

export interface SchedulerInfo {
  queue_depth: number;
  completed: number;
  failed: number;
}

export interface ResourcesInfo {
  cpu: number;
  memory: number;
  disk: number;
}

export interface WorkerInfo {
  worker_id: string;
  status: string;
  progress: number;
  best_metric: number | null;
  last_event_at: string;
  node_count: number;
}

export interface WorkerSummary {
  source: string;
  total: number;
  active: number;
  completed: number;
  failed: number;
  idle: number;
  event_count: number;
  generated_at: string;
  global_best_metric: number | null;
  global_best_stage_id: string;
  global_best_worker_id: string;
}

export interface Alert {
  type: string;
  message: string;
  severity: 'info' | 'warn' | 'error';
  created_at: string;
}

export interface PerformanceInfo {
  latency: number;
  throughput: number;
  samples: PerformanceSample[];
}

export interface PerformanceSample {
  timestamp: string;
  value: number;
  label: string;
}

export interface Command {
  command_id: string;
  text: string;
  status: string;
  created_at: string;
}

export interface Job {
  job_id: string;
  status: string;
  command_id: string;
  launched_at: string;
}

export interface PromoteInfo {
  handoff: string;
  command_id: string;
  job_id: string;
}

export interface TransportInfo {
  timeline_limit: number;
  pipeline_lane_limit: number;
  pipeline_event_limit: number;
  monitor_sample_limit: number;
}

export interface StatePatch {
  signature?: string;
  module_etags?: Record<string, string>;
  not_modified?: boolean;
  modules?: Record<string, unknown>;
}

export interface ChatMessage {
  message_id: string;
  role: 'user' | 'assistant' | 'platform' | 'tool';
  content: string;
  created_at: string;
  tool_events?: ChatToolEvent[];
  decision?: DecisionCard;
  route?: 'agent' | 'chat';
  idempotency_key?: string;
  // Index into the gateway log parser's runs[] when this message came from
  // (or was matched to) a parsed agent run; anchors the execution timeline.
  runIndex?: number;
}

export interface ChatToolEvent {
  event_id: string;
  type: string;
  status: string;
  name: string;
  message: string;
  summary: string;
  error: string;
  path: string;
  title: string;
  created_at: string;
  steps: ChatToolStep[];
}

export interface ChatToolStep {
  step_id: string;
  action: string;
  content: string;
  status: string;
  created_at: string;
}

export interface DecisionCard {
  decision_id: string;
  title: string;
  body: string;
  options: DecisionOption[];
  resolved: boolean;
  expired: boolean;
  created_at: string;
}

export interface DecisionOption {
  key: string;
  label: string;
}

// ── Gateway file-tree types (new SSE-based sync) ──

export type FileNodeType = 'file' | 'dir' | 'symlink';

export interface FileNode {
  name: string;
  path: string;
  type: FileNodeType;
  size: number;
  mtime: number;
}

export interface FileTreeResponse {
  root: string;
  tree: FileNode[];
}

export interface FileTreeEvent {
  session_id?: string;
  kind: 'tree' | 'changes';
  root?: string;
  tree?: FileNode[];
  added?: FileNode[];
  removed?: FileNode[];
  modified?: FileNode[];
  overflow?: boolean;
  timestamp?: string;
}

// Nested tree node built from the flat FileNode[] for rendering.
export interface FileTreeNode {
  name: string;
  path: string;
  type: FileNodeType;
  size: number;
  mtime: number;
  children?: FileTreeNode[];
}

export type View = 'l0' | 'l1';
export type FrontTab = 'agent-map' | 'key-report';
export type L1Tab = 'board' | 'optimization' | 'workspace' | 'key-report' | 'logs';
export type Theme = 'scienceflow-dark' | 'paper-light';
export type FontSize = 'small' | 'default' | 'large';
export type L1Scope = 'task' | 'node';
export type ChatRunState = 'idle' | 'running' | 'finalizing' | 'cancelling' | 'completed' | 'failed' | 'cancelled';
export type ChatRouteMode = 'chat' | 'agent';
export type AuthStatus = 'idle' | 'loading' | 'authenticated' | 'unauthenticated';

export interface AuthUser {
  username: string;
  display_name: string;
  created_at: string;
}

export interface AuthState {
  status: AuthStatus;
  token: string;
  user: AuthUser | null;
}

export interface AgentMapModel {
  mode: 'repl' | 'lnr' | 'empty';
  agents: AgentMapAgent[];
  visibleAgents: AgentMapAgent[];
  workers: AgentMapWorker[];
  workerSummary: WorkerSummary;
  selectedAgent: AgentMapAgent | null;
  elapsed: string;
  stats: AgentMapStats;
  board: AgentMapBoard;
  inboxItems: AgentMapInboxItem[];
}

export interface AgentMapAgent {
  id: string;
  worker_id: string;
  label: string;
  kind: 'repl' | 'lnr' | '';
  pointId: string;
  color: string;
  statusClass: string;
  actorState: string;
  issue: AgentMapIssue | null;
  progress: number;
  thought: string;
  thoughtLines: string[];
  bestCurve: AgentMapCurve;
  spark: number[];
  last_event_at: string;
  stage_count: number;
  current_stage: string;
  sendContext: string;
  route: string;
  runState: string;
  recentEvents?: AgentMapEvent[];
}

export interface AgentMapWorker {
  worker_id: string;
  statusClass: string;
  progress: number;
  thought: string;
  bestCurve: AgentMapCurve;
  last_event_at: string;
  actorState: string;
  issue: AgentMapIssue | null;
}

export interface AgentMapIssue {
  kind: string;
  severity: string;
  title: string;
  body: string;
  worker_id: string;
}

export interface AgentMapCurve {
  values: number[];
  best: number | null;
  gain: number | null;
  hasMetric: boolean;
  lowerIsBetter: boolean;
}

export interface AgentMapStats {
  total: number;
  active: number;
  completed: number;
  failed: number;
}

export interface AgentMapBoard {
  title: string;
  rows: string[];
  stats: [string, string][];
}

export interface AgentMapInboxItem {
  id: string;
  kind: string;
  worker_id: string;
  title: string;
  body: string;
  meta: string;
}

export interface AgentMapEvent {
  type: string;
  message: string;
  status: string;
  created_at: string;
}
