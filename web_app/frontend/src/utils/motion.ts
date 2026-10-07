const ROOM_WAYPOINTS = Object.freeze({
  idle: waypoint('idle', 50.0, 50.5, 50.0, 38.5, 'right'),
  desk: waypoint('desk', 50.0, 58.0, 50.0, 45.5, 'right'),
  board: waypoint('board', 50.0, 39.8, 50.0, 27.8, 'right'),
  tools: waypoint('tools', 72.5, 45.5, 72.5, 33.5, 'left'),
  report: waypoint('report', 78.0, 67.2, 78.0, 55.2, 'left'),
  blocked: waypoint('blocked', 39.0, 55.0, 39.0, 43.0, 'right'),
  left: waypoint('left', 27.2, 66.8, 27.2, 53.8, 'right'),
  right: waypoint('right', 72.0, 66.8, 72.0, 53.8, 'left'),
  up: waypoint('up', 50.0, 39.8, 50.0, 27.8, 'right'),
  down: waypoint('down', 50.0, 76.0, 50.0, 64.0, 'right'),
  center: waypoint('center', 50.0, 50.5, 50.0, 38.5, 'right'),
});

const DIRECTION_STEPS = Object.freeze({
  left: { dx: -18, dy: 0, facing: 'left' },
  right: { dx: 18, dy: 0, facing: 'right' },
  up: { dx: 0, dy: -14, facing: 'right' },
  down: { dx: 0, dy: 14, facing: 'right' },
});

const WAYPOINT_ALIASES: [string, RegExp][] = [
  ['blocked', /(?:\bblocked\b|\bblock\b|\braise\b|\bhand\b|举手|卡住|阻塞|问题|求助|报错|失败|异常)/i],
  ['board', /(?:\bboard\b|\bscreen\b|\bblackboard\b|\bwhiteboard\b|黑板|白板|中控|大屏|屏幕|看板)/i],
  ['tools', /(?:\btool(?:s)?\b|\bterminal\b|\bcompute\b|\binspect\b|\bworkbench\b|工具台|工具|终端|执行|检查|计算|运行)/i],
  ['report', /(?:\breport\b|\bsummary\b|\bresult(?:s)?\b|报告|结果|总结|产出)/i],
  ['right', /(?:\bright\b|\bright\s+side\b|\beast\b|右边|右侧|右面|向右|往右)/i],
  ['left', /(?:\bleft\b|\bleft\s+side\b|\bwest\b|左边|左侧|左面|向左|往左)/i],
  ['up', /(?:\bup\b|\bnorth\b|\bforward\b|上边|上面|上方|向上|往上|前进|往前|向前)/i],
  ['down', /(?:\bdown\b|\bsouth\b|\bback\b|\bbackward\b|下边|下面|下方|向下|往下|后退|往后|向后)/i],
  ['center', /(?:\bcenter\b|\bmiddle\b|中间|中央|居中|中心)/i],
  ['desk', /(?:\bdesk\b|\bworkstation\b|\bseat\b|\bwork\b|工位|座位|工作台|干活)/i],
  ['idle', /(?:\bidle\b|\brest\b|\bwait\b|\bstandby\b|\bhome\b|休息|等待|待机|回来|回去|原位)/i],
];

interface Waypoint {
  id: string;
  x: number;
  y: number;
  bubbleX: number;
  bubbleY: number;
  svgX: number;
  svgY: number;
  facing: string;
}

interface RoomPose {
  id: string;
  x: number;
  y: number;
  bubbleX: number;
  bubbleY: number;
  svgX: number;
  svgY: number;
  targetId: string;
  action: string;
  restAction: string;
  facing: string;
  moving: boolean;
  renderX: number;
  renderY: number;
}

interface MotionIntent {
  targetId: string;
  explicit: boolean;
  localOnly: boolean;
  route: string;
  rawText: string;
  commandText: string;
  createdAt: number;
}

interface AgentActor {
  worker_id?: string;
  id?: string;
  pointId?: string;
  kind?: string;
  actorState?: string;
  statusClass?: string;
  issue?: { body?: string } | null;
  route?: string;
  x?: number;
  y?: number;
  bubbleX?: number;
  bubbleY?: number;
}

interface CollabModel {
  mode?: string;
  selectedAgent?: { route?: string } | null;
}

function waypoint(id: string, x: number, y: number, bubbleX: number, bubbleY: number, facing: string): Waypoint {
  return { id, x, y, bubbleX, bubbleY, svgX: Math.round((x / 100) * 1672), svgY: Math.round((y / 100) * 941), facing };
}

function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.max(min, Math.min(max, value));
}

function fallbackPose(actor: AgentActor): RoomPose {
  return {
    id: actor.pointId || actor.id || 'static',
    x: Number(actor.x || 50),
    y: Number(actor.y || 50),
    bubbleX: Number(actor.bubbleX || actor.x || 50),
    bubbleY: Number(actor.bubbleY || (Number(actor.y || 50) - 10)),
    svgX: Math.round((Number(actor.x || 50) / 100) * 1672),
    svgY: Math.round((Number(actor.y || 50) / 100) * 941),
    targetId: actor.pointId || actor.id || 'static',
    action: normalizedActorState(actor.actorState),
    restAction: normalizedActorState(actor.actorState),
    facing: 'right',
    moving: false,
    renderX: Number(actor.x || 50),
    renderY: Number(actor.y || 50),
  };
}

function normalizedActorState(value?: string): string {
  const state = String(value || 'idle').trim().toLowerCase();
  if (['walking', 'working', 'talking', 'blocked', 'failed', 'done', 'idle'].includes(state)) return state;
  if (state === 'active' || state === 'running') return 'working';
  if (state === 'completed' || state === 'complete') return 'done';
  return 'idle';
}

function targetFromText(value: string): string {
  const text = String(value || '').trim();
  if (!text) return '';
  const direct = text.toLowerCase().replace(/^[\s:：,，.。-]+|[\s:：,，.。-]+$/g, '');
  if (direct in ROOM_WAYPOINTS) return direct;
  for (const [targetId, pattern] of WAYPOINT_ALIASES) {
    if (pattern.test(text)) return targetId;
  }
  return '';
}

function directionalTarget(targetId: string, basePose: { x: number; y: number } | null): Waypoint | null {
  const step = (DIRECTION_STEPS as Record<string, { dx: number; dy: number; facing: string }>)[targetId];
  if (!step || !basePose) return null;
  const x = clamp(Number(basePose.x || ROOM_WAYPOINTS.idle.x) + step.dx, 18, 82);
  const y = clamp(Number(basePose.y || ROOM_WAYPOINTS.idle.y) + step.dy, 36, 84);
  return waypoint(targetId, x, y, x, y - 12, step.facing);
}

function isReplActor(actor: AgentActor, mode: string): boolean {
  if (!actor) return false;
  if (!(mode === 'repl' || mode === 'empty')) return false;
  return actor.kind === 'repl' || actor.id === 'repl-agent' || actor.worker_id === 'Agent';
}

function targetForActor(actor: AgentActor, collabModel: CollabModel, motionIntent: MotionIntent | null): string {
  const stateAction = normalizedActorState(actor.actorState);
  if (motionIntent?.targetId && motionIntent.targetId in ROOM_WAYPOINTS) return motionIntent.targetId;
  if (stateAction === 'blocked' || stateAction === 'failed') return 'blocked';
  if (stateAction === 'working') return 'tools';
  if (stateAction === 'talking') return 'board';
  if (stateAction === 'done') return 'report';
  if (actor?.route === 'agent' || collabModel?.selectedAgent?.route === 'agent') return 'desk';
  return 'idle';
}

export function parseRoomMotionIntent(text: string, collabMode: string = 'repl'): MotionIntent | null {
  const original = String(text || '').trim();
  if (!original) return null;
  const routeMatch = original.match(/^@(agent|chat)\b\s*/i);
  const route = routeMatch ? routeMatch[1].toLowerCase() : '';
  const withoutRoute = routeMatch ? original.slice(routeMatch[0].length).trim() : original;
  const commandMatch = withoutRoute.match(/^\/(?:move|go|walk|goto|room|agent)\b(?:\s+(.+))?$/i);
  if (commandMatch) {
    const targetId = targetFromText(commandMatch[1] || withoutRoute.replace(/^\/(?:move|go|walk|goto|room|agent)\b\s*/i, '').trim()) || 'idle';
    return { targetId, explicit: true, localOnly: collabMode === 'repl' || collabMode === 'empty', route, rawText: original, commandText: withoutRoute, createdAt: Date.now() };
  }
  const targetId = targetFromText(withoutRoute);
  if (!targetId) return null;
  return { targetId, explicit: false, localOnly: false, route, rawText: original, commandText: '', createdAt: Date.now() };
}

export function resolveAgentRoomPose(
  actor: AgentActor,
  collabModel: CollabModel,
  motionIntent: MotionIntent | null,
  previousPose: { x: number; y: number } | null,
): RoomPose {
  const fallback = fallbackPose(actor);
  const mode = String(collabModel?.mode || '');
  if (!isReplActor(actor, mode)) {
    return { ...fallback, targetId: actor.pointId || actor.id || 'static', action: normalizedActorState(actor.actorState), restAction: normalizedActorState(actor.actorState), facing: fallback.facing, moving: false };
  }
  const stateAction = normalizedActorState(actor.actorState);
  const targetId = targetForActor(actor, collabModel, motionIntent);
  const previous = previousPose && Number.isFinite(Number(previousPose.x)) && Number.isFinite(Number(previousPose.y)) ? previousPose : null;
  const base = previous || { x: fallback.x, y: fallback.y };
  const target = directionalTarget(targetId, base) || (ROOM_WAYPOINTS as Record<string, Waypoint>)[targetId] || ROOM_WAYPOINTS.idle;
  const moving = Boolean(previous && (Math.abs(Number(previous.x) - target.x) > 0.1 || Math.abs(Number(previous.y) - target.y) > 0.1));
  const facing = moving && previous ? (target.x >= Number(previous.x) ? 'right' : 'left') : target.facing || 'right';
  return { ...target, targetId, action: moving ? 'walking' : stateAction, restAction: stateAction, facing, moving, renderX: moving ? Number(previous!.x) : target.x, renderY: moving ? Number(previous!.y) : target.y };
}
