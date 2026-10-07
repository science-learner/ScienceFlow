import { useEffect, useMemo, useState } from 'react';
import { useAppStore } from '../store/useAppStore';
import { useT } from '../i18n/useT';
import { buildAgentMapModel } from '../utils/collab';
import { resolveAgentRoomPose } from '../utils/motion';
import * as api from '../api/client';
import type { AgentMapModel, AgentMapAgent } from '../types';
import clsx from 'clsx';

const ACTOR_POINTS: Record<string, { x: number; y: number; bubbleX: number; bubbleY: number }> = {
  worker0: { x: 27.2, y: 66.8, bubbleX: 27.2, bubbleY: 53.8 },
  worker1: { x: 72.0, y: 66.8, bubbleX: 72.0, bubbleY: 53.8 },
  worker2: { x: 27.2, y: 86.9, bubbleX: 27.2, bubbleY: 73.9 },
  worker3: { x: 71.3, y: 86.8, bubbleX: 71.3, bubbleY: 73.8 },
  coordinator: { x: 50.0, y: 50.5, bubbleX: 50.0, bubbleY: 38.5 },
};

export function AgentMapView() {
  const {
    currentState, chatBusy, chatRunState, chatQueueDepth,
    chatRouteMode, chatProgressLabel, chatWarmup,
    toolEvents,
    setView, setL1Tab, setL1Scope,
    agentPositionId, agentMapBackground,
  } = useAppStore();
  const t = useT();
  const [customBackground, setCustomBackground] = useState('');

  // Custom backgrounds live outside the bundle (<program dir>/themes/
  // backgrounds), so they are fetched through the app bindings as data URIs.
  useEffect(() => {
    if (!agentMapBackground) {
      setCustomBackground('');
      return;
    }
    let cancelled = false;
    api.fetchThemeBackgroundData(agentMapBackground)
      .then((data) => { if (!cancelled) setCustomBackground(data); })
      .catch(() => { if (!cancelled) setCustomBackground(''); });
    return () => { cancelled = true; };
  }, [agentMapBackground]);

  // Bundled default background (also the fallback when a chosen
  // themes/backgrounds image is missing or unreadable).
  const backgroundSrc = customBackground || './assets/scienceflow-lab.png';

  const model = useMemo(() => {
    return buildAgentMapModel(currentState, {
      busy: chatBusy,
      warmup: chatWarmup,
      runState: chatRunState,
      queueDepth: chatQueueDepth,
      routeMode: chatRouteMode,
      progressLabel: chatProgressLabel,
      toolEvents: toolEvents || [],
      assistantPreview: '',
      lastEventAt: '',
      injectionState: 'none',
      settingsReady: true,
      elapsed: currentState?.monitor?.runtime?.elapsed || '',
      messages: [],
      agentPositionId,
    });
  }, [currentState, chatBusy, chatRunState, chatQueueDepth, chatRouteMode, chatProgressLabel, chatWarmup, toolEvents, agentPositionId]);

  return (
    <div className="agent-map-stage-only">
      <picture>
        <img
          className="agent-map-photo"
          src={backgroundSrc}
          alt={t.agentMap.imgAlt}
        />
      </picture>
      <AgentMapOverlay model={model} />
      <AgentMapActorLayer model={model} />
      <button
        className="agent-map-hotspot agent-map-hotspot-board"
        type="button"
        aria-label={t.agentMap.openBoardAria}
        onClick={() => { setL1Scope('task'); setL1Tab('workspace'); setView('l1'); }}
      >
        <span>{t.agentMap.openBoard}</span>
      </button>
      <button
        className="agent-map-hotspot agent-map-hotspot-report"
        type="button"
        aria-label={t.agentMap.keyReportAria}
        onClick={() => { setL1Scope('task'); setL1Tab('key-report'); setView('l1'); }}
      >
        <span>{t.agentMap.keyReport}</span>
      </button>
    </div>
  );
}

function AgentMapOverlay({ model }: { model: AgentMapModel }) {
  const agents = model.agents || [];
  const photoPoints: Record<string, { x: number; y: number; r: number }> = {
    worker0: { x: 454, y: 585, r: 43 },
    worker1: { x: 1204, y: 585, r: 43 },
    worker2: { x: 454, y: 774, r: 44 },
    worker3: { x: 1192, y: 774, r: 44 },
    coordinator: { x: 836, y: 432, r: 38 },
  };

  if (model.mode === 'repl' || model.mode === 'empty') {
    return <ReplBoardOverlay model={model} />;
  }

  const hub = [836, 590];
  const hubActive = agents.some((a) => a.statusClass && a.statusClass !== 'idle');

  return (
    <svg className="agent-map-overlay" viewBox="0 0 1672 941" preserveAspectRatio="xMidYMid meet" aria-hidden="true">
      <defs>
        <filter id="agentPhotoGlow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
      </defs>
      <g filter="url(#agentPhotoGlow)">
        {agents.map((agent) => {
          const pt = photoPoints[agent.pointId || agent.id] || photoPoints.coordinator;
          const color = agent.color || '#5cc8ff';
          const moving = agent.statusClass !== 'idle';
          const opacity = agent.statusClass === 'idle' ? '0.26' : agent.statusClass === 'waiting' ? '0.52' : '0.76';
          return (
            <line key={agent.id} className={moving ? 'agent-map-flow' : ''}
              x1={hub[0]} y1={hub[1]} x2={pt.x} y2={pt.y}
              stroke={color} strokeOpacity={opacity} strokeWidth="3" strokeLinecap="round" />
          );
        })}
        <circle className={hubActive ? 'agent-map-pulse' : ''}
          cx={hub[0]} cy={hub[1]} r="7"
          fill={hubActive ? '#91ffc0' : '#7890a0'}
          opacity={hubActive ? '1' : '0.52'} />
      </g>
    </svg>
  );
}

function ReplBoardOverlay({ model }: { model: AgentMapModel }) {
  const t = useT();
  const board = model.board || {};
  const agent = model.selectedAgent || model.agents?.[0];
  const rows = (board.rows?.length ? board.rows : [agent?.thought || 'ready']).slice(-4);
  const stats = (board.stats || []).slice(0, 6);
  const statX = [548, 666, 786, 906, 1024, 1112];
  const color = agent?.color || '#62d884';

  return (
    <svg className="agent-map-overlay" viewBox="0 0 1672 941" preserveAspectRatio="xMidYMid meet" aria-hidden="true">
      <g fontFamily="var(--mono)">
        <rect className="agent-map-board-screen" x="500" y="145" width="656" height="204" rx="8" />
        <text x="534" y="173" fill="#e5f2ff" fontSize="15">{agent?.label || 'Agent'}</text>
        {rows.map((line: string, i: number) => (
          <text key={i} x="536" y={234 + i * 18}
            fill={i === rows.length - 1 ? '#f5fbff' : '#a9b8c6'} fontSize="13">
            {line}
          </text>
        ))}
        {stats.map(([label, value], i: number) => (
          <g key={i}>
            <text x={statX[i] || 548} y="325" fill="#8796a6" fontSize="11">{label}</text>
            <text x={statX[i] || 548} y="343" fill={i === 3 ? color : '#e6f2ff'} fontSize="14">{value || t.common.dash}</text>
          </g>
        ))}
      </g>
    </svg>
  );
}

function AgentMapActorLayer({ model }: { model: AgentMapModel }) {
  const { selectedAgentWorkerId, selectAgentMapWorker } = useAppStore();
  const t = useT();
  const agents = model.visibleAgents || [];

  return (
    <div className="agent-map-actor-layer" aria-hidden="false">
      {agents.map((agent) => {
        const pt = ACTOR_POINTS[agent.pointId || agent.id] || ACTOR_POINTS.coordinator;
        if (!pt) return null;
        const selected = agent.worker_id === selectedAgentWorkerId || agent.id === selectedAgentWorkerId;
        const color = agent.color || '#5cc8ff';

        return (
          <button
            key={agent.worker_id || agent.id}
            className={clsx(
              'agent-map-actor',
              `acting-${agent.actorState || 'idle'}`,
              'facing-right',
              selected && 'is-selected'
            )}
            style={{
              '--actor-x': `${pt.x}%`,
              '--actor-y': `${pt.y}%`,
              '--actor-color': color,
            } as React.CSSProperties}
            onClick={() => selectAgentMapWorker(agent.worker_id || agent.id)}
            title={`${agent.label} · ${agent.actorState || agent.statusClass || t.common.idle}`}
          >
            <span className="agent-sprite" aria-hidden="true" />
            <span className="agent-actor-badge">{agent.label || agent.worker_id || agent.id}</span>
          </button>
        );
      })}

      {/* Action bubbles for issues */}
      {agents
        .filter((a) => a.issue)
        .map((agent) => {
          const pt = ACTOR_POINTS[agent.pointId || agent.id];
          if (!pt) return null;
          return (
            <button
              key={`bubble-${agent.worker_id || agent.id}`}
              className="agent-map-action-bubble"
              style={{
                '--bubble-x': `${pt.bubbleX}%`,
                '--bubble-y': `${pt.bubbleY - 10}%`,
              } as React.CSSProperties}
              onClick={() => selectAgentMapWorker(agent.worker_id || agent.id)}
              title={agent.issue?.body || t.agentMap.openIssue}
            >
              {agent.issue?.title?.slice(0, 18) || t.agentMap.needsAttention}
            </button>
          );
        })}
    </div>
  );
}
