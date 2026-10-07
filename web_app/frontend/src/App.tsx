import { useEffect, useCallback, useRef, useState } from 'react';
import { useAppStore } from './store/useAppStore';
import { useT } from './i18n/useT';
import { Topbar } from './components/Topbar';
import { ChatRail } from './components/ChatRail';
import { BatchPanel } from './components/BatchPanel';
import { AgentMapView } from './components/AgentMapView';
import { ReportViewer } from './components/ReportViewer';
import { L1Workspace } from './components/L1Workspace';
import { WorkspaceTabs } from './components/WorkspaceTabs';
import { SettingsModal } from './components/SettingsModal';
import { StatePanel } from './components/StatePanel';
import { SessionPanel } from './components/SessionPanel';
import { LoginPage } from './components/LoginPage';
import * as api from './api/client';
import { setGatewayURL } from './api/gateway';
import { useGatewayStore } from './store/useGatewayStore';
import { useGatewayStream } from './hooks/useGatewayStream';
import { useFileSync } from './hooks/useFileSync';
import { usePolling } from './hooks/usePolling';
import clsx from 'clsx';
import type { Run, TimelineEvent, AuthStatus, AuthUser } from './types';

const STATE_POLL_MS = 10000;
const LS_RAIL_KEY = 'scienceflow.chatRailWidth';

export default function App() {
  const {
    currentView, setView, frontTab, setFrontTab,
    batchPanelCollapsed, setBatchPanelCollapsed,
    settingsPanelOpen, statePanelOpen, sessionsPanelOpen,
    refreshState,
    chatSessionId, setChatSessionId,
    setSessionsPanelOpen,
    currentState,
    selectedRunIndex,
    reportList, selectedReportPath, fetchReportContent,
  } = useAppStore();
  const t = useT();

  const initialized = useRef(false);
  const [stateEventsCollapsed, setStateEventsCollapsed] = useState(true);
  const [authStatus, setAuthStatus] = useState<AuthStatus>('loading');
  const [authUser, setAuthUser] = useState<AuthUser | null>(null);

  useEffect(() => {
    const cached = api.loadAuth();
    if (cached) {
      setAuthStatus('authenticated');
      setAuthUser(cached.user);
      api.validateAuth().then((ok) => {
        if (!ok) {
          api.clearAuth();
          setAuthStatus('unauthenticated');
          setAuthUser(null);
        }
      });
      // Reconnect the log gateway with saved credentials.
      const gwUser = localStorage.getItem('scienceflow.gwUser');
      const gwPass = localStorage.getItem('scienceflow.gwPass');
      if (gwUser && gwPass) {
        void useGatewayStore.getState().connect(gwUser, gwPass).then(() => {
          const gw = useGatewayStore.getState();
          if (gw.sessionId) {
            useAppStore.getState().setChatSessionId(gw.sessionId);
          }
        });
      }
    } else {
      setAuthStatus('unauthenticated');
    }

    const savedGateway = localStorage.getItem('scienceflow.gatewayBase');
    if (savedGateway) setGatewayURL(savedGateway);
  }, []);

  const handleLoginSuccess = (_token: string, user: AuthUser, userName: string, password: string) => {
    setAuthStatus('authenticated');
    setAuthUser(user);
    // Connect to the log gateway (login → create session → start SSE stream).
    // The gateway session ID becomes the app's chatSessionId for agent invoke.
    void useGatewayStore.getState().connect(userName, password).then(() => {
      const gw = useGatewayStore.getState();
      if (gw.sessionId) {
        useAppStore.getState().setChatSessionId(gw.sessionId);
      }
    });
  };

  const handleLogout = () => {
    api.clearAuth();
    setAuthStatus('unauthenticated');
    setAuthUser(null);
  };

  useEffect(() => {
    if (authStatus !== 'authenticated' || initialized.current) return;
    initialized.current = true;
    (async () => {
      // All interactions go through the gateway; no :8200 chat session setup.
      // The gateway connection (login → session → SSE) is triggered in
      // handleLoginSuccess via useGatewayStore.connect().
    })();
  }, [authStatus]);

  useGatewayStream();
  useFileSync();
  usePolling(refreshState, { interval: STATE_POLL_MS, enabled: authStatus === 'authenticated' });

  const handleKeyboard = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape') {
      if (settingsPanelOpen) {
        useAppStore.getState().setSettingsPanelOpen(false);
        return;
      }
      if (statePanelOpen) {
        useAppStore.getState().setStatePanelOpen(false);
        return;
      }
      if (sessionsPanelOpen) {
        setSessionsPanelOpen(false);
        return;
      }
      if (currentView === 'l1') {
        setView('l0');
        setFrontTab('agent-map');
        setBatchPanelCollapsed(true);
      }
    }
  }, [currentView, settingsPanelOpen, statePanelOpen, sessionsPanelOpen, setView, setFrontTab, setBatchPanelCollapsed, setSessionsPanelOpen]);

  useEffect(() => {
    document.addEventListener('keydown', handleKeyboard);
    return () => document.removeEventListener('keydown', handleKeyboard);
  }, [handleKeyboard]);

  useEffect(() => {
    try {
      const saved = localStorage.getItem(LS_RAIL_KEY);
      if (saved) {
        document.documentElement.style.setProperty('--chat-rail-w', `${saved}px`);
      }
    } catch { /* silent */ }
  }, []);

  useEffect(() => {
    const resizer = document.getElementById('railResizer');
    if (!resizer) return;

    let startX = 0;
    let startWidth = 0;

    const onPointerDown = (e: PointerEvent) => {
      startX = e.clientX;
      startWidth =
        parseInt(
          getComputedStyle(document.documentElement)
            .getPropertyValue('--chat-rail-w')
            .trim(),
        ) || 380;
      document.body.classList.add('rail-resizing');
      try {
        resizer.setPointerCapture(e.pointerId);
      } catch { /* silent */ }

      const onPointerMove = (ev: PointerEvent) => {
        const dx = startX - ev.clientX;
        const newWidth = Math.max(380, Math.min(720, startWidth + dx));
        document.documentElement.style.setProperty('--chat-rail-w', `${newWidth}px`);
        try {
          localStorage.setItem(LS_RAIL_KEY, String(newWidth));
        } catch { /* silent */ }
      };

      const onPointerUp = () => {
        document.body.classList.remove('rail-resizing');
        window.removeEventListener('pointermove', onPointerMove);
        window.removeEventListener('pointerup', onPointerUp);
      };

      window.addEventListener('pointermove', onPointerMove);
      window.addEventListener('pointerup', onPointerUp, { once: true });
    };

    resizer.addEventListener('pointerdown', onPointerDown);
    return () => resizer.removeEventListener('pointerdown', onPointerDown);
  }, [authStatus]);

  const state = currentState;
  const summary = state?.summary;
  const runs: Run[] = state?.runs?.length ? state.runs : [fallbackRun(state)];
  const runCount = runs.length;
  const subtitle = summary
    ? `${summary.nodes || 0} nodes \u00b7 ${summary.runs || 0} runs \u00b7 ${summary.status || 'attached'}`
    : 'No workspace attached';
  const timelineEvents: TimelineEvent[] = state?.timeline || [];

  const handleSelectRun = (index: number) => {
    useAppStore.getState().selectRun(index);
  };

  const handleDoubleClickRun = (index: number) => {
    useAppStore.getState().selectRun(index);
    if (currentView === 'l1') {
      setView('l0');
    } else {
      useAppStore.getState().setL1Scope('task');
      useAppStore.getState().setL1Tab('workspace');
      setView('l1');
    }
  };

  if (authStatus === 'loading') {
    return (
      <div style={{
        position: 'fixed', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: '#0a0d14',
      }}>
        <div style={{ textAlign: 'center' }}>
          <p style={{ color: 'rgba(255,255,255,0.25)', fontFamily: 'var(--mono)', fontSize: 14 }}>Initializing...</p>
        </div>
      </div>
    );
  }

  if (authStatus === 'unauthenticated') {
    return <LoginPage onSuccess={handleLoginSuccess} />;
  }

  return (
    <div className="app" data-view={currentView} data-front-panel={frontTab}>
      <Topbar authUser={authUser} onLogout={handleLogout} />
      {settingsPanelOpen && <SettingsModal />}
      {statePanelOpen && <StatePanel />}
      {sessionsPanelOpen && <SessionPanel />}

      <main className="viewport">
        <div
          className={clsx(
            'workspace-frame',
            (currentView === 'l0' || currentView === 'l1') && 'batch-hidden',
            batchPanelCollapsed ? 'batch-collapsed' : 'batch-open',
          )}
        >
          {currentView !== 'l1' && (
            <BatchPanel
              collapsed={batchPanelCollapsed}
              runCount={runCount}
              subtitle={subtitle}
              runs={runs}
              selectedRunIndex={selectedRunIndex}
              onToggle={() => setBatchPanelCollapsed(!batchPanelCollapsed)}
              onSelectRun={handleSelectRun}
              onDoubleClick={handleDoubleClickRun}
            />
          )}

          <div className="workspace-main">
            {/* L0 Page */}
            <section
              className={clsx('page', currentView === 'l0' && 'active')}
              data-page="l0"
            >
              <div className="front-stage">
                <section className="card front-card front-hero">
                  <WorkspaceTabs />
                  <div className="card-body">
                    <div className="report-shell">
                      <div className="report-panel">
                        <div
                          className={clsx(
                            'tab-panel',
                            frontTab === 'agent-map' && 'active',
                          )}
                          data-tab-panel="front-panel:agent-map"
                        >
                          <AgentMapView />
                        </div>
                        <div
                          className={clsx(
                            'tab-panel',
                            'agent-map-report-tab',
                            frontTab === 'key-report' && 'active',
                          )}
                          data-tab-panel="front-panel:key-report"
                        >
                          <div className="doc-frame">
                            <div className="doc-toolbar">
                              <div className="doc-toolbar-actions">
                                <button
                                  className="btn"
                                  onClick={() => setFrontTab('agent-map')}
                                >
                                  &larr; {t.topbar.backToAgentMap}
                                </button>
                                <button className="doc-action" type="button" onClick={() => window.print()}>{t.reportViewer.pdf}</button>
                              </div>
                              {reportList.length > 0 ? (
                                <select
                                  className="report-select"
                                  value={selectedReportPath}
                                  onChange={(e) => fetchReportContent(e.target.value)}
                                  style={{ fontFamily: 'var(--mono)', fontSize: 12, padding: '2px 6px', background: 'var(--surface)', color: 'var(--text)', border: '1px solid var(--line)', borderRadius: 5 }}
                                >
                                  {reportList.map((r) => (
                                    <option key={r.path} value={r.path}>
                                      {r.relative_path || r.title || r.filename}
                                    </option>
                                  ))}
                                </select>
                              ) : (
                                <span className="doc-toolbar-title">
                                  {state?.report?.title || t.reportViewer.keyReport}
                                </span>
                              )}
                            </div>
                            <div className="doc-scroll">
                              <article className="doc-page">
                                <ReportViewer />
                              </article>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </section>
              </div>
            </section>

            {/* L1 Page */}
            <section
              className={clsx('page', currentView === 'l1' && 'active')}
              data-page="l1"
            >
              <L1Workspace />
            </section>
          </div>

          <div
            className="rail-resizer"
            id="railResizer"
            role="separator"
            aria-orientation="vertical"
          />

          <ChatRail />
        </div>
      </main>
    </div>
  );
}

function fallbackRun(state: import('./types').ScienceFlowState | null): Run {
  return {
    run_id: state?.task?.task_id || 'current-run',
    status: state?.summary?.status || 'attached',
    best_metric: state?.summary?.best_metric ?? null,
    elapsed: state?.summary?.elapsed || '',
    created_at: '',
  };
}
