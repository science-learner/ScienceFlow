import { useEffect, useState } from 'react';
import { Plus, RefreshCw, Trash2 } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';
import { useGatewayStore } from '../store/useGatewayStore';
import { useT, useLang } from '../i18n/useT';
import { formatSessionTime } from '../utils/sessionTime';
import { truncateText } from '../utils/helpers';

export function SessionPanel() {
  const { setSessionsPanelOpen, setChatSessionId, setChatMessages, setChatBusy, setChatRunState, clearTimeline } = useAppStore();
  const { sessionList, fetchSessionList, switchSession, deleteSession, prepareNewSession, sessionId: gwSessionId } = useGatewayStore();
  const t = useT();
  const lang = useLang();

  const [loading, setLoading] = useState(false);
  const [switching, setSwitching] = useState(false);
  const [switchError, setSwitchError] = useState('');
  const [deletingId, setDeletingId] = useState('');
  const [confirmDeleteId, setConfirmDeleteId] = useState('');

  useEffect(() => {
    setLoading(true);
    fetchSessionList().finally(() => setLoading(false));
  }, [fetchSessionList]);

  const handleCreate = () => {
    if (loading || switching || deletingId) return;
    setConfirmDeleteId('');
    prepareNewSession();
    setChatSessionId('');
    setChatMessages([]);
    setChatBusy(false);
    setChatRunState('idle');
    clearTimeline();
    setSessionsPanelOpen(false);
  };

  const handleRefresh = async () => {
    if (loading || switching || deletingId) return;
    setLoading(true);
    try {
      await fetchSessionList();
    } finally {
      setLoading(false);
    }
  };

  const handleSelect = async (sessionId: string) => {
    if (switching || deletingId || sessionId === gwSessionId) return;
    setSwitching(true);
    setSwitchError('');
    try {
      const session = await switchSession(sessionId);
      if (session) {
        setChatMessages([]);
        setChatSessionId(session.session_id);
        // 切换后先回到 idle，等新会话的 run_state 事件再更新徽章，避免残留上一会话的“运行中”
        setChatBusy(false);
        setChatRunState('idle');
        clearTimeline();
        setSessionsPanelOpen(false);
      } else {
        setSwitchError(useGatewayStore.getState().lastError || t.sessionPanel.switchFailed);
      }
    } finally {
      setSwitching(false);
    }
  };

  const handleDelete = async (sessionId: string) => {
    if (loading || switching || deletingId) return;
    if (confirmDeleteId !== sessionId) {
      setConfirmDeleteId(sessionId);
      return;
    }
    setConfirmDeleteId('');
    setDeletingId(sessionId);
    try {
      const session = await deleteSession(sessionId);
      if (session) {
        setChatSessionId(session.session_id);
        clearTimeline();
      } else if (sessionId === gwSessionId) {
        setChatSessionId('');
        clearTimeline();
      }
    } finally {
      setDeletingId('');
    }
  };

  return (
    <div className="settings-modal">
      <div className="settings-scrim" onClick={() => setSessionsPanelOpen(false)} />
      <div className="settings-card">
        <div className="card-head">
          <span className="card-title">{t.sessionPanel.chatSessions}</span>
          <button className="btn" onClick={() => setSessionsPanelOpen(false)}>{t.sessionPanel.close}</button>
        </div>
        <div className="settings-body">
          <div className="session-actions session-panel-actions">
            {switchError && <div className="session-switch-error" role="alert">{switchError}</div>}
            <button className="btn session-action-btn" onClick={handleCreate} disabled={loading || switching || !!deletingId}>
              <Plus size={14} />
              <span>{t.sessionPanel.newSession}</span>
            </button>
            <button className="btn session-action-btn" onClick={handleRefresh} disabled={loading || switching || !!deletingId} aria-label={t.sessionPanel.refresh} title={t.sessionPanel.refresh}>
              <RefreshCw size={14} />
              <span>{t.sessionPanel.refresh}</span>
            </button>
          </div>
          {loading ? (
            <div className="dim" style={{ padding: 20, textAlign: 'center' }}>{t.sessionPanel.loading}</div>
          ) : sessionList.length === 0 ? (
            <div className="dim" style={{ padding: 20, textAlign: 'center' }}>{t.sessionPanel.noSessions}</div>
          ) : (
            sessionList.map((s) => {
              const isActive = s.session_id === gwSessionId;
              const agentStatus = s.agent?.status || 'idle';
              const isRunning = agentStatus === 'active';
              return (
                <div
                  key={s.session_id}
                  className="session-card"
                  role="button"
                  tabIndex={switching ? -1 : 0}
                  style={{ background: isActive ? 'rgba(92,200,255,0.12)' : undefined }}
                  onClick={() => handleSelect(s.session_id)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault();
                      void handleSelect(s.session_id);
                    }
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
                    <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: isActive ? 'var(--accent)' : 'var(--muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {s.session_id}
                    </span>
                    <span className={isRunning ? 'session-agent-badge running' : 'session-agent-badge'} style={{ flexShrink: 0 }}>
                      {isRunning ? '● running' : 'idle'}
                    </span>
                  </div>
                  <div style={{ fontSize: 13 }}>
                    <span title={s.name || undefined}>{s.name ? truncateText(s.name, 20) : t.common.newSession}</span>
                  </div>
                  <div className="session-card-foot">
                    {s.last_active ? (
                      <span className="dim" style={{ fontSize: 11 }}>
                        {formatSessionTime(s.last_active, lang)}
                      </span>
                    ) : <span />}
                    <div className="session-delete-actions">
                      {confirmDeleteId === s.session_id && (
                        <button
                          type="button"
                          className="session-cancel-delete-btn"
                          aria-label={t.sessionPanel.cancel}
                          title={t.sessionPanel.cancel}
                          disabled={!!deletingId || switching}
                          onClick={(event) => {
                            event.stopPropagation();
                            setConfirmDeleteId('');
                          }}
                        >
                          {t.sessionPanel.cancel}
                        </button>
                      )}
                      <button
                        type="button"
                        className={confirmDeleteId === s.session_id ? 'session-delete-btn confirming' : 'session-delete-btn'}
                        aria-label={confirmDeleteId === s.session_id ? t.sessionPanel.confirmDelete : t.sessionPanel.delete}
                        title={confirmDeleteId === s.session_id ? t.sessionPanel.confirmDelete : t.sessionPanel.delete}
                        disabled={!!deletingId || switching}
                        onClick={(event) => {
                          event.stopPropagation();
                          void handleDelete(s.session_id);
                        }}
                      >
                        <Trash2 size={12} />
                        {confirmDeleteId === s.session_id && <span>{t.sessionPanel.confirm}</span>}
                      </button>
                    </div>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}
