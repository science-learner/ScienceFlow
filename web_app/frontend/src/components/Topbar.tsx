import { useAppStore } from '../store/useAppStore';
import { useGatewayStore } from '../store/useGatewayStore';
import { useT } from '../i18n/useT';
import type { Lang } from '../i18n/translations';
import type { AuthUser } from '../types';
import { LogOut, Settings, User, Languages } from 'lucide-react';
import clsx from 'clsx';

interface TopbarProps {
  authUser?: AuthUser | null;
  onLogout?: () => void;
}

export function Topbar({ authUser, onLogout }: TopbarProps) {
  const {
    currentState, currentView, batchPanelCollapsed,
    setSettingsPanelOpen,
    language, setLanguage,
    setView, setFrontTab, setBatchPanelCollapsed,
  } = useAppStore();
  const gatewayStatus = useGatewayStore((s) => s.status);
  const gatewayError = useGatewayStore((s) => s.lastError);
  const t = useT();

  const runCount = currentState?.runs?.length || 0;
  const showBatchToggle = currentView !== 'l0' && currentView !== 'l1';

  // Gateway connection indicator (updated by the settings "test connection"
  // action and the gateway's own status events).
  const statusKey: 'statusConnected' | 'statusConnecting' | 'statusUnauthorized' | 'statusDisconnected' =
    gatewayStatus === 'connected' ? 'statusConnected'
      : gatewayStatus === 'connecting' ? 'statusConnecting'
        : gatewayStatus === 'unauthorized' ? 'statusUnauthorized'
          : 'statusDisconnected';
  const statusClass =
    gatewayStatus === 'connected' ? 'is-connected'
      : gatewayStatus === 'connecting' ? 'is-connecting'
        : gatewayStatus === 'unauthorized' ? 'is-unauthorized'
          : 'is-offline';

  const toggleLang = () => {
    const next: Lang = language === 'en-US' ? 'zh-CN' : 'en-US';
    setLanguage(next);
  };

  const goToAgentMap = () => {
    setView('l0');
    setFrontTab('agent-map');
    if (!batchPanelCollapsed) setBatchPanelCollapsed(true);
  };

  return (
    <header className="topbar">
      <div className="brand">
        <button className="brand-logo" type="button" onClick={goToAgentMap} title={t.topbar.backToAgentMap} aria-label={t.topbar.backToAgentMap}>
          <img src="/favicon.ico" alt={t.topbar.title} className="brand-logo-image" />
        </button>
        <div className="brand-copy">
          <h1>{t.topbar.title}</h1>
          <p>{t.topbar.workspaceReady}</p>
        </div>
      </div>
      <div className="status-strip"></div>
      <div className="top-actions">
        {(authUser || onLogout) && (
          <div className="topbar-user-group">
            {authUser && (
              <span className="topbar-user">
                <User size={14} className="topbar-user-icon" />
                <span className="topbar-user-name">{authUser.display_name || authUser.username}</span>
              </span>
            )}
            {onLogout && (
              <button className="btn topbar-logout" type="button" onClick={onLogout} title={t.topbar.logout} aria-label={t.topbar.logout}>
                <LogOut size={16} />
              </button>
            )}
          </div>
        )}
        <button className="btn lang-toggle" type="button"
          title={language === 'en-US' ? t.topbar.switchToChinese : t.topbar.switchToEnglish}
          aria-label={language === 'en-US' ? t.topbar.switchToChinese : t.topbar.switchToEnglish}
          onClick={toggleLang}>
          <Languages size={14} />
          {language === 'en-US' ? '\u4E2D\u6587' : 'EN'}
        </button>
        <div className="top-context-strip">
          <span
            className={clsx('top-context-item', 'status', statusClass)}
            title={gatewayError || t.topbar[statusKey]}
          >
            {t.topbar[statusKey]}
          </span>
        </div>
        {showBatchToggle && (
          <button className="btn top-batch-toggle" type="button" onClick={() => setBatchPanelCollapsed(!batchPanelCollapsed)}>
            <span>{t.topbar.batch}</span>
            <span className="batch-rail-count">{runCount}</span>
          </button>
        )}
        <button className="btn icon-btn" id="frontOpenSettings" type="button" aria-label={t.topbar.settings}
          onClick={() => setSettingsPanelOpen(true)}>
          <Settings size={17} />
        </button>
      </div>
    </header>
  );
}
