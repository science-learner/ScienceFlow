import clsx from 'clsx';
import { useAppStore } from '../store/useAppStore';
import { useT } from '../i18n/useT';
import { shortenId } from '../utils/helpers';
import type { L1Tab } from '../types';

// Persistent workspace tab bar. Rendered as the card head of both the Agent
// Map hero (L0) and the L1 workspace card, so the tabs stay visible across
// views with the Agent Map as a peer tab of Workspace / Key Report / etc.
export function WorkspaceTabs() {
  const {
    currentView, l1Tab, l1Scope, selectedNodeIndex, currentState,
    setView, setFrontTab, setL1Tab, setBatchPanelCollapsed,
  } = useAppStore();
  const t = useT();

  const onAgentMap = currentView === 'l0';

  const openAgentMap = () => {
    setView('l0');
    setFrontTab('agent-map');
    setBatchPanelCollapsed(true);
  };

  const openL1 = (tab: L1Tab) => {
    setL1Tab(tab);
    setView('l1');
  };

  const scopeLabel = l1Scope === 'node'
    ? `${t.l1Workspace.node} ${shortenId(currentState?.nodes?.[selectedNodeIndex]?.node_id || '', 12)}`
    : t.l1Workspace.taskScope;

  const tab = (key: string, active: boolean, label: string, onClick: () => void) => (
    <button
      className={clsx(active && 'active')}
      data-tab-target={key}
      role="tab"
      aria-selected={active}
      onClick={onClick}
    >
      {label}
    </button>
  );

  return (
    <div className="card-head l1-card-head">
      <div className="l1-head-left">
        <div className="seg" data-tab-group="l1" role="tablist">
          {tab('agent-map', onAgentMap, t.l1Workspace.agentMapTab, openAgentMap)}
          {tab('workspace', !onAgentMap && l1Tab === 'workspace', t.l1Workspace.workspace, () => openL1('workspace'))}
          {tab('key-report', !onAgentMap && l1Tab === 'key-report', t.reportViewer.keyReport, () => openL1('key-report'))}
          {tab('optimization', !onAgentMap && l1Tab === 'optimization', t.l1Workspace.lineage, () => openL1('optimization'))}
          {tab('logs', !onAgentMap && l1Tab === 'logs', t.l1Workspace.logs, () => openL1('logs'))}
        </div>
      </div>
      <span className="card-subtitle" data-l1-scope-label>{onAgentMap ? '' : scopeLabel}</span>
    </div>
  );
}
