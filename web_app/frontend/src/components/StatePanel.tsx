import { useAppStore } from '../store/useAppStore';
import { useT } from '../i18n/useT';
import { valueOrDash } from '../utils/helpers';

export function StatePanel() {
  const {
    currentState, moduleEtags, transportMetrics,
    statePanelOpen, setStatePanelOpen,
    activeTaskRoot,
  } = useAppStore();
  const t = useT();

  const handleClose = () => setStatePanelOpen(false);
  const taskRoot = activeTaskRoot();

  const transportEntries: [string, string][] = [];
  if (transportMetrics) {
    const tm = transportMetrics as Record<string, unknown>;
    if (tm.mode) transportEntries.push(['Mode', String(tm.mode)]);
    if (tm.cache) transportEntries.push(['Cache', String(tm.cache)]);
    if (tm.build_ms !== undefined) transportEntries.push(['Build', `${Number(tm.build_ms)}ms`]);
    if (tm.response_ms !== undefined) transportEntries.push(['Response', `${Number(tm.response_ms)}ms`]);
    if (tm.payload_bytes !== undefined) transportEntries.push(['Payload', `${Number(tm.payload_bytes)} bytes`]);
    if (tm.modules_count !== undefined) transportEntries.push(['Modules', String(tm.modules_count)]);
  }

  const scheduler = currentState?.monitor?.scheduler;
  const runtime = currentState?.monitor?.runtime;

  const schedulerEntries: [string, string][] = [];
  if (scheduler) {
    schedulerEntries.push(['Queue depth', String(scheduler.queue_depth)]);
    schedulerEntries.push(['Completed', String(scheduler.completed)]);
    schedulerEntries.push(['Failed', String(scheduler.failed)]);
  }
  if (runtime) {
    schedulerEntries.push(['Runtime status', runtime.status || t.common.dash]);
    schedulerEntries.push(['Elapsed', runtime.elapsed || t.common.dash]);
  }

  const etagEntries: [string, string][] = [];
  if (moduleEtags && moduleEtags.size > 0) {
    moduleEtags.forEach((etag, key) => {
      etagEntries.push([key, etag]);
    });
  }

  return (
    <div className="settings-modal" data-state-modal>
      <div className="settings-scrim" data-state-close onClick={handleClose}></div>
      <section className="settings-card state-card" role="dialog" aria-modal="true">
        <div className="card-head">
          <span className="card-title">{t.statePanel.stateDetails}</span>
          <button className="btn" type="button" data-state-close onClick={handleClose}>
            {t.statePanel.close}
          </button>
        </div>
        <div className="settings-body">
          <div className="settings-section">
            <div className="settings-section-title">{t.statePanel.transport}</div>
            <div className="kv" data-state-transport-details>
              {transportEntries.length === 0 ? (
                <div className="kv-row"><span>{t.common.dash}</span><span>{t.statePanel.noTransport}</span></div>
              ) : (
                transportEntries.map(([key, value]) => (
                  <div className="kv-row" key={key}>
                    <span>{key}</span>
                    <span>{valueOrDash(value)}</span>
                  </div>
                ))
              )}
            </div>
          </div>

          <div className="settings-section">
            <div className="settings-section-title">{t.statePanel.scheduler}</div>
            <div className="kv" data-state-scheduler-details>
              {schedulerEntries.length === 0 ? (
                <div className="kv-row"><span>{t.common.dash}</span><span>{t.statePanel.noScheduler}</span></div>
              ) : (
                schedulerEntries.map(([key, value]) => (
                  <div className="kv-row" key={key}>
                    <span>{key}</span>
                    <span>{valueOrDash(value)}</span>
                  </div>
                ))
              )}
            </div>
          </div>

          <div className="settings-section">
            <div className="settings-section-title">{t.statePanel.moduleEtags}</div>
            <div className="kv" data-state-modules-details>
              {etagEntries.length === 0 ? (
                <div className="kv-row"><span>{t.common.dash}</span><span>{t.statePanel.noEtags}</span></div>
              ) : (
                etagEntries.map(([key, value]) => (
                  <div className="kv-row" key={key}>
                    <span>{key}</span>
                    <span>{valueOrDash(value)}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        <div className="settings-footer state-footer">
          <div className="settings-section-title">{t.statePanel.workspace}</div>
          <textarea
            className="state-workspace-text"
            data-state-workspace
            readOnly
            spellCheck={false}
            value={taskRoot || t.statePanel.noTaskRoot}
          />
        </div>
      </section>
    </div>
  );
}
