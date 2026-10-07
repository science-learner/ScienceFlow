import clsx from 'clsx';
import { useT } from '../i18n/useT';
import { metricText } from '../utils/helpers';
import type { Run } from '../types';

interface Props {
  collapsed: boolean;
  runCount: number;
  subtitle: string;
  runs: Run[];
  selectedRunIndex: number;
  onToggle: () => void;
  onSelectRun: (index: number) => void;
  onDoubleClick: (index: number) => void;
}

export function BatchPanel({
  collapsed,
  runCount,
  subtitle,
  runs,
  selectedRunIndex,
  onToggle,
  onSelectRun,
  onDoubleClick,
}: Props) {
  const t = useT();
  return (
    <aside
      className={clsx(
        'card',
        'front-card',
        'front-batch-card',
        collapsed && 'is-collapsed'
      )}
      data-batch-panel
      aria-expanded={!collapsed}
    >
      <button
        className="batch-rail-toggle"
        type="button"
        data-batch-toggle
        onClick={onToggle}
      >
        <span className="batch-rail-icon">&gt;</span>
        <span className="batch-rail-label">{t.batchPanel.batchRailLabel}</span>
        <span className="batch-rail-count" data-batch-rail-count>
          {runCount}
        </span>
      </button>
      <div className="card-head front-batch-head">
        <span className="card-title">{t.batchPanel.currentBatch}</span>
        <span className="card-subtitle" data-batch-subtitle>
          {subtitle}
        </span>
        <button
          className="batch-collapse-btn"
          type="button"
          data-batch-toggle
          onClick={onToggle}
        >
          <span>&lt;</span>
        </button>
      </div>
      <div className="card-body stack front-batch-body">
        <div className="front-runs-section">
          <div className="section-title">{t.batchPanel.runs}</div>
          <div className="run-list">
            {runs.map((run, i) => (
              <button
                key={run.run_id || i}
                className={clsx(
                  'run-card',
                  i === selectedRunIndex && 'active'
                )}
                data-live-run-index={i}
                onClick={() => onSelectRun(i)}
                onDoubleClick={() => onDoubleClick(i)}
              >
                <div className="run-title">
                  <strong>{run.run_id || 'current-run'}</strong>
                  <span className={clsx('pill', run.status === 'completed' ? 'green' : 'cyan')}>
                    {run.status || t.common.attached}
                  </span>
                </div>
                <div className="mini-row">
                  <span>{t.batchPanel.bestEffective}</span>
                  <span className="score">{metricText(run.best_metric)}</span>
                </div>
                {run.elapsed && (
                  <div className="mini-row">
                    <span>{t.batchPanel.elapsed}</span>
                    <span>{run.elapsed}</span>
                  </div>
                )}
              </button>
            ))}
          </div>
        </div>
      </div>
    </aside>
  );
}
