import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useEffect } from 'react';
import { useAppStore } from '../store/useAppStore';
import { useT } from '../i18n/useT';
import { sanitizeReportMarkdown, getDefaultReport } from '../utils/helpers';

function resolveImageSrc(src: string | undefined): string {
  if (!src) return '';
  if (src.startsWith('data:') || src.startsWith('http://') || src.startsWith('https://')) {
    return src;
  }
  const base = import.meta.env.DEV ? 'http://localhost:46000' : '';
  return `${base}/api/workspace/file/raw?path=${encodeURIComponent(src)}`;
}

export function ReportViewer() {
  const {
    currentState, reportList, selectedReportPath,
    reportContent, chatSessionId, fetchReports, clearReportState,
  } = useAppStore();
  const t = useT();

  useEffect(() => {
    clearReportState();
    fetchReports();
  }, [chatSessionId]);

  const report = currentState?.report;
  const content = reportContent || report?.content || getDefaultReport();
  const timestamp = report?.updated_at || currentState?.generated_at || '';
  const selectedItem = reportList.find((r) => r.report_key === selectedReportPath || r.path === selectedReportPath);
  const selectedTitle = selectedItem?.title || report?.title || t.reportViewer.keyReport;

  return (
    <>
      <div className="doc-meta">{t.reportViewer.scienceflow}{timestamp ? ` · ${timestamp}` : ''}</div>
      <h1>{selectedTitle}</h1>
      <p className="doc-lead">
        {currentState?.summary?.status
          ? `Status: ${currentState.summary.status}. ${currentState.summary.nodes ?? 0} nodes, ${currentState.summary.runs ?? 0} runs.`
          : t.reportViewer.defaultLead}
      </p>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          pre: ({ children, ...props }) => (
            <pre {...props}>{children}</pre>
          ),
          code: ({ className, children, ...props }) => {
            const inline = !className;
            if (inline) {
              return <code {...props}>{children}</code>;
            }
            return (
              <code className={className} {...props}>
                {children}
              </code>
            );
          },
          table: ({ children }) => (
            <div style={{ overflowX: 'auto' }}>
              <table>{children}</table>
            </div>
          ),
          img: ({ src, alt, ...props }) => (
            <img
              src={resolveImageSrc(src)}
              alt={alt || ''}
              style={{ maxWidth: '100%', height: 'auto', borderRadius: 6, margin: '12px 0' }}
              loading="lazy"
              {...props}
            />
          ),
        }}
      >
        {sanitizeReportMarkdown(content)}
      </ReactMarkdown>
    </>
  );
}
