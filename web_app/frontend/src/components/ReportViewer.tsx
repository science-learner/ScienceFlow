import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useEffect, useState } from 'react';
import { useAppStore } from '../store/useAppStore';
import { useGatewayStore } from '../store/useGatewayStore';
import { workspaceDownloadFile } from '../api/gateway';
import { useT } from '../i18n/useT';
import { sanitizeReportMarkdown, getDefaultReport } from '../utils/helpers';

// Report images live in the session workspace, so they must be fetched
// through the Go bridge (WorkspaceDownload) and rendered from a blob URL.
// Requesting them directly from the page origin would 404 (desktop/server
// builds have no /api route on the UI origin).

function normalizeWorkspacePath(path: string): string {
  const out: string[] = [];
  for (const seg of path.split('/')) {
    if (!seg || seg === '.') continue;
    if (seg === '..') out.pop();
    else out.push(seg);
  }
  return out.join('/');
}

function resolveWorkspaceImage(src: string, reportPath: string): string {
  if (src.startsWith('/')) return normalizeWorkspacePath(src);
  const dir = reportPath.includes('/') ? reportPath.slice(0, reportPath.lastIndexOf('/')) : '';
  return normalizeWorkspacePath(dir ? `${dir}/${src}` : src);
}

interface WorkspaceImageProps {
  src?: string;
  alt?: string;
  reportPath: string;
}

function WorkspaceImage({ src, alt, reportPath }: WorkspaceImageProps) {
  const token = useGatewayStore((s) => s.token);
  const sessionId = useGatewayStore((s) => s.sessionId);
  const [url, setUrl] = useState('');
  const [failed, setFailed] = useState(false);
  const external = !!src && (src.startsWith('data:') || src.startsWith('http://') || src.startsWith('https://'));

  useEffect(() => {
    if (!src || external || !token || !sessionId) return;
    let cancelled = false;
    let objectUrl = '';
    workspaceDownloadFile(token, sessionId, resolveWorkspaceImage(src, reportPath))
      .then((blob) => {
        if (cancelled) return;
        objectUrl = URL.createObjectURL(blob);
        setUrl(objectUrl);
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [src, external, reportPath, token, sessionId]);

  const style = { maxWidth: '100%', height: 'auto', borderRadius: 6, margin: '12px 0' };
  if (external) return <img src={src} alt={alt || ''} style={style} loading="lazy" />;
  if (failed) {
    return <span className="dim" style={{ fontSize: 11, fontFamily: 'var(--mono)' }}>{alt || src}</span>;
  }
  if (!url) return null;
  return <img src={url} alt={alt || ''} style={style} loading="lazy" />;
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
          img: ({ src, alt }) => (
            <WorkspaceImage
              src={typeof src === 'string' ? src : undefined}
              alt={alt || ''}
              reportPath={selectedItem?.path || ''}
            />
          ),
        }}
      >
        {sanitizeReportMarkdown(content)}
      </ReactMarkdown>
    </>
  );
}
