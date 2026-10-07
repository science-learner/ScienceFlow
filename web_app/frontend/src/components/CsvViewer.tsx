import { useMemo } from 'react';
import Papa from 'papaparse';

interface CsvViewerProps {
  content: string;
  maxRows?: number;
}

export function CsvViewer({ content, maxRows = 2000 }: CsvViewerProps) {
  const { headers, rows, truncated, totalRows } = useMemo(() => {
    const result = Papa.parse(content, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: false,
    });

    if (result.errors.length > 0 && !result.data.length) {
      return { headers: [], rows: [], truncated: false, totalRows: 0 };
    }

    const allRows = result.data as Record<string, string>[];
    const h = result.meta.fields || [];
    const total = allRows.length;

    if (allRows.length <= maxRows) {
      return { headers: h, rows: allRows, truncated: false, totalRows: total };
    }

    return {
      headers: h,
      rows: allRows.slice(0, maxRows),
      truncated: true,
      totalRows: total,
    };
  }, [content, maxRows]);

  if (!headers.length) {
    return (
      <div className="csv-viewer csv-empty">
        <div className="code" style={{ whiteSpace: 'pre-wrap', fontFamily: 'var(--mono)', fontSize: 12 }}>
          {content.slice(0, 5000) || '(empty CSV)'}
        </div>
      </div>
    );
  }

  return (
    <div className="csv-viewer">
      <div className="csv-toolbar">
        <span className="csv-row-count">
          {truncated ? `Showing ${maxRows} of ${totalRows} rows` : `${totalRows} rows`}
          &nbsp;&middot;&nbsp;
          {headers.length} columns
        </span>
      </div>
      <div className="csv-table-wrap">
        <table className="csv-table">
          <thead>
            <tr>
              <th className="csv-row-num">#</th>
              {headers.map((h) => (
                <th key={h} title={h}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i}>
                <td className="csv-row-num dim">{i + 1}</td>
                {headers.map((h) => (
                  <td key={h} title={row[h] || ''}>
                    {row[h] ?? ''}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {truncated && (
        <div className="csv-truncated dim">
          &hellip; {totalRows - maxRows} more rows not shown. Use download for full file.
        </div>
      )}
    </div>
  );
}
