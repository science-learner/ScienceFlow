const DEFAULT_REPORT = `No logs yet — send a message to start`;

export function getDefaultReport(): string {
  return DEFAULT_REPORT;
}

export function sanitizeReportMarkdown(md: string): string {
  return (md || '')
    .replace(/\[LLM memory view:[^\]]*\]\s*/g, '')
    .replace(/<\|tool_calls\|>[\s\S]*?(?:<\|[\/]tool_calls\|>|$)/g, '')
    .replace(/<｜｜DSML｜｜tool_calls>[\s\S]*?<[\/]｜｜DSML｜｜tool_calls>/g, '')
    .replace(/Platform Chat Update[\s\S]*?(?=\n#|\n---|\n\*\*|$)/g, '')
    .replace(/\b[a-zA-Z]:[\\/][^\s<>"|?*\n]+/g, (match) => match.replace(/[\\/]/g, '/'));
}

export function escapeHtml(text: string): string {
  return String(text || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

export function metricText(value: unknown): string {
  if (value === null || value === undefined) return '—';
  const num = Number(value);
  if (Number.isFinite(num)) {
    if (num < 0.001 && num > -0.001) return num.toExponential(3);
    if (num < 1 && num > -1) return num.toFixed(4);
    return num.toFixed(num > 1000 ? 1 : 3);
  }
  return String(value);
}

export function valueOrDash(value: unknown): string {
  if (value === null || value === undefined || value === '') return '—';
  return String(value);
}

// 按显示字符截取文本（中英文/emoji 均算 1 个字符），超出 max 时截断并以“...”结尾。
export function truncateText(text: string, max: number): string {
  const chars = Array.from(text || '');
  if (chars.length <= max) return text || '';
  return chars.slice(0, max).join('') + '...';
}

export function hasDisplayValue(value: unknown): boolean {
  return value !== null && value !== undefined && value !== '';
}

export function shortenId(id: string, maxLen = 12): string {
  const s = String(id || '');
  if (s.length <= maxLen) return s;
  return `${s.slice(0, maxLen - 4)}..${s.slice(-4)}`;
}
