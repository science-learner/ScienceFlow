import type { Lang } from '../i18n/translations';

const pad = (n: number) => String(n).padStart(2, '0');
const dayKey = (d: Date) => `${d.getFullYear()}-${d.getMonth()}-${d.getDate()}`;
const hm = (d: Date) => `${pad(d.getHours())}:${pad(d.getMinutes())}`;
const enMonth = (d: Date) => d.toLocaleDateString('en-US', { month: 'short' });

// 会话列表的绝对时间：今天显示 HH:mm；昨天显示“昨天 HH:mm”；
// 其它（同年）显示“9月21日 15:17”；跨年再补年份。
export function formatSessionTime(rfc3339: string, lang: Lang): string {
  const d = new Date(rfc3339);
  if (isNaN(d.getTime())) return '--';
  const now = new Date();
  const yesterday = new Date(now);
  yesterday.setDate(now.getDate() - 1);
  if (dayKey(d) === dayKey(now)) return hm(d);
  if (dayKey(d) === dayKey(yesterday)) return lang === 'zh-CN' ? `昨天 ${hm(d)}` : `Yesterday ${hm(d)}`;
  if (d.getFullYear() === now.getFullYear()) {
    return lang === 'zh-CN'
      ? `${d.getMonth() + 1}月${d.getDate()}日 ${hm(d)}`
      : `${enMonth(d)} ${d.getDate()}, ${hm(d)}`;
  }
  return lang === 'zh-CN'
    ? `${d.getFullYear()}年${d.getMonth() + 1}月${d.getDate()}日 ${hm(d)}`
    : `${enMonth(d)} ${d.getDate()}, ${d.getFullYear()} ${hm(d)}`;
}
