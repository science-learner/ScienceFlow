import { useAppStore } from '../store/useAppStore';
import { translations, type Translations, type Lang } from './translations';

export function useT(): Translations {
  const lang: Lang = useAppStore((s) => s.language);
  return translations[lang] || translations['zh-CN'];
}

export function useLang(): Lang {
  return useAppStore((s) => s.language);
}
