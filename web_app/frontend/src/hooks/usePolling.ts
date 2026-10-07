import { useEffect, useRef } from 'react';

interface UsePollingOptions {
  interval: number;
  enabled: boolean;
}

export function usePolling(
  callback: (options?: { force?: boolean; compact?: boolean }) => Promise<void>,
  options: UsePollingOptions,
) {
  const timerRef = useRef<number | null>(null);
  const busyRef = useRef(false);

  useEffect(() => {
    if (!options.enabled) return;

    const tick = async () => {
      if (busyRef.current) {
        timerRef.current = window.setTimeout(tick, options.interval);
        return;
      }
      busyRef.current = true;
      try {
        await callback({ compact: true });
      } finally {
        busyRef.current = false;
        timerRef.current = window.setTimeout(tick, options.interval);
      }
    };

    timerRef.current = window.setTimeout(tick, options.interval);

    return () => {
      if (timerRef.current) window.clearTimeout(timerRef.current);
    };
  }, [callback, options.interval, options.enabled]);

  // Pause when hidden
  useEffect(() => {
    const onVisibilityChange = () => {
      if (document.hidden) {
        if (timerRef.current) window.clearTimeout(timerRef.current);
      } else {
        timerRef.current = window.setTimeout(async () => {
          busyRef.current = true;
          await callback({ compact: true });
          busyRef.current = false;
        }, 200);
      }
    };
    document.addEventListener('visibilitychange', onVisibilityChange);
    return () => document.removeEventListener('visibilitychange', onVisibilityChange);
  }, [callback]);
}
