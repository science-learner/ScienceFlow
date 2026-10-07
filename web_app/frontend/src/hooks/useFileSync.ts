import { useEffect } from 'react';
import { Events } from '@wailsio/runtime';
import { useGatewayStore } from '../store/useGatewayStore';
import type { FileTreeEvent } from '../types';

// useFileSync listens for "gateway-file" Wails events emitted by the Go SSE
// bridge. The gateway sends two kinds of file events over the SSE connection:
//   - kind="tree":    full file tree (sent immediately on connect)
//   - kind="changes": incremental deltas (added/removed/modified)
//
// On "tree" events we replace the entire file tree in the store.
// On "changes" events we apply deltas to the existing flat map.
// If overflow is true, we fall back to a full REST re-fetch via GET /sessions/{id}/files.
export function useFileSync() {
  useEffect(() => {
    const off = Events.On('gateway-file', (ev) => {
      const e = ev.data as FileTreeEvent;
      const store = useGatewayStore.getState();
      if (e.session_id && store.sessionId && e.session_id !== store.sessionId) return;

      if (e.kind === 'tree') {
        store.applyFileTree(e.root || '', e.tree || []);
      } else if (e.kind === 'changes') {
        if (e.overflow) {
          // Too many changes — re-fetch full tree.
          store.refetchFileTree();
        } else {
          store.applyFileChanges(e.added || [], e.removed || [], e.modified || []);
        }
      }
    });

    return off;
  }, []);
}
