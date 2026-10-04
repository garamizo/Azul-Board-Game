import { api, ApiError } from './api';
import type { GameView } from './types';

export type LinkStatus = 'live' | 'reconnecting' | 'signed-out';

export interface Handlers {
  state(view: GameView): void;
  deleted(): void;
  status(status: LinkStatus): void;
}

const DELAYS = [1000, 2000, 5000, 10000];

/// Live game updates. Reconnects with backoff; before reconnecting it checks
/// /api/me (EventSource cannot see a 401 itself: api.me runs the reload-once
/// path) and the game (a 404 means it was deleted).
export function subscribe(id: string, on: Handlers): () => void {
  let source: EventSource | null = null;
  let stopped = false;
  let attempt = 0;

  const open = () => {
    if (stopped) return;
    const es = new EventSource(`/api/games/${id}/events`);
    source = es;
    es.addEventListener('state', (e) => {
      attempt = 0;
      on.status('live');
      on.state(JSON.parse((e as MessageEvent).data));
    });
    es.addEventListener('deleted', () => {
      stopped = true;
      es.close();
      on.deleted();
    });
    es.onerror = async () => {
      es.close();
      if (stopped) return;
      on.status('reconnecting');
      await new Promise((r) => setTimeout(r, DELAYS[Math.min(attempt++, DELAYS.length - 1)]));
      if (stopped) return;
      // Each probe can outlive an unsubscribe: re-check `stopped` after it.
      try {
        await api.me();
      } catch (err) {
        if (stopped) return;
        if (err instanceof ApiError && err.status === 401) {
          // api.me already reloaded the page once; if we are still here,
          // reloading again will not help.
          stopped = true;
          on.status('signed-out');
          return;
        }
      }
      if (stopped) return;
      try {
        await api.game(id);
      } catch (err) {
        if (stopped) return;
        if (err instanceof ApiError && err.status === 404) {
          stopped = true;
          on.deleted();
          return;
        }
      }
      open();
    };
  };

  open();
  return () => {
    stopped = true;
    source?.close();
  };
}
