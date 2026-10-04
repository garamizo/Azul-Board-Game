import { clearReloadStamp, handleUnauthorized } from './session';
import type { GameSummary, GameView, MoveBody } from './types';

export class ApiError extends Error {
  constructor(public status: number, public body: any) {
    super(`HTTP ${status}${body?.error ? `: ${body.error}` : ''}`);
  }
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = { 'X-Requested-With': 'XMLHttpRequest' };
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  const res = await fetch(path, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
    credentials: 'same-origin',
  });
  if (res.status === 401) {
    handleUnauthorized();
    throw new ApiError(401, null);
  }
  const data = res.status === 204 ? null : await res.json().catch(() => null);
  if (!res.ok) throw new ApiError(res.status, data);
  clearReloadStamp();
  return data as T;
}

export function message(e: unknown): string {
  if (e instanceof ApiError) {
    if (e.status === 401) return 'Signed out. Reload the page to sign in again.';
    if (e.status === 503) return "The server can't verify sign-ins right now. Try again in a moment.";
    return e.body?.error ? `Not allowed: ${e.body.error}` : `Server error (${e.status})`;
  }
  return e instanceof Error ? e.message : String(e);
}

export const api = {
  me: () => request<{ email: string }>('GET', '/api/me'),
  games: () => request<GameSummary[]>('GET', '/api/games'),
  create: (players: number) => request<GameView>('POST', '/api/games', { players }),
  game: (id: string) => request<GameView>('GET', `/api/games/${id}`),
  claim: (id: string, seat: number) => request<GameView>('POST', `/api/games/${id}/seats/${seat}/claim`, {}),
  release: (id: string, seat: number) => request<GameView>('POST', `/api/games/${id}/seats/${seat}/release`, {}),
  setKind: (id: string, seat: number, kind: 'open' | 'bot') =>
    request<GameView>('POST', `/api/games/${id}/seats/${seat}/kind`, { kind }),
  start: (id: string) => request<GameView>('POST', `/api/games/${id}/start`, {}),
  toBot: (id: string, seat: number) => request<GameView>('POST', `/api/games/${id}/seats/${seat}/to-bot`, {}),
  takeBack: (id: string, seat: number) => request<GameView>('POST', `/api/games/${id}/seats/${seat}/take-back`, {}),
  remove: (id: string) => request<null>('DELETE', `/api/games/${id}`),
  move: (id: string, body: MoveBody) => request<GameView>('POST', `/api/games/${id}/moves`, body),
};
