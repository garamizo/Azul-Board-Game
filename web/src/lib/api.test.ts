import { afterEach, describe, expect, it, vi } from 'vitest';
import { api, ApiError, message } from './api';

afterEach(() => vi.restoreAllMocks());

function respond(status: number, body: unknown) {
  return vi.spyOn(globalThis, 'fetch').mockResolvedValue(
    new Response(body === null ? null : JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } }));
}

describe('api', () => {
  it('sends the Access and JSON headers', async () => {
    const f = respond(201, { id: 'abc' });
    await api.create(3);
    const [url, init] = f.mock.calls[0];
    expect(url).toBe('/api/games');
    const headers = init!.headers as Record<string, string>;
    expect(headers['X-Requested-With']).toBe('XMLHttpRequest');
    expect(headers['Content-Type']).toBe('application/json');
    expect(init!.body).toBe('{"players":3}');
  });

  it('seat actions send an empty JSON body', async () => {
    const f = respond(200, {});
    await api.claim('abc', 1);
    expect(f.mock.calls[0][1]!.body).toBe('{}');
  });

  it('401 goes through the session handler (stamps sessionStorage, reloads once)', async () => {
    sessionStorage.clear();
    respond(401, null);
    await expect(api.games()).rejects.toBeInstanceOf(ApiError);
    expect(sessionStorage.getItem('azul.reloadedForAuth')).not.toBeNull();
    sessionStorage.clear();
  });

  it('errors carry status and body', async () => {
    respond(409, { error: 'stale', view: { version: 9 } });
    const err = await api.game('abc').catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(409);
    expect(err.body.view.version).toBe(9);
    expect(message(err)).toContain('stale');
  });

  it('503 has a friendly message', async () => {
    respond(503, { error: 'auth-unavailable' });
    expect(message(await api.games().catch((e) => e))).toMatch(/verify sign-ins/);
  });
});
