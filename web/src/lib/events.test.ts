import { afterEach, describe, expect, it, vi } from 'vitest';
import { subscribe } from './events';
import { api, ApiError } from './api';

class FakeSource {
  static last: FakeSource;
  listeners = new Map<string, (e: MessageEvent) => void>();
  onerror: (() => void) | null = null;
  closed = false;
  constructor(public url: string) { FakeSource.last = this; }
  addEventListener(name: string, fn: (e: MessageEvent) => void) { this.listeners.set(name, fn); }
  close() { this.closed = true; }
  emit(name: string, data: unknown) { this.listeners.get(name)!({ data: JSON.stringify(data) } as MessageEvent); }
}

afterEach(() => { vi.restoreAllMocks(); vi.useRealTimers(); });

describe('subscribe', () => {
  it('delivers states and reconnects after an error', async () => {
    vi.useFakeTimers();
    vi.stubGlobal('EventSource', FakeSource);
    vi.spyOn(api, 'me').mockResolvedValue({ email: 'a@x' });
    vi.spyOn(api, 'game').mockResolvedValue({} as never);
    const states: number[] = [];
    const statuses: string[] = [];
    const stop = subscribe('abc', { state: (v) => states.push(v.version), deleted: () => {}, status: (s) => statuses.push(s) });
    const first = FakeSource.last;
    expect(first.url).toBe('/api/games/abc/events');
    first.emit('state', { version: 3 });
    first.onerror!();
    expect(first.closed).toBe(true);
    await vi.advanceTimersByTimeAsync(1000);
    expect(FakeSource.last).not.toBe(first);
    FakeSource.last.emit('state', { version: 4 });
    expect(states).toEqual([3, 4]);
    expect(statuses).toEqual(['live', 'reconnecting', 'live']);
    stop();
    expect(FakeSource.last.closed).toBe(true);
  });

  it('a deleted game stops reconnecting', async () => {
    vi.useFakeTimers();
    vi.stubGlobal('EventSource', FakeSource);
    vi.spyOn(api, 'me').mockResolvedValue({ email: 'a@x' });
    vi.spyOn(api, 'game').mockRejectedValue(new ApiError(404, { error: 'not-found' }));
    const deleted = vi.fn();
    subscribe('abc', { state: () => {}, deleted, status: () => {} });
    const first = FakeSource.last;
    first.onerror!();
    await vi.advanceTimersByTimeAsync(1000);
    expect(deleted).toHaveBeenCalled();
    expect(FakeSource.last).toBe(first);
  });

  it('an expired session stops reconnecting and says so', async () => {
    vi.useFakeTimers();
    vi.stubGlobal('EventSource', FakeSource);
    vi.spyOn(api, 'me').mockRejectedValue(new ApiError(401, null));
    const statuses: string[] = [];
    subscribe('abc', { state: () => {}, deleted: () => {}, status: (s) => statuses.push(s) });
    const first = FakeSource.last;
    first.onerror!();
    await vi.advanceTimersByTimeAsync(1000);
    expect(statuses).toEqual(['reconnecting', 'signed-out']);
    expect(FakeSource.last).toBe(first);
  });

  it('the deleted event ends the subscription', () => {
    vi.stubGlobal('EventSource', FakeSource);
    const deleted = vi.fn();
    subscribe('abc', { state: () => {}, deleted, status: () => {} });
    FakeSource.last.emit('deleted', {});
    expect(deleted).toHaveBeenCalled();
    expect(FakeSource.last.closed).toBe(true);
  });
});
