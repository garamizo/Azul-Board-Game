import { describe, expect, it, vi } from 'vitest';
import { clearReloadStamp, handleUnauthorized } from './session';

function fakeWindow() {
  const store = new Map<string, string>();
  return {
    sessionStorage: {
      getItem: (k: string) => store.get(k) ?? null,
      setItem: (k: string, v: string) => void store.set(k, v),
      removeItem: (k: string) => void store.delete(k),
    },
    location: { reload: vi.fn() },
  } as unknown as Window;
}

describe('session expiry', () => {
  it('reloads once, then stops', () => {
    const win = fakeWindow();
    expect(handleUnauthorized(win)).toBe('reloading');
    expect(handleUnauthorized(win)).toBe('signed-out');
    expect(win.location.reload).toHaveBeenCalledTimes(1);
  });

  it('a successful request re-arms the reload', () => {
    const win = fakeWindow();
    handleUnauthorized(win);
    clearReloadStamp(win);
    expect(handleUnauthorized(win)).toBe('reloading');
    expect(win.location.reload).toHaveBeenCalledTimes(2);
  });

  it('storage that throws never reloads in a loop', () => {
    const win = { sessionStorage: { getItem: () => { throw new Error('blocked'); } }, location: { reload: vi.fn() } } as unknown as Window;
    expect(handleUnauthorized(win)).toBe('signed-out');
    expect(win.location.reload).not.toHaveBeenCalled();
  });
});
