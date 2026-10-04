import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { clearToasts, dismiss, toast, toasts } from './toasts.svelte';

beforeEach(() => { vi.useFakeTimers(); clearToasts(); });
afterEach(() => { clearToasts(); vi.useRealTimers(); });

describe('toasts', () => {
  it('keeps three, dropping the oldest at once', () => {
    for (const t of ['a', 'b', 'c', 'd']) toast(t);
    expect(toasts.items.map((t) => t.text)).toEqual(['b', 'c', 'd']);
  });

  it('each goes away after 3 s', () => {
    toast('a');
    vi.advanceTimersByTime(2999);
    expect(toasts.items).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(toasts.items).toHaveLength(0);
  });

  it('dismissing early clears its timer', () => {
    toast('a');
    dismiss(toasts.items[0].id);
    expect(toasts.items).toHaveLength(0);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('a string is a line with no name; a narration keeps its parts', () => {
    toast('The board changed', 'warn');
    toast({ who: 'bob', text: 'took 2 red', color: 2 });
    expect(toasts.items[0]).toMatchObject({ who: null, text: 'The board changed', kind: 'warn' });
    expect(toasts.items[1]).toMatchObject({ who: 'bob', text: 'took 2 red', color: 2, kind: 'info' });
  });

  it('clearToasts empties the list and its timers', () => {
    toast('a'); toast('b');
    clearToasts();
    expect(toasts.items).toHaveLength(0);
    expect(vi.getTimerCount()).toBe(0);
  });
});
