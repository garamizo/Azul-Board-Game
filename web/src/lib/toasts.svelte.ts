import type { Narration } from './narrate';

export const MAX_TOASTS = 3;
export const TOAST_MS = 3000;
/// The last part of a bubble's life, spent fading out (Toasts.svelte).
export const FADE_MS = 300;
export interface Toast extends Narration { id: number; kind: 'info' | 'warn'; leaving: boolean }

/// The bubbles standing, oldest first. A ring like catan's: a fourth displaces
/// the oldest at once, so a burst of moves never queues behind a timer.
export const toasts = $state<{ items: Toast[] }>({ items: [] });
const timers = new Map<number, ReturnType<typeof setTimeout>[]>();
let nextId = 1;

export function toast(n: Narration | string, kind: 'info' | 'warn' = 'info'): void {
  const item: Toast = { ...(typeof n === 'string' ? { who: null, text: n } : n), id: nextId++, kind, leaving: false };
  while (toasts.items.length >= MAX_TOASTS) dismiss(toasts.items[0].id);
  toasts.items.push(item);
  timers.set(item.id, [
    setTimeout(() => {
      const live = toasts.items.find((t) => t.id === item.id);
      if (live) live.leaving = true;
    }, TOAST_MS - FADE_MS),
    setTimeout(() => dismiss(item.id), TOAST_MS),
  ]);
}

export function dismiss(id: number): void {
  timers.get(id)?.forEach(clearTimeout);
  timers.delete(id);
  toasts.items = toasts.items.filter((t) => t.id !== id);
}

export function clearToasts(): void {
  for (const pending of timers.values()) pending.forEach(clearTimeout);
  timers.clear();
  toasts.items = [];
}
