import type { Narration } from './narrate';

export const MAX_TOASTS = 3;
export const TOAST_MS = 3000;
export interface Toast extends Narration { id: number; kind: 'info' | 'warn' }

/// The bubbles standing, oldest first. A ring like catan's: a fourth displaces
/// the oldest at once, so a burst of moves never queues behind a timer.
export const toasts = $state<{ items: Toast[] }>({ items: [] });
const timers = new Map<number, ReturnType<typeof setTimeout>>();
let nextId = 1;

export function toast(n: Narration | string, kind: 'info' | 'warn' = 'info'): void {
  const item: Toast = { ...(typeof n === 'string' ? { who: null, text: n } : n), id: nextId++, kind };
  while (toasts.items.length >= MAX_TOASTS) dismiss(toasts.items[0].id);
  toasts.items.push(item);
  timers.set(item.id, setTimeout(() => dismiss(item.id), TOAST_MS));
}

export function dismiss(id: number): void {
  clearTimeout(timers.get(id));
  timers.delete(id);
  toasts.items = toasts.items.filter((t) => t.id !== id);
}

export function clearToasts(): void {
  for (const timer of timers.values()) clearTimeout(timer);
  timers.clear();
  toasts.items = [];
}
