import type { GameView, LegalView, WallRowView } from './types';

export const FLOOR = 5;
export const FIRST = 5;

export type Source = { factory: number; color: number };
export type TakeSel = { phase: 'take'; source: Source | null; row: number | null };
export type WallSel = { phase: 'wall'; activeRow: number | null; columns: (number | null)[] };
export type Selection = TakeSel | WallSel;
type WallOptions = (WallRowView | null)[];

/// Null when it is not the viewer's turn.
export function initial(view: GameView): Selection | null {
  const legal = view.legal;
  if (!legal) return null;
  if (legal.takes) return { phase: 'take', source: null, row: null };
  const columns: (number | null)[] = [null, null, null, null, null];
  return { phase: 'wall', columns, activeRow: firstOpenRow(legal.wall!, columns) };
}

export const canPick = (legal: LegalView, factory: number, color: number): boolean =>
  !!legal.takes?.some((t) => t[0] === factory && t[1] === color);

export function legalRows(legal: LegalView, source: Source | null): number[] {
  if (!source || !legal.takes) return [];
  return legal.takes.filter((t) => t[0] === source.factory && t[1] === source.color).map((t) => t[2]);
}

export function tapSource(sel: TakeSel, legal: LegalView, factory: number, color: number): TakeSel {
  if (sel.source?.factory === factory && sel.source.color === color) return { phase: 'take', source: null, row: null };
  if (!canPick(legal, factory, color)) return sel;
  const rows = legalRows(legal, { factory, color });
  // One destination (the FIRST marker alone, or floor only): choose it now.
  return { phase: 'take', source: { factory, color }, row: rows.length === 1 ? rows[0] : null };
}

export function tapRow(sel: TakeSel, legal: LegalView, row: number): TakeSel {
  if (!sel.source || !legalRows(legal, sel.source).includes(row)) return sel;
  return { ...sel, row: sel.row === row ? null : row };
}

export function takeMove(sel: Selection | null): [number, number, number] | null {
  return sel?.phase === 'take' && sel.source && sel.row !== null ? [sel.source.factory, sel.source.color, sel.row] : null;
}

/// Targets still open for `row`: its own options minus wall columns already
/// chosen by another completed line of the same colour.
export function wallTargets(wall: WallOptions, sel: WallSel, row: number): number[] {
  const option = wall[row];
  if (!option) return [];
  const taken = new Set<number>();
  wall.forEach((o, r) => {
    const c = sel.columns[r];
    if (r !== row && o && c !== null && c !== FLOOR && o.color === option.color) taken.add(c);
  });
  return option.targets.filter((t) => t === FLOOR || !taken.has(t));
}

export function firstOpenRow(wall: WallOptions, columns: (number | null)[]): number | null {
  const i = wall.findIndex((o, r) => o !== null && columns[r] === null);
  return i < 0 ? null : i;
}

export function tapWallTarget(sel: WallSel, wall: WallOptions, row: number, target: number): WallSel {
  if (!wallTargets(wall, sel, row).includes(target)) return sel;
  const columns = [...sel.columns];
  columns[row] = columns[row] === target ? null : target;
  return { phase: 'wall', columns, activeRow: firstOpenRow(wall, columns) ?? row };
}

export const wallComplete = (sel: WallSel, wall: WallOptions): boolean =>
  wall.every((o, r) => o === null || sel.columns[r] !== null);

export const wallColumns = (sel: WallSel, wall: WallOptions): number[] =>
  wall.map((o, r) => (o === null ? -1 : (sel.columns[r] as number)));

/// Where the selected tiles would land on the viewer's board.
export function ghost(view: GameView, sel: TakeSel, seat: number):
  { row: number; color: number; placed: number; overflow: number } | null {
  const board = view.board;
  if (!board || !sel.source || sel.row === null) return null;
  const { factory, color } = sel.source;
  if (color === FIRST) return { row: FLOOR, color, placed: 0, overflow: 0 };
  const count = factory < board.factories.length ? board.factories[factory][color] : board.center[color];
  if (sel.row === FLOOR) return { row: FLOOR, color, placed: 0, overflow: count };
  const line = board.players[seat].lines[sel.row];
  const have = line && line[0] === color ? line[1] : 0;
  const placed = Math.min(count, sel.row + 1 - have);
  return { row: sel.row, color, placed, overflow: count - placed };
}
