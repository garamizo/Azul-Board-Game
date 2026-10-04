import { describe, expect, it } from 'vitest';
import type { GameView, LegalView, WallRowView } from './types';
import {
  FLOOR, ghost, initial, tapRow, tapSource, tapWallTarget, takeMove, wallColumns, wallComplete, wallTargets,
  type TakeSel, type WallSel,
} from './selection';

const takeLegal: LegalView = {
  takes: [[0, 1, 0], [0, 1, 5], [0, 2, 3], [0, 2, 5], [5, 5, 5]],  // centre = 5 here
  wall: null,
};

function view(legal: LegalView | null): GameView {
  return {
    id: 'g', status: 'playing', version: 4, numPlayers: 2, creator: 'a', you: { email: 'a', seat: 0 },
    seats: [], legal, lastMove: null, result: null, autoPlay: false,
    board: {
      round: 1, phase: 'take', activeSeat: 0, factories: [[0, 3, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
      center: [0, 0, 0, 0, 0], centerHasFirst: true, bag: [0, 0, 0, 0, 0], discard: [0, 0, 0, 0, 0],
      players: [{ score: 0, lines: [null, null, null, [2, 1], null], wall: Array(5).fill([-1, -1, -1, -1, -1]), floor: [], hasFirst: false }],
    },
  };
}

describe('take phase', () => {
  it('source, then row, then a move', () => {
    let sel = initial(view(takeLegal)) as TakeSel;
    expect(sel).toEqual({ phase: 'take', source: null, row: null });
    sel = tapSource(sel, takeLegal, 0, 1);
    expect(sel.source).toEqual({ factory: 0, color: 1 });
    expect(takeMove(sel)).toBeNull();
    sel = tapRow(sel, takeLegal, 0);
    expect(takeMove(sel)).toEqual([0, 1, 0]);
  });

  it('ignores illegal sources and rows; tapping again cancels', () => {
    let sel = initial(view(takeLegal)) as TakeSel;
    expect(tapSource(sel, takeLegal, 1, 0)).toBe(sel);
    sel = tapSource(sel, takeLegal, 0, 2);
    expect(tapRow(sel, takeLegal, 0)).toBe(sel);
    expect(tapSource(sel, takeLegal, 0, 2).source).toBeNull();
  });

  it('the FIRST marker alone goes straight to the floor', () => {
    const sel = tapSource(initial(view(takeLegal)) as TakeSel, takeLegal, 5, 5);
    expect(takeMove(sel)).toEqual([5, 5, FLOOR]);
  });

  it('ghost shows what fits and what overflows', () => {
    const v = view(takeLegal);
    // 3 yellow onto line 0 (capacity 1): 1 placed, 2 to the floor.
    expect(ghost(v, { phase: 'take', source: { factory: 0, color: 1 }, row: 0 }, 0)).toEqual({ row: 0, color: 1, placed: 1, overflow: 2 });
    // 1 red onto line 3 that already holds 1 red (capacity 4).
    expect(ghost(v, { phase: 'take', source: { factory: 0, color: 2 }, row: 3 }, 0)).toEqual({ row: 3, color: 2, placed: 1, overflow: 0 });
    expect(ghost(v, { phase: 'take', source: { factory: 0, color: 1 }, row: FLOOR }, 0)).toEqual({ row: FLOOR, color: 1, placed: 0, overflow: 3 });
  });
});

describe('wall phase', () => {
  // Rows 1 and 3 both completed in red (2); row 4 blue (0) can only go to the floor.
  const wall: (WallRowView | null)[] = [null, { color: 2, targets: [0, 2, FLOOR] }, null, { color: 2, targets: [0, 4, FLOOR] }, { color: 0, targets: [FLOOR] }];
  const legal: LegalView = { takes: null, wall };

  it('starts on the first completed row', () => {
    const sel = initial(view(legal)) as WallSel;
    expect(sel.activeRow).toBe(1);
    expect(wallComplete(sel, wall)).toBe(false);
  });

  it('same colour cannot take the same column', () => {
    let sel = initial(view(legal)) as WallSel;
    sel = tapWallTarget(sel, wall, 1, 0);
    expect(wallTargets(wall, sel, 3)).toEqual([4, FLOOR]);
    expect(tapWallTarget(sel, wall, 3, 0)).toBe(sel);
    sel = tapWallTarget(sel, wall, 3, 4);
    sel = tapWallTarget(sel, wall, 4, FLOOR);
    expect(wallComplete(sel, wall)).toBe(true);
    expect(wallColumns(sel, wall)).toEqual([-1, 0, -1, 4, FLOOR]);
  });

  it('tapping a chosen target again clears it', () => {
    let sel = tapWallTarget(initial(view(legal)) as WallSel, wall, 1, 2);
    sel = tapWallTarget(sel, wall, 1, 2);
    expect(sel.columns[1]).toBeNull();
    expect(sel.activeRow).toBe(1);
  });
});

describe('not my turn', () => {
  it('has no selection', () => {
    expect(initial(view(null))).toBeNull();
  });
});
