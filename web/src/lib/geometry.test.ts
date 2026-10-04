import { describe, expect, it } from 'vitest';
import { factoryTiles, floorDisplay, lineBox, lineCell, wallCell } from './geometry';

describe('geometry (board2.png, 900x600, from azul/models.py)', () => {
  it('places pattern lines right-aligned and the wall on the right', () => {
    expect(lineCell(0, 0)).toEqual({ x: 390, y: 60 });
    expect(lineCell(4, 4)).toEqual({ x: 50, y: 400 });
    expect(lineBox(2)).toEqual({ x: 175, y: 185, w: 255, h: 85 });
    expect(wallCell(0, 0)).toEqual({ x: 510, y: 60 });
    expect(wallCell(4, 4)).toEqual({ x: 850, y: 400 });
  });

  it('floor overflow: marker first, 7 slots, the rest as a count', () => {
    const shown = floorDisplay([0, 0, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4], true);
    expect(shown.tiles).toEqual([5, 0, 0, 1, 2, 2, 2]);
    expect(shown.extra).toBe(6);
    expect(floorDisplay([], false)).toEqual({ tiles: [], extra: 0 });
  });

  it('expands factory counts in colour order', () => {
    expect(factoryTiles([0, 2, 1, 0, 1])).toEqual([1, 1, 2, 4]);
  });
});
