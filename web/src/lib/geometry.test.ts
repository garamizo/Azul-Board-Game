import { describe, expect, it } from 'vitest';
import { factoryTiles, floorDisplay, floorPenalty, lineBox, lineCell, ringLayout, wallCell, wallColor } from './geometry';

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

describe('printed board and scoring', () => {
  it('wall colours follow board2.png', () => {
    expect([0, 1, 2, 3, 4].map((c) => wallColor(0, c))).toEqual([0, 1, 2, 3, 4]);
    expect([0, 1, 2, 3, 4].map((c) => wallColor(1, c))).toEqual([4, 0, 1, 2, 3]);
  });

  it('floor penalty mirrors FloorToScore, capped at -14', () => {
    expect([0, 1, 2, 3, 4, 5, 6, 7, 9].map(floorPenalty)).toEqual([0, -1, -2, -4, -6, -8, -11, -14, -14]);
  });
});

describe('ringLayout', () => {
  for (const n of [5, 7, 9]) {
    it(`${n} factories fit around the centre`, () => {
      const { d, r, disc, centres } = ringLayout(n);
      expect(centres).toHaveLength(n);
      expect((d / 100) * 410).toBeGreaterThanOrEqual(90);
      expect(centres[0].x).toBeCloseTo(50);
      expect(centres[0].y).toBeLessThan(50);           // factory 1 at 12 o'clock
      expect(centres[1].x).toBeGreaterThan(50);         // then clockwise
      for (let i = 0; i < n; i++) {
        const a = centres[i], b = centres[(i + 1) % n];
        expect(Math.hypot(a.x - b.x, a.y - b.y)).toBeGreaterThanOrEqual(d + 4 - 1e-9);  // circles apart
        expect(a.x - d / 2).toBeGreaterThanOrEqual(0);
        expect(a.x + d / 2).toBeLessThanOrEqual(100);
        expect(a.y - d / 2).toBeGreaterThanOrEqual(0);
        expect(a.y + d / 2).toBeLessThanOrEqual(100);
      }
      expect(disc / 2 + d / 2).toBeLessThan(r);         // disc does not touch any factory
    });
  }
});
