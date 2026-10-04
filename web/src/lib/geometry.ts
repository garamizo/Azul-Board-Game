// Coordinates in board2.png's own 900x600 space, taken from azul/models.py
// (which draws at 2/3 scale): tile centres, 85 px apart, tiles 75 px wide.
export const TILE = 75;
export const FLOOR_SLOTS = 7;
export const COLOR_NAMES = ['blue', 'yellow', 'red', 'black', 'white', 'first player'];

export interface Point { x: number; y: number }
export interface Box { x: number; y: number; w: number; h: number }

/// i-th tile of pattern line `row`, counted from the right end.
export const lineCell = (row: number, i: number): Point => ({ x: 390 - 85 * i, y: 60 + 85 * row });
export const lineBox = (row: number): Box => ({ x: 430 - 85 * (row + 1), y: 15 + 85 * row, w: 85 * (row + 1), h: 85 });
export const wallCell = (row: number, col: number): Point => ({ x: 510 + 85 * col, y: 60 + 85 * row });
export const wallBox = (row: number, col: number): Box => ({ x: 467.5 + 85 * col, y: 17.5 + 85 * row, w: 85, h: 85 });
export const floorCell = (i: number): Point => ({ x: 50 + 90 * i, y: 550 });
export const FLOOR_BOX: Box = { x: 9, y: 505, w: 630, h: 90 };

export const tileHref = (color: number): string =>
  `/assets/sprites/tile_${['blue', 'yellow', 'red', 'black', 'white', 'first'][color]}.png`;

/// Top-left corner for a tile image centred on p.
export const tileAt = (p: Point, size = TILE) => ({ x: p.x - size / 2, y: p.y - size / 2, width: size, height: size });

/// The FIRST marker (5) first, then ordinary tiles; at most 7 slots, the rest as `extra`.
export function floorDisplay(floor: number[], hasFirst: boolean): { tiles: number[]; extra: number } {
  const all = hasFirst ? [5, ...floor] : [...floor];
  return { tiles: all.slice(0, FLOOR_SLOTS), extra: Math.max(0, all.length - FLOOR_SLOTS) };
}

export function factoryTiles(counts: number[]): number[] {
  return counts.flatMap((n, color) => Array<number>(n).fill(color));
}

/// Display colours for the mini boards and bubbles, matching the tile sprites;
/// index 5 is the first-player marker.
export const TILE_COLORS = ['#2f6fd0', '#e7c23a', '#c8382e', '#2b2b2b', '#e8e2d6', '#f6efe4'];

/// The printed wall (board2.png): row r, column c holds colour (c − r) mod 5.
export const wallColor = (row: number, col: number): number => (col - row + 5) % 5;

const FLOOR_SCORE = [0, -1, -2, -4, -6, -8, -11, -14];
/// Mirrors FloorToScore (AzulLibrary/Logic.cs); n counts the first-player marker.
export const floorPenalty = (n: number): number => FLOOR_SCORE[Math.min(n, FLOOR_SCORE.length - 1)];

export interface RingLayout { d: number; r: number; disc: number; centres: Point[] }

/// Factories on a ring around the centre, in percent of a square box: the
/// largest factory (capped at 24 %) whose neighbours stay 4 % apart as circles,
/// factory 1 at 12 o'clock, then clockwise.
export function ringLayout(n: number): RingLayout {
  const s = Math.sin(Math.PI / n);
  const d = Math.min(24, (98 * s - 4) / (1 + s));
  const r = 50 - d / 2 - 1;
  const centres = Array.from({ length: n }, (_, i) => {
    const a = (2 * Math.PI * i) / n - Math.PI / 2;
    return { x: 50 + r * Math.cos(a), y: 50 + r * Math.sin(a) };
  });
  return { d, r, disc: 2 * (r - d / 2) - 4, centres };
}
