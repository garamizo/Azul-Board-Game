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
