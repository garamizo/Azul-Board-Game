// Mirrors server/AzulServer/Games/Views.cs (System.Text.Json web defaults).
export type SeatKind = 'open' | 'human' | 'bot';
export type GameStatus = 'lobby' | 'playing' | 'finished';

export interface SeatView { idx: number; kind: SeatKind; email: string | null }
export interface PlayerView {
  score: number;
  lines: ([number, number] | null)[];  // [color, count] per pattern line
  wall: number[][];                     // -1 empty, else colour
  floor: number[];                      // ordinary tiles, colour order; FIRST marker is hasFirst
  hasFirst: boolean;
}
export interface BoardView {
  round: number;
  phase: 'take' | 'wall' | 'over';
  activeSeat: number;
  factories: number[][];  // per factory, 5 colour counts
  center: number[];       // 5 colour counts
  centerHasFirst: boolean;
  bag: number[];
  discard: number[];
  players: PlayerView[];
}
export interface WallRowView { color: number; targets: number[] }
export interface LegalView {
  takes: [number, number, number][] | null;  // [factory, color, row]; factory === factories.length is the centre
  wall: (WallRowView | null)[] | null;
}
export interface LastMoveView {
  version: number; seat: number; kind: 'take' | 'wall';
  factory: number | null; color: number | null; row: number | null; tiles: number | null; columns: number[] | null;
}
export interface ResultView { scores: number[]; winners: number[]; reason: string }
export interface GameView {
  id: string; status: GameStatus; version: number; numPlayers: number; creator: string;
  you: { email: string; seat: number | null };
  seats: SeatView[];
  board: BoardView | null;
  legal: LegalView | null;
  lastMove: LastMoveView | null;
  result: ResultView | null;
}
export interface GameSummary {
  id: string; status: GameStatus; numPlayers: number; creator: string;
  seats: SeatView[]; round: number | null; updatedAt: string;
}
export type MoveBody =
  | { version: number; requestId: string; kind: 'take'; factory: number; color: number; row: number }
  | { version: number; requestId: string; kind: 'wall'; columns: number[] };
