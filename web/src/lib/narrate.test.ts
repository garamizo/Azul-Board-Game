import { describe, expect, it } from 'vitest';
import { narrate } from './narrate';
import type { GameView, LastMoveView, PlayerView } from './types';

const empty = [-1, -1, -1, -1, -1];
const player = (score = 0): PlayerView =>
  ({ score, lines: [null, null, null, null, null], wall: [empty, empty, empty, empty, empty], floor: [], hasFirst: false });

function view(version: number, you: number | null = 0): GameView {
  return {
    id: 'g', status: 'playing', version, numPlayers: 2, creator: 'a@x', you: { email: 'a@x', seat: you },
    seats: [{ idx: 0, kind: 'human', email: 'a@x' }, { idx: 1, kind: 'bot', email: null }],
    legal: null, lastMove: null, result: null, autoPlay: false,
    board: {
      round: 1, phase: 'take', activeSeat: 0,
      factories: [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
      center: [0, 0, 0, 0, 0], centerHasFirst: true, bag: [0, 0, 0, 0, 0], discard: [0, 0, 0, 0, 0],
      players: [player(), player()],
    },
  };
}
const take = (version: number, seat: number, factory: number, color: number, row: number, tiles: number): LastMoveView =>
  ({ version, seat, kind: 'take', factory, color, row, tiles, columns: null });
const wall = (version: number, seat: number, columns: number[]): LastMoveView =>
  ({ version, seat, kind: 'wall', factory: null, color: null, row: null, tiles: null, columns });
const after = (prev: GameView, move: LastMoveView, edit: (v: GameView) => void = () => {}): GameView => {
  const v = structuredClone(prev);
  v.version = move.version;
  v.lastMove = move;
  edit(v);
  return v;
};

describe('narrate: takes', () => {
  it('from a factory to a line', () => {
    const p = view(7);
    expect(narrate(p, after(p, take(8, 1, 2, 0, 3, 3)))).toEqual([
      { who: 'Bot 2', text: 'took 3 blue from factory 3 → line 4', color: 0 },
    ]);
  });

  it('from the centre to the floor, with the marker', () => {
    const p = view(7);
    const n = after(p, take(8, 1, 5, 2, 5, 2), (v) => { v.board!.centerHasFirst = false; });
    expect(narrate(p, n)[0].text).toBe('took 2 red from the centre → floor (+ first player)');
  });

  it('the first-player marker alone', () => {
    const p = view(7);
    expect(narrate(p, after(p, take(8, 1, 5, 5, 5, 0)))[0]).toEqual({ who: 'Bot 2', text: 'took the first-player marker', color: 5 });
  });

  it('a version gap infers nothing: the marker may have gone with an earlier move', () => {
    const p = view(6);
    const n = after(p, take(8, 1, 5, 2, 3, 2), (v) => { v.board!.centerHasFirst = false; });
    expect(narrate(p, n)[0].text).toBe('took 2 red from the centre → line 4');
  });
});

describe('narrate: wall moves', () => {
  it('tiles placed, with the score change', () => {
    const p = view(7);
    const n = after(p, wall(8, 1, [0, -1, 2, -1, -1]), (v) => { v.board!.players[1].score = 7; });
    expect(narrate(p, n)[0].text).toBe('placed 2 tiles on their wall (+7)');
  });

  it('one tile', () => {
    const p = view(7);
    expect(narrate(p, after(p, wall(8, 1, [3, -1, -1, -1, -1])))[0].text).toBe('placed 1 tile on their wall');
  });

  it('lines sent to the floor', () => {
    const p = view(7);
    p.board!.players[1].score = 5;
    const n = after(p, wall(8, 1, [5, -1, -1, -1, -1]), (v) => { v.board!.players[1].score = 3; });
    expect(narrate(p, n)[0].text).toBe('sent their completed lines to the floor (−2)');
  });

  it('a forced scoring turn', () => {
    const p = view(7);
    expect(narrate(p, after(p, wall(8, 1, [-1, -1, -1, -1, -1])))[0].text).toBe('scored the round');
  });

  it('a version gap shows no score change', () => {
    const p = view(5);
    const n = after(p, wall(8, 1, [0, -1, -1, -1, -1]), (v) => { v.board!.players[1].score = 9; });
    expect(narrate(p, n)[0].text).toBe('placed 1 tile on their wall');
  });
});

describe('narrate: who and when', () => {
  it('a new round', () => {
    const p = view(7);
    const n = after(p, wall(8, 1, [-1, -1, -1, -1, -1]), (v) => { v.board!.round = 2; });
    expect(narrate(p, n)[1]).toEqual({ who: null, text: 'Round 2 begins' });
  });

  it('no new round once the game is over', () => {
    const p = view(7);
    const n = after(p, wall(8, 1, [-1, -1, -1, -1, -1]), (v) => { v.board!.round = 2; v.status = 'finished'; });
    expect(narrate(p, n)).toHaveLength(1);
  });

  it('your own move made by hand is not narrated', () => {
    const p = view(7);
    expect(narrate(p, after(p, take(8, 0, 1, 1, 0, 1)))).toEqual([]);
  });

  it('your forced move is narrated', () => {
    const p = view(7);
    p.autoPlay = true;
    expect(narrate(p, after(p, take(8, 0, 5, 5, 5, 0)))[0].who).toBe('You');
  });

  it('a bot playing your seat is narrated', () => {
    const p = view(7);
    p.seats[0] = { idx: 0, kind: 'bot', email: 'a@x' };
    expect(narrate(p, after(p, take(8, 0, 1, 1, 0, 1)))[0].who).toBe('a (bot)');
  });

  it('spectators hear every seat', () => {
    const p = view(7, null);
    expect(narrate(p, after(p, take(8, 0, 1, 1, 0, 1)))[0].who).toBe('a');
  });

  it('a lastMove older than the view is not narrated', () => {
    const p = view(7);
    const n = after(p, take(8, 1, 1, 1, 0, 1));
    n.version = 9;
    expect(narrate(p, n)).toEqual([]);
  });

  it('no board, no narration', () => {
    const p = view(7);
    const n = after(p, take(8, 1, 1, 1, 0, 1));
    p.board = null;
    expect(narrate(p, n)).toEqual([]);
  });
});
