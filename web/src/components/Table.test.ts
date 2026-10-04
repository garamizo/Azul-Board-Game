import { fireEvent, render } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { tick } from 'svelte';
import Table from './Table.svelte';
import type { GameView, MoveBody, PlayerView } from '../lib/types';

function myTurn(): GameView {
  const empty = [-1, -1, -1, -1, -1];
  const player = { score: 0, lines: [null, null, null, null, null], wall: [empty, empty, empty, empty, empty], floor: [], hasFirst: false };
  return {
    id: 'abcdefghij', status: 'playing', version: 7, numPlayers: 2, creator: 'a@x', you: { email: 'a@x', seat: 0 },
    seats: [{ idx: 0, kind: 'human', email: 'a@x' }, { idx: 1, kind: 'bot', email: null }],
    legal: { takes: [[0, 1, 0], [0, 1, 5]], wall: null }, lastMove: null, result: null, autoPlay: false,
    board: {
      round: 1, phase: 'take', activeSeat: 0,
      factories: [[0, 2, 0, 1, 1], [1, 1, 1, 1, 0], [0, 0, 4, 0, 0], [2, 2, 0, 0, 0], [0, 0, 0, 2, 2]],
      center: [0, 0, 0, 0, 0], centerHasFirst: true, bag: [0, 0, 0, 0, 0], discard: [0, 0, 0, 0, 0],
      players: [player, player],
    },
  };
}

describe('Table', () => {
  it('Confirm is enabled only once a move is chosen, and sends it', async () => {
    const send = vi.fn(async (_: MoveBody) => 'ok' as const);
    const { container, getByRole } = render(Table, { view: myTurn(), send });
    const confirm = getByRole('button', { name: 'Confirm' }) as HTMLButtonElement;
    expect(confirm.disabled).toBe(true);
    await fireEvent.click(container.querySelector('[data-factory="0"][data-color="1"]')!);
    expect(confirm.disabled).toBe(true);
    await fireEvent.click(container.querySelector('[data-row="0"]')!);
    expect(confirm.disabled).toBe(false);
    await fireEvent.click(confirm);
    expect(send).toHaveBeenCalledTimes(1);
    expect(send.mock.calls[0][0]).toMatchObject({ version: 7, kind: 'take', factory: 0, color: 1, row: 0 });
  });

  it('Confirm is disabled while a move is pending', async () => {
    let release!: () => void;
    const send = vi.fn(() => new Promise<'ok'>((r) => { release = () => r('ok'); }));
    const { container, getByRole } = render(Table, { view: myTurn(), send });
    await fireEvent.click(container.querySelector('[data-factory="0"][data-color="1"]')!);
    await fireEvent.click(container.querySelector('[data-row="0"]')!);
    const confirm = getByRole('button', { name: 'Confirm' }) as HTMLButtonElement;
    await fireEvent.click(confirm);
    await fireEvent.click(confirm);
    expect(send).toHaveBeenCalledTimes(1);
    expect(confirm.disabled).toBe(true);
    release();
  });

  it('shows neither the discard nor the bag', () => {
    const v = myTurn();
    v.board!.bag = [12, 10, 9, 14, 11];
    v.board!.discard = [0, 1, 0, 0, 2];
    const { container, queryByTestId } = render(Table, { view: v, send: vi.fn() });
    expect(queryByTestId('supply')).toBeNull();
    expect(container.textContent).not.toMatch(/discard|bag/i);
  });

  it('a forced turn shows the auto-play step, not Your turn or Confirm', () => {
    const v = myTurn();
    v.legal = null;
    v.autoPlay = true;
    v.board!.phase = 'wall';
    const { getByTestId, queryByTestId, queryByRole } = render(Table, { view: v, send: vi.fn() });
    expect(getByTestId('auto-play').textContent).toContain('Scoring your wall…');
    expect(queryByTestId('your-turn')).toBeNull();
    expect(queryByRole('button', { name: 'Confirm' })).toBeNull();
  });

  it('a forced take says the first-player marker is being taken', () => {
    const v = { ...myTurn(), legal: null, autoPlay: true };
    const { getByTestId } = render(Table, { view: v, send: vi.fn() });
    expect(getByTestId('auto-play').textContent).toContain('Taking the first-player marker…');
  });

  it('your turn is a pill', () => {
    const { getByTestId } = render(Table, { view: myTurn(), send: vi.fn() });
    expect(getByTestId('your-turn').classList.contains('pill')).toBe(true);
  });

  it('spectators see no Confirm button', () => {
    const v = { ...myTurn(), you: { email: 'z@x', seat: null }, legal: null };
    const { queryByRole } = render(Table, { view: v, send: vi.fn() });
    expect(queryByRole('button', { name: 'Confirm' })).toBeNull();
  });
});

describe('Table: tile flight', () => {
  const empty = [-1, -1, -1, -1, -1];
  const fresh = (): PlayerView => ({ score: 0, lines: [null, null, null, null, null], wall: [empty, empty, empty, empty, empty], floor: [], hasFirst: false });
  function watching(version: number): GameView {
    const v = myTurn();
    v.version = version;
    v.legal = null;
    v.board!.players = [fresh(), fresh()];
    return v;
  }
  function botTook(version: number, edit: (v: GameView) => void = () => {}): GameView {
    const v = watching(version);
    v.board!.players[1] = { ...fresh(), lines: [null, null, null, [2, 3], null], floor: [2] };
    v.lastMove = { version, seat: 1, kind: 'take', factory: 2, color: 2, row: 3, tiles: 4, columns: null };
    edit(v);
    return v;
  }
  let land: () => void;
  beforeEach(() => {
    const finished = new Promise<void>((r) => (land = r));
    (Element.prototype as unknown as { animate: unknown }).animate = vi.fn(() => ({ finished, cancel: vi.fn() }));
    vi.spyOn(Element.prototype, 'getBoundingClientRect').mockReturnValue(new DOMRect(0, 0, 50, 50));
  });
  afterEach(() => {
    delete (Element.prototype as unknown as { animate?: unknown }).animate;
    vi.restoreAllMocks();
  });
  const sprites = () => document.querySelectorAll('.flight img.sprite');

  it('a consecutive take flies the tiles, hides them until they land, then shows them', async () => {
    const { container, rerender } = render(Table, { view: watching(7), send: vi.fn() });
    await rerender({ view: botTook(8), send: vi.fn() });
    await vi.waitFor(() => expect(sprites()).toHaveLength(4));   // 3 to line 4, 1 to the floor
    expect(container.querySelectorAll('.others-desktop image.tile.arriving')).toHaveLength(4);
    land();
    await vi.waitFor(() => expect(sprites()).toHaveLength(0));
    expect(container.querySelectorAll('image.tile.arriving')).toHaveLength(0);
  });

  it('reduced motion flies nothing even with Web Animations', async () => {
    vi.spyOn(window, 'matchMedia').mockImplementation((q: string) =>
      ({ matches: q.includes('reduce'), media: q, addEventListener() {}, removeEventListener() {} }) as unknown as MediaQueryList);
    const { container, rerender } = render(Table, { view: watching(7), send: vi.fn() });
    await rerender({ view: botTook(8), send: vi.fn() });
    await tick();
    expect(sprites()).toHaveLength(0);
    expect(container.querySelectorAll('image.tile.arriving')).toHaveLength(0);
  });

  it('a version jump flies nothing and hides nothing (guard: passes before the feature too)', async () => {
    const { container, rerender } = render(Table, { view: watching(6), send: vi.fn() });
    await rerender({ view: botTook(8), send: vi.fn() });
    await tick();
    expect(sprites()).toHaveLength(0);
    expect(container.querySelectorAll('image.tile.arriving')).toHaveLength(0);
  });

  it('a newer take replaces the flight in the air', async () => {
    const { container, rerender } = render(Table, { view: watching(7), send: vi.fn() });
    const v8 = botTook(8);
    await rerender({ view: v8, send: vi.fn() });
    await vi.waitFor(() => expect(sprites()).toHaveLength(4));
    const v9 = botTook(9, (v) => {
      v.board!.players[0] = { ...fresh(), lines: [[1, 1], null, null, null, null] };
      v.lastMove = { version: 9, seat: 0, kind: 'take', factory: 0, color: 1, row: 0, tiles: 1, columns: null };
    });
    await rerender({ view: v9, send: vi.fn() });
    await vi.waitFor(() => expect(sprites()).toHaveLength(1));    // only the new take's tile
    expect(container.querySelectorAll('.others-desktop image.tile.arriving')).toHaveLength(0);
    expect(container.querySelectorAll('.mine image.tile.arriving')).toHaveLength(1);
  });

  it('a newer non-take version cancels the flight', async () => {
    const { container, rerender } = render(Table, { view: watching(7), send: vi.fn() });
    await rerender({ view: botTook(8), send: vi.fn() });
    await vi.waitFor(() => expect(sprites()).toHaveLength(4));
    const v9 = botTook(9, (v) => {
      v.lastMove = { version: 9, seat: 1, kind: 'wall', factory: null, color: null, row: null, tiles: null, columns: [-1, -1, -1, -1, -1] };
    });
    await rerender({ view: v9, send: vi.fn() });
    await tick();
    expect(sprites()).toHaveLength(0);
    expect(container.querySelectorAll('image.tile.arriving')).toHaveLength(0);
  });

  it('taking the first-player marker alone flies the marker to the floor', async () => {
    const { container, rerender } = render(Table, { view: watching(7), send: vi.fn() });
    const v8 = watching(8);
    v8.board!.centerHasFirst = false;
    v8.board!.players[1] = { ...fresh(), hasFirst: true };
    v8.lastMove = { version: 8, seat: 1, kind: 'take', factory: 5, color: 5, row: 5, tiles: 0, columns: null };
    await rerender({ view: v8, send: vi.fn() });
    await vi.waitFor(() => expect(sprites()).toHaveLength(1));
    expect((sprites()[0] as HTMLImageElement).src).toContain('tile_first.png');
    expect(container.querySelectorAll('.others-desktop image.tile.floor.arriving')).toHaveLength(1);
  });
});
