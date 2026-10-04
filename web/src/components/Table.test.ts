import { fireEvent, render } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import Table from './Table.svelte';
import type { GameView, MoveBody } from '../lib/types';

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

  it('shows bag and discard counts', () => {
    const v = myTurn();
    v.board!.bag = [12, 10, 9, 14, 11];
    v.board!.discard = [0, 1, 0, 0, 2];
    const { getByTestId } = render(Table, { view: v, send: vi.fn() });
    const text = getByTestId('supply').textContent!.replace(/\s+/g, ' ');
    expect(text).toContain('Bag 12 10 9 14 11');
    expect(text).toContain('Discard 0 1 0 0 2');
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

  it('spectators see no Confirm button', () => {
    const v = { ...myTurn(), you: { email: 'z@x', seat: null }, legal: null };
    const { queryByRole } = render(Table, { view: v, send: vi.fn() });
    expect(queryByRole('button', { name: 'Confirm' })).toBeNull();
  });
});
