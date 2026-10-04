import { fireEvent, render } from '@testing-library/svelte';
import { tick } from 'svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Handlers } from '../lib/events';
import type { GameView } from '../lib/types';

const live = vi.hoisted(() => ({ on: null as Handlers | null }));
vi.mock('../lib/events', () => ({
  subscribe: (_id: string, on: Handlers) => { live.on = on; return () => {}; },
}));

import GamePage from './GamePage.svelte';
import { api, ApiError } from '../lib/api';
import { sounds } from '../lib/sound.svelte';
import { clearToasts, toasts } from '../lib/toasts.svelte';

function playing(version: number, seat: number | null): GameView {
  const empty = [-1, -1, -1, -1, -1];
  const player = { score: 0, lines: [null, null, null, null, null], wall: [empty, empty, empty, empty, empty], floor: [], hasFirst: false };
  return {
    id: 'abcdefghij', status: 'playing', version, numPlayers: 2, creator: 'a@x',
    you: { email: seat === null ? 'z@x' : 'a@x', seat },
    seats: [{ idx: 0, kind: 'human', email: 'a@x' }, { idx: 1, kind: 'bot', email: null }],
    legal: seat === 0 ? { takes: [[0, 1, 0], [0, 1, 5]], wall: null } : null, lastMove: null, result: null, autoPlay: false,
    board: {
      round: 1, phase: 'take', activeSeat: 0,
      factories: [[0, 2, 0, 1, 1], [1, 1, 1, 1, 0], [0, 0, 4, 0, 0], [2, 2, 0, 0, 0], [0, 0, 0, 2, 2]],
      center: [0, 0, 0, 0, 0], centerHasFirst: true, bag: [0, 0, 0, 0, 0], discard: [0, 0, 0, 0, 0],
      players: [player, player],
    },
  };
}

function finished(version: number, seat: number | null, winners: number[]): GameView {
  return { ...playing(version, seat), status: 'finished', legal: null, result: { scores: [10, 5], winners, reason: 'normal' } };
}

afterEach(() => { vi.restoreAllMocks(); live.on = null; clearToasts(); });

describe('GamePage', () => {
  it('a 409 shows "The board changed" as one warning bubble, not a banner', async () => {
    vi.spyOn(api, 'move').mockRejectedValue(new ApiError(409, { error: 'stale', view: playing(8, 0) }));
    const { container, getByRole, queryAllByText } = render(GamePage, { id: 'abcdefghij' });
    live.on!.state(playing(7, 0));
    await tick();
    await fireEvent.click(container.querySelector('[data-factory="0"][data-color="1"]')!);
    await fireEvent.click(container.querySelector('[data-row="0"]')!);
    await fireEvent.click(getByRole('button', { name: 'Confirm' }));
    await vi.waitFor(() => expect(queryAllByText('The board changed')).toHaveLength(1));
    live.on!.state(playing(8, 0));  // not newer: no second bubble
    await tick();
    expect(queryAllByText('The board changed')).toHaveLength(1);
    expect(container.querySelector('.banner')).toBeNull();
  });

  it("an opponent's move becomes a bubble; your own does not", async () => {
    const { findAllByTestId, queryAllByTestId } = render(GamePage, { id: 'abcdefghij' });
    live.on!.state(playing(7, 0));
    const bot = playing(8, 0);
    bot.lastMove = { version: 8, seat: 1, kind: 'take', factory: 2, color: 2, row: 3, tiles: 4, columns: null };
    live.on!.state(bot);
    const bubbles = await findAllByTestId('toast');
    expect(bubbles[0].textContent).toContain('Bot 2');
    expect(bubbles[0].textContent).toContain('took 4 red from factory 3 → line 4');
    const mine = playing(9, 0);
    mine.lastMove = { version: 9, seat: 0, kind: 'take', factory: 0, color: 1, row: 0, tiles: 2, columns: null };
    live.on!.state(mine);
    await tick();
    expect(queryAllByTestId('toast')).toHaveLength(1);
  });

  it('a move answered after you left the page adds no bubble', async () => {
    let reject!: (e: unknown) => void;
    vi.spyOn(api, 'move').mockReturnValue(new Promise((_, r) => { reject = r; }));
    const { container, getByRole, unmount } = render(GamePage, { id: 'abcdefghij' });
    live.on!.state(playing(7, 0));
    await tick();
    await fireEvent.click(container.querySelector('[data-factory="0"][data-color="1"]')!);
    await fireEvent.click(container.querySelector('[data-row="0"]')!);
    await fireEvent.click(getByRole('button', { name: 'Confirm' }));
    unmount();
    reject(new ApiError(409, { error: 'stale', view: playing(8, 0) }));
    await new Promise((r) => setTimeout(r, 0));
    expect(toasts.items).toHaveLength(0);
  });

  it('at most three bubbles stand at once, newest last', async () => {
    const { queryAllByTestId } = render(GamePage, { id: 'abcdefghij' });
    live.on!.state(playing(7, 0));
    for (let v = 8; v <= 12; v++) {
      const next = playing(v, 0);
      next.lastMove = { version: v, seat: 1, kind: 'take', factory: 0, color: v % 5, row: 5, tiles: v, columns: null };
      live.on!.state(next);
    }
    await tick();
    const shown = queryAllByTestId('toast');
    expect(shown).toHaveLength(3);
    expect(shown[2].textContent).toContain('took 12');
  });

  it('spectators hear no win or lose sound; players do', async () => {
    const play = vi.spyOn(sounds, 'play').mockImplementation(() => {});
    render(GamePage, { id: 'abcdefghij' });
    live.on!.state(playing(7, null));
    live.on!.state(finished(8, null, [0]));
    expect(play).not.toHaveBeenCalledWith('win');
    expect(play).not.toHaveBeenCalledWith('lose');

    render(GamePage, { id: 'abcdefghij' });
    live.on!.state(playing(7, 0));
    live.on!.state(finished(8, 0, [0]));
    expect(play).toHaveBeenCalledWith('win');
  });
});
