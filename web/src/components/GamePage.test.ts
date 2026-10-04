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

afterEach(() => { vi.restoreAllMocks(); live.on = null; });

describe('GamePage', () => {
  it('"The board changed" goes away once a newer version arrives', async () => {
    vi.spyOn(api, 'move').mockRejectedValue(new ApiError(409, { error: 'stale', view: playing(8, 0) }));
    const { container, getByRole, queryByText } = render(GamePage, { id: 'abcdefghij' });
    live.on!.state(playing(7, 0));
    await tick();
    await fireEvent.click(container.querySelector('[data-factory="0"][data-color="1"]')!);
    await fireEvent.click(container.querySelector('[data-row="0"]')!);
    await fireEvent.click(getByRole('button', { name: 'Confirm' }));
    await vi.waitFor(() => expect(queryByText('The board changed')).not.toBeNull());
    live.on!.state(playing(8, 0));  // not newer: the notice stays
    await tick();
    expect(queryByText('The board changed')).not.toBeNull();
    live.on!.state(playing(9, 0));
    await tick();
    expect(queryByText('The board changed')).toBeNull();
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
