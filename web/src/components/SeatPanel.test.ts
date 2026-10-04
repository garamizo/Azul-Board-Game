import { render } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import SeatPanel from './SeatPanel.svelte';
import type { GameView } from '../lib/types';

function lobby(you: string): GameView {
  return {
    id: 'abcdefghij', status: 'lobby', version: 2, numPlayers: 3, creator: 'a@x',
    you: { email: you, seat: you === 'a@x' ? 0 : null },
    seats: [{ idx: 0, kind: 'human', email: 'a@x' }, { idx: 1, kind: 'open', email: null }, { idx: 2, kind: 'bot', email: null }],
    board: null, legal: null, lastMove: null, result: null, autoPlay: false,
  };
}

describe('SeatPanel', () => {
  it('the creator manages seats and starts', () => {
    const { getByRole, getAllByRole, queryByRole } = render(SeatPanel, { view: lobby('a@x'), run: vi.fn() });
    getByRole('button', { name: 'Start' });
    getByRole('button', { name: 'Make bot' });
    getByRole('button', { name: 'Make open' });
    getByRole('button', { name: 'Leave' });
    getByRole('button', { name: 'Delete game' });
    expect(queryByRole('button', { name: 'Take this seat' })).toBeNull();
    expect(getAllByRole('listitem')).toHaveLength(3);
  });

  it('a visitor can take the open seat and nothing else', () => {
    const { getByRole, queryByRole } = render(SeatPanel, { view: lobby('b@x'), run: vi.fn() });
    getByRole('button', { name: 'Take this seat' });
    expect(queryByRole('button', { name: 'Start' })).toBeNull();
    expect(queryByRole('button', { name: 'Make bot' })).toBeNull();
  });
});
