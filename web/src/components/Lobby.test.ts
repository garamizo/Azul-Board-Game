import { render, within } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import Lobby from './Lobby.svelte';
import { api } from '../lib/api';
import type { GameSummary } from '../lib/types';

afterEach(() => vi.restoreAllMocks());

describe('Lobby', () => {
  it('an unseated creator still finds the game under Your games', async () => {
    const game: GameSummary = {
      id: 'abcdefghij', status: 'lobby', numPlayers: 2, creator: 'me@x', round: null, updatedAt: 't',
      seats: [{ idx: 0, kind: 'open', email: null }, { idx: 1, kind: 'human', email: 'bob@x' }],
    };
    vi.spyOn(api, 'games').mockResolvedValue([game]);
    vi.spyOn(api, 'me').mockResolvedValue({ email: 'me@x' });
    const { findByRole, queryByRole } = render(Lobby);
    const heading = await findByRole('heading', { name: 'Your games' });
    within(heading.closest('section')!).getByRole('link');
    expect(queryByRole('heading', { name: 'Open to join' })).toBeNull();
  });
});
