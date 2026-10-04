import { render } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import App from './App.svelte';
import { api } from './lib/api';

afterEach(() => vi.restoreAllMocks());

describe('App header', () => {
  it('links to Playhub and the Azul leaderboard when the server names the hub', async () => {
    vi.spyOn(api, 'games').mockResolvedValue([]);
    vi.spyOn(api, 'me').mockResolvedValue({ email: 'me@x', hubUrl: 'https://play.example' });
    const { findByRole } = render(App);
    expect((await findByRole('link', { name: 'Playhub' })).getAttribute('href')).toBe('https://play.example');
    expect((await findByRole('link', { name: 'Leaderboard' })).getAttribute('href')).toBe('https://play.example/games/azul');
  });

  it('shows no hub links without a hub', async () => {
    vi.spyOn(api, 'games').mockResolvedValue([]);
    vi.spyOn(api, 'me').mockResolvedValue({ email: 'me@x', hubUrl: null });
    const { findAllByText, queryByRole } = render(App);
    await findAllByText('me@x');
    expect(queryByRole('link', { name: 'Playhub' })).toBeNull();
    expect(queryByRole('link', { name: 'Leaderboard' })).toBeNull();
  });
});
