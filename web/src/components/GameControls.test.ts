import { render } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import GameControls from './GameControls.svelte';
import type { GameView, SeatView } from '../lib/types';

function game(you: string, seats: SeatView[], status: GameView['status'] = 'playing'): GameView {
  const seat = seats.find((s) => s.email === you)?.idx ?? null;
  return {
    id: 'abcdefghij', status, version: 9, numPlayers: seats.length, creator: 'a@x', you: { email: you, seat },
    seats, board: null, legal: null, lastMove: null, result: null,
  };
}

const names = (container: HTMLElement) => [...container.querySelectorAll('button')].map((b) => b.textContent!.trim());

describe('GameControls', () => {
  it('a seated creator can hand any human seat to a bot and delete', () => {
    const { container } = render(GameControls, { view: game('a@x', [
      { idx: 0, kind: 'human', email: 'a@x' }, { idx: 1, kind: 'human', email: 'b@x' }, { idx: 2, kind: 'bot', email: null }]), run: vi.fn() });
    expect(names(container)).toEqual(['Hand my seat to a bot', 'Bot for b', 'Delete game']);
  });

  it('an unseated creator still manages the game', () => {
    const { container } = render(GameControls, { view: game('a@x', [
      { idx: 0, kind: 'human', email: 'b@x' }, { idx: 1, kind: 'human', email: 'c@x' }]), run: vi.fn() });
    expect(names(container)).toEqual(['Bot for b', 'Bot for c', 'Delete game']);
  });

  it('a player only manages their own seat', () => {
    const seats: SeatView[] = [{ idx: 0, kind: 'human', email: 'a@x' }, { idx: 1, kind: 'human', email: 'b@x' }];
    expect(names(render(GameControls, { view: game('b@x', seats), run: vi.fn() }).container)).toEqual(['Hand my seat to a bot']);
    const handed: SeatView[] = [seats[0], { idx: 1, kind: 'bot', email: 'b@x' }];
    expect(names(render(GameControls, { view: game('b@x', handed), run: vi.fn() }).container)).toEqual(['Take my seat back']);
  });

  it('a finished game can only be deleted, by its creator', () => {
    const seats: SeatView[] = [{ idx: 0, kind: 'human', email: 'a@x' }, { idx: 1, kind: 'human', email: 'b@x' }];
    expect(names(render(GameControls, { view: game('a@x', seats, 'finished'), run: vi.fn() }).container)).toEqual(['Delete game']);
    expect(names(render(GameControls, { view: game('b@x', seats, 'finished'), run: vi.fn() }).container)).toEqual([]);
  });
});
