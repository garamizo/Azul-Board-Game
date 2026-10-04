import { render } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import OpponentCard from './OpponentCard.svelte';
import type { PlayerView } from '../lib/types';

const empty = [-1, -1, -1, -1, -1];
const player: PlayerView = {
  score: 23,
  lines: [[1, 1], null, [2, 2], null, [0, 5]],
  wall: [[0, -1, -1, -1, -1], empty, empty, empty, empty],
  floor: [0, 3],
  hasFirst: true,
};

describe('OpponentCard', () => {
  it('shows pattern lines, wall and floor', () => {
    const { container, getByRole, getByText } = render(OpponentCard, { player, name: 'alice', active: false, seat: 1, onOpen: vi.fn() });
    getByRole('button', { name: "alice's board, 23 points" });
    expect(container.querySelectorAll('.line i')).toHaveLength(15);
    expect(container.querySelectorAll('.line i:not(.empty)')).toHaveLength(8);
    const third = [...container.querySelectorAll('.line')[2].querySelectorAll('i')].map((i) => i.classList.contains('empty'));
    expect(third).toEqual([true, false, false]);   // right-aligned
    expect(container.querySelectorAll('.wall i')).toHaveLength(25);
    expect(container.querySelectorAll('.wall i:not(.tint)')).toHaveLength(1);
    expect(container.querySelectorAll('.floor i')).toHaveLength(7);
    expect(container.querySelectorAll('.floor i:not(.empty)')).toHaveLength(3);
    getByText('−4');                                // 2 tiles + marker
    expect(container.querySelector('[data-flight-dest="1:4"]')).not.toBeNull();
    expect(container.querySelector('[data-flight-dest="1:floor"]')).not.toBeNull();
  });

  it('an empty floor shows no penalty', () => {
    const { container } = render(OpponentCard, {
      player: { ...player, floor: [], hasFirst: false }, name: 'b', active: false, seat: 2, onOpen: vi.fn() });
    expect(container.querySelector('.penalty')).toBeNull();
  });
});
