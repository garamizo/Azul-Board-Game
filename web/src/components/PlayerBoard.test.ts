import { render } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import PlayerBoard from './PlayerBoard.svelte';
import type { PlayerView } from '../lib/types';

const player: PlayerView = {
  score: 12,
  lines: [[1, 1], null, [2, 2], null, null],
  wall: [[-1, 0, -1, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]],
  floor: [0, 0, 1, 2, 2, 3, 3, 4, 4],
  hasFirst: true,
};

describe('PlayerBoard', () => {
  it('draws lines, wall, floor and the overflow badge', () => {
    const { container, getByText } = render(PlayerBoard, { player, name: 'alice' });
    expect(container.querySelectorAll('image.tile.line')).toHaveLength(3);
    expect(container.querySelectorAll('image.tile.wall')).toHaveLength(1);
    expect(container.querySelectorAll('image.tile.floor')).toHaveLength(7);
    getByText('+3');
    getByText('alice: 12');
  });

  it('only an interactive board has tap targets', () => {
    const passive = render(PlayerBoard, { player, name: 'a' });
    expect(passive.container.querySelector('[data-row]')).toBeNull();
    const active = render(PlayerBoard, { player, name: 'a', interactive: true, legalRows: [1, 5] });
    expect(active.container.querySelectorAll('[data-row]')).toHaveLength(6);
    expect(active.container.querySelectorAll('.hit.legal')).toHaveLength(2);
  });
});
