import { render } from '@testing-library/svelte';
import { tick } from 'svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import ScoreChip from './ScoreChip.svelte';

const withMotion = () => {
  (Element.prototype as unknown as { animate: unknown }).animate = vi.fn();
};
afterEach(() => { delete (Element.prototype as unknown as { animate?: unknown }).animate; vi.restoreAllMocks(); });

describe('ScoreChip', () => {
  it('without motion the number changes and nothing pops', async () => {
    const { container, rerender, getByText } = render(ScoreChip, { name: 'bob', score: 5, active: false });
    await rerender({ name: 'bob', score: 12, active: false });
    await tick();
    getByText('12');
    expect(container.querySelector('.pop')).toBeNull();
  });

  it('shows the score and pops the change', async () => {
    withMotion();
    const { container, rerender, getByText } = render(ScoreChip, { name: 'bob', score: 5, active: false });
    getByText('5');
    expect(container.querySelector('.pop')).toBeNull();
    await rerender({ name: 'bob', score: 12, active: false });
    await tick();
    getByText('+7');
    await rerender({ name: 'bob', score: 10, active: false });
    await tick();
    getByText('−2');
  });

  it('the active seat glows', () => {
    const { container } = render(ScoreChip, { name: 'bob', score: 0, active: true });
    expect(container.querySelector('.chip.glow')).not.toBeNull();
  });
});
