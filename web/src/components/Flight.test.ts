import { render } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import Flight from './Flight.svelte';

let finish: () => void;
beforeEach(() => {
  const finished = new Promise<void>((r) => (finish = r));
  (Element.prototype as unknown as { animate: unknown }).animate = vi.fn(() => ({ finished, cancel: vi.fn() }));
});
afterEach(() => {
  delete (Element.prototype as unknown as { animate?: unknown }).animate;
  vi.restoreAllMocks();
  document.body.innerHTML = '';
});

describe('Flight', () => {
  it('flies one sprite per arriving tile, then lands', async () => {
    document.body.insertAdjacentHTML('beforeend',
      '<div data-flight-source="2"></div><div data-flight-dest="1:3"></div><div data-flight-dest="1:floor"></div>');
    vi.spyOn(Element.prototype, 'getBoundingClientRect').mockReturnValue(new DOMRect(10, 10, 40, 40));
    const onLanded = vi.fn();
    render(Flight, { plan: { id: 8, seat: 1, source: 2, color: 0, line: { row: 3, count: 2 }, floor: [0, 5] }, onLanded });
    await vi.waitFor(() => expect(document.querySelectorAll('.flight img.sprite')).toHaveLength(4));
    expect(onLanded).not.toHaveBeenCalled();
    finish();
    await vi.waitFor(() => expect(onLanded).toHaveBeenCalledTimes(1));
    expect(document.querySelectorAll('.flight img.sprite')).toHaveLength(0);
  });

  it('flies to the visible twin of a destination', async () => {
    document.body.insertAdjacentHTML('beforeend',
      '<div data-flight-source="2"></div><div class="hidden" data-flight-dest="1:3"></div><div class="shown" data-flight-dest="1:3"></div>');
    vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function (this: Element) {
      if (this.classList.contains('hidden')) return new DOMRect(0, 0, 0, 0);
      return this.classList.contains('shown') ? new DOMRect(300, 400, 50, 50) : new DOMRect(10, 10, 40, 40);
    });
    render(Flight, { plan: { id: 8, seat: 1, source: 2, color: 0, line: { row: 3, count: 1 }, floor: [] }, onLanded: vi.fn() });
    const animate = Element.prototype.animate as unknown as ReturnType<typeof vi.fn>;
    await vi.waitFor(() => expect(animate).toHaveBeenCalledTimes(1));
    const keyframes = animate.mock.calls[0][0] as Keyframe[];
    expect(keyframes[1].transform).toBe('translate(307px, 407px) scale(0.85)');  // 300 + 25 − 18, 400 + 25 − 18
  });

  it('the largest take still lands within 650 ms', async () => {
    document.body.insertAdjacentHTML('beforeend',
      '<div data-flight-source="2"></div><div data-flight-dest="1:4"></div><div data-flight-dest="1:floor"></div>');
    vi.spyOn(Element.prototype, 'getBoundingClientRect').mockReturnValue(new DOMRect(10, 10, 40, 40));
    render(Flight, { plan: { id: 8, seat: 1, source: 2, color: 0, line: { row: 4, count: 5 }, floor: [5, 0, 0, 0, 0, 0, 0] }, onLanded: vi.fn() });
    const animate = Element.prototype.animate as unknown as ReturnType<typeof vi.fn>;
    await vi.waitFor(() => expect(animate).toHaveBeenCalledTimes(12));
    for (const [, timing] of animate.mock.calls as [Keyframe[], KeyframeAnimationOptions][]) {
      expect((timing.delay as number) + (timing.duration as number)).toBeLessThanOrEqual(650);
    }
  });

  it('a failing animation still lands, so no tile stays hidden', async () => {
    document.body.insertAdjacentHTML('beforeend', '<div data-flight-source="2"></div><div data-flight-dest="1:3"></div>');
    vi.spyOn(Element.prototype, 'getBoundingClientRect').mockReturnValue(new DOMRect(10, 10, 40, 40));
    (Element.prototype as unknown as { animate: unknown }).animate = vi.fn(() => { throw new Error('no animations here'); });
    const onLanded = vi.fn();
    render(Flight, { plan: { id: 8, seat: 1, source: 2, color: 0, line: { row: 3, count: 2 }, floor: [] }, onLanded });
    await vi.waitFor(() => expect(onLanded).toHaveBeenCalledTimes(1));
    expect(document.querySelectorAll('.flight img.sprite')).toHaveLength(0);
  });

  it('nothing to fly to still lands', async () => {
    const onLanded = vi.fn();  // no source or destination elements on the page
    render(Flight, { plan: { id: 8, seat: 1, source: 2, color: 0, line: { row: 3, count: 2 }, floor: [] }, onLanded });
    await vi.waitFor(() => expect(onLanded).toHaveBeenCalledTimes(1));
    expect(document.querySelectorAll('.flight img.sprite')).toHaveLength(0);
  });
});
