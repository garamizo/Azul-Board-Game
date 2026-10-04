<script lang="ts">
  import { tick } from 'svelte';
  import { tileHref } from '../lib/geometry';
  import type { FlightPlan } from '../lib/flight';

  let { plan, onLanded }: { plan: FlightPlan; onLanded: () => void } = $props();
  let overlay: HTMLDivElement;
  const SIZE = 36;

  /// The first match that is laid out: the phone or desktop twin that is
  /// `display: none` has an empty box.
  function visible(selector: string): DOMRect | null {
    for (const el of document.querySelectorAll(selector)) {
      const r = el.getBoundingClientRect();
      if (r.width > 0 && r.height > 0) return r;
    }
    return null;
  }
  const mid = (r: DOMRect) => ({ x: r.left + r.width / 2 - SIZE / 2, y: r.top + r.height / 2 - SIZE / 2 });

  $effect(() => {
    const p = plan;
    const sprites: HTMLImageElement[] = [];
    const flights: Animation[] = [];
    let cancelled = false;
    (async () => {
      try {
        await tick();
        if (cancelled) return;
        const from = visible(`[data-flight-source="${p.source}"]`);
        const legs = [
          { to: p.line ? visible(`[data-flight-dest="${p.seat}:${p.line.row}"]`) : null,
            colors: p.line ? Array<number>(p.line.count).fill(p.color) : [] },
          { to: visible(`[data-flight-dest="${p.seat}:floor"]`), colors: p.floor },
        ];
        let k = 0;
        for (const leg of legs) {
          if (!from || !leg.to) continue;
          const a = mid(from), b = mid(leg.to);
          for (const color of leg.colors) {
            const img = document.createElement('img');
            img.className = 'sprite';
            img.alt = '';
            img.src = tileHref(color);
            overlay.append(img);
            sprites.push(img);
            const dx = ((k % 3) - 1) * 12, dy = (Math.floor(k / 3) % 2) * 12 - 6;
            flights.push(img.animate([
              { transform: `translate(${a.x + dx}px, ${a.y + dy}px) scale(1)` },
              { transform: `translate(${b.x}px, ${b.y}px) scale(0.85)` },
            ], { duration: 450, delay: Math.min(k, 5) * 40, easing: 'ease-in-out', fill: 'both' }));  // ≤ 650 ms
            k++;
          }
        }
        await Promise.all(flights.map((f) => f.finished.catch(() => {})));
      } catch {
        // The animation is decoration; the board must not wait on it.
      } finally {
        // Land even if the flight failed: the arriving tiles are hidden until then.
        if (!cancelled) {
          sprites.forEach((s) => s.remove());
          onLanded();
        }
      }
    })();
    return () => {
      cancelled = true;
      flights.forEach((f) => f.cancel());
      sprites.forEach((s) => s.remove());
    };
  });
</script>

<div class="flight" bind:this={overlay} aria-hidden="true"></div>

<style>
  .flight { position: fixed; inset: 0; pointer-events: none; z-index: 8; }
  .flight :global(.sprite) { position: absolute; left: 0; top: 0; width: 36px; height: 36px; }
</style>
