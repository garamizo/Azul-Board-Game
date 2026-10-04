<script lang="ts">
  import type { BoardView } from '../lib/types';
  import { ringLayout } from '../lib/geometry';
  import FactoryView from './FactoryView.svelte';
  import CenterView from './CenterView.svelte';

  interface Props {
    board: BoardView;
    selected: (factory: number) => number | null;
    canPick: (factory: number, color: number) => boolean;
    onPick: (factory: number, color: number) => void;
  }
  let { board, selected, canPick, onPick }: Props = $props();
  const centre = $derived(board.factories.length);
  const ring = $derived(ringLayout(board.factories.length));
</script>

<!-- One DOM, two layouts: a grid, or (wide desktop) a ring around a felt disc.
     The ring's custom properties are inert until the container query uses them. -->
<div class="tray" style:--d={`${ring.d}%`} style:--disc={`${ring.disc}%`}>
  <div class="factories">
    {#each board.factories as counts, i}
      <div class="slot" style:--x={`${ring.centres[i].x}%`} style:--y={`${ring.centres[i].y}%`}>
        <FactoryView index={i} {counts} selectedColor={selected(i)}
          canPick={(c) => canPick(i, c)} onPick={(c) => onPick(i, c)} />
      </div>
    {/each}
    <div class="centre">
      <CenterView index={centre} counts={board.center} hasFirst={board.centerHasFirst}
        selectedColor={selected(centre)} canPick={(c) => canPick(centre, c)} onPick={(c) => onPick(centre, c)} />
    </div>
  </div>
</div>

<style>
  .factories { display: grid; grid-template-columns: repeat(auto-fill, minmax(64px, 1fr)); gap: 6px; }
  .centre { grid-column: 1 / -1; margin-top: 2px; }
  @media (min-width: 900px) {
    /* Only a desktop tray is a container, so on a wide phone the ring query never matches. */
    .tray { container: market / inline-size; }
    .factories { grid-template-columns: repeat(auto-fill, minmax(96px, 1fr)); gap: 8px; }
  }
  @container market (min-width: 410px) {
    .factories { display: block; position: relative; aspect-ratio: 1; }
    .slot { position: absolute; width: var(--d); left: calc(var(--x) - var(--d) / 2); top: calc(var(--y) - var(--d) / 2); }
    .centre { position: absolute; width: var(--disc); aspect-ratio: 1; margin: 0;
      left: calc(50% - var(--disc) / 2); top: calc(50% - var(--disc) / 2); }
    .centre :global(.center) { width: 100%; height: 100%; border-radius: 50%; padding: 15%; }
    .centre :global(.group), .centre :global(.group img) { width: 36px; height: 36px; }
    /* Factory boxes overlap at the corners; only their tiles take taps. */
    .slot :global(svg.factory) { pointer-events: none; }
    .slot :global(svg.factory .tile) { pointer-events: auto; }
  }
</style>
