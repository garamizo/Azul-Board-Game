<script lang="ts">
  import { tileHref } from '../lib/geometry';

  interface Props {
    index: number;  // the centre's factory index (= number of factories)
    counts: number[];
    hasFirst: boolean;
    selectedColor?: number | null;
    canPick?: (color: number) => boolean;
    onPick?: (color: number) => void;
  }
  let { index, counts, hasFirst, selectedColor = null, canPick = () => false, onPick }: Props = $props();
</script>

<div class="center" role="group" aria-label="centre">
  {#if hasFirst}
    <button class="group" class:selected={selectedColor === 5} disabled={!canPick(5)}
      data-factory={index} data-color="5" onclick={() => onPick?.(5)} title="First-player marker only">
      <img src={tileHref(5)} alt="first-player marker" /><span>only</span>
    </button>
  {/if}
  {#each counts as n, color}
    {#if n > 0}
      <button class="group" class:selected={selectedColor === color} disabled={!canPick(color)}
        data-factory={index} data-color={color} onclick={() => onPick?.(color)}>
        <img src={tileHref(color)} alt={`colour ${color}`} /><span>×{n}</span>
      </button>
    {/if}
  {/each}
  {#if !hasFirst && counts.every((n) => n === 0)}<span class="muted">Centre is empty</span>{/if}
</div>

<style>
  .center { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; min-height: 48px; }
  .group { display: inline-flex; align-items: center; gap: 4px; padding: 4px 8px; }
  .group img { width: 32px; height: 32px; }
  .group.selected { border-color: #1f5fa8; box-shadow: 0 0 0 2px #1f5fa8; }
  .muted { color: var(--muted); }
</style>
