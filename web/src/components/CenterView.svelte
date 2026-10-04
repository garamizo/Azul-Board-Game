<script lang="ts">
  import { COLOR_NAMES, tileHref } from '../lib/geometry';

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

<div class="center" role="group" aria-label="centre" data-flight-source={index}>
  {#if hasFirst}
    <button class="group" class:selected={selectedColor === 5} disabled={!canPick(5)}
      data-factory={index} data-color="5" onclick={() => onPick?.(5)}
      title="First-player marker only" aria-label="take only the first-player marker">
      <img src={tileHref(5)} alt="" />
    </button>
  {/if}
  {#each counts as n, color}
    {#if n > 0}
      <button class="group" class:selected={selectedColor === color} disabled={!canPick(color)}
        data-factory={index} data-color={color} onclick={() => onPick?.(color)}
        aria-label={`take ${n} ${COLOR_NAMES[color]} from the centre`}>
        <img src={tileHref(color)} alt="" /><span class="badge">{n}</span>
      </button>
    {/if}
  {/each}
  {#if !hasFirst && counts.every((n) => n === 0)}<span class="empty">Centre is empty</span>{/if}
</div>

<style>
  .center { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: center; align-content: center;
    min-height: 64px; padding: 12px; border-radius: 14px;
    background: radial-gradient(circle at 50% 40%, color-mix(in srgb, var(--felt) 80%, white), var(--felt));
    box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.35); }
  .group { position: relative; padding: 0; border: none; background: transparent; width: 36px; height: 36px; border-radius: 6px; }
  .group:disabled { opacity: 1; }  /* not your turn: still readable */
  .group img { display: block; width: 36px; height: 36px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.45); }
  .badge { position: absolute; right: -6px; bottom: -6px; min-width: 18px; height: 18px; padding: 0 4px; border-radius: 9px;
    background: var(--card); color: var(--fg); font-size: 12px; font-weight: 700; line-height: 18px; text-align: center;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.4); }
  .group.selected { outline: 3px solid var(--accent); outline-offset: 2px; }
  .empty { color: rgba(255, 255, 255, 0.8); font-size: 0.9em; }
  @media (min-width: 900px) { .group, .group img { width: 44px; height: 44px; } }
</style>
