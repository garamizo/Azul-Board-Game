<script lang="ts">
  import { factoryTiles, tileHref } from '../lib/geometry';

  interface Props {
    index: number;
    counts: number[];
    selectedColor?: number | null;
    canPick?: (color: number) => boolean;
    onPick?: (color: number) => void;
  }
  let { index, counts, selectedColor = null, canPick = () => false, onPick }: Props = $props();
  const tiles = $derived(factoryTiles(counts));
  const slots = [[35, 35], [95, 35], [35, 95], [95, 95]];
</script>

<svg viewBox="0 0 130 130" class="factory" data-flight-source={index} class:empty={tiles.length === 0} role="group" aria-label={`factory ${index + 1}`}>
  <image href="/assets/sprites/factory.png" width="130" height="130" />
  {#each tiles as color, i}
    <image class="tile" class:selected={selectedColor === color} class:pickable={canPick(color)}
      href={tileHref(color)} x={slots[i][0] - 25} y={slots[i][1] - 25} width="50" height="50"
      data-factory={index} data-color={color} role="button" tabindex="0"
      aria-label={`take colour ${color} from factory ${index + 1}`}
      onclick={() => onPick?.(color)}
      onkeydown={(e) => { if (e.key === 'Enter') onPick?.(color); }} />
  {/each}
</svg>

<style>
  .factory { width: 100%; height: auto; display: block; }
  .factory.empty { opacity: 0.35; }
  .tile.pickable { cursor: pointer; }
  .tile.selected { outline: 3px solid var(--accent); filter: drop-shadow(0 0 6px var(--accent)); }
</style>
