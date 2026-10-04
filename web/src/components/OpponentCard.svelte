<script lang="ts">
  import type { PlayerView } from '../lib/types';

  let { player, name, active, onOpen }: { player: PlayerView; name: string; active: boolean; onOpen: () => void } = $props();
  const colors = ['#2f6fd0', '#e7c23a', '#c8382e', '#2b2b2b', '#e8e2d6'];
</script>

<button class="card" class:active onclick={onOpen} aria-label={`${name}'s board`}>
  <span class="name truncate">{name}</span>
  <span class="score">{player.score}</span>
  <span class="mini" aria-hidden="true">
    {#each player.wall as row}
      {#each row as cell}<i style:background={cell >= 0 ? colors[cell] : 'transparent'}></i>{/each}
    {/each}
  </span>
</button>

<style>
  .card { display: grid; grid-template-columns: minmax(0, 1fr) auto auto; gap: 8px; align-items: center; width: 100%; text-align: left; }
  .card.active { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent); }
  .score { font-weight: 700; }
  .mini { display: grid; grid-template-columns: repeat(5, 8px); gap: 1px; }
  .mini i { width: 8px; height: 8px; border: 1px solid var(--line); }
</style>
