<script lang="ts">
  import type { GameView } from '../lib/types';
  import { seatName } from '../lib/names';
  import { sounds, soundState } from '../lib/sound.svelte';

  let { view }: { view: GameView } = $props();
  const board = $derived(view.board!);
  const yourTurn = $derived(!!view.legal);
</script>

<div class="status" data-testid="status" data-version={view.version}>
  <span>Round {board.round}</span>
  {#if view.status === 'finished'}
    <strong>Game over</strong>
  {:else if view.autoPlay}
    <strong data-testid="auto-play">{board.phase === 'wall' ? 'Scoring your wall…'
      : board.centerHasFirst ? 'Taking the first-player marker…' : 'Your only move is being played…'}</strong>
  {:else if yourTurn}
    <strong data-testid="your-turn">Your turn{board.phase === 'wall' ? ': place your tiles' : ''}</strong>
  {:else}
    <span class="truncate">{seatName(view, board.activeSeat)} is {board.phase === 'wall' ? 'placing tiles' : 'choosing'}…</span>
  {/if}
  <span class="scores">
    {#each board.players as p, i}
      <span class="score truncate" class:active={i === board.activeSeat}>{seatName(view, i)} {p.score}</span>
    {/each}
  </span>
  <button class="mute" onclick={() => sounds.toggle()} aria-label={soundState.muted ? 'Unmute' : 'Mute'}>
    {soundState.muted ? '🔇' : '🔊'}
  </button>
</div>

<style>
  .status { display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; padding: 8px 0; min-width: 0; }
  .scores { display: flex; flex-wrap: wrap; gap: 6px 10px; min-width: 0; }
  .score { max-width: 12em; color: var(--muted); }
  .score.active { color: var(--fg); font-weight: 700; }
  .mute { margin-left: auto; padding: 4px 8px; }
</style>
