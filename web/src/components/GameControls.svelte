<script lang="ts">
  import type { GameView } from '../lib/types';
  import { api } from '../lib/api';
  import { seatName } from '../lib/names';

  interface Props { view: GameView; run: (action: () => Promise<unknown>) => void }
  let { view, run }: Props = $props();

  const me = $derived(view.you.email);
  const mySeat = $derived(view.you.seat);
  const isCreator = $derived(view.creator === me);
  const playing = $derived(view.status === 'playing');
  const handedToBot = $derived(mySeat !== null && view.seats[mySeat].kind === 'bot');
  const otherHumans = $derived(view.seats.filter((s) => s.kind === 'human' && s.email !== me));
</script>

<div class="controls">
  {#if playing && mySeat !== null}
    {#if handedToBot}
      <button onclick={() => run(() => api.takeBack(view.id, mySeat))}>Take my seat back</button>
    {:else}
      <button onclick={() => { if (confirm('Let a bot play your seat?')) run(() => api.toBot(view.id, mySeat)); }}>Hand my seat to a bot</button>
    {/if}
  {/if}
  {#if playing && isCreator}
    {#each otherHumans as seat (seat.idx)}
      <button onclick={() => { if (confirm(`Let a bot play ${seatName(view, seat.idx)}'s seat?`)) run(() => api.toBot(view.id, seat.idx)); }}>
        Bot for {seatName(view, seat.idx)}
      </button>
    {/each}
  {/if}
  {#if isCreator}
    <button class="danger" onclick={() => { if (confirm('Delete this game?')) run(() => api.remove(view.id)); }}>Delete game</button>
  {/if}
</div>

<style>
  .controls { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
  .controls button { max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .danger { color: var(--danger); }
</style>
