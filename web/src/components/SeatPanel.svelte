<script lang="ts">
  import type { GameView } from '../lib/types';
  import { api } from '../lib/api';

  interface Props { view: GameView; run: (action: () => Promise<unknown>) => void }
  let { view, run }: Props = $props();

  const me = $derived(view.you.email);
  const isCreator = $derived(view.creator === me);
  const seated = $derived(view.you.seat !== null);
  const anyHuman = $derived(view.seats.some((s) => s.kind === 'human'));
  let copied = $state(false);

  async function share() {
    try { await navigator.clipboard.writeText(location.href); copied = true; } catch { /* clipboard blocked */ }
  }
</script>

<section class="lobby">
  <h2>Waiting for players</h2>
  <p>Share this page's address with friends. Empty seats become bots when the game starts.
    <button onclick={share}>{copied ? 'Copied' : 'Copy link'}</button></p>
  <ol class="seats">
    {#each view.seats as seat}
      <li>
        <span class="who truncate">
          {#if seat.kind === 'human'}{seat.email === me ? 'You' : seat.email}{:else if seat.kind === 'bot'}Bot{:else}Open seat{/if}
        </span>
        <span class="actions">
          {#if seat.kind === 'open' && !seated}
            <button class="primary" onclick={() => run(() => api.claim(view.id, seat.idx))}>Take this seat</button>
          {/if}
          {#if isCreator && seat.kind === 'open'}
            <button onclick={() => run(() => api.setKind(view.id, seat.idx, 'bot'))}>Make bot</button>
          {/if}
          {#if isCreator && seat.kind === 'bot'}
            <button onclick={() => run(() => api.setKind(view.id, seat.idx, 'open'))}>Make open</button>
          {/if}
          {#if seat.kind === 'human' && seat.email === me}
            <button onclick={() => run(() => api.release(view.id, seat.idx))}>Leave</button>
          {:else if seat.kind === 'human' && isCreator}
            <button onclick={() => run(() => api.release(view.id, seat.idx))}>Remove</button>
          {/if}
        </span>
      </li>
    {/each}
  </ol>
  {#if isCreator}
    <div class="creator">
      <button class="primary" disabled={!anyHuman} onclick={() => run(() => api.start(view.id))}>Start</button>
      <button class="danger" onclick={() => { if (confirm('Delete this game?')) run(() => api.remove(view.id)); }}>Delete game</button>
    </div>
  {/if}
</section>

<style>
  .seats { padding-left: 1.2em; display: grid; gap: 8px; }
  .seats li { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: space-between; }
  .who { max-width: 60vw; }
  .actions { display: flex; gap: 6px; flex-wrap: wrap; }
  .creator { display: flex; gap: 8px; margin-top: 12px; }
  .danger { color: var(--danger); }
</style>
