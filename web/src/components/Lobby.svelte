<script lang="ts">
  import { onMount } from 'svelte';
  import { api, message } from '../lib/api';
  import { flash, navigate } from '../lib/router.svelte';
  import type { GameSummary } from '../lib/types';

  let games = $state<GameSummary[]>([]);
  let me = $state('');
  let error = $state('');
  let busy = $state(false);
  const notice = flash.text;
  flash.text = '';

  async function load() {
    try {
      [games, me] = await Promise.all([api.games(), api.me().then((m) => m.email)]);
      error = '';
    } catch (e) {
      error = message(e);
    }
  }

  async function create(players: number) {
    busy = true;
    try {
      navigate(`/g/${(await api.create(players)).id}`);
    } catch (e) {
      error = message(e);
    } finally {
      busy = false;
    }
  }

  onMount(() => {
    load();
    const timer = setInterval(load, 10_000);
    return () => clearInterval(timer);
  });

  // An unseated creator still owns the game (can seat people, start or delete it).
  const isMine = (g: GameSummary) => g.creator === me || g.seats.some((s) => s.email === me);
  const mine = $derived(games.filter((g) => g.status !== 'finished' && isMine(g)));
  const joinable = $derived(games.filter((g) => g.status === 'lobby' && !isMine(g) && g.seats.some((s) => s.kind === 'open')));
  const watchable = $derived(games.filter((g) => g.status === 'playing' && !isMine(g)));
  const finished = $derived(games.filter((g) => g.status === 'finished'));
  const sections = $derived([
    { title: 'Your games', list: mine },
    { title: 'Open to join', list: joinable },
    { title: 'Watch', list: watchable },
    { title: 'Recent results', list: finished },
  ]);
  const label = (g: GameSummary) =>
    `${g.numPlayers} players · ${g.seats.filter((s) => s.kind === 'human').map((s) => s.email?.split('@')[0]).join(', ')}` +
    (g.round ? ` · round ${g.round}` : '');
</script>

{#if notice}<div class="banner">{notice}</div>{/if}
{#if error}<div class="banner error">{error}</div>{/if}

<section>
  <h2>New game</h2>
  <div class="new">
    {#each [2, 3, 4] as n}
      <button class="primary" disabled={busy} onclick={() => create(n)}>{n} players</button>
    {/each}
  </div>
</section>

{#each sections as section}
  {#if section.list.length > 0}
    <section>
      <h2>{section.title}</h2>
      <ul class="games">
        {#each section.list as g (g.id)}
          <li><a href={`/g/${g.id}`} class="truncate" onclick={(e) => { e.preventDefault(); navigate(`/g/${g.id}`); }}>{label(g)}</a></li>
        {/each}
      </ul>
    </section>
  {/if}
{/each}

<style>
  .new { display: flex; gap: 8px; flex-wrap: wrap; }
  .games { padding-left: 1.2em; display: grid; gap: 6px; }
  .games a { display: block; max-width: 100%; }
</style>
