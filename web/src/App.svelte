<script lang="ts">
  import { onMount } from 'svelte';
  import Lobby from './components/Lobby.svelte';
  import GamePage from './components/GamePage.svelte';
  import { navigate, route } from './lib/router.svelte';
  import { api } from './lib/api';

  let email = $state<string | null>(null);
  let hubUrl = $state<string | null>(null);
  onMount(async () => {
    try {
      const me = await api.me();
      email = me.email;
      hubUrl = me.hubUrl;
    } catch { /* pages show the error */ }
  });
</script>

<header class="top">
  <div class="inner">
    <a class="logo" href="/" onclick={(e) => { e.preventDefault(); navigate('/'); }}>Azul</a>
    {#if hubUrl}
      <a href={hubUrl}>Playhub</a>
      <a href={`${hubUrl}/games/azul`}>Leaderboard</a>
    {/if}
    {#if email}
      <span class="who truncate" title={email}>{email}</span>
      <a href="/cdn-cgi/access/logout">Sign out</a>
    {/if}
  </div>
</header>

<main>
  {#if route.gameId}
    {#key route.gameId}<GamePage id={route.gameId} />{/key}
  {:else}
    <Lobby />
  {/if}
</main>

<style>
  .top { background: var(--header); color: var(--header-ink); box-shadow: var(--shadow); }
  .inner { display: flex; flex-wrap: wrap; gap: 4px 10px; align-items: center; padding: 10px 12px; max-width: 1280px; margin: 0 auto; }
  .top a { color: var(--header-ink); white-space: nowrap; }
  .logo { font-family: var(--display); font-weight: 700; font-size: 1.4em; letter-spacing: 0.14em;
    text-transform: uppercase; text-decoration: none; margin-right: auto; }
  .who { max-width: 45vw; min-width: 0; opacity: 0.75; }
</style>
