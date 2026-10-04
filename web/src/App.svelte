<script lang="ts">
  import { onMount } from 'svelte';
  import Lobby from './components/Lobby.svelte';
  import GamePage from './components/GamePage.svelte';
  import { navigate, route } from './lib/router.svelte';
  import { api } from './lib/api';

  let email = $state<string | null>(null);
  onMount(async () => {
    try { email = (await api.me()).email; } catch { /* pages show the error */ }
  });
</script>

<header class="top">
  <div class="inner">
    <a class="logo" href="/" onclick={(e) => { e.preventDefault(); navigate('/'); }}>Azul</a>
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
  .inner { display: flex; gap: 10px; align-items: center; padding: 10px 12px; max-width: 1280px; margin: 0 auto; }
  .top a { color: var(--header-ink); }
  .logo { font-family: var(--display); font-weight: 700; font-size: 1.4em; letter-spacing: 0.14em;
    text-transform: uppercase; text-decoration: none; margin-right: auto; }
  .who { max-width: 45vw; opacity: 0.75; }
</style>
