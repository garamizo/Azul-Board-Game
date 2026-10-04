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
  <a class="logo" href="/" onclick={(e) => { e.preventDefault(); navigate('/'); }}>Azul</a>
  {#if email}
    <span class="who truncate" title={email}>{email}</span>
    <a href="/cdn-cgi/access/logout">Sign out</a>
  {/if}
</header>

<main>
  {#if route.gameId}
    {#key route.gameId}<GamePage id={route.gameId} />{/key}
  {:else}
    <Lobby />
  {/if}
</main>

<style>
  .top { display: flex; gap: 10px; align-items: center; padding: 10px 12px; max-width: 1280px; margin: 0 auto; }
  .logo { font-weight: 800; font-size: 1.3em; color: var(--accent); text-decoration: none; margin-right: auto; }
  .who { max-width: 45vw; color: var(--muted); }
</style>
