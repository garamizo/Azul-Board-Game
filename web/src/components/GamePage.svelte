<script lang="ts">
  import { onMount } from 'svelte';
  import { api, ApiError, message } from '../lib/api';
  import { subscribe, type LinkStatus } from '../lib/events';
  import { navigate } from '../lib/router.svelte';
  import { sounds } from '../lib/sound.svelte';
  import type { GameView, MoveBody } from '../lib/types';
  import SeatPanel from './SeatPanel.svelte';
  import Table from './Table.svelte';
  import GameControls from './GameControls.svelte';
  import Toasts from './Toasts.svelte';
  import { narrate } from '../lib/narrate';
  import { clearToasts, toast } from '../lib/toasts.svelte';

  let { id }: { id: string } = $props();
  let view = $state<GameView | null>(null);
  let link = $state<'connecting' | LinkStatus>('connecting');
  let error = $state('');
  // Set on unmount: a request still in flight must not write a bubble into the
  // next page or navigate away from it.
  let disposed = false;

  function accept(next: GameView) {
    if (disposed) return;
    const prev = view;
    if (prev && next.version <= prev.version) return;
    view = next;
    if (!prev) return;
    const last = next.lastMove;
    if (last && last.version === next.version && last.seat !== next.you.seat)
      sounds.play(next.seats[last.seat]?.kind === 'bot' ? 'botMove' : 'select');
    if (next.legal && !next.autoPlay && !prev.legal) sounds.play('yourTurn');
    if (next.status === 'finished' && prev.status !== 'finished') {
      if (next.you.seat !== null)  // spectators neither win nor lose
        sounds.play(next.result?.winners.includes(next.you.seat) ? 'win' : 'lose');
    } else if (next.board && prev.board && next.board.round > prev.board.round) sounds.play('score');
    for (const n of narrate(prev, next)) toast(n);
  }

  function failed(e: unknown) {
    if (disposed) return;
    if (e instanceof ApiError && e.status === 409 && e.body?.view) {
      accept(e.body.view);
      toast('The board changed', 'warn');
    } else if (e instanceof ApiError && e.status === 404) {
      navigate('/', 'This game was deleted');
    } else {
      error = message(e);
    }
  }

  /// Seat and lobby actions.
  async function run(action: () => Promise<unknown>) {
    try {
      const result = await action();
      error = '';
      if (result && typeof result === 'object' && 'version' in result) accept(result as GameView);
      else if (result === null) navigate('/', 'Game deleted');
    } catch (e) {
      failed(e);
    }
  }

  async function send(body: MoveBody): Promise<'ok' | 'invalid' | 'other'> {
    try {
      accept(await api.move(id, body));
      error = '';
      return 'ok';
    } catch (e) {
      failed(e);
      return e instanceof ApiError && e.status === 400 ? 'invalid' : 'other';
    }
  }

  onMount(() => {
    const unsubscribe = subscribe(id, {
      state: accept,
      deleted: () => navigate('/', 'This game was deleted'),
      status: (s) => (link = s),
    });
    // Bubbles belong to this game: none follow you to the lobby or another game.
    return () => { disposed = true; unsubscribe(); clearToasts(); };
  });

  $effect(() => {
    const yours = !!view?.legal;
    const update = () => { document.title = yours && document.hidden ? '● Your turn — Azul' : 'Azul'; };
    update();
    document.addEventListener('visibilitychange', update);
    return () => { document.removeEventListener('visibilitychange', update); document.title = 'Azul'; };
  });

</script>

{#if link === 'reconnecting'}<div class="banner">Reconnecting…</div>{/if}
{#if link === 'signed-out'}<div class="banner error">Signed out. Reload the page to sign in again.</div>{/if}
{#if error}<div class="banner error">{error}</div>{/if}
<Toasts />

{#if !view}
  <p>Loading…</p>
{:else if view.status === 'lobby'}
  <SeatPanel {view} {run} />
{:else}
  <Table {view} {send} />
  <GameControls {view} {run} />
{/if}
