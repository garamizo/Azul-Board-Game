<script lang="ts">
  import type { GameView, MoveBody } from '../lib/types';
  import Market from './Market.svelte';
  import Flight from './Flight.svelte';
  import type { FlightPlan } from '../lib/flight';
  import { reducedMotion } from '../lib/motion';
  import { floorDisplay } from '../lib/geometry';
  import PlayerBoard from './PlayerBoard.svelte';
  import StatusBar from './StatusBar.svelte';
  import WallChooser from './WallChooser.svelte';
  import OpponentCard from './OpponentCard.svelte';
  import Sheet from './Sheet.svelte';
  import { seatName } from '../lib/names';
  import { oneAtATime } from '../lib/submit';
  import { sounds } from '../lib/sound.svelte';
  import {
    arrivals, canPick, ghost, initial, legalRows, tapRow, tapSource, tapWallTarget, takeMove, wallColumns, wallComplete,
    type Arrivals, type Selection,
  } from '../lib/selection';

  interface Props { view: GameView; send: (body: MoveBody) => Promise<'ok' | 'invalid' | 'other'> }
  let { view, send }: Props = $props();

  let sel = $state<Selection | null>(null);
  let pending = $state(false);
  let shake = $state(false);
  let openSeat = $state<number | null>(null);
  let seenVersion = -1;
  let shownBefore: GameView | null = null;  // the version rendered before this one
  let flight = $state<FlightPlan | null>(null);
  let arriving = $state<{ seat: number; a: Arrivals } | null>(null);

  // A new version resets the selection, and a take that follows the version on
  // screen flies its tiles (a jump after a gap just shows the new state).
  $effect.pre(() => {
    if (view.version !== seenVersion) {
      const before = shownBefore;
      seenVersion = view.version;
      shownBefore = view;
      sel = initial(view);
      flight = null;
      arriving = null;
      const m = view.lastMove;
      if (before?.board && view.board && view.version === before.version + 1 && m?.version === view.version
          && m.kind === 'take' && !reducedMotion()) {
        const p = view.board.players[m.seat];
        const a = arrivals(before.board.players[m.seat], p);
        const shown = floorDisplay(p.floor, p.hasFirst).tiles;
        arriving = { seat: m.seat, a };
        flight = {
          id: view.version, seat: m.seat, source: m.factory!, color: m.color!,
          line: a.line ? { row: a.line.row, count: Math.min(a.line.to - a.line.from, 5) } : null,
          floor: a.floor.map((i) => shown[i]),
        };
      }
    }
  });
  const landing = (seat: number) => (arriving?.seat === seat ? arriving.a : null);

  const board = $derived(view.board!);
  const mySeat = $derived(view.you.seat);
  const legal = $derived(view.legal);
  const takeSel = $derived(sel?.phase === 'take' ? sel : null);
  const wallSel = $derived(sel?.phase === 'wall' ? sel : null);
  const ready = $derived(
    !!legal && (takeMove(sel) !== null || (!!wallSel && !!legal.wall && wallComplete(wallSel, legal.wall))));
  const others = $derived(board.players.map((_, i) => i).filter((i) => i !== mySeat));
  const pulse = (seat: number) =>
    view.lastMove && view.lastMove.version === view.version && view.lastMove.seat === seat && view.lastMove.kind === 'take'
      ? view.lastMove.row : null;

  function pick(factory: number, color: number) {
    if (!legal || !takeSel) return;
    sel = tapSource(takeSel, legal, factory, color);
    sounds.play('select');
  }

  const confirm = oneAtATime(async () => {
    if (!legal || !ready) return;
    pending = true;
    try {
      const requestId = crypto.randomUUID();
      const move = takeMove(sel);
      const body: MoveBody = move
        ? { version: view.version, requestId, kind: 'take', factory: move[0], color: move[1], row: move[2] }
        : { version: view.version, requestId, kind: 'wall', columns: wallColumns(wallSel!, legal.wall!) };
      const result = await send(body);
      if (result === 'invalid') {
        sounds.play('invalid');
        shake = true;
        setTimeout(() => (shake = false), 400);
      }
    } finally {
      pending = false;
    }
  });
</script>

<div class="table">
  <div class="status-area"><StatusBar {view} /></div>

  <section class="market panel" aria-label="factories">
    <Market {board}
      selected={(f) => (takeSel?.source?.factory === f ? takeSel.source.color : null)}
      canPick={(f, c) => !!legal && canPick(legal, f, c)}
      onPick={pick} />
  </section>

  {#if mySeat !== null}
    <section class="mine panel" class:shake aria-label="your board">
      <PlayerBoard player={board.players[mySeat]} name={seatName(view, mySeat)} seat={mySeat} arriving={landing(mySeat)}
        interactive={!!legal}
        legalRows={legal && takeSel ? legalRows(legal, takeSel.source) : []}
        ghost={takeSel ? ghost(view, takeSel, mySeat) : null}
        wall={legal?.wall ?? null} wallSel={wallSel}
        pulseRow={pulse(mySeat)}
        onRow={(row) => { if (legal && takeSel) sel = tapRow(takeSel, legal, row); }}
        onWallCell={(row, col) => { if (legal?.wall && wallSel) sel = tapWallTarget(wallSel, legal.wall, row, col); }} />
      {#if legal?.wall && wallSel}
        <WallChooser wall={legal.wall} sel={wallSel}
          onPick={(row, target) => { sel = tapWallTarget(wallSel, legal.wall!, row, target); }} />
      {/if}
      {#if legal}
        <div class="actions">
          <button class="primary" disabled={!ready || pending} onclick={confirm}>Confirm</button>
          {#if sel && (takeSel?.source || wallSel?.columns.some((c) => c !== null))}
            <button onclick={() => (sel = initial(view))}>Clear</button>
          {/if}
        </div>
      {/if}
    </section>
  {/if}

  <section class="others" aria-label="other players">
    <div class="others-desktop">
      {#each others as i}
        <div class="panel opp"><PlayerBoard player={board.players[i]} name={seatName(view, i)} seat={i} arriving={landing(i)} pulseRow={pulse(i)} /></div>
      {/each}
    </div>
    <div class="others-phone">
      {#each others as i}
        <OpponentCard player={board.players[i]} name={seatName(view, i)} seat={i}
          active={i === board.activeSeat && view.status === 'playing'} onOpen={() => (openSeat = i)} />
      {/each}
    </div>
  </section>

  {#if view.result}
    <section class="result panel" data-testid="result">
      <h2>{view.result.winners.includes(mySeat ?? -1) ? 'You win!' : 'Game over'}</h2>
      <ol>
        {#each view.result.scores as score, i}
          <li class:winner={view.result.winners.includes(i)}>{seatName(view, i)}: {score}</li>
        {/each}
      </ol>
    </section>
  {/if}
</div>

{#if flight}
  {#key flight.id}
    <Flight plan={flight} onLanded={() => { flight = null; arriving = null; }} />
  {/key}
{/if}

{#if openSeat !== null}
  <Sheet title={`${seatName(view, openSeat)}'s board`} onClose={() => (openSeat = null)}>
    <PlayerBoard player={board.players[openSeat]} name={seatName(view, openSeat)} seat={openSeat} />
  </Sheet>
{/if}

<style>
  .table { display: grid; gap: 12px; grid-template-columns: minmax(0, 1fr); }
  .actions { display: flex; gap: 8px; margin-top: 8px; }
  .others-desktop { display: none; }
  .others-phone { display: grid; gap: 6px; }
  .result .winner { font-weight: 700; }
  .result h2 { font-family: var(--display); margin-top: 0; }
  .shake { animation: shake 300ms; }
  @keyframes shake { 25% { transform: translateX(-6px); } 75% { transform: translateX(6px); } }
  @media (min-width: 900px) {
    .table { grid-template-columns: minmax(280px, 2fr) 3fr; grid-template-areas: 'status status' 'market mine' 'others others' 'result result'; }
    .status-area { grid-area: status; }
    .market { grid-area: market; }
    .mine { grid-area: mine; }
    .others { grid-area: others; }
    .result { grid-area: result; }
    /* Always three slots, so an opponent's board is the same size at any player count. */
    .others-desktop { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
    .others-phone { display: none; }
  }
</style>
