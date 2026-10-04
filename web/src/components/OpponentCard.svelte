<script lang="ts">
  import type { PlayerView } from '../lib/types';
  import { FLOOR_SLOTS, TILE_COLORS, floorDisplay, floorPenalty, wallColor } from '../lib/geometry';

  let { player, name, active, seat, onOpen }:
    { player: PlayerView; name: string; active: boolean; seat: number; onOpen: () => void } = $props();
  const rows = [0, 1, 2, 3, 4];
  const floor = $derived(floorDisplay(player.floor, player.hasFirst));
  const penalty = $derived(floorPenalty(player.floor.length + (player.hasFirst ? 1 : 0)));
  /// Colour in slot i of pattern line `row` (slots counted from the left), or null.
  const filled = (row: number, i: number): number | null => {
    const line = player.lines[row];
    return line && i >= row + 1 - line[1] ? line[0] : null;
  };
</script>

<button class="card" class:active onclick={onOpen} aria-label={`${name}'s board, ${player.score} points`}>
  <span class="head">
    <span class="name truncate">{name}</span>
    {#if player.hasFirst}<span class="marker" title="first player">1</span>{/if}
    <span class="score">{player.score}</span>
  </span>
  <span class="grids" aria-hidden="true">
    <span class="lines">
      {#each rows as row}
        <span class="line" data-flight-dest={`${seat}:${row}`}>
          {#each Array(row + 1) as _, i}
            {@const c = filled(row, i)}
            <i class:empty={c === null} style:background={c === null ? null : TILE_COLORS[c]}></i>
          {/each}
        </span>
      {/each}
    </span>
    <span class="wall">
      {#each rows as row}
        {#each rows as col}
          {@const c = player.wall[row][col]}
          <i class:tint={c < 0} style:background={TILE_COLORS[c >= 0 ? c : wallColor(row, col)]}></i>
        {/each}
      {/each}
    </span>
  </span>
  <span class="floor" aria-hidden="true" data-flight-dest={`${seat}:floor`}>
    {#each Array(FLOOR_SLOTS) as _, i}
      {@const c = floor.tiles[i]}
      <i class:empty={c === undefined} class:first={c === 5}
        style:background={c === undefined || c === 5 ? null : TILE_COLORS[c]}>{c === 5 ? '1' : ''}</i>
    {/each}
    {#if floor.extra > 0}<span class="extra">+{floor.extra}</span>{/if}
    {#if penalty < 0}<span class="penalty">{`−${-penalty}`}</span>{/if}
  </span>
</button>

<style>
  .card { --cell: 11px; display: grid; gap: 6px; width: 100%; text-align: left; padding: 10px 12px;
    background: var(--card); border-radius: 14px; box-shadow: var(--shadow); }
  .card.active { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent); }
  .head { display: flex; align-items: center; gap: 8px; min-width: 0; }
  .name { flex: 1; }
  .score { font-family: var(--display); font-weight: 700; font-size: 1.15em; }
  .marker { width: 1.4em; height: 1.4em; border-radius: 50%; display: inline-grid; place-items: center;
    background: #fff; color: #2b2118; border: 1px solid var(--line); font-size: 0.8em; font-weight: 700; }
  .grids { display: flex; gap: 14px; align-items: flex-start; }
  .lines, .wall { display: grid; gap: 2px; }
  .line { display: flex; justify-content: flex-end; gap: 2px; }
  .wall { grid-template-columns: repeat(5, var(--cell)); }
  i { display: block; width: var(--cell); height: var(--cell); border-radius: 2px; flex: none; }
  i.empty { border: 1px solid var(--line); }
  i.tint { opacity: 0.22; }
  .floor { display: flex; align-items: center; gap: 2px; }
  .floor i.first { background: #fff; color: #2b2118; border: 1px solid var(--line); font: 700 8px/1 system-ui;
    display: grid; place-items: center; font-style: normal; }
  .extra { margin-left: 4px; color: var(--muted); font-size: 0.85em; }
  .penalty { margin-left: 6px; color: var(--danger); font-weight: 700; font-size: 0.9em; }
  @media (min-width: 400px) { .card { --cell: 14px; } }
</style>
