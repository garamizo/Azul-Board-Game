<script lang="ts">
  import type { PlayerView, WallRowView } from '../lib/types';
  import { FLOOR_BOX, floorCell, floorDisplay, lineBox, lineCell, tileAt, tileHref, wallBox, wallCell } from '../lib/geometry';
  import { wallTargets, type Arrivals, type WallSel } from '../lib/selection';

  interface Props {
    player: PlayerView;
    name: string;
    interactive?: boolean;
    legalRows?: number[];
    ghost?: { row: number; color: number; placed: number; overflow: number } | null;
    wall?: (WallRowView | null)[] | null;
    wallSel?: WallSel | null;
    pulseRow?: number | null;
    onRow?: (row: number) => void;
    onWallCell?: (row: number, col: number) => void;
    seat?: number;
    arriving?: Arrivals | null;
  }
  let { player, name, interactive = false, legalRows = [], ghost = null, wall = null, wallSel = null,
        pulseRow = null, onRow, onWallCell, seat = undefined, arriving = null }: Props = $props();

  const rows = [0, 1, 2, 3, 4];
  const floor = $derived(floorDisplay(player.floor, player.hasFirst));
  const ghostFloor = $derived(ghost ? ghost.overflow : 0);
  const key = (fn: () => void) => (e: KeyboardEvent) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fn(); } };
  const targets = (row: number) => (wall && wallSel ? wallTargets(wall, wallSel, row) : []);
</script>

<svg viewBox="0 0 900 600" class="board" role="group" aria-label={`${name}'s board`}>
  <image href="/assets/sprites/board2.png" width="900" height="600" />
  <text x="20" y="48" class="label">{name}: {player.score}</text>
  {#if seat !== undefined}
    {#each rows as row}
      {@const box = lineBox(row)}
      <rect class="dest" data-flight-dest={`${seat}:${row}`} x={box.x} y={box.y} width={box.w} height={box.h} />
    {/each}
    <rect class="dest" data-flight-dest={`${seat}:floor`} x={FLOOR_BOX.x} y={FLOOR_BOX.y} width={FLOOR_BOX.w} height={FLOOR_BOX.h} />
  {/if}

  {#each rows as row}
    {@const line = player.lines[row]}
    {@const box = lineBox(row)}
    {#if line}
      {#each Array(line[1]) as _, i}
        <image class="tile line" class:arriving={arriving?.line?.row === row && i >= arriving.line.from}
          href={tileHref(line[0])} {...tileAt(lineCell(row, i))} />
      {/each}
    {/if}
    {#if ghost && ghost.row === row}
      {#each Array(ghost.placed) as _, i}
        <image class="tile ghost" href={tileHref(ghost.color)} {...tileAt(lineCell(row, (line?.[1] ?? 0) + i))} />
      {/each}
    {/if}
    {#if interactive}
      <rect class="hit" class:legal={legalRows.includes(row)} class:chosen={ghost?.row === row} class:pulse={pulseRow === row}
        data-row={row} x={box.x} y={box.y} width={box.w} height={box.h} rx="8"
        role="button" tabindex="0" aria-label={`pattern line ${row + 1}`}
        onclick={() => onRow?.(row)} onkeydown={key(() => onRow?.(row))} />
    {:else if pulseRow === row}
      <rect class="hit pulse" x={box.x} y={box.y} width={box.w} height={box.h} rx="8" />
    {/if}
  {/each}

  {#each rows as row}
    {#each rows as col}
      {@const cell = player.wall[row][col]}
      {#if cell >= 0}
        <image class="tile wall" href={tileHref(cell)} {...tileAt(wallCell(row, col))} />
      {:else if wallSel && wall?.[row] && wallSel.columns[row] === col}
        <image class="tile ghost" href={tileHref(wall[row]!.color)} {...tileAt(wallCell(row, col))} />
      {/if}
      {#if interactive && targets(row).includes(col)}
        {@const b = wallBox(row, col)}
        <rect class="hit target" class:chosen={wallSel?.columns[row] === col}
          data-wall-row={row} data-wall-col={col} x={b.x} y={b.y} width={b.w} height={b.h} rx="8"
          role="button" tabindex="0" aria-label={`wall row ${row + 1} column ${col + 1}`}
          onclick={() => onWallCell?.(row, col)} onkeydown={key(() => onWallCell?.(row, col))} />
      {/if}
    {/each}
  {/each}

  {#each floor.tiles as color, i}
    <image class="tile floor" class:arriving={!!arriving?.floor.includes(i)} href={tileHref(color)} {...tileAt(floorCell(i))} />
  {/each}
  {#if floor.extra > 0}
    <text x="660" y="565" class="extra">+{floor.extra}</text>
  {/if}
  {#if ghost && ghostFloor > 0}
    <text x="660" y="530" class="ghost-count">+{ghostFloor} to floor</text>
  {/if}
  {#if interactive}
    <rect class="hit" class:legal={legalRows.includes(5)} class:chosen={ghost?.row === 5}
      data-row="5" x={FLOOR_BOX.x} y={FLOOR_BOX.y} width={FLOOR_BOX.w} height={FLOOR_BOX.h} rx="8"
      role="button" tabindex="0" aria-label="floor"
      onclick={() => onRow?.(5)} onkeydown={key(() => onRow?.(5))} />
  {/if}
</svg>

<style>
  .board { width: 100%; height: auto; display: block; user-select: none; }
  .label { font-family: var(--display); font-size: 40px; font-weight: 700; fill: #2b2118; paint-order: stroke; stroke: #f6efe4; stroke-width: 6px; }
  .extra, .ghost-count { font-size: 34px; font-weight: 700; fill: #a8321f; }
  .ghost { opacity: 0.5; }
  .dest { fill: transparent; pointer-events: none; }
  .tile.line, .tile.floor { transition: opacity 120ms ease-in; }
  .tile.arriving { opacity: 0; }
  .hit { fill: transparent; stroke: transparent; stroke-width: 6; cursor: default; }
  .hit.legal, .hit.target { fill: color-mix(in srgb, var(--accent) 16%, transparent);
    stroke: color-mix(in srgb, var(--accent) 60%, transparent); stroke-width: 3; cursor: pointer;
    animation: legal 1.6s ease-in-out infinite alternate; }
  .hit.chosen { fill: color-mix(in srgb, var(--accent) 28%, transparent); stroke: var(--accent); stroke-width: 4; animation: none; }
  .hit.pulse { animation: pulse 600ms ease-out; }
  @keyframes legal { from { fill: color-mix(in srgb, var(--accent) 10%, transparent); }
    to { fill: color-mix(in srgb, var(--accent) 22%, transparent); } }
  @keyframes pulse { from { fill: rgba(255, 196, 0, 0.55); } to { fill: transparent; } }
</style>
