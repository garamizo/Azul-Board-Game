<script lang="ts">
  import type { WallRowView } from '../lib/types';
  import { COLOR_NAMES } from '../lib/geometry';
  import { FLOOR, wallTargets, type WallSel } from '../lib/selection';

  let { wall, sel, onPick }: { wall: (WallRowView | null)[]; sel: WallSel; onPick: (row: number, target: number) => void } = $props();
</script>

<div class="chooser">
  {#each wall as option, row}
    {#if option}
      <div class="row" class:active={sel.activeRow === row}>
        <span>Line {row + 1} ({COLOR_NAMES[option.color]}):</span>
        {#each wallTargets(wall, sel, row) as target}
          <button class:chosen={sel.columns[row] === target} data-wall-target={`${row}-${target}`}
            onclick={() => onPick(row, target)}>
            {target === FLOOR ? 'floor' : `col ${target + 1}`}
          </button>
        {/each}
      </div>
    {/if}
  {/each}
</div>

<style>
  .chooser { display: grid; gap: 6px; }
  .row { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
  .row.active span { font-weight: 700; }
  button.chosen { background: var(--accent); color: white; }
</style>
