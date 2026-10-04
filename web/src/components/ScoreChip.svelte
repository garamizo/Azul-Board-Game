<script lang="ts">
  import { Tween } from 'svelte/motion';
  import { untrack } from 'svelte';
  import { reducedMotion } from '../lib/motion';

  let { name, score, active }: { name: string; score: number; active: boolean } = $props();
  const shown = Tween.of(() => score, { duration: reducedMotion() ? 0 : 400 });
  let pops = $state<{ id: number; delta: number }[]>([]);
  let last = untrack(() => score);
  let nextId = 0;

  $effect(() => {
    const now = score;
    if (now === last) return;
    const pop = { id: nextId++, delta: now - last };
    last = now;
    if (reducedMotion()) return;  // §7: no pops without motion; the number just changes
    untrack(() => { pops = [...pops, pop]; });
    setTimeout(() => { pops = pops.filter((p) => p.id !== pop.id); }, 700);
  });
</script>

<span class="chip" class:active class:glow={active}>
  <span class="name truncate">{name}</span>
  <b>{Math.round(shown.current)}</b>
  {#each pops as p (p.id)}
    <span class="pop" class:down={p.delta < 0}>{p.delta > 0 ? `+${p.delta}` : `−${-p.delta}`}</span>
  {/each}
</span>

<style>
  .chip { position: relative; display: inline-flex; gap: 4px; align-items: baseline; max-width: 12em;
    padding: 2px 8px; border-radius: 999px; color: var(--muted); }
  .chip.active { color: var(--fg); font-weight: 700; }
  b { font-family: var(--display); }
  .pop { position: absolute; right: -6px; top: -4px; font-weight: 700; color: var(--accent); pointer-events: none;
    animation: pop 700ms ease-out forwards; }
  .pop.down { color: var(--danger); }
  @keyframes pop { from { opacity: 1; transform: translateY(0); } to { opacity: 0; transform: translateY(-16px); } }
</style>
