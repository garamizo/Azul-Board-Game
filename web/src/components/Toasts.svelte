<script lang="ts">
  import { TILE_COLORS } from '../lib/geometry';
  import { dismiss, toasts } from '../lib/toasts.svelte';
</script>

<!-- Taps go through a bubble to the board under it; only × catches them. -->
<div class="toasts" role="status" aria-live="polite">
  {#each toasts.items as t (t.id)}
    <div class="toast" class:warn={t.kind === 'warn'} data-testid="toast">
      {#if t.color !== undefined}<i class="dot" style:background={TILE_COLORS[t.color]}></i>{/if}
      <span class="msg">
        {#if t.who}<strong class="who truncate" title={t.who}>{t.who}</strong>{/if}
        <span class="text">{t.text}</span>
      </span>
      <button class="close" aria-label="Dismiss" onclick={() => dismiss(t.id)}>×</button>
    </div>
  {/each}
</div>

<style>
  .toasts { position: fixed; top: calc(8px + env(safe-area-inset-top, 0px)); left: 50%; transform: translateX(-50%);
    width: min(92vw, 420px); display: grid; gap: 6px; z-index: 9; pointer-events: none; }
  .toast { display: flex; align-items: center; gap: 8px; padding: 6px 4px 6px 12px; background: var(--card); color: var(--fg);
    border: 1px solid var(--line); border-radius: 12px; box-shadow: var(--shadow); animation: toast-in 150ms ease-out; }
  .toast.warn { background: var(--warn-bg); }
  .dot { width: 12px; height: 12px; border-radius: 3px; flex: none; border: 1px solid rgba(0, 0, 0, 0.25); }
  .msg { display: flex; gap: 0.35em; min-width: 0; flex: 1; align-items: baseline; }
  .who { max-width: 40%; flex: none; }
  .text { display: -webkit-box; -webkit-line-clamp: 2; line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
  .close { pointer-events: auto; flex: none; padding: 2px 8px; border: none; background: transparent; color: var(--muted);
    font-size: 1.2em; line-height: 1; }
  @keyframes toast-in { from { opacity: 0; transform: translateY(-6px); } }
</style>
