<script lang="ts">
  import type { Snippet } from 'svelte';
  let { title, onClose, children }: { title: string; onClose: () => void; children: Snippet } = $props();
</script>

<div class="backdrop" role="presentation" onclick={onClose}>
  <div class="sheet" role="dialog" aria-label={title} tabindex="-1"
    onclick={(e) => e.stopPropagation()} onkeydown={(e) => { if (e.key === 'Escape') onClose(); }}>
    <header><strong class="truncate">{title}</strong><button onclick={onClose}>Close</button></header>
    {@render children()}
  </div>
</div>

<style>
  .backdrop { position: fixed; inset: 0; background: rgba(0, 0, 0, 0.4); display: flex; align-items: flex-end; z-index: 10; }
  .sheet { background: var(--bg); width: 100%; max-height: 90vh; overflow: auto; padding: 12px; border-radius: 12px 12px 0 0; }
  header { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 8px; }
</style>
