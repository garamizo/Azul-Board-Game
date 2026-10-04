/// <reference types="vitest/config" />
import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { svelteTesting } from '@testing-library/svelte/vite';

export default defineConfig({
  // svelteTesting: Svelte's browser build under Vitest and automatic cleanup
  // of mounted components after each test.
  plugins: [svelte(), svelteTesting()],
  server: {
    port: 5173,
    proxy: { '/api': { target: 'http://127.0.0.1:5080' } },
  },
  test: {
    environment: 'jsdom',
    include: ['src/**/*.test.ts'],
    setupFiles: ['src/test-setup.ts'],
  },
});
