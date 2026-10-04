# Azul Web — Plan 3 of 4: Web Client Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Svelte single-page app that plays Azul against friends and bots from phone or desktop: lobby, seat panel, an SVG table with tap-to-select moves for both phases, live updates, sounds, and session-expiry handling, proven by Playwright end to end.

**Architecture:** Vite + Svelte 5 (runes) + TypeScript, no SvelteKit. Pure logic (API client, session, SSE client, selection state machine, board geometry) lives in `src/lib` with Vitest tests; components render SVG boards from the server's `GameView`. The server serves the built bundle (`web/dist`) on the same origin.

**Tech Stack:** svelte 5.57.1, vite 8.3.2, @sveltejs/vite-plugin-svelte 7.3.1, typescript 6.0.3, svelte-check 4.7.6, vitest 5.0.3, jsdom 30.1.1, @testing-library/svelte 5.4.2, @playwright/test 1.63.0. Node 25 on the host.

**Spec:** `docs/superpowers/specs/2026-10-03-azul-web-design.md` (§5). Requires Plans 1–2.

## Global Constraints

Everything in Plans 1–2's Global Constraints applies. In addition:

- The JSON contract is Plan 2 Task 10's views; `web/src/lib/types.ts` mirrors it exactly (camelCase).
- Every `fetch` sends `X-Requested-With: XMLHttpRequest`; state-changing calls send `Content-Type: application/json` (a body of `{}` when there is nothing to say).
- No horizontal scrolling at 360 px width (spec §5.3). Breakpoint: 900 px.
- `localStorage`/`sessionStorage` access is always wrapped in try/catch.
- Sprites and sounds are copied from `../assets` into `web/public/assets` by `npm run assets` (run by `dev` and `build`); `web/public/assets/` is git-ignored.
- Ports: Vite dev 5173 (proxies `/api` to the dev server on 5080); e2e server 5081. Both fixed: parallel worktrees collide.

### Deviations from the spec found while planning

1. The 1.5 MB fanfare wav is served as is: no ffmpeg in the build image, and it loads once, lazily.
2. "Animate `lastMove`" is a 600 ms highlight pulse of the destination line (or wall row) plus the move sound, not a tile flight across SVGs.
3. A wall turn also lists each completed line as a row of chips (`col 1`, `col 3`, `floor`) under the board, in addition to tapping wall cells: the floor has no single cell to tap per line.
4. Dev identity: `?as=<email>` sets the `azul_dev_user` cookie (ignored by the server outside dev mode).

## Review Focus

1. Double-tapping Confirm, or tapping it while the previous request is in flight, must send one move → Task 15 `submit.test.ts` and Task 17 `Table.test.ts` "Confirm is disabled while a move is pending".
2. A floor with more than 7 tiles must render 7 slots and a `+N` badge without widening the board → Task 16 `geometry.test.ts` "floor overflow" and Task 17 `PlayerBoard.test.ts` "+N badge".
3. Access session expiry mid-game: the page reloads once to sign in again and never loops → Task 15 `session.test.ts`.
4. A 4-player game on a 360 px phone with a very long email in a seat: no horizontal scroll, names truncated → Task 19 `layout.spec.ts`.
5. The SSE stream drops (server restart, phone sleep): the page reconnects and keeps playing; a deleted game sends the viewer back to the lobby → Task 19 `game.spec.ts` (restart mid-game) and `deleted.spec.ts`.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `web/package.json`, `package-lock.json`, `vite.config.ts`, `svelte.config.js`, `tsconfig.json`, `index.html` | Toolchain. |
| `web/scripts/copy-assets.mjs` | Copies sprites and sounds from `../assets`. |
| `web/src/main.ts`, `web/src/App.svelte`, `web/src/app.css` | Entry, header, routing between lobby and game. |
| `web/src/lib/types.ts` | `GameView` and friends. |
| `web/src/lib/api.ts` | Fetch wrapper, `ApiError`, `message()`, endpoint functions. |
| `web/src/lib/session.ts` | 401 → reload once. |
| `web/src/lib/events.ts` | SSE client with reconnect. |
| `web/src/lib/router.svelte.ts` | Path routing, flash message. |
| `web/src/lib/submit.ts` | One-at-a-time guard. |
| `web/src/lib/names.ts` | Seat display names. |
| `web/src/lib/geometry.ts` | Board coordinates (900×600), floor overflow, sprite paths. |
| `web/src/lib/selection.ts` | Selection state machine for both phases, ghost preview. |
| `web/src/lib/sound.svelte.ts` | Sounds, mute toggle. |
| `web/src/components/*.svelte` | `Lobby`, `GamePage`, `SeatPanel`, `Table`, `StatusBar`, `FactoryView`, `CenterView`, `PlayerBoard`, `WallChooser`, `OpponentCard`, `Sheet`. |
| `web/src/**/*.test.ts` | Vitest. |
| `web/playwright.config.ts`, `web/e2e/*.ts` | Playwright e2e against a published server in Docker. |
| `Makefile` | `web-test`, `e2e-publish`, `e2e-server-start/restart/stop`, `e2e`. |

---

### Task 15: Scaffold, API client, session, SSE client, submit guard

**Files:**
- Create: `web/package.json`, `web/vite.config.ts`, `web/svelte.config.js`, `web/tsconfig.json`, `web/index.html`, `web/scripts/copy-assets.mjs`, `web/src/main.ts`, `web/src/App.svelte` (placeholder), `web/src/app.css`, `web/src/lib/types.ts`, `web/src/lib/api.ts`, `web/src/lib/session.ts`, `web/src/lib/events.ts`, `web/src/lib/router.svelte.ts`, `web/src/lib/submit.ts`, tests `web/src/lib/session.test.ts`, `web/src/lib/api.test.ts`, `web/src/lib/submit.test.ts`, `web/src/lib/events.test.ts`
- Modify: `.gitignore` (add `web/public/assets/`), `Makefile` (`web-test`)

**Interfaces:**
- Produces: types in `types.ts` (below); `api` object; `ApiError(status, body)`; `message(e)`; `handleUnauthorized(win?)`, `clearReloadStamp(win?)`; `subscribe(id, handlers) → () => void`; `route`, `navigate(path)`, `flash`; `oneAtATime(fn)`.

- [ ] **Step 1: Toolchain files**

`web/package.json`:

```json
{
  "name": "azul-web",
  "private": true,
  "type": "module",
  "scripts": {
    "assets": "node scripts/copy-assets.mjs",
    "dev": "npm run assets && vite",
    "build": "npm run assets && vite build",
    "check": "svelte-check --tsconfig ./tsconfig.json",
    "test": "vitest run",
    "e2e": "playwright test"
  },
  "devDependencies": {
    "@playwright/test": "1.63.0",
    "@sveltejs/vite-plugin-svelte": "7.3.1",
    "@testing-library/svelte": "5.4.2",
    "@tsconfig/svelte": "5.0.8",
    "jsdom": "30.1.1",
    "svelte": "5.57.1",
    "svelte-check": "4.7.6",
    "typescript": "6.0.3",
    "vite": "8.3.2",
    "vitest": "5.0.3"
  }
}
```

`web/vite.config.ts`:

```ts
/// <reference types="vitest/config" />
import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  plugins: [svelte()],
  // Component tests need Svelte's browser build under Vitest.
  resolve: process.env.VITEST ? { conditions: ['browser'] } : undefined,
  server: {
    port: 5173,
    proxy: { '/api': { target: 'http://127.0.0.1:5080' } },
  },
  test: {
    environment: 'jsdom',
    include: ['src/**/*.test.ts'],
  },
});
```

`web/svelte.config.js`:

```js
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

export default { preprocess: vitePreprocess() };
```

`web/tsconfig.json`:

```json
{
  "extends": "@tsconfig/svelte/tsconfig.json",
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "noEmit": true,
    "verbatimModuleSyntax": true,
    "isolatedModules": true,
    "types": ["vite/client"]
  },
  "include": ["src/**/*.ts", "src/**/*.svelte", "e2e/**/*.ts", "vite.config.ts", "playwright.config.ts"]
}
```

`web/index.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Azul</title>
    <link rel="icon" href="/assets/sprites/tile_blue.png" />
  </head>
  <body>
    <div id="app"></div>
    <script type="module" src="/src/main.ts"></script>
  </body>
</html>
```

`web/scripts/copy-assets.mjs`:

```js
// Copies the desktop game's sprites and sounds into public/assets.
import { cpSync, mkdirSync } from 'node:fs';

const src = new URL('../../assets/', import.meta.url);
const dst = new URL('../public/assets/', import.meta.url);
mkdirSync(new URL('sprites/', dst), { recursive: true });
mkdirSync(new URL('sounds/', dst), { recursive: true });

const sprites = ['board2.png', 'factory.png', 'tile_blue.png', 'tile_yellow.png', 'tile_red.png',
  'tile_black.png', 'tile_white.png', 'tile_first.png'];
for (const f of sprites) cpSync(new URL(`sprites/${f}`, src), new URL(`sprites/${f}`, dst));

// name in the web app -> file in assets/sounds (the desktop game's choices, azul/game.py)
const sounds = {
  select: 'mixkit-poker-card-placement-2001.wav',
  invalid: 'mixkit-video-game-mystery-alert-234.wav',
  yourTurn: 'mixkit-paper-slide-1530.wav',
  botMove: 'mixkit-retro-confirmation-tone-2860.wav',
  score: 'mixkit-small-win-2020.wav',
  win: 'mixkit-medieval-show-fanfare-announcement-226.wav',
  lose: 'Bidibodi_bidibu_radio.mp3',
};
for (const [name, file] of Object.entries(sounds)) {
  cpSync(new URL(`sounds/${file}`, src), new URL(`sounds/${name}${file.slice(file.lastIndexOf('.'))}`, dst));
}
```

Append `web/public/assets/` to `.gitignore`. Then:

```bash
cd /home/garamizo/Azul-Board-Game-web/web
npm install
npx playwright install chromium
```

Expected: `package-lock.json` created; no peer-dependency errors.

Add to the `Makefile` (and to `.PHONY`):

```make
web-test:
	cd web && npm run check && npm test
```

- [ ] **Step 2: Types, router, entry, placeholder app**

`web/src/lib/types.ts`:

```ts
// Mirrors server/AzulServer/Games/Views.cs (System.Text.Json web defaults).
export type SeatKind = 'open' | 'human' | 'bot';
export type GameStatus = 'lobby' | 'playing' | 'finished';

export interface SeatView { idx: number; kind: SeatKind; email: string | null }
export interface PlayerView {
  score: number;
  lines: ([number, number] | null)[];  // [color, count] per pattern line
  wall: number[][];                     // -1 empty, else colour
  floor: number[];                      // ordinary tiles, colour order; FIRST marker is hasFirst
  hasFirst: boolean;
}
export interface BoardView {
  round: number;
  phase: 'take' | 'wall' | 'over';
  activeSeat: number;
  factories: number[][];  // per factory, 5 colour counts
  center: number[];       // 5 colour counts
  centerHasFirst: boolean;
  bag: number[];
  discard: number[];
  players: PlayerView[];
}
export interface WallRowView { color: number; targets: number[] }
export interface LegalView {
  takes: [number, number, number][] | null;  // [factory, color, row]; factory === factories.length is the centre
  wall: (WallRowView | null)[] | null;
}
export interface LastMoveView {
  version: number; seat: number; kind: 'take' | 'wall';
  factory: number | null; color: number | null; row: number | null; tiles: number | null; columns: number[] | null;
}
export interface ResultView { scores: number[]; winners: number[]; reason: string }
export interface GameView {
  id: string; status: GameStatus; version: number; numPlayers: number; creator: string;
  you: { email: string; seat: number | null };
  seats: SeatView[];
  board: BoardView | null;
  legal: LegalView | null;
  lastMove: LastMoveView | null;
  result: ResultView | null;
}
export interface GameSummary {
  id: string; status: GameStatus; numPlayers: number; creator: string;
  seats: SeatView[]; round: number | null; updatedAt: string;
}
export type MoveBody =
  | { version: number; requestId: string; kind: 'take'; factory: number; color: number; row: number }
  | { version: number; requestId: string; kind: 'wall'; columns: number[] };
```

`web/src/lib/router.svelte.ts`:

```ts
function parse(path: string): string | null {
  const m = /^\/g\/([a-z2-7]{10})$/.exec(path);
  return m ? m[1] : null;
}

export const route = $state({ gameId: parse(location.pathname) });
/// One-shot message for the next page (e.g. "This game was deleted").
export const flash = $state({ text: '' });

export function navigate(path: string, message = ''): void {
  flash.text = message;
  history.pushState({}, '', path);
  route.gameId = parse(location.pathname);
}

window.addEventListener('popstate', () => { route.gameId = parse(location.pathname); });
```

`web/src/main.ts`:

```ts
import { mount } from 'svelte';
import App from './App.svelte';
import './app.css';

// Dev mode only: /?as=bob@example.com picks who you are (the server ignores
// the cookie unless it runs without Access).
const as = new URLSearchParams(location.search).get('as');
if (as) document.cookie = `azul_dev_user=${encodeURIComponent(as)}; path=/; SameSite=Strict`;

mount(App, { target: document.getElementById('app')! });
```

`web/src/App.svelte` (placeholder; Task 18 replaces it):

```svelte
<h1>Azul</h1>
```

`web/src/app.css`:

```css
:root {
  --bg: #f6efe4; --fg: #2b2118; --muted: #7a6a5a; --accent: #1f5fa8; --danger: #a8321f;
  --card: #fffaf2; --line: #e0d4c3;
  font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
  color: var(--fg); background: var(--bg);
}
* { box-sizing: border-box; }
body { margin: 0; }
main { padding: 0 12px 24px; max-width: 1280px; margin: 0 auto; }
button { font: inherit; padding: 8px 14px; border-radius: 8px; border: 1px solid var(--line);
  background: var(--card); color: var(--fg); cursor: pointer; }
button.primary { background: var(--accent); color: white; border-color: var(--accent); }
button:disabled { opacity: 0.45; cursor: default; }
.banner { padding: 8px 12px; border-radius: 8px; background: #fff3cd; margin: 8px 0; }
.banner.error { background: #f8d7da; }
.truncate { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0; }
```

- [ ] **Step 3: Write the failing tests**

`web/src/lib/session.test.ts`:

```ts
import { describe, expect, it, vi } from 'vitest';
import { clearReloadStamp, handleUnauthorized } from './session';

function fakeWindow() {
  const store = new Map<string, string>();
  return {
    sessionStorage: {
      getItem: (k: string) => store.get(k) ?? null,
      setItem: (k: string, v: string) => void store.set(k, v),
      removeItem: (k: string) => void store.delete(k),
    },
    location: { reload: vi.fn() },
  } as unknown as Window;
}

describe('session expiry', () => {
  it('reloads once, then stops', () => {
    const win = fakeWindow();
    expect(handleUnauthorized(win)).toBe('reloading');
    expect(handleUnauthorized(win)).toBe('signed-out');
    expect(win.location.reload).toHaveBeenCalledTimes(1);
  });

  it('a successful request re-arms the reload', () => {
    const win = fakeWindow();
    handleUnauthorized(win);
    clearReloadStamp(win);
    expect(handleUnauthorized(win)).toBe('reloading');
    expect(win.location.reload).toHaveBeenCalledTimes(2);
  });

  it('storage that throws never reloads in a loop', () => {
    const win = { sessionStorage: { getItem: () => { throw new Error('blocked'); } }, location: { reload: vi.fn() } } as unknown as Window;
    expect(handleUnauthorized(win)).toBe('signed-out');
    expect(win.location.reload).not.toHaveBeenCalled();
  });
});
```

`web/src/lib/api.test.ts`:

```ts
import { afterEach, describe, expect, it, vi } from 'vitest';
import { api, ApiError, message } from './api';

afterEach(() => vi.restoreAllMocks());

function respond(status: number, body: unknown) {
  return vi.spyOn(globalThis, 'fetch').mockResolvedValue(
    new Response(body === null ? null : JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } }));
}

describe('api', () => {
  it('sends the Access and JSON headers', async () => {
    const f = respond(201, { id: 'abc' });
    await api.create(3);
    const [url, init] = f.mock.calls[0];
    expect(url).toBe('/api/games');
    const headers = init!.headers as Record<string, string>;
    expect(headers['X-Requested-With']).toBe('XMLHttpRequest');
    expect(headers['Content-Type']).toBe('application/json');
    expect(init!.body).toBe('{"players":3}');
  });

  it('seat actions send an empty JSON body', async () => {
    const f = respond(200, {});
    await api.claim('abc', 1);
    expect(f.mock.calls[0][1]!.body).toBe('{}');
  });

  it('401 goes through the session handler (stamps sessionStorage, reloads once)', async () => {
    sessionStorage.clear();
    respond(401, null);
    await expect(api.games()).rejects.toBeInstanceOf(ApiError);
    expect(sessionStorage.getItem('azul.reloadedForAuth')).not.toBeNull();
    sessionStorage.clear();
  });

  it('errors carry status and body', async () => {
    respond(409, { error: 'stale', view: { version: 9 } });
    const err = await api.game('abc').catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(409);
    expect(err.body.view.version).toBe(9);
    expect(message(err)).toContain('stale');
  });

  it('503 has a friendly message', async () => {
    respond(503, { error: 'auth-unavailable' });
    expect(message(await api.games().catch((e) => e))).toMatch(/verify sign-ins/);
  });
});
```

`web/src/lib/submit.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { oneAtATime } from './submit';

describe('oneAtATime', () => {
  it('ignores calls while one is in flight', async () => {
    let calls = 0;
    let release!: () => void;
    const run = oneAtATime(() => { calls++; return new Promise<void>((r) => { release = r; }); });
    const first = run();
    const second = run();
    expect(calls).toBe(1);
    expect(await second).toBeUndefined();
    release();
    await first;
    run();
    expect(calls).toBe(2);
  });

  it('frees itself after a failure', async () => {
    let calls = 0;
    const run = oneAtATime(async () => { calls++; throw new Error('x'); });
    await expect(run()).rejects.toThrow('x');
    await expect(run()).rejects.toThrow('x');
    expect(calls).toBe(2);
  });
});
```

`web/src/lib/events.test.ts`:

```ts
import { afterEach, describe, expect, it, vi } from 'vitest';
import { subscribe } from './events';
import { api, ApiError } from './api';

class FakeSource {
  static last: FakeSource;
  listeners = new Map<string, (e: MessageEvent) => void>();
  onerror: (() => void) | null = null;
  closed = false;
  constructor(public url: string) { FakeSource.last = this; }
  addEventListener(name: string, fn: (e: MessageEvent) => void) { this.listeners.set(name, fn); }
  close() { this.closed = true; }
  emit(name: string, data: unknown) { this.listeners.get(name)!({ data: JSON.stringify(data) } as MessageEvent); }
}

afterEach(() => { vi.restoreAllMocks(); vi.useRealTimers(); });

describe('subscribe', () => {
  it('delivers states and reconnects after an error', async () => {
    vi.useFakeTimers();
    vi.stubGlobal('EventSource', FakeSource);
    vi.spyOn(api, 'me').mockResolvedValue({ email: 'a@x' });
    vi.spyOn(api, 'game').mockResolvedValue({} as never);
    const states: number[] = [];
    const statuses: string[] = [];
    const stop = subscribe('abc', { state: (v) => states.push(v.version), deleted: () => {}, status: (s) => statuses.push(s) });
    const first = FakeSource.last;
    expect(first.url).toBe('/api/games/abc/events');
    first.emit('state', { version: 3 });
    first.onerror!();
    expect(first.closed).toBe(true);
    await vi.advanceTimersByTimeAsync(1000);
    expect(FakeSource.last).not.toBe(first);
    FakeSource.last.emit('state', { version: 4 });
    expect(states).toEqual([3, 4]);
    expect(statuses).toEqual(['live', 'reconnecting', 'live']);
    stop();
    expect(FakeSource.last.closed).toBe(true);
  });

  it('a deleted game stops reconnecting', async () => {
    vi.useFakeTimers();
    vi.stubGlobal('EventSource', FakeSource);
    vi.spyOn(api, 'me').mockResolvedValue({ email: 'a@x' });
    vi.spyOn(api, 'game').mockRejectedValue(new ApiError(404, { error: 'not-found' }));
    const deleted = vi.fn();
    subscribe('abc', { state: () => {}, deleted, status: () => {} });
    const first = FakeSource.last;
    first.onerror!();
    await vi.advanceTimersByTimeAsync(1000);
    expect(deleted).toHaveBeenCalled();
    expect(FakeSource.last).toBe(first);
  });

  it('the deleted event ends the subscription', () => {
    vi.stubGlobal('EventSource', FakeSource);
    const deleted = vi.fn();
    subscribe('abc', { state: () => {}, deleted, status: () => {} });
    FakeSource.last.emit('deleted', {});
    expect(deleted).toHaveBeenCalled();
    expect(FakeSource.last.closed).toBe(true);
  });
});
```

- [ ] **Step 4: Run them to verify they fail**

Run: `cd web && npm test`
Expected: FAIL — modules `./session`, `./api`, `./submit`, `./events` not found.

- [ ] **Step 5: Implement**

`web/src/lib/session.ts`:

```ts
const KEY = 'azul.reloadedForAuth';

/// An expired Access session answers 401 (we send X-Requested-With). Reload
/// once to go through the sign-in page; a second 401 before any success
/// means reloading will not help.
export function handleUnauthorized(win: Window = window): 'reloading' | 'signed-out' {
  try {
    if (win.sessionStorage.getItem(KEY)) return 'signed-out';
    win.sessionStorage.setItem(KEY, String(Date.now()));
  } catch {
    return 'signed-out';
  }
  win.location.reload();
  return 'reloading';
}

export function clearReloadStamp(win: Window = window): void {
  try { win.sessionStorage.removeItem(KEY); } catch { /* storage blocked */ }
}
```

`web/src/lib/api.ts`:

```ts
import { clearReloadStamp, handleUnauthorized } from './session';
import type { GameSummary, GameView, MoveBody } from './types';

export class ApiError extends Error {
  constructor(public status: number, public body: any) {
    super(`HTTP ${status}${body?.error ? `: ${body.error}` : ''}`);
  }
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = { 'X-Requested-With': 'XMLHttpRequest' };
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  const res = await fetch(path, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
    credentials: 'same-origin',
  });
  if (res.status === 401) {
    handleUnauthorized();
    throw new ApiError(401, null);
  }
  const data = res.status === 204 ? null : await res.json().catch(() => null);
  if (!res.ok) throw new ApiError(res.status, data);
  clearReloadStamp();
  return data as T;
}

export function message(e: unknown): string {
  if (e instanceof ApiError) {
    if (e.status === 401) return 'Signed out. Reload the page to sign in again.';
    if (e.status === 503) return "The server can't verify sign-ins right now. Retrying…";
    return e.body?.error ? `Not allowed: ${e.body.error}` : `Server error (${e.status})`;
  }
  return e instanceof Error ? e.message : String(e);
}

export const api = {
  me: () => request<{ email: string }>('GET', '/api/me'),
  games: () => request<GameSummary[]>('GET', '/api/games'),
  create: (players: number) => request<GameView>('POST', '/api/games', { players }),
  game: (id: string) => request<GameView>('GET', `/api/games/${id}`),
  claim: (id: string, seat: number) => request<GameView>('POST', `/api/games/${id}/seats/${seat}/claim`, {}),
  release: (id: string, seat: number) => request<GameView>('POST', `/api/games/${id}/seats/${seat}/release`, {}),
  setKind: (id: string, seat: number, kind: 'open' | 'bot') =>
    request<GameView>('POST', `/api/games/${id}/seats/${seat}/kind`, { kind }),
  start: (id: string) => request<GameView>('POST', `/api/games/${id}/start`, {}),
  toBot: (id: string, seat: number) => request<GameView>('POST', `/api/games/${id}/seats/${seat}/to-bot`, {}),
  takeBack: (id: string, seat: number) => request<GameView>('POST', `/api/games/${id}/seats/${seat}/take-back`, {}),
  remove: (id: string) => request<null>('DELETE', `/api/games/${id}`),
  move: (id: string, body: MoveBody) => request<GameView>('POST', `/api/games/${id}/moves`, body),
};
```

`web/src/lib/submit.ts`:

```ts
/// Wraps an async action so calls made while it runs are ignored (a double
/// tap on Confirm sends one move).
export function oneAtATime<T>(fn: () => Promise<T>): () => Promise<T | undefined> {
  let running = false;
  return async () => {
    if (running) return undefined;
    running = true;
    try {
      return await fn();
    } finally {
      running = false;
    }
  };
}
```

`web/src/lib/events.ts`:

```ts
import { api, ApiError } from './api';
import type { GameView } from './types';

export interface Handlers {
  state(view: GameView): void;
  deleted(): void;
  status(status: 'live' | 'reconnecting'): void;
}

const DELAYS = [1000, 2000, 5000, 10000];

/// Live game updates. Reconnects with backoff; before reconnecting it checks
/// /api/me (EventSource cannot see a 401 itself: api.me runs the reload-once
/// path) and the game (a 404 means it was deleted).
export function subscribe(id: string, on: Handlers): () => void {
  let source: EventSource | null = null;
  let stopped = false;
  let attempt = 0;

  const open = () => {
    if (stopped) return;
    const es = new EventSource(`/api/games/${id}/events`);
    source = es;
    es.addEventListener('state', (e) => {
      attempt = 0;
      on.status('live');
      on.state(JSON.parse((e as MessageEvent).data));
    });
    es.addEventListener('deleted', () => {
      stopped = true;
      es.close();
      on.deleted();
    });
    es.onerror = async () => {
      es.close();
      if (stopped) return;
      on.status('reconnecting');
      await new Promise((r) => setTimeout(r, DELAYS[Math.min(attempt++, DELAYS.length - 1)]));
      if (stopped) return;
      try {
        await api.me();
      } catch (err) {
        if (err instanceof ApiError && err.status === 401) return;  // reloading, or signed out
      }
      try {
        await api.game(id);
      } catch (err) {
        if (err instanceof ApiError && err.status === 404) {
          stopped = true;
          on.deleted();
          return;
        }
      }
      open();
    };
  };

  open();
  return () => {
    stopped = true;
    source?.close();
  };
}
```

- [ ] **Step 6: Run the tests and the checker**

Run: `cd web && npm test && npm run check && npm run build`
Expected: all Vitest tests pass; svelte-check reports 0 errors; build writes `web/dist/index.html`.

- [ ] **Step 7: Commit**

```bash
cd /home/garamizo/Azul-Board-Game-web
git add -A web .gitignore Makefile
git commit -m "web: scaffold, API client, session expiry, SSE client

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 16: Board geometry and the selection state machine

**Files:**
- Create: `web/src/lib/geometry.ts`, `web/src/lib/selection.ts`, `web/src/lib/names.ts`, tests `web/src/lib/geometry.test.ts`, `web/src/lib/selection.test.ts`

**Interfaces:**
- Consumes: `types.ts`.
- Produces: geometry (`TILE`, `lineCell`, `lineBox`, `wallCell`, `wallBox`, `floorCell`, `FLOOR_BOX`, `FLOOR_SLOTS`, `floorDisplay`, `tileHref`, `factoryTiles`, `COLOR_NAMES`); selection (`FLOOR`, `FIRST`, `Source`, `TakeSel`, `WallSel`, `Selection`, `initial`, `canPick`, `legalRows`, `tapSource`, `tapRow`, `takeMove`, `wallTargets`, `tapWallTarget`, `wallComplete`, `wallColumns`, `ghost`); names (`seatName`).

- [ ] **Step 1: Write the failing tests**

`web/src/lib/geometry.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { factoryTiles, floorDisplay, lineBox, lineCell, wallCell } from './geometry';

describe('geometry (board2.png, 900x600, from azul/models.py)', () => {
  it('places pattern lines right-aligned and the wall on the right', () => {
    expect(lineCell(0, 0)).toEqual({ x: 390, y: 60 });
    expect(lineCell(4, 4)).toEqual({ x: 50, y: 400 });
    expect(lineBox(2)).toEqual({ x: 175, y: 185, w: 255, h: 85 });
    expect(wallCell(0, 0)).toEqual({ x: 510, y: 60 });
    expect(wallCell(4, 4)).toEqual({ x: 850, y: 400 });
  });

  it('floor overflow: marker first, 7 slots, the rest as a count', () => {
    const shown = floorDisplay([0, 0, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4], true);
    expect(shown.tiles).toEqual([5, 0, 0, 1, 2, 2, 2]);
    expect(shown.extra).toBe(6);
    expect(floorDisplay([], false)).toEqual({ tiles: [], extra: 0 });
  });

  it('expands factory counts in colour order', () => {
    expect(factoryTiles([0, 2, 1, 0, 1])).toEqual([1, 1, 2, 4]);
  });
});
```

`web/src/lib/selection.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import type { GameView, LegalView, WallRowView } from './types';
import {
  FLOOR, ghost, initial, tapRow, tapSource, tapWallTarget, takeMove, wallColumns, wallComplete, wallTargets,
  type TakeSel, type WallSel,
} from './selection';

const takeLegal: LegalView = {
  takes: [[0, 1, 0], [0, 1, 5], [0, 2, 3], [0, 2, 5], [5, 5, 5]],  // centre = 5 here
  wall: null,
};

function view(legal: LegalView | null): GameView {
  return {
    id: 'g', status: 'playing', version: 4, numPlayers: 2, creator: 'a', you: { email: 'a', seat: 0 },
    seats: [], legal, lastMove: null, result: null,
    board: {
      round: 1, phase: 'take', activeSeat: 0, factories: [[0, 3, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
      center: [0, 0, 0, 0, 0], centerHasFirst: true, bag: [0, 0, 0, 0, 0], discard: [0, 0, 0, 0, 0],
      players: [{ score: 0, lines: [null, null, null, [2, 1], null], wall: Array(5).fill([-1, -1, -1, -1, -1]), floor: [], hasFirst: false }],
    },
  };
}

describe('take phase', () => {
  it('source, then row, then a move', () => {
    let sel = initial(view(takeLegal)) as TakeSel;
    expect(sel).toEqual({ phase: 'take', source: null, row: null });
    sel = tapSource(sel, takeLegal, 0, 1);
    expect(sel.source).toEqual({ factory: 0, color: 1 });
    expect(takeMove(sel)).toBeNull();
    sel = tapRow(sel, takeLegal, 0);
    expect(takeMove(sel)).toEqual([0, 1, 0]);
  });

  it('ignores illegal sources and rows; tapping again cancels', () => {
    let sel = initial(view(takeLegal)) as TakeSel;
    expect(tapSource(sel, takeLegal, 1, 0)).toBe(sel);
    sel = tapSource(sel, takeLegal, 0, 2);
    expect(tapRow(sel, takeLegal, 0)).toBe(sel);
    expect(tapSource(sel, takeLegal, 0, 2).source).toBeNull();
  });

  it('the FIRST marker alone goes straight to the floor', () => {
    const sel = tapSource(initial(view(takeLegal)) as TakeSel, takeLegal, 5, 5);
    expect(takeMove(sel)).toEqual([5, 5, FLOOR]);
  });

  it('ghost shows what fits and what overflows', () => {
    const v = view(takeLegal);
    // 3 yellow onto line 0 (capacity 1): 1 placed, 2 to the floor.
    expect(ghost(v, { phase: 'take', source: { factory: 0, color: 1 }, row: 0 }, 0)).toEqual({ row: 0, color: 1, placed: 1, overflow: 2 });
    // 1 red onto line 3 that already holds 1 red (capacity 4).
    expect(ghost(v, { phase: 'take', source: { factory: 0, color: 2 }, row: 3 }, 0)).toEqual({ row: 3, color: 2, placed: 1, overflow: 0 });
    expect(ghost(v, { phase: 'take', source: { factory: 0, color: 1 }, row: FLOOR }, 0)).toEqual({ row: FLOOR, color: 1, placed: 0, overflow: 3 });
  });
});

describe('wall phase', () => {
  // Rows 1 and 3 both completed in red (2); row 4 blue (0) can only go to the floor.
  const wall: (WallRowView | null)[] = [null, { color: 2, targets: [0, 2, FLOOR] }, null, { color: 2, targets: [0, 4, FLOOR] }, { color: 0, targets: [FLOOR] }];
  const legal: LegalView = { takes: null, wall };

  it('starts on the first completed row', () => {
    const sel = initial(view(legal)) as WallSel;
    expect(sel.activeRow).toBe(1);
    expect(wallComplete(sel, wall)).toBe(false);
  });

  it('same colour cannot take the same column', () => {
    let sel = initial(view(legal)) as WallSel;
    sel = tapWallTarget(sel, wall, 1, 0);
    expect(wallTargets(wall, sel, 3)).toEqual([4, FLOOR]);
    expect(tapWallTarget(sel, wall, 3, 0)).toBe(sel);
    sel = tapWallTarget(sel, wall, 3, 4);
    sel = tapWallTarget(sel, wall, 4, FLOOR);
    expect(wallComplete(sel, wall)).toBe(true);
    expect(wallColumns(sel, wall)).toEqual([-1, 0, -1, 4, FLOOR]);
  });

  it('tapping a chosen target again clears it', () => {
    let sel = tapWallTarget(initial(view(legal)) as WallSel, wall, 1, 2);
    sel = tapWallTarget(sel, wall, 1, 2);
    expect(sel.columns[1]).toBeNull();
    expect(sel.activeRow).toBe(1);
  });
});

describe('not my turn', () => {
  it('has no selection', () => {
    expect(initial(view(null))).toBeNull();
  });
});
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd web && npm test`
Expected: FAIL — `./geometry` and `./selection` not found.

- [ ] **Step 3: Implement**

`web/src/lib/geometry.ts`:

```ts
// Coordinates in board2.png's own 900x600 space, taken from azul/models.py
// (which draws at 2/3 scale): tile centres, 85 px apart, tiles 75 px wide.
export const TILE = 75;
export const FLOOR_SLOTS = 7;
export const COLOR_NAMES = ['blue', 'yellow', 'red', 'black', 'white', 'first player'];

export interface Point { x: number; y: number }
export interface Box { x: number; y: number; w: number; h: number }

/// i-th tile of pattern line `row`, counted from the right end.
export const lineCell = (row: number, i: number): Point => ({ x: 390 - 85 * i, y: 60 + 85 * row });
export const lineBox = (row: number): Box => ({ x: 430 - 85 * (row + 1), y: 15 + 85 * row, w: 85 * (row + 1), h: 85 });
export const wallCell = (row: number, col: number): Point => ({ x: 510 + 85 * col, y: 60 + 85 * row });
export const wallBox = (row: number, col: number): Box => ({ x: 467.5 + 85 * col, y: 17.5 + 85 * row, w: 85, h: 85 });
export const floorCell = (i: number): Point => ({ x: 50 + 90 * i, y: 550 });
export const FLOOR_BOX: Box = { x: 9, y: 505, w: 630, h: 90 };

export const tileHref = (color: number): string =>
  `/assets/sprites/tile_${['blue', 'yellow', 'red', 'black', 'white', 'first'][color]}.png`;

/// Top-left corner for a tile image centred on p.
export const tileAt = (p: Point, size = TILE) => ({ x: p.x - size / 2, y: p.y - size / 2, width: size, height: size });

/// The FIRST marker (5) first, then ordinary tiles; at most 7 slots, the rest as `extra`.
export function floorDisplay(floor: number[], hasFirst: boolean): { tiles: number[]; extra: number } {
  const all = hasFirst ? [5, ...floor] : [...floor];
  return { tiles: all.slice(0, FLOOR_SLOTS), extra: Math.max(0, all.length - FLOOR_SLOTS) };
}

export function factoryTiles(counts: number[]): number[] {
  return counts.flatMap((n, color) => Array<number>(n).fill(color));
}
```

`web/src/lib/selection.ts`:

```ts
import type { GameView, LegalView, WallRowView } from './types';

export const FLOOR = 5;
export const FIRST = 5;

export type Source = { factory: number; color: number };
export type TakeSel = { phase: 'take'; source: Source | null; row: number | null };
export type WallSel = { phase: 'wall'; activeRow: number | null; columns: (number | null)[] };
export type Selection = TakeSel | WallSel;
type WallOptions = (WallRowView | null)[];

/// Null when it is not the viewer's turn.
export function initial(view: GameView): Selection | null {
  const legal = view.legal;
  if (!legal) return null;
  if (legal.takes) return { phase: 'take', source: null, row: null };
  const columns: (number | null)[] = [null, null, null, null, null];
  return { phase: 'wall', columns, activeRow: firstOpenRow(legal.wall!, columns) };
}

export const canPick = (legal: LegalView, factory: number, color: number): boolean =>
  !!legal.takes?.some((t) => t[0] === factory && t[1] === color);

export function legalRows(legal: LegalView, source: Source | null): number[] {
  if (!source || !legal.takes) return [];
  return legal.takes.filter((t) => t[0] === source.factory && t[1] === source.color).map((t) => t[2]);
}

export function tapSource(sel: TakeSel, legal: LegalView, factory: number, color: number): TakeSel {
  if (sel.source?.factory === factory && sel.source.color === color) return { phase: 'take', source: null, row: null };
  if (!canPick(legal, factory, color)) return sel;
  const rows = legalRows(legal, { factory, color });
  // One destination (the FIRST marker alone, or floor only): choose it now.
  return { phase: 'take', source: { factory, color }, row: rows.length === 1 ? rows[0] : null };
}

export function tapRow(sel: TakeSel, legal: LegalView, row: number): TakeSel {
  if (!sel.source || !legalRows(legal, sel.source).includes(row)) return sel;
  return { ...sel, row: sel.row === row ? null : row };
}

export function takeMove(sel: Selection | null): [number, number, number] | null {
  return sel?.phase === 'take' && sel.source && sel.row !== null ? [sel.source.factory, sel.source.color, sel.row] : null;
}

/// Targets still open for `row`: its own options minus wall columns already
/// chosen by another completed line of the same colour.
export function wallTargets(wall: WallOptions, sel: WallSel, row: number): number[] {
  const option = wall[row];
  if (!option) return [];
  const taken = new Set<number>();
  wall.forEach((o, r) => {
    const c = sel.columns[r];
    if (r !== row && o && c !== null && c !== FLOOR && o.color === option.color) taken.add(c);
  });
  return option.targets.filter((t) => t === FLOOR || !taken.has(t));
}

export function firstOpenRow(wall: WallOptions, columns: (number | null)[]): number | null {
  const i = wall.findIndex((o, r) => o !== null && columns[r] === null);
  return i < 0 ? null : i;
}

export function tapWallTarget(sel: WallSel, wall: WallOptions, row: number, target: number): WallSel {
  if (!wallTargets(wall, sel, row).includes(target)) return sel;
  const columns = [...sel.columns];
  columns[row] = columns[row] === target ? null : target;
  return { phase: 'wall', columns, activeRow: firstOpenRow(wall, columns) ?? row };
}

export const wallComplete = (sel: WallSel, wall: WallOptions): boolean =>
  wall.every((o, r) => o === null || sel.columns[r] !== null);

export const wallColumns = (sel: WallSel, wall: WallOptions): number[] =>
  wall.map((o, r) => (o === null ? -1 : (sel.columns[r] as number)));

/// Where the selected tiles would land on the viewer's board.
export function ghost(view: GameView, sel: TakeSel, seat: number):
  { row: number; color: number; placed: number; overflow: number } | null {
  const board = view.board;
  if (!board || !sel.source || sel.row === null) return null;
  const { factory, color } = sel.source;
  if (color === FIRST) return { row: FLOOR, color, placed: 0, overflow: 0 };
  const count = factory < board.factories.length ? board.factories[factory][color] : board.center[color];
  if (sel.row === FLOOR) return { row: FLOOR, color, placed: 0, overflow: count };
  const line = board.players[seat].lines[sel.row];
  const have = line && line[0] === color ? line[1] : 0;
  const placed = Math.min(count, sel.row + 1 - have);
  return { row: sel.row, color, placed, overflow: count - placed };
}
```

`web/src/lib/names.ts`:

```ts
import type { GameView } from './types';

const local = (email: string) => email.split('@')[0];

export function seatName(view: GameView, idx: number): string {
  const seat = view.seats[idx];
  if (view.you.seat === idx && seat.kind === 'human') return 'You';
  if (seat.kind === 'human' && seat.email) return local(seat.email);
  if (seat.kind === 'bot') return seat.email ? `${local(seat.email)} (bot)` : `Bot ${idx + 1}`;
  return 'Open seat';
}
```

- [ ] **Step 4: Run the tests**

Run: `cd web && npm test && npm run check`
Expected: all pass; 0 svelte-check errors.

- [ ] **Step 5: Commit**

```bash
cd /home/garamizo/Azul-Board-Game-web
git add -A web
git commit -m "web: board geometry and move selection state machine

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 17: Table components (boards, factories, centre, status, wall chooser)

**Files:**
- Create: `web/src/components/PlayerBoard.svelte`, `FactoryView.svelte`, `CenterView.svelte`, `StatusBar.svelte`, `WallChooser.svelte`, `OpponentCard.svelte`, `Sheet.svelte`, `Table.svelte`, `web/src/lib/sound.svelte.ts`; tests `web/src/components/PlayerBoard.test.ts`, `web/src/components/Table.test.ts`

**Interfaces:**
- Consumes: Tasks 15–16.
- Produces: `Table` props `{ view: GameView; send: (body: MoveBody) => Promise<'ok' | 'invalid' | 'other'> }`; `sounds.play(name)`, `sounds.toggle()`, `soundState.muted`.

- [ ] **Step 1: Write the failing component tests**

`web/src/components/PlayerBoard.test.ts`:

```ts
import { render } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import PlayerBoard from './PlayerBoard.svelte';
import type { PlayerView } from '../lib/types';

const player: PlayerView = {
  score: 12,
  lines: [[1, 1], null, [2, 2], null, null],
  wall: [[-1, 0, -1, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]],
  floor: [0, 0, 1, 2, 2, 3, 3, 4, 4],
  hasFirst: true,
};

describe('PlayerBoard', () => {
  it('draws lines, wall, floor and the overflow badge', () => {
    const { container, getByText } = render(PlayerBoard, { player, name: 'alice' });
    expect(container.querySelectorAll('image.tile.line')).toHaveLength(3);
    expect(container.querySelectorAll('image.tile.wall')).toHaveLength(1);
    expect(container.querySelectorAll('image.tile.floor')).toHaveLength(7);
    getByText('+3');
    getByText('alice: 12');
  });

  it('only an interactive board has tap targets', () => {
    const passive = render(PlayerBoard, { player, name: 'a' });
    expect(passive.container.querySelector('[data-row]')).toBeNull();
    const active = render(PlayerBoard, { player, name: 'a', interactive: true, legalRows: [1, 5] });
    expect(active.container.querySelectorAll('[data-row]')).toHaveLength(6);
    expect(active.container.querySelectorAll('.hit.legal')).toHaveLength(2);
  });
});
```

`web/src/components/Table.test.ts`:

```ts
import { fireEvent, render } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import Table from './Table.svelte';
import type { GameView, MoveBody } from '../lib/types';

function myTurn(): GameView {
  const empty = [-1, -1, -1, -1, -1];
  const player = { score: 0, lines: [null, null, null, null, null], wall: [empty, empty, empty, empty, empty], floor: [], hasFirst: false };
  return {
    id: 'abcdefghij', status: 'playing', version: 7, numPlayers: 2, creator: 'a@x', you: { email: 'a@x', seat: 0 },
    seats: [{ idx: 0, kind: 'human', email: 'a@x' }, { idx: 1, kind: 'bot', email: null }],
    legal: { takes: [[0, 1, 0], [0, 1, 5]], wall: null }, lastMove: null, result: null,
    board: {
      round: 1, phase: 'take', activeSeat: 0,
      factories: [[0, 2, 0, 1, 1], [1, 1, 1, 1, 0], [0, 0, 4, 0, 0], [2, 2, 0, 0, 0], [0, 0, 0, 2, 2]],
      center: [0, 0, 0, 0, 0], centerHasFirst: true, bag: [0, 0, 0, 0, 0], discard: [0, 0, 0, 0, 0],
      players: [player, player],
    },
  };
}

describe('Table', () => {
  it('Confirm is enabled only once a move is chosen, and sends it', async () => {
    const send = vi.fn(async (_: MoveBody) => 'ok' as const);
    const { container, getByRole } = render(Table, { view: myTurn(), send });
    const confirm = getByRole('button', { name: 'Confirm' }) as HTMLButtonElement;
    expect(confirm.disabled).toBe(true);
    await fireEvent.click(container.querySelector('[data-factory="0"][data-color="1"]')!);
    expect(confirm.disabled).toBe(true);
    await fireEvent.click(container.querySelector('[data-row="0"]')!);
    expect(confirm.disabled).toBe(false);
    await fireEvent.click(confirm);
    expect(send).toHaveBeenCalledTimes(1);
    expect(send.mock.calls[0][0]).toMatchObject({ version: 7, kind: 'take', factory: 0, color: 1, row: 0 });
  });

  it('Confirm is disabled while a move is pending', async () => {
    let release!: () => void;
    const send = vi.fn(() => new Promise<'ok'>((r) => { release = () => r('ok'); }));
    const { container, getByRole } = render(Table, { view: myTurn(), send });
    await fireEvent.click(container.querySelector('[data-factory="0"][data-color="1"]')!);
    await fireEvent.click(container.querySelector('[data-row="0"]')!);
    const confirm = getByRole('button', { name: 'Confirm' }) as HTMLButtonElement;
    await fireEvent.click(confirm);
    await fireEvent.click(confirm);
    expect(send).toHaveBeenCalledTimes(1);
    expect(confirm.disabled).toBe(true);
    release();
  });

  it('spectators see no Confirm button', () => {
    const v = { ...myTurn(), you: { email: 'z@x', seat: null }, legal: null };
    const { queryByRole } = render(Table, { view: v, send: vi.fn() });
    expect(queryByRole('button', { name: 'Confirm' })).toBeNull();
  });
});
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd web && npm test`
Expected: FAIL — `./PlayerBoard.svelte` and `./Table.svelte` not found.

- [ ] **Step 3: Implement the sound module**

`web/src/lib/sound.svelte.ts`:

```ts
const FILES = {
  select: 'select.wav', invalid: 'invalid.wav', yourTurn: 'yourTurn.wav', botMove: 'botMove.wav',
  score: 'score.wav', win: 'win.wav', lose: 'lose.mp3',
} as const;
export type SoundName = keyof typeof FILES;

const KEY = 'azul.muted';
function readMuted(): boolean {
  try { return localStorage.getItem(KEY) === '1'; } catch { return false; }
}

export const soundState = $state({ muted: readMuted() });

// Browsers play audio only after a user gesture.
let unlocked = false;
if (typeof window !== 'undefined') window.addEventListener('pointerdown', () => { unlocked = true; }, { once: true });

const cache = new Map<SoundName, HTMLAudioElement>();

export const sounds = {
  play(name: SoundName): void {
    if (soundState.muted || !unlocked) return;
    let audio = cache.get(name);
    if (!audio) {
      audio = new Audio(`/assets/sounds/${FILES[name]}`);
      cache.set(name, audio);
    }
    audio.currentTime = 0;
    void audio.play().catch(() => { /* autoplay refused */ });
  },
  toggle(): void {
    soundState.muted = !soundState.muted;
    try { localStorage.setItem(KEY, soundState.muted ? '1' : '0'); } catch { /* storage blocked */ }
  },
};
```

- [ ] **Step 4: Implement the components**

`web/src/components/PlayerBoard.svelte`:

```svelte
<script lang="ts">
  import type { PlayerView, WallRowView } from '../lib/types';
  import { FLOOR_BOX, floorCell, floorDisplay, lineBox, lineCell, tileAt, tileHref, wallBox, wallCell } from '../lib/geometry';
  import { wallTargets, type WallSel } from '../lib/selection';

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
  }
  let { player, name, interactive = false, legalRows = [], ghost = null, wall = null, wallSel = null,
        pulseRow = null, onRow, onWallCell }: Props = $props();

  const rows = [0, 1, 2, 3, 4];
  const floor = $derived(floorDisplay(player.floor, player.hasFirst));
  const ghostFloor = $derived(ghost ? ghost.overflow : 0);
  const key = (fn: () => void) => (e: KeyboardEvent) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fn(); } };
  const targets = (row: number) => (wall && wallSel ? wallTargets(wall, wallSel, row) : []);
</script>

<svg viewBox="0 0 900 600" class="board" role="group" aria-label={`${name}'s board`}>
  <image href="/assets/sprites/board2.png" width="900" height="600" />
  <text x="20" y="48" class="label">{name}: {player.score}</text>

  {#each rows as row}
    {@const line = player.lines[row]}
    {@const box = lineBox(row)}
    {#if line}
      {#each Array(line[1]) as _, i}
        <image class="tile line" href={tileHref(line[0])} {...tileAt(lineCell(row, i))} />
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
    <image class="tile floor" href={tileHref(color)} {...tileAt(floorCell(i))} />
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
  .label { font-size: 40px; font-weight: 700; fill: #2b2118; paint-order: stroke; stroke: #f6efe4; stroke-width: 6px; }
  .extra, .ghost-count { font-size: 34px; font-weight: 700; fill: #a8321f; }
  .ghost { opacity: 0.5; }
  .hit { fill: transparent; stroke: transparent; stroke-width: 6; cursor: default; }
  .hit.legal { stroke: #1f5fa8; stroke-dasharray: 12 8; cursor: pointer; }
  .hit.target { stroke: #1f5fa8; stroke-dasharray: 12 8; cursor: pointer; }
  .hit.chosen { stroke: #1f5fa8; stroke-dasharray: none; fill: rgba(31, 95, 168, 0.12); }
  .hit.pulse { animation: pulse 600ms ease-out; }
  @keyframes pulse { from { fill: rgba(255, 196, 0, 0.55); } to { fill: transparent; } }
</style>
```

`web/src/components/FactoryView.svelte`:

```svelte
<script lang="ts">
  import { factoryTiles, tileHref } from '../lib/geometry';

  interface Props {
    index: number;
    counts: number[];
    selectedColor?: number | null;
    canPick?: (color: number) => boolean;
    onPick?: (color: number) => void;
  }
  let { index, counts, selectedColor = null, canPick = () => false, onPick }: Props = $props();
  const tiles = $derived(factoryTiles(counts));
  const slots = [[35, 35], [95, 35], [35, 95], [95, 95]];
</script>

<svg viewBox="0 0 130 130" class="factory" class:empty={tiles.length === 0} role="group" aria-label={`factory ${index + 1}`}>
  <image href="/assets/sprites/factory.png" width="130" height="130" />
  {#each tiles as color, i}
    <image class="tile" class:selected={selectedColor === color} class:pickable={canPick(color)}
      href={tileHref(color)} x={slots[i][0] - 25} y={slots[i][1] - 25} width="50" height="50"
      data-factory={index} data-color={color} role="button" tabindex="0"
      aria-label={`take colour ${color} from factory ${index + 1}`}
      onclick={() => onPick?.(color)}
      onkeydown={(e) => { if (e.key === 'Enter') onPick?.(color); }} />
  {/each}
</svg>

<style>
  .factory { width: 100%; height: auto; display: block; }
  .factory.empty { opacity: 0.35; }
  .tile.pickable { cursor: pointer; }
  .tile.selected { outline: 3px solid #1f5fa8; filter: drop-shadow(0 0 6px #1f5fa8); }
</style>
```

`web/src/components/CenterView.svelte`:

```svelte
<script lang="ts">
  import { tileHref } from '../lib/geometry';

  interface Props {
    index: number;  // the centre's factory index (= number of factories)
    counts: number[];
    hasFirst: boolean;
    selectedColor?: number | null;
    canPick?: (color: number) => boolean;
    onPick?: (color: number) => void;
  }
  let { index, counts, hasFirst, selectedColor = null, canPick = () => false, onPick }: Props = $props();
</script>

<div class="center" role="group" aria-label="centre">
  {#if hasFirst}
    <button class="group" class:selected={selectedColor === 5} disabled={!canPick(5)}
      data-factory={index} data-color="5" onclick={() => onPick?.(5)} title="First-player marker only">
      <img src={tileHref(5)} alt="first-player marker" /><span>only</span>
    </button>
  {/if}
  {#each counts as n, color}
    {#if n > 0}
      <button class="group" class:selected={selectedColor === color} disabled={!canPick(color)}
        data-factory={index} data-color={color} onclick={() => onPick?.(color)}>
        <img src={tileHref(color)} alt={`colour ${color}`} /><span>×{n}</span>
      </button>
    {/if}
  {/each}
  {#if !hasFirst && counts.every((n) => n === 0)}<span class="muted">Centre is empty</span>{/if}
</div>

<style>
  .center { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; min-height: 48px; }
  .group { display: inline-flex; align-items: center; gap: 4px; padding: 4px 8px; }
  .group img { width: 32px; height: 32px; }
  .group.selected { border-color: #1f5fa8; box-shadow: 0 0 0 2px #1f5fa8; }
  .muted { color: var(--muted); }
</style>
```

`web/src/components/StatusBar.svelte`:

```svelte
<script lang="ts">
  import type { GameView } from '../lib/types';
  import { seatName } from '../lib/names';
  import { sounds, soundState } from '../lib/sound.svelte';

  let { view }: { view: GameView } = $props();
  const board = $derived(view.board!);
  const yourTurn = $derived(!!view.legal);
</script>

<div class="status" data-testid="status" data-version={view.version}>
  <span>Round {board.round}</span>
  {#if view.status === 'finished'}
    <strong>Game over</strong>
  {:else if yourTurn}
    <strong data-testid="your-turn">Your turn{board.phase === 'wall' ? ': place your tiles' : ''}</strong>
  {:else}
    <span class="truncate">{seatName(view, board.activeSeat)} is {board.phase === 'wall' ? 'placing tiles' : 'choosing'}…</span>
  {/if}
  <span class="scores">
    {#each board.players as p, i}
      <span class="score truncate" class:active={i === board.activeSeat}>{seatName(view, i)} {p.score}</span>
    {/each}
  </span>
  <button class="mute" onclick={() => sounds.toggle()} aria-label={soundState.muted ? 'Unmute' : 'Mute'}>
    {soundState.muted ? '🔇' : '🔊'}
  </button>
</div>

<style>
  .status { display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; padding: 8px 0; min-width: 0; }
  .scores { display: flex; flex-wrap: wrap; gap: 6px 10px; min-width: 0; }
  .score { max-width: 12em; color: var(--muted); }
  .score.active { color: var(--fg); font-weight: 700; }
  .mute { margin-left: auto; padding: 4px 8px; }
</style>
```

`web/src/components/WallChooser.svelte`:

```svelte
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
```

`web/src/components/Sheet.svelte`:

```svelte
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
```

`web/src/components/OpponentCard.svelte`:

```svelte
<script lang="ts">
  import type { PlayerView } from '../lib/types';

  let { player, name, active, onOpen }: { player: PlayerView; name: string; active: boolean; onOpen: () => void } = $props();
  const colors = ['#2f6fd0', '#e7c23a', '#c8382e', '#2b2b2b', '#e8e2d6'];
</script>

<button class="card" class:active onclick={onOpen} aria-label={`${name}'s board`}>
  <span class="name truncate">{name}</span>
  <span class="score">{player.score}</span>
  <span class="mini" aria-hidden="true">
    {#each player.wall as row}
      {#each row as cell}<i style:background={cell >= 0 ? colors[cell] : 'transparent'}></i>{/each}
    {/each}
  </span>
</button>

<style>
  .card { display: grid; grid-template-columns: minmax(0, 1fr) auto auto; gap: 8px; align-items: center; width: 100%; text-align: left; }
  .card.active { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent); }
  .score { font-weight: 700; }
  .mini { display: grid; grid-template-columns: repeat(5, 8px); gap: 1px; }
  .mini i { width: 8px; height: 8px; border: 1px solid var(--line); }
</style>
```

`web/src/components/Table.svelte`:

```svelte
<script lang="ts">
  import type { GameView, MoveBody } from '../lib/types';
  import FactoryView from './FactoryView.svelte';
  import CenterView from './CenterView.svelte';
  import PlayerBoard from './PlayerBoard.svelte';
  import StatusBar from './StatusBar.svelte';
  import WallChooser from './WallChooser.svelte';
  import OpponentCard from './OpponentCard.svelte';
  import Sheet from './Sheet.svelte';
  import { seatName } from '../lib/names';
  import { oneAtATime } from '../lib/submit';
  import { sounds } from '../lib/sound.svelte';
  import {
    canPick, ghost, initial, legalRows, tapRow, tapSource, tapWallTarget, takeMove, wallColumns, wallComplete,
    type Selection,
  } from '../lib/selection';

  interface Props { view: GameView; send: (body: MoveBody) => Promise<'ok' | 'invalid' | 'other'> }
  let { view, send }: Props = $props();

  let sel = $state<Selection | null>(null);
  let pending = $state(false);
  let shake = $state(false);
  let openSeat = $state<number | null>(null);
  let seenVersion = -1;

  // A new version resets the selection.
  $effect.pre(() => {
    if (view.version !== seenVersion) {
      seenVersion = view.version;
      sel = initial(view);
    }
  });

  const board = $derived(view.board!);
  const centre = $derived(board.factories.length);
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

  <section class="market" aria-label="factories">
    <div class="factories">
      {#each board.factories as counts, i}
        <FactoryView index={i} {counts}
          selectedColor={takeSel?.source?.factory === i ? takeSel.source.color : null}
          canPick={(c) => !!legal && canPick(legal, i, c)} onPick={(c) => pick(i, c)} />
      {/each}
    </div>
    <CenterView index={centre} counts={board.center} hasFirst={board.centerHasFirst}
      selectedColor={takeSel?.source?.factory === centre ? takeSel.source.color : null}
      canPick={(c) => !!legal && canPick(legal, centre, c)} onPick={(c) => pick(centre, c)} />
  </section>

  {#if mySeat !== null}
    <section class="mine" class:shake aria-label="your board">
      <PlayerBoard player={board.players[mySeat]} name={seatName(view, mySeat)}
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
        <PlayerBoard player={board.players[i]} name={seatName(view, i)} pulseRow={pulse(i)} />
      {/each}
    </div>
    <div class="others-phone">
      {#each others as i}
        <OpponentCard player={board.players[i]} name={seatName(view, i)} active={i === board.activeSeat}
          onOpen={() => (openSeat = i)} />
      {/each}
    </div>
  </section>

  {#if view.result}
    <section class="result" data-testid="result">
      <h2>{view.result.winners.includes(mySeat ?? -1) ? 'You win!' : 'Game over'}</h2>
      <ol>
        {#each view.result.scores as score, i}
          <li class:winner={view.result.winners.includes(i)}>{seatName(view, i)}: {score}</li>
        {/each}
      </ol>
    </section>
  {/if}
</div>

{#if openSeat !== null}
  <Sheet title={`${seatName(view, openSeat)}'s board`} onClose={() => (openSeat = null)}>
    <PlayerBoard player={board.players[openSeat]} name={seatName(view, openSeat)} />
  </Sheet>
{/if}

<style>
  .table { display: grid; gap: 12px; grid-template-columns: minmax(0, 1fr); }
  .factories { display: grid; grid-template-columns: repeat(auto-fill, minmax(64px, 1fr)); gap: 6px; margin-bottom: 8px; }
  .actions { display: flex; gap: 8px; margin-top: 8px; }
  .others-desktop { display: none; }
  .others-phone { display: grid; gap: 6px; }
  .result .winner { font-weight: 700; }
  .shake { animation: shake 300ms; }
  @keyframes shake { 25% { transform: translateX(-6px); } 75% { transform: translateX(6px); } }
  @media (min-width: 900px) {
    .table { grid-template-columns: minmax(280px, 2fr) 3fr; grid-template-areas: 'status status' 'market mine' 'others others' 'result result'; }
    .status-area { grid-area: status; }
    .market { grid-area: market; }
    .mine { grid-area: mine; }
    .others { grid-area: others; }
    .result { grid-area: result; }
    .others-desktop { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }
    .others-phone { display: none; }
  }
</style>
```

- [ ] **Step 5: Run the tests and the checker**

Run: `cd web && npm test && npm run check`
Expected: all Vitest tests pass; svelte-check 0 errors (warnings about a11y on SVG `rect` with `role="button"` are acceptable only if they are warnings, not errors).

- [ ] **Step 6: Commit**

```bash
cd /home/garamizo/Azul-Board-Game-web
git add -A web
git commit -m "web: SVG table, factories, centre, wall chooser, sounds

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 18: Lobby, seat panel, game page, app shell

**Files:**
- Create: `web/src/components/Lobby.svelte`, `web/src/components/SeatPanel.svelte`, `web/src/components/GamePage.svelte`
- Modify: `web/src/App.svelte`
- Test: `web/src/components/SeatPanel.test.ts`

**Interfaces:**
- Consumes: everything above.
- Produces: the running app (used by Task 19's e2e). Buttons the e2e relies on: lobby `2 players` / `3 players` / `4 players`; seat panel `Take this seat`, `Make bot`, `Make open`, `Leave`, `Remove`, `Start`, `Delete game`; game page `Hand my seat to a bot`, `Take my seat back`.

- [ ] **Step 1: Write the failing test**

`web/src/components/SeatPanel.test.ts`:

```ts
import { render } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import SeatPanel from './SeatPanel.svelte';
import type { GameView } from '../lib/types';

function lobby(you: string): GameView {
  return {
    id: 'abcdefghij', status: 'lobby', version: 2, numPlayers: 3, creator: 'a@x',
    you: { email: you, seat: you === 'a@x' ? 0 : null },
    seats: [{ idx: 0, kind: 'human', email: 'a@x' }, { idx: 1, kind: 'open', email: null }, { idx: 2, kind: 'bot', email: null }],
    board: null, legal: null, lastMove: null, result: null,
  };
}

describe('SeatPanel', () => {
  it('the creator manages seats and starts', () => {
    const { getByRole, getAllByRole, queryByRole } = render(SeatPanel, { view: lobby('a@x'), run: vi.fn() });
    getByRole('button', { name: 'Start' });
    getByRole('button', { name: 'Make bot' });
    getByRole('button', { name: 'Make open' });
    getByRole('button', { name: 'Leave' });
    getByRole('button', { name: 'Delete game' });
    expect(queryByRole('button', { name: 'Take this seat' })).toBeNull();
    expect(getAllByRole('listitem')).toHaveLength(3);
  });

  it('a visitor can take the open seat and nothing else', () => {
    const { getByRole, queryByRole } = render(SeatPanel, { view: lobby('b@x'), run: vi.fn() });
    getByRole('button', { name: 'Take this seat' });
    expect(queryByRole('button', { name: 'Start' })).toBeNull();
    expect(queryByRole('button', { name: 'Make bot' })).toBeNull();
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd web && npm test`
Expected: FAIL — `./SeatPanel.svelte` not found.

- [ ] **Step 3: Implement**

`web/src/components/SeatPanel.svelte`:

```svelte
<script lang="ts">
  import type { GameView } from '../lib/types';
  import { api } from '../lib/api';

  interface Props { view: GameView; run: (action: () => Promise<unknown>) => void }
  let { view, run }: Props = $props();

  const me = $derived(view.you.email);
  const isCreator = $derived(view.creator === me);
  const seated = $derived(view.you.seat !== null);
  const anyHuman = $derived(view.seats.some((s) => s.kind === 'human'));
  let copied = $state(false);

  async function share() {
    try { await navigator.clipboard.writeText(location.href); copied = true; } catch { /* clipboard blocked */ }
  }
</script>

<section class="lobby">
  <h2>Waiting for players</h2>
  <p>Share this page's address with friends. Empty seats become bots when the game starts.
    <button onclick={share}>{copied ? 'Copied' : 'Copy link'}</button></p>
  <ol class="seats">
    {#each view.seats as seat}
      <li>
        <span class="who truncate">
          {#if seat.kind === 'human'}{seat.email === me ? 'You' : seat.email}{:else if seat.kind === 'bot'}Bot{:else}Open seat{/if}
        </span>
        <span class="actions">
          {#if seat.kind === 'open' && !seated}
            <button class="primary" onclick={() => run(() => api.claim(view.id, seat.idx))}>Take this seat</button>
          {/if}
          {#if isCreator && seat.kind === 'open'}
            <button onclick={() => run(() => api.setKind(view.id, seat.idx, 'bot'))}>Make bot</button>
          {/if}
          {#if isCreator && seat.kind === 'bot'}
            <button onclick={() => run(() => api.setKind(view.id, seat.idx, 'open'))}>Make open</button>
          {/if}
          {#if seat.kind === 'human' && seat.email === me}
            <button onclick={() => run(() => api.release(view.id, seat.idx))}>Leave</button>
          {:else if seat.kind === 'human' && isCreator}
            <button onclick={() => run(() => api.release(view.id, seat.idx))}>Remove</button>
          {/if}
        </span>
      </li>
    {/each}
  </ol>
  {#if isCreator}
    <div class="creator">
      <button class="primary" disabled={!anyHuman} onclick={() => run(() => api.start(view.id))}>Start</button>
      <button class="danger" onclick={() => { if (confirm('Delete this game?')) run(() => api.remove(view.id)); }}>Delete game</button>
    </div>
  {/if}
</section>

<style>
  .seats { padding-left: 1.2em; display: grid; gap: 8px; }
  .seats li { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: space-between; }
  .who { max-width: 60vw; }
  .actions { display: flex; gap: 6px; flex-wrap: wrap; }
  .creator { display: flex; gap: 8px; margin-top: 12px; }
  .danger { color: var(--danger); }
</style>
```

`web/src/components/GamePage.svelte`:

```svelte
<script lang="ts">
  import { onMount } from 'svelte';
  import { api, ApiError, message } from '../lib/api';
  import { subscribe } from '../lib/events';
  import { navigate } from '../lib/router.svelte';
  import { sounds } from '../lib/sound.svelte';
  import type { GameView, MoveBody } from '../lib/types';
  import SeatPanel from './SeatPanel.svelte';
  import Table from './Table.svelte';

  let { id }: { id: string } = $props();
  let view = $state<GameView | null>(null);
  let link = $state<'connecting' | 'live' | 'reconnecting'>('connecting');
  let notice = $state('');
  let error = $state('');

  function accept(next: GameView) {
    const prev = view;
    if (prev && next.version <= prev.version) return;
    view = next;
    if (!prev) return;
    const last = next.lastMove;
    if (last && last.version === next.version && last.seat !== next.you.seat)
      sounds.play(next.seats[last.seat]?.kind === 'bot' ? 'botMove' : 'select');
    if (next.legal && !prev.legal) sounds.play('yourTurn');
    if (next.status === 'finished' && prev.status !== 'finished')
      sounds.play(next.you.seat !== null && next.result?.winners.includes(next.you.seat) ? 'win' : 'lose');
    else if (next.board && prev.board && next.board.round > prev.board.round) sounds.play('score');
  }

  function failed(e: unknown) {
    if (e instanceof ApiError && e.status === 409 && e.body?.view) {
      accept(e.body.view);
      notice = 'The board changed';
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
      notice = '';
      return 'ok';
    } catch (e) {
      failed(e);
      return e instanceof ApiError && e.status === 400 ? 'invalid' : 'other';
    }
  }

  onMount(() => subscribe(id, {
    state: accept,
    deleted: () => navigate('/', 'This game was deleted'),
    status: (s) => (link = s),
  }));

  $effect(() => {
    const yours = !!view?.legal;
    const update = () => { document.title = yours && document.hidden ? '● Your turn — Azul' : 'Azul'; };
    update();
    document.addEventListener('visibilitychange', update);
    return () => { document.removeEventListener('visibilitychange', update); document.title = 'Azul'; };
  });

  const mySeat = $derived(view?.you.seat ?? null);
  const handedToBot = $derived(view && mySeat !== null ? view.seats[mySeat].kind === 'bot' : false);
</script>

{#if link === 'reconnecting'}<div class="banner">Reconnecting…</div>{/if}
{#if notice}<div class="banner">{notice}</div>{/if}
{#if error}<div class="banner error">{error}</div>{/if}

{#if !view}
  <p>Loading…</p>
{:else if view.status === 'lobby'}
  <SeatPanel {view} {run} />
{:else}
  <Table {view} {send} />
  {#if view.status === 'playing' && mySeat !== null}
    <p class="seat-actions">
      {#if handedToBot}
        <button onclick={() => run(() => api.takeBack(id, mySeat!))}>Take my seat back</button>
      {:else}
        <button onclick={() => { if (confirm('Let a bot play your seat?')) run(() => api.toBot(id, mySeat!)); }}>Hand my seat to a bot</button>
      {/if}
    </p>
  {/if}
{/if}

<style>
  .seat-actions { margin-top: 16px; }
</style>
```

`web/src/components/Lobby.svelte`:

```svelte
<script lang="ts">
  import { onMount } from 'svelte';
  import { api, message } from '../lib/api';
  import { flash, navigate } from '../lib/router.svelte';
  import type { GameSummary } from '../lib/types';

  let games = $state<GameSummary[]>([]);
  let me = $state('');
  let error = $state('');
  let busy = $state(false);
  const notice = flash.text;
  flash.text = '';

  async function load() {
    try {
      [games, me] = await Promise.all([api.games(), api.me().then((m) => m.email)]);
      error = '';
    } catch (e) {
      error = message(e);
    }
  }

  async function create(players: number) {
    busy = true;
    try {
      navigate(`/g/${(await api.create(players)).id}`);
    } catch (e) {
      error = message(e);
    } finally {
      busy = false;
    }
  }

  onMount(() => {
    load();
    const timer = setInterval(load, 10_000);
    return () => clearInterval(timer);
  });

  const isMine = (g: GameSummary) => g.seats.some((s) => s.email === me);
  const mine = $derived(games.filter((g) => g.status !== 'finished' && isMine(g)));
  const joinable = $derived(games.filter((g) => g.status === 'lobby' && !isMine(g) && g.seats.some((s) => s.kind === 'open')));
  const watchable = $derived(games.filter((g) => g.status === 'playing' && !isMine(g)));
  const finished = $derived(games.filter((g) => g.status === 'finished'));
  const sections = $derived([
    { title: 'Your games', list: mine },
    { title: 'Open to join', list: joinable },
    { title: 'Watch', list: watchable },
    { title: 'Recent results', list: finished },
  ]);
  const label = (g: GameSummary) =>
    `${g.numPlayers} players · ${g.seats.filter((s) => s.kind === 'human').map((s) => s.email?.split('@')[0]).join(', ')}` +
    (g.round ? ` · round ${g.round}` : '');
</script>

{#if notice}<div class="banner">{notice}</div>{/if}
{#if error}<div class="banner error">{error}</div>{/if}

<section>
  <h2>New game</h2>
  <div class="new">
    {#each [2, 3, 4] as n}
      <button class="primary" disabled={busy} onclick={() => create(n)}>{n} players</button>
    {/each}
  </div>
</section>

{#each sections as section}
  {#if section.list.length > 0}
    <section>
      <h2>{section.title}</h2>
      <ul class="games">
        {#each section.list as g (g.id)}
          <li><a href={`/g/${g.id}`} class="truncate" onclick={(e) => { e.preventDefault(); navigate(`/g/${g.id}`); }}>{label(g)}</a></li>
        {/each}
      </ul>
    </section>
  {/if}
{/each}

<style>
  .new { display: flex; gap: 8px; flex-wrap: wrap; }
  .games { padding-left: 1.2em; display: grid; gap: 6px; }
  .games a { display: block; max-width: 100%; }
</style>
```

`web/src/App.svelte` (replace):

```svelte
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
```

- [ ] **Step 4: Run the tests, checker and build**

Run: `cd web && npm test && npm run check && npm run build`
Expected: all pass, 0 errors, `web/dist` built.

- [ ] **Step 5: Manual smoke in dev**

```bash
make dev-server            # terminal 1 (serves web/dist on :5080)
# browser: http://127.0.0.1:5080/?as=alice@example.com → New game → 2 players → Start → play a few moves
```

Expected: the lobby lists the game; the table renders boards and factories; a move goes through and the bot answers within a few seconds.

- [ ] **Step 6: Commit**

```bash
cd /home/garamizo/Azul-Board-Game-web
git add -A web
git commit -m "web: lobby, seat panel, game page, app shell

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 19: Playwright end to end

**Files:**
- Create: `web/playwright.config.ts`, `web/e2e/global-setup.ts`, `web/e2e/global-teardown.ts`, `web/e2e/helpers.ts`, `web/e2e/game.spec.ts`, `web/e2e/layout.spec.ts`, `web/e2e/deleted.spec.ts`
- Modify: `Makefile` (`e2e-publish`, `e2e-server-start`, `e2e-server-restart`, `e2e-server-stop`, `e2e`)

**Interfaces:**
- Consumes: the built app and the published server in dev mode on `http://127.0.0.1:5081`.

- [ ] **Step 1: Makefile targets**

Add (and list them in `.PHONY`):

```make
# End-to-end: the published server in dev mode, in the ASP.NET runtime image,
# on 127.0.0.1:5081 with a fresh database. Fixed port and container name per
# checkout directory; two worktrees running e2e at once collide on the port.
E2E_NAME := azul-e2e-$(notdir $(CURDIR))
E2E_WAIT = for i in $$(seq 1 60); do curl -sf http://127.0.0.1:5081/api/health >/dev/null && exit 0; sleep 1; done; docker logs $(E2E_NAME); exit 1

e2e-publish: | $(NUGET_DIR)
	$(DOTNET) publish server/AzulServer/AzulServer.csproj -c Release -o /src/.data/e2e-server

e2e-server-start: e2e-publish
	rm -rf .data/e2e-db && mkdir -p .data/e2e-db
	docker rm -f $(E2E_NAME) >/dev/null 2>&1 || true
	docker run -d --name $(E2E_NAME) --user $(UID):$(GID) -p 127.0.0.1:5081:8080 \
		-e ASPNETCORE_HTTP_PORTS=8080 -e AZUL_DATA_DIR=/data -e AZUL_WEB_ROOT=/web \
		-e AZUL_BOT_THINK_SECONDS=0.2 -e AZUL_MIN_MOVE_DELAY_SECONDS=0.2 \
		-v $(CURDIR)/.data/e2e-server:/app:ro -v $(CURDIR)/.data/e2e-db:/data -v $(CURDIR)/web/dist:/web:ro \
		mcr.microsoft.com/dotnet/aspnet:10.0 dotnet /app/AzulServer.dll >/dev/null
	@$(E2E_WAIT)

e2e-server-restart:
	docker restart $(E2E_NAME) >/dev/null
	@$(E2E_WAIT)

e2e-server-stop:
	docker rm -f $(E2E_NAME) >/dev/null 2>&1 || true

e2e:
	cd web && npm run build && npx playwright test
```

- [ ] **Step 2: Playwright config and helpers**

`web/playwright.config.ts`:

```ts
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: 'e2e',
  timeout: 300_000,
  workers: 1,
  retries: 0,
  use: { baseURL: 'http://127.0.0.1:5081', trace: 'retain-on-failure', actionTimeout: 10_000 },
  globalSetup: './e2e/global-setup.ts',
  globalTeardown: './e2e/global-teardown.ts',
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
});
```

`web/e2e/global-setup.ts`:

```ts
import { execSync } from 'node:child_process';

export default function setup() {
  execSync('make -C .. e2e-server-start', { stdio: 'inherit' });
}
```

`web/e2e/global-teardown.ts`:

```ts
import { execSync } from 'node:child_process';

export default function teardown() {
  execSync('make -C .. e2e-server-stop', { stdio: 'inherit' });
}
```

`web/e2e/helpers.ts`:

```ts
import { expect, type Browser, type Page } from '@playwright/test';

export const BASE = 'http://127.0.0.1:5081';

export async function person(browser: Browser, email: string, viewport: { width: number; height: number }) {
  const context = await browser.newContext({ viewport });
  await context.addCookies([{ name: 'azul_dev_user', value: email, url: BASE }]);
  return context.newPage();
}

export async function view(page: Page, id: string) {
  return (await page.request.get(`/api/games/${id}`)).json();
}

export async function newGame(page: Page, players: number): Promise<string> {
  await page.goto('/');
  await page.getByRole('button', { name: `${players} players` }).click();
  await page.waitForURL(/\/g\/[a-z2-7]{10}$/);
  return page.url().split('/').pop()!;
}

/// Plays the viewer's turn through the UI: the first legal take that is not
/// to the floor (else any); on a wall turn, the first open target of each
/// completed line. Returns false when the turn went away meanwhile (the
/// server auto-plays forced moves after 0.2 s), so the caller just looks again.
export async function playTurn(page: Page, id: string): Promise<boolean> {
  const v = await view(page, id);
  try {
    await expect(page.getByTestId('status')).toHaveAttribute('data-version', String(v.version), { timeout: 15_000 });
    const confirm = page.getByRole('button', { name: 'Confirm' });
    if (v.legal.takes) {
      const [f, c, r] = v.legal.takes.find((t: number[]) => t[2] !== 5) ?? v.legal.takes[0];
      await page.locator(`[data-factory="${f}"][data-color="${c}"]`).first().click();
      if (!(await confirm.isEnabled())) await page.locator(`[data-row="${r}"]`).first().click();
    } else {
      for (let row = 0; row < 5; row++) {
        if (v.legal.wall[row]) await page.locator(`[data-wall-target^="${row}-"]`).first().click();
      }
    }
    await confirm.click();
    await expect.poll(async () => (await view(page, id)).version, { timeout: 15_000 }).toBeGreaterThan(v.version);
    return true;
  } catch (e) {
    if ((await view(page, id)).version > v.version) return false;  // someone (the server) moved first
    throw e;
  }
}

export async function noHorizontalScroll(page: Page) {
  const [scroll, width] = await page.evaluate(() => [document.documentElement.scrollWidth, window.innerWidth]);
  expect(scroll).toBeLessThanOrEqual(width);
}
```

- [ ] **Step 3: Write the e2e tests**

`web/e2e/game.spec.ts`:

```ts
import { execSync } from 'node:child_process';
import { expect, test } from '@playwright/test';
import { newGame, noHorizontalScroll, person, playTurn, view } from './helpers';

test('two people and a bot finish a 3-player game, surviving a server restart', async ({ browser }) => {
  const alice = await person(browser, 'alice@example.com', { width: 360, height: 740 });   // phone
  const bob = await person(browser, 'bob@example.com', { width: 1280, height: 800 });      // desktop
  const id = await newGame(alice, 3);
  await bob.goto(`/g/${id}`);
  await bob.getByRole('button', { name: 'Take this seat' }).first().click();
  await alice.getByRole('button', { name: 'Start' }).click();
  await expect(alice.getByTestId('status')).toBeVisible();
  await expect(bob.getByTestId('status')).toBeVisible();

  let moves = 0;
  let restarted = false;
  for (let i = 0; i < 1000; i++) {
    const v = await view(alice, id);
    if (v.status === 'finished') break;
    if (!restarted && moves >= 6) {
      execSync('make -C .. e2e-server-restart', { stdio: 'inherit' });
      restarted = true;
      await expect(alice.getByText('Reconnecting…')).toBeHidden({ timeout: 30_000 });
      await expect(bob.getByText('Reconnecting…')).toBeHidden({ timeout: 30_000 });
    }
    const who = v.legal ? alice : (await view(bob, id)).legal ? bob : null;
    if (who) {
      if (await playTurn(who, id)) moves++;
    } else {
      await alice.waitForTimeout(150);
    }
  }
  expect(restarted).toBe(true);
  await expect(alice.getByTestId('result')).toBeVisible({ timeout: 15_000 });
  await expect(bob.getByTestId('result')).toBeVisible({ timeout: 15_000 });
  await noHorizontalScroll(alice);
});
```

`web/e2e/layout.spec.ts`:

```ts
import { expect, test } from '@playwright/test';
import { newGame, noHorizontalScroll, person } from './helpers';

test('4 players on a 360 px phone with a very long email', async ({ browser }) => {
  const long = 'someone.with.an.extraordinarily.long.address.for.testing@example-with-a-long-domain.com';
  const alice = await person(browser, 'alice@example.com', { width: 360, height: 740 });
  const other = await person(browser, long, { width: 360, height: 740 });
  const id = await newGame(alice, 4);
  await other.goto(`/g/${id}`);
  await other.getByRole('button', { name: 'Take this seat' }).first().click();
  await noHorizontalScroll(alice);
  await alice.getByRole('button', { name: 'Start' }).click();
  await expect(alice.getByTestId('status')).toBeVisible();
  await expect(other.getByTestId('status')).toBeVisible();
  await noHorizontalScroll(alice);
  await noHorizontalScroll(other);
  await expect(alice.locator('svg.factory')).toHaveCount(9);
});
```

`web/e2e/deleted.spec.ts`:

```ts
import { expect, test } from '@playwright/test';
import { newGame, person } from './helpers';

test('a deleted game sends watchers back to the lobby', async ({ browser }) => {
  const alice = await person(browser, 'alice@example.com', { width: 1280, height: 800 });
  const bob = await person(browser, 'bob@example.com', { width: 1280, height: 800 });
  const id = await newGame(alice, 2);
  await bob.goto(`/g/${id}`);
  await expect(bob.getByRole('button', { name: 'Take this seat' })).toBeVisible();
  alice.once('dialog', (d) => d.accept());
  await alice.getByRole('button', { name: 'Delete game' }).click();
  await expect(bob).toHaveURL(/\/$/, { timeout: 15_000 });
  await expect(bob.getByText('This game was deleted')).toBeVisible();
});
```

- [ ] **Step 4: Run the e2e suite**

Run: `make e2e`
Expected: 3 tests pass. The 3-player game takes 1–3 minutes.

If `deleted.spec.ts` fails because bob's page stays on the game, check that `GameService.Delete` publishes `deleted` (Plan 2 Task 13) and that `subscribe` calls `navigate('/', …)`; fix the defect, do not loosen the test.

- [ ] **Step 5: Commit**

```bash
cd /home/garamizo/Azul-Board-Game-web
git add -A web Makefile
git commit -m "web: Playwright e2e (full game with restart, phone layout, deletion)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
