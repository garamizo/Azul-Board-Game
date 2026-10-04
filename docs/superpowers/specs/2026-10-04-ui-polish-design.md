# Azul web — UI polish, compact opponents and move bubbles — design

Branch `feat/ui-polish`. Client-only (`web/`); no server or engine change.

## 1. Goal and requirements

What the user asked for (2026-10-04):

1. **No discard pile in the UI**, the same treatment the bag already got (`a848a18`, `b665f4a`).
2. **Phone: the compact opponent view shows both grids and the floor** — pattern lines, wall and floor,
   not only the mini wall it shows today.
3. **Bubble notifications like catan's** (`catanatron/ui/src/components/Snackbar.tsx`): short
   move narration that appears, stacks a few deep and fades.
4. **A prettier UI**, taking all five suggestions offered in chat:
   - A. Azulejo theme (blue header, tiled background, cards with shadow, display font for scores);
   - B. the centre drawn as a round table with the factories around it;
   - C. motion: tiles fly to the line, scores count up with a "+n" pop, the active player glows;
   - D. your-turn emphasis: status pill, softly pulsing legal targets instead of dashed outlines;
   - E. dark mode following the system setting.

Assumptions (not stated by the user; open to correction at spec review):

- Bubbles narrate other players' moves and round changes. Moves on **your own seat** are narrated
  only when you did not make them by hand: a forced turn the server played for you, or a bot that
  has taken over your seat (§4.1). Spectators get every seat.
- The round table (B) is used only where the market is wide enough for it (≈ 1120 px viewport and
  up, §6). A phone ring of 9 factories would be a ~340 px square of small factories above your
  board; narrower screens keep the factory grid with the centre as a felt panel.
- Dark mode follows `prefers-color-scheme`; no in-app toggle.
- `prefers-reduced-motion: reduce` turns off every animation in C and D (state changes still show
  instantly).

Success criteria: the existing unit and e2e suites stay green (adjusted where they assert the
discard); no horizontal scroll at 360 px with 4 players; desktop factories stay ≥ 90 px wide at
900 and 1440 px for 2–4 players (the current `layout.spec.ts` bar); bubbles cap at three.

Non-goals: drag and drop, an in-app theme toggle, a move log/drawer, server-side changes, new
sounds.

## 2. Discard (req. 1)

Remove the `.supply` block (`data-testid="supply"`) from `web/src/components/Table.svelte` and its
CSS. `BoardView.discard` and `bag` stay in `types.ts` (the server still sends them). The unit test
`Table.test.ts` "shows discard counts but not the bag" becomes "shows neither the discard nor the
bag": no `supply` test id, no text matching `/discard|bag/i`.

## 3. Compact opponent card (req. 2)

`web/src/components/OpponentCard.svelte`, used only below 900 px (`.others-phone`). Still a single
`<button>` that opens the full board in the `Sheet`.

```
┌──────────────────────────────────────────────┐
│ alice (bot)                       ⓵  23     │  name (truncated) · first-player marker if held · score
│            ■        ■ □ □ □ □                │
│          □ □        □ ■ □ □ □                │  pattern lines, right-aligned staircase
│        ■ ■ ■        □ □ □ □ □                │  + 5×5 wall
│      □ □ □ □        □ □ □ □ □                │
│    ■ ■ ■ ■ ■        □ □ □ □ □                │
│ ■ ■ □ □ □ □ □  −2                            │  floor: 7 slots, then the floor penalty
└──────────────────────────────────────────────┘
```

- **Cells** are `<i>` squares, 11 px with a 2 px gap (14 px at ≥ 400 px viewport width). Grid
  width: lines 5 × 13 = 65 px, gap 14 px, wall 65 px → 144 px; fits a 360 px screen with room for
  the header row above it.
- **Pattern line** `r` has `r + 1` slots, right-aligned. Filled slots (`player.lines[r] = [color,
  count]`, the rightmost `count` slots) take the tile colour; empty slots are outlined.
- **Wall**: a filled cell (`wall[r][c] >= 0`) takes its colour. An empty cell is tinted with the
  colour that belongs there at ~20 % opacity, like the printed board. The layout, read off
  `board2.png` with colours 0–4 = blue, yellow, red, black, white: `colour(r, c) = (c − r + 5) % 5`.
- **Floor**: `floorDisplay(player.floor, player.hasFirst)` (existing helper, FIRST marker first),
  7 slots; `+n` after the slots when `extra > 0`; then the penalty, e.g. `−2`, from a new helper
  `floorPenalty(n)` in `lib/geometry.ts`, where `n = floor.length + (hasFirst ? 1 : 0)` — the
  engine's `floor.Sum()`, which counts the marker. Values mirror `FloorToScore`
  (`AzulLibrary/Logic.cs`): 0, −1, −2, −4, −6, −8, −11, and −14 for 7 or more. No text at 0.
- **Colours** come from one table `TILE_COLORS` in `lib/geometry.ts`, also used by the toasts and
  the theme; the FIRST marker draws as a white square with a dark "1".
- **Active seat**: the existing accent ring, plus the glow from §7.3.
- The button's `aria-label` becomes `"<name>'s board, <score> points"` (today `"<name>'s board"`;
  no test queries the card by that name). The mini grids inside are `aria-hidden`.

## 4. Bubbles (req. 3)

### 4.1 Narration — `web/src/lib/narrate.ts` (pure)

```ts
export interface Narration { who: string | null; text: string; color?: number }
export function narrate(prev: GameView, next: GameView): Narration[]
```

Returns zero or more bubbles for the change from `prev` to `next`, in this order. `who` is the
player name, kept apart from `text` so the bubble can truncate a long name without losing the
move (§4.3); `color` is the taken tile colour, for the bubble's colour dot.

**Consecutive or not.** `consecutive = next.version === prev.version + 1`. The server coalesces
event-stream notifications (`EventHub.cs`), so a client can jump several versions; then `prev` is
not the state just before `next.lastMove`, and anything inferred by comparing the two views (the
first-player marker leaving the centre, a score change) could belong to a different move. Those
inferred parts are added **only when consecutive**; the rest of the sentence comes from
`lastMove` alone and is always correct. Skipped moves are not narrated (the view does not carry
them; recovering them is a server change, a non-goal).

1. **The move**, when `next.lastMove` exists and `lastMove.version === next.version`, and the seat
   passes the own-seat rule below. `who = seatName(next, seat)`.
   - take, `color === 5`: `"took the first-player marker"`.
   - take, other colours: `"took <tiles> <colour> from factory <factory + 1> → line <row + 1>"`;
     from the centre (`factory === next.board.factories.length`; the factory count never changes
     within a game): `"… from the centre"`; `row === 5`: `"… → floor"`. `tiles` is everything of
     that colour taken; overflow is not described. When consecutive and
     `prev.board.centerHasFirst && !next.board.centerHasFirst`, append `" (+ first player)"`.
     Colour names from `COLOR_NAMES`; `color` set on the narration.
   - wall (`columns`: −1 = no completed line, 0–4 = wall column, 5 = floor):
     - some column in 0–4: `"placed <k> tile(s) on their wall"`, k = how many;
     - none in 0–4 but some 5: `"sent their completed lines to the floor"`;
     - all −1 (a forced scoring turn): `"scored the round"`.
     When consecutive and the seat's score changed, append `" (+n)"` / `" (−n)"`.
2. **A new round**: `next.board.round > prev.board.round` and `next.status !== 'finished'` →
   `{ who: null, text: "Round <round> begins" }` (correct even across a gap).

**Own-seat rule.** A move on `next.you.seat` is narrated only when it was not made by hand from
this client: `next.seats[seat].kind === 'bot'` (a bot took over your seat), or `prev.autoPlay` was
true (the server played your forced turn). Otherwise the user made it and is not told about it.
A move you made by hand in *another* tab or device is therefore not narrated either — the view
does not say which client sent a move; accepted as a limitation. Spectators (`you.seat === null`)
get every seat.

No bubble when `prev.board` or `next.board` is null (lobby → playing is not narrated).

### 4.2 Toast store — `web/src/lib/toasts.svelte.ts`

Module-level `$state` list, so `Toasts.svelte` and `GamePage.svelte` share it without props.

```ts
export interface Toast extends Narration { id: number; kind: 'info' | 'warn' }
export const toasts: { items: Toast[] };
export function toast(n: Narration | string, kind?: 'info' | 'warn'): void;  // string → { who: null, text }
export function dismiss(id: number): void;
export function clearToasts(): void;
```

- **At most three** (`MAX_TOASTS = 3`). Adding a fourth removes the oldest at once (catan's ring:
  a burst never queues behind a timer).
- Each toast removes itself after **3 s** (`setTimeout`, cleared when dismissed early). Same
  duration as catan's `autoHideDuration`.
- `clearToasts()` runs when `GamePage` unmounts, so bubbles do not follow you to the lobby or to
  another game.

### 4.3 View — `web/src/components/Toasts.svelte`

- Mounted once in `GamePage.svelte`. `position: fixed`, top centre, below the header
  (`top: 8px` plus safe-area inset), `z-index` above the table and below the `Sheet` (10).
  Width `min(92vw, 420px)`. Layout per bubble: `who` in bold, truncated with ellipsis at 40 % of
  the bubble width (full name in `title`), then `text`, which wraps to at most two lines
  (`-webkit-line-clamp: 2`). So a very long email local part (the 360 px e2e fixture) still leaves
  the move readable.
- New bubbles appear **under** the ones already up (catan's top-anchored order).
- Each bubble: card background, shadow, 12 px radius, a colour dot when `color` is set (from
  `TILE_COLORS`), × button `aria-label="Dismiss"`. `warn` bubbles use `--warn-bg`.
- Container `role="status"` `aria-live="polite"`, so screen readers announce moves.
- Svelte `fly`/`fade` transitions, 150 ms; none under reduced motion.

### 4.4 Wiring — `GamePage.svelte`

- In `accept(next)`, after the sounds, when `prev` exists: `for (const n of narrate(prev, next))
  toast(n)`.
- `"The board changed"` (409) becomes `toast('The board changed', 'warn')` instead of the
  `notice` banner, and the `notice` state goes away. The "Reconnecting…", "Signed out" and error
  banners stay: they describe a condition that lasts, a bubble that fades would hide it.
- The existing unit test `"The board changed" goes away once a newer version arrives` changes
  meaning: the bubble now goes away on its own after 3 s or on dismiss. It is replaced by: the
  bubble appears on a 409, a not-newer state does not add a second line, and it disappears after
  3 s (fake timers).

## 5. Theme (A) and dark mode (E)

### 5.1 Tokens — `web/src/app.css`

All colour in components goes through custom properties on `:root`; the literals in components
(`#1f5fa8` in `PlayerBoard`, `FactoryView`, `CenterView`; `#2b2118`/`#f6efe4` in the board label;
the banner colours) are replaced with tokens.

| token | light | dark |
|---|---|---|
| `--bg` | `#f3ead9` | `#0f1a2b` |
| `--bg-pattern` | blue motif at 5 % | light motif at 4 % |
| `--fg` | `#2b2118` | `#ece4d6` |
| `--muted` | `#7a6a5a` | `#a49a8b` |
| `--card` | `#fffaf2` | `#18263b` |
| `--line` | `#e0d4c3` | `#2b3b55` |
| `--accent` | `#1f5fa8` | `#6aa3ea` |
| `--accent-ink` | `#ffffff` | `#0f1a2b` |
| `--header` | `#163a6b` | `#0a1322` |
| `--header-ink` | `#f6efe4` | `#ece4d6` |
| `--danger` | `#a8321f` | `#ef7a66` |
| `--warn-bg` | `#fff3cd` | `#4a3d12` |
| `--error-bg` | `#f8d7da` | `#4d1f22` |
| `--felt` | `#2d6a4f` | `#1f4d3a` |
| `--glow` | `rgba(255, 196, 0, .55)` | same |
| `--shadow` | `0 2px 10px rgba(43,33,24,.12)` | `0 2px 12px rgba(0,0,0,.45)` |

Dark values apply under `@media (prefers-color-scheme: dark)`; `color-scheme: light dark` on
`:root` so form controls and scrollbars follow. The board, factory and tile sprites are photos of
the physical game and are not recoloured; on the dark background they read as pieces on a table.
`PlayerBoard`'s name/score label keeps its light stroke since it is drawn on the sprite.

### 5.2 Look

- **Header** (`App.svelte`): full-width `--header` bar, the logo "AZUL" in the display font with
  letter-spacing, email and Sign out in `--header-ink`. Content stays in the 1280 px column.
- **Background**: `body` gets `--bg` plus an inline SVG data-URI azulejo motif (a 48 px
  four-petal tile) as `background-image`, opacity built into the SVG colour.
- **Panels**: `.market`, `.mine`, each desktop opponent board and each phone card sit on `--card`
  with `--shadow`, 14 px radius, 10–12 px padding. Lobby list items and the seat panel use the same
  `.panel` class.
- **Display font**: Cinzel (600), self-hosted through `@fontsource/cinzel` (new dependency,
  bundled by Vite into `dist/`, no request to a third party; the app sits behind Cloudflare Access
  and should not depend on Google Fonts). Used for the logo, the status bar scores, the result
  heading and the board labels. Body text stays `system-ui`.
- **Buttons**: 10 px radius; `primary` with a subtle vertical gradient of `--accent`; focus ring
  `2px solid var(--accent)` with offset.

## 6. Round table (B)

### 6.1 `web/src/components/Market.svelte` (new)

`Table.svelte`'s market section (factories + centre) moves into `Market.svelte`, so the ring
geometry has one home. Props: `board`, `selectedColor(factory)`, `canPick(factory, color)`,
`onPick(factory, color)` — the callbacks `Table` passes to `FactoryView`/`CenterView` today.

**One DOM, two layouts, chosen by a container query** on the market's inner wrapper
(`container: market / inline-size`), not by viewport width:

- **Ring** when the viewport is **≥ 900 px** *and* the wrapper is **≥ 410 px** wide
  (`@media (min-width: 900px) { @container market (min-width: 410px) { … } }`). The media gate
  matters: below 900 px the table is one column, so on a 480 px phone the wrapper alone would be
  ≈ 430 px and qualify. With today's desktop grid (`minmax(280px, 2fr) 3fr`, unchanged) the ring
  starts at a viewport of about 1120 px; at 1440 px the wrapper is ≈ 474 px. Playwright's default
  Desktop Chrome viewport (1280 × 720) is in ring mode, so `game.spec.ts` plays a whole game
  through it.
- **Grid** below that (phones, and desktop 900–1120 px): the factory grid as today, the centre
  below it as a felt panel.

Why not the ring at every desktop width: at 900 px the market wrapper is ≈ 320 px, where a
9-factory ring gives 70 px factories, under the 90 px bar in `layout.spec.ts`; widening the market
column instead shrinks your board below the "opponent ≤ 0.6 × your board" bar in the same test.
The desktop grid columns therefore stay as they are.

**Ring geometry** — pure helper `ringLayout(n)` in `lib/geometry.ts`, all values in percent of the
square box (`aspect-ratio: 1`):

- factory diameter `d` = the largest value with a 4 % gap between neighbours,
  `2R·sin(π/n) ≥ d + 4` where `R = 50 − d/2 − 1`, capped at 24 %:
  n = 5 → 24, n = 7 → 24, n = 9 → 22.0;
- ring radius `R` = 37, 37, 38;
- each factory's centre: factory 1 at 12 o'clock, then clockwise (so "factory 2" in a bubble is
  the next one clockwise);
- centre disc diameter `2(R − d/2) − 4` = 46, 46, 50.

At the 410 px threshold a 9-factory ring has 0.22 × 410 = 90 px factories; 5 or 7 factories,
98 px. Each factory gets inline custom properties (`--x`, `--y`, `--d`); only the `@container`
rule uses them for absolute placement, so the grid layout ignores them.

The geometry separates the factories as **circles** (the factory sprite is round), not as boxes:
neighbouring centres are ≥ `d + 4` % apart. Their square SVG boxes do overlap at the corners
(n = 9: factories 4/5 and 6/7). So in ring mode the factory `<svg>` gets `pointer-events: none`
and its tile images `pointer-events: auto`: a transparent corner of one factory never swallows a
tap meant for its neighbour's tile. (Tile images sit at 10–120 of the 130-unit box, inside the
circle, so they never overlap a neighbour's.)

**Centre** (`CenterView`): in ring mode a felt disc (`--felt`, soft inner shadow) centred in the
box; in grid mode a felt rounded rectangle. Its groups become a tile with a count badge in the
corner (`×n` → a small pill on the tile), 36 px, so six groups (5 colours + the FIRST marker) fit
two rows of three inside the disc's inscribed square (0.707 × 46 % × 410 px ≈ 133 px). "Centre is
empty" in light text on the felt. The `button`s with `data-factory`/`data-color` stay, which the
e2e helper and unit tests click.

## 7. Motion (C)

All of §7 is skipped when `matchMedia('(prefers-reduced-motion: reduce)')` matches (checked through
a tiny `lib/motion.ts` `reducedMotion()` helper so tests can stub it).

### 7.1 Tile flight — `web/src/components/Flight.svelte`

- **Trigger**: in `Table.svelte`, a new version where `view.version === prevVersion + 1`
  (consecutive, so the previous render is exactly the state before the move), `lastMove` is a
  take with `lastMove.version === view.version`, by **any** seat (your confirm gets the flight
  too). Not consecutive → no flight, no hiding; the state just changes.
- **What arrives where** — pure helper `arrivals(prev: PlayerView, next: PlayerView)` in
  `lib/selection.ts` (beside `ghost`, which already computes the same split for the preview):
  - `line: { row, from, to } | null` — the pattern line whose count grew: `from` = previous count
    (0 if it held nothing), `to` = new count. Tiles beyond the line's capacity are not here.
  - `floor: number[]` — indices in `floorDisplay(next.floor, next.hasFirst).tiles` (colour order,
    FIRST marker first, at most 7) that are new. Computed by **colour count**, not by position,
    because the server rebuilds the floor in colour order (`Projection.cs`): for each colour,
    new = next count − prev count, and the new tiles of that colour are the **last** ones of its
    run in the display; the marker is new when `!prev.hasFirst && next.hasFirst`. Indices ≥ 7
    (shown as `+n`) are dropped.
- **Source**: the element `[data-flight-source="<factory>"]` — the factory `svg` (it stays mounted
  when empty) or the centre container. Its bounding box is read after the DOM update (`tick()`);
  the box centre is the start point, spread ±12 px per tile.
- **Destinations**: `[data-flight-dest="<seat>:<row>"]` for a pattern line and
  `[data-flight-dest="<seat>:floor"]`, which each visible representation exposes:
  - `PlayerBoard`: an invisible `<rect>` per line (`lineBox(row)`) and for the floor
    (`FLOOR_BOX`), always rendered, not only when interactive. They carry no `data-row`, so the
    existing `PlayerBoard.test.ts` counts of `[data-row]` stay 0 / 6.
  - `OpponentCard`: each line's row and the floor strip.
  The first match with a non-zero bounding box is used, so the `display: none` desktop or phone
  duplicate is ignored (a card scrolled off screen still counts; the flight ends off screen).
  No match → no flight for that part.
- **Flight**: one sprite per arriving tile, capped at 5 to the line and 7 to the floor (the marker
  flies with the floor tiles, using `tile_first.png`), absolutely positioned `<img>`s in a fixed
  overlay, animated with the Web Animations API to the destination centre over **450 ms**,
  `ease-in-out`, 40 ms stagger, scale 1 → 0.85, then removed. Total ≤ 650 ms.
- **Landing**: until the flight lands, `PlayerBoard` draws the arriving tiles (line indices
  `from … to − 1`, the floor indices from `arrivals`) at opacity 0, then fades them in over
  120 ms. `Table` passes the `arrivals` result down as a prop (`arriving`) to the board of that
  seat; the phone card shows tiles immediately (its cells are too small for the effect to read).
- A newer version mid-flight cancels the pending flight: overlay cleared, `arriving` reset to null.

### 7.2 Scores

- Status bar scores and board labels count up/down to a new value with `svelte/motion`'s `Tween`
  (400 ms). In the SVG label the number is part of the text node, so the tween drives it.
- A **"+n" / "−n" pop** floats up 16 px and fades (700 ms) next to the score in the status bar
  when a seat's score changes between versions. Green-ish `--accent` for +, `--danger` for −.
- The round-scoring sound keeps its existing trigger.

### 7.3 Active player glow

The active seat's status-bar score chip, desktop board panel and phone card get a slow (2 s,
infinite, alternate) box-shadow pulse in `--glow`. Not during `finished`.

## 8. Your-turn emphasis (D)

- **Status pill**: when it is your turn (`view.legal`), the "Your turn…" text sits in a pill with
  `--accent` background and `--accent-ink` text, plus a one-time 600 ms scale-in when the turn
  starts. The `data-testid="your-turn"` stays on the same element.
- **Legal targets**: `PlayerBoard`'s `.hit.legal` and `.hit.target` drop the dashed stroke for a
  1.6 s pulsing fill of `--accent` at 10–22 % opacity and a solid 3 px `--accent` stroke at 60 %.
  `.hit.chosen` stays a solid stroke with a stronger fill. Under reduced motion: a static 16 % fill.
- **Pickable tiles**: factory tiles and centre groups that `canPick` lift 2 px with a soft shadow
  on hover/focus (desktop pointer only, `@media (hover: hover)`); the selected colour keeps its
  ring, now in `--accent`.
- **Not your turn**: factories are not dimmed (they must stay readable), but the cursor stays
  default and no lift.

## 9. Files

| file | change |
|---|---|
| `web/src/app.css` | tokens, dark mode, background, panel, buttons, font import |
| `web/src/App.svelte` | header bar |
| `web/src/components/Table.svelte` | no supply; uses `Market`; flight trigger; prev snapshot; panels |
| `web/src/components/Market.svelte` | **new**: ring / grid by container query |
| `web/src/components/CenterView.svelte` | felt look, flight source attr, tokens |
| `web/src/components/FactoryView.svelte` | flight source attr, hover lift, tokens |
| `web/src/components/PlayerBoard.svelte` | flight dest rects, landing fade, legal pulse, label tween, tokens |
| `web/src/components/OpponentCard.svelte` | lines + wall + floor, dest attrs, glow |
| `web/src/components/StatusBar.svelte` | pill, tweened scores, pops, glow |
| `web/src/components/Flight.svelte` | **new**: overlay |
| `web/src/components/Toasts.svelte` | **new** |
| `web/src/components/GamePage.svelte` | narrate → toast; 409 as toast; mount `Toasts`; clear on unmount |
| `web/src/components/Lobby.svelte`, `SeatPanel.svelte`, `Sheet.svelte`, `GameControls.svelte` | tokens, `.panel` |
| `web/src/lib/narrate.ts` | **new**, pure |
| `web/src/lib/toasts.svelte.ts` | **new** |
| `web/src/lib/motion.ts` | **new**: `reducedMotion()` |
| `web/src/lib/geometry.ts` | `TILE_COLORS`, `wallColor(r, c)`, `floorPenalty(n)`, `ringLayout(n)` |
| `web/package.json` | `@fontsource/cinzel` |

## 10. Testing

Unit (Vitest, jsdom), written before the code:

- `narrate.test.ts`: each take form (factory, centre, floor, first-player marker, "+ first
  player"); each wall form (wall columns, floor only, all −1) with +, − and unchanged score; round
  change; own hand-made move not narrated, own forced move (`prev.autoPlay`) and own bot-seat move
  narrated; spectator narrates all; null boards; `lastMove.version !== next.version` → no move
  bubble; **a version gap**: player A took the marker and B's centre take arrives two versions
  later → B's bubble has no "+ first player" and no score suffix.
- `selection.test.ts` (`arrivals`): line fill from empty and from partial; overflow split between
  line and floor; floor colour reorder (blue arriving on a floor holding white → the blue index is
  new, white's is not); marker arriving with a centre take; more than 7 floor tiles.
- `toasts.test.ts`: cap of three drops the oldest; auto-dismiss at 3 s (fake timers); early dismiss
  clears its timer; `clearToasts`.
- `geometry.test.ts`: `wallColor` matches the printed board's first two rows; `floorPenalty` for
  0, 1, 3, 6, 7 and 9 (−14 cap); `ringLayout(n)` for 5/7/9: neighbouring factories do not overlap, every factory
  is inside the box, neighbouring factory **circles** are ≥ `d + 4` apart, the disc does not
  touch any factory circle, `d(n)` × 410 px ≥ 90 px.
- `OpponentCard.test.ts` (new): renders 15 line slots, 25 wall cells, 7 floor slots; filled counts
  match a fixture; penalty text; first-player marker shown when held.
- `Table.test.ts`: no discard, no bag (replaces the current test).
- `GamePage.test.ts`: an opponent's move from the event stream adds a bubble with the sentence;
  your own does not; the 409 bubble; three-cap through the page.
- Flight: a unit test with `reducedMotion()` stubbed false checks that a consecutive take creates
  one overlay image per arriving tile (line + floor, with the caps) and removes them
  (`Element.animate` stubbed in jsdom); with it stubbed true, none; a non-consecutive version,
  none.

e2e (Playwright, existing harness):

- `layout.spec.ts`: existing tests stay (360 px no scroll with 4 players, desktop factory ≥ 90 px,
  opponent width ≤ 0.6 × mine). Add: the phone opponent card shows lines, wall and floor (count of
  cells) and still no horizontal scroll; at 480 px the grid layout is used (no ring on a wide
  phone); at 1440 px the factories sit on a ring (all ≥ 90 px wide, neighbouring centres at least
  one factory width apart, i.e. circles do not overlap) with the centre disc inside the ring; at
  900 px the grid layout is used and the factory ≥ 90 px and opponent ≤ 0.6 × your board bars
  still hold.
- `game.spec.ts`: a full game still completes. Playwright runs with `reducedMotion: 'reduce'` in
  the existing tests so flights do not slow the suite or intercept clicks; one new test runs with
  motion on and checks that an opponent bot's move produces a bubble.
- Dark mode: one e2e screenshot-free check with `colorScheme: 'dark'` that `body`'s computed
  background is the dark `--bg`.

Shared resources: the e2e server is a fixed container name on port 5081 (`Makefile`
`e2e-server-start`), so `make e2e` in this worktree and in `../Azul-Board-Game-ui` or the main
checkout at the same time would collide. Run them one at a time.

## 11. Review log

Codex adversarial review of the first draft (2026-10-04); all ten findings accepted.

1. Floor penalty used −3 per tile past seven and ignored the cap → mirror `FloorToScore`
   (−14 cap) with the marker counted (§3).
2. Landing detection by floor index hid existing tiles, since the floor is colour-ordered →
   `arrivals()` diffs colour counts and the marker (§7.1).
3. Flights sent overflow to the line → separate line and floor destinations from `arrivals()`
   (§7.1).
4. Nine-factory square boxes overlap at the corners, contradicting a box test → separation is
   between circles, tests assert that, and SVG boxes do not take pointer events in ring mode (§6.1,
   §10).
5. The container query alone put the ring on 430 px-wide phones → gated by `min-width: 900px` too,
   tested at 480 px (§6.1, §10).
6. Version gaps misattributed the marker and score changes → inferred parts only for consecutive
   versions, with a gap test (§4.1).
7. A wall move need not tile the wall → wording from `columns` (wall / floor / scored) (§4.1).
8. Own-seat suppression also hid forced turns and bot-seat moves → narrate those (§4.1).
9. No path for the colour dot through `string[]` → structured `Narration` (§4.1, §4.2).
10. One-line truncation could cut the whole move behind a long name → name truncated separately,
    text wraps to two lines (§4.3).
