# Azul on the web — design

Date: 2026-10-03. Branch: `feat/web` (worktree `../Azul-Board-Game-web`).

## 1. Goal and requirements

Publish the Azul game at **https://azul.signalwave.dev** so the user's invited friends can play it
online from phones and desktops.

What the user asked for:

- R1. Behind Cloudflare Tunnel + Cloudflare Access, admitting the **`players` rule group** (currently
  4 emails). Catan's Access application moves to the same group, so both apps admit the same people.
- R2. A proper web port (not a streamed desktop window).
- R3. Friends play online: a lobby where a game is created, friends take seats from their own
  devices, and empty seats are bots. 2–4 players.
- R4. Plays well on phone and desktop.
- R5. Keeps the engine's current rules, including its end-of-round **wall phase** where each player
  picks a wall column for each completed pattern line (the AI is built for this variant).
- R6. Games in progress survive a server restart or redeploy.

Assumptions (not stated by the user, open to correction):

- A1. The C# engine (`AzulLibrary`) and its `MCTS_Stochastic` AI stay the source of truth for rules
  and bots. The pure-Python `azul/logic.py`/`azul/ai.py` are not used.
- A2. The desktop pygame app keeps working; it is not removed.
- A3. Existing sprites (`assets/sprites/board2.png`, `factory.png`, `tile_*.png`) and sounds are reused.
- A4. Bag and discard tile counts are shown to everyone (the counts are public in the physical game).
- A5. Any invited person can watch any game read-only; only seated players move.
- A6. The origin checks the Access JWT on every path except `/api/health` (§4.1).

Success criteria:

- S1. Two invited people on different devices plus bots finish a 3- or 4-player game at
  azul.signalwave.dev, with every move checked by the server.
- S2. Killing and restarting the stack mid-game resumes the same game, including a pending bot turn.
- S3. `scripts/serve-check.sh` passes all probes (§6.4), and an email outside the group is refused by
  Access.
- S4. With 4 concurrent 4-player games whose seats are all bots, the API answers `GET /api/games/{id}`
  with p95 under 200 ms on this machine and every game keeps advancing (measured by the load script,
  §7).

Out of scope: chat, turn timers, push notifications, undo, standard (fixed-wall) rules, ratings,
spectators outside the invite list, horizontal scaling (one app instance only).

## 2. Architecture

One container runs an **ASP.NET Core** (.NET 10 LTS) server that hosts the engine in-process,
serves a **Svelte + Vite + TypeScript** single-page app from the same origin, stores games in
**SQLite**, and pushes updates over **Server-Sent Events**. A `cloudflared` container is the tunnel
connector.

```
browser ──https──> Cloudflare Access ──> Tunnel ──> cloudflared ──http://app:8080──> AzulServer
                                                                                      ├─ AccessAuth (JWT)
                                                                                      ├─ REST /api + SSE
                                                                                      ├─ GameService ── GameStore (SQLite, /data)
                                                                                      ├─ BotScheduler (MCTS workers)
                                                                                      └─ static web/dist
```

Rejected alternatives: FastAPI + pythonnet (two runtimes, thread crossing into the CLR, and it would
not fix the engine's shared state, §3.1); engine in the browser via WASM (the server would still have
to hold authority, persistence and bots). Codex reviewed the approach on 2026-10-03 and confirmed the
choice, with the engine work in §3 as the first milestone.

### 2.1 Repository layout

| Path | Content |
| --- | --- |
| `AzulLibrary/` | The engine: `net10.0` **class library** (was `Exe`, `net7.0`). Changes in §3. |
| `AzulBench/` | New console project: today's `Test.cs` top-level statements and the CsvHelper-based benchmark code from `Utils.cs`. References `AzulLibrary`. |
| `AzulLibrary.Tests/` | xUnit tests for the engine changes. |
| `server/AzulServer/` | ASP.NET Core minimal-API app. |
| `server/AzulServer.Tests/` | xUnit + `WebApplicationFactory` integration tests. |
| `web/` | Svelte SPA; Vitest unit tests; Playwright e2e. |
| `Azul.sln` | All .NET projects (replaces `Azul-Board-Game.sln`). |
| `Dockerfile`, `docker-compose.serve.yml`, `.env.serve.example`, `.dockerignore` | Serve stack. |
| `scripts/serve-check.sh`, `scripts/bot-load.sh` | Deploy probes; load measurement. |
| `Makefile` | `test`, `build`, `serve`, `serve-check`, `serve-logs`, `serve-down`. |
| `docs/deploy.md` | Publishing and day-to-day operation. |

`AzulBench/`, `AzulLibrary.Tests/` and `server/` sit outside `AzulLibrary/` so the library's default
source glob does not pick them up.

The desktop app (`azul/ai_wrapper.py`, `azul/logic_wrapper.py`) loads
`AzulLibrary/bin/Release/net7.0`; both paths change to `net10.0`.

### 2.2 Toolchain

There is no .NET SDK on the host. Every .NET build and test runs in the pinned
`mcr.microsoft.com/dotnet/sdk:10.0` container through `make` targets (the checkout is bind-mounted,
NuGet cache in a named volume), so nothing is installed on the host. Node 25 is on the host and runs
`web/` directly; the Docker build uses a pinned `node` image.

## 3. Engine changes (`AzulLibrary`), first milestone

Each change has a test in `AzulLibrary.Tests/`. Codex verified these problems in the source on
2026-10-03.

### 3.1 No shared mutable state between games

- `GameUtils.Game<M>.rng` (`Utils.cs:239`) is a static `Random` shared by all games. It becomes an
  **instance** field. `TicTacToe`, `RandomGameLogic` and `Ai.cs` call sites switch to the instance.
- `Game.rowIdxArray` and `Game.colorIdxArray` (`Logic.cs:26-27`) are static and shuffled during move
  generation (`Logic.cs:644-646`). They become instance fields.
- `MCTS_Stochastic.c` (`Ai.cs:185`) is a mutable static; it becomes `const` or readonly (it is never
  reassigned). The static stopwatch/search state of `ParanoidID`/`MaxnID` (`Ai.cs:19`, `Ai.cs:112`)
  is not used by the server and is left alone; the server never calls those classes.
- Rule: one MCTS tree is touched by one thread only. The server never shares a tree or a `Game`
  between threads (§5.3).

Test: two games stepped from two threads at once for many moves, each compared against the same
game stepped alone with the same seed, end in identical states.

### 3.2 Side-effect-free `Clone()`

`DeepCopier` (`DeepCopy/CopierGenerator.cs:75`) calls the public parameterless constructor before
copying fields, and `Game()` deals a fresh 2-player game (`Logic.cs:41-73`): every MCTS copy draws
from the RNG and resets buffers. Replace it:

- `Game.Clone()` copies every field explicitly (arrays deep-copied; scratch buffers newly allocated)
  through a private constructor that does no dealing. The clone gets its own `Random` seeded from
  `Random.Shared` (simulations must not advance the authoritative game's RNG).
- `GameUtils.Game<M>` gains `public abstract Game<M> Clone()`; `TicTacToe` implements it.
- Every `DeepCopier.Copy(...)` call is replaced with `Clone()`: `Ai.cs:37, 89, 150, 217, 233, 289,
  580, 596, 631`, `TicTacToe.cs:191` (`TicTacToe.GetGreedyMove`) and `Test.cs:484` (moving to
  `AzulBench`). The `using DeepCopy;` directives (`Ai.cs:5`, `Logic.cs:8`, `TicTacToe.cs:2`,
  `Test.cs:10`) go too, and then the `DeepCopy/` folder is deleted.

Tests: a clone equals the original field by field (a test-only structural comparer, not
`Game.Equals`, which ignores bag, discard, phase and more — `Logic.cs:679`); mutating the clone leaves
the original unchanged; cloning does not change the original's next random draw.

### 3.3 Save format

`Player.grid` and `Player.line` are `int[,]` (`Logic.cs:1492-1493`), which System.Text.Json cannot
serialize. Add an explicit, versioned snapshot:

```csharp
public sealed record GameSnapshot(int Format, int NumPlayers, int ActivePlayer, int Step,
    ulong ChanceHash, int RoundIdx, bool IsRegularPhase, int NewRoundPlayer,
    int CountPlayerClearedRound, bool IsFinished, int[][] Factories, int[] Bag, int[] Discarded,
    PlayerSnapshot[] Players);
public sealed record PlayerSnapshot(int Score, int[][] Grid, int[][] Line, int[] Floor);
```

- `Factories` is the engine's array as is: entries `0..numFactories-1` are factories (5 colour
  counts), entry `numFactories` is the centre (5 colour counts + the FIRST slot). `Floor` is the
  engine's per-colour counts (6 entries, index 5 = FIRST). `Grid` cells are `-1` (empty) or the colour
  `0..4`; `Line[row][color]` is a count.
- `Game.ToSnapshot()` and `Game.FromSnapshot(GameSnapshot)`. `FromSnapshot` goes through the private
  non-dealing constructor and **validates invariants**; a bad snapshot throws:
  - shapes: `NumPlayers` 2..4, factory count matching `NumPlayers` (5/7/9), 5×5 grids and lines,
    6-entry floors;
  - ranges: all counts ≥ 0; grid cells in `-1..4`; `ActivePlayer`, `NewRoundPlayer` in range;
    `CountPlayerClearedRound` in `0..NumPlayers`;
  - per player: each pattern line holds at most one colour and at most `row+1` tiles; no colour
    appears twice in a wall row or a wall column; a line's colour is not already on that wall row;
  - tile conservation for the 5 ordinary colours: factories + centre + pattern lines + wall cells of
    that colour + floors + bag + discard = 20 per colour;
  - FIRST: at most one marker in the centre's FIRST slot or in one floor's index 5. During the take
    phase it is in exactly one place; it is absent only during the wall phase after its holder has
    resolved their wall turn;
  - phase: in the take phase `CountPlayerClearedRound` is 0; `IsFinished` implies the wall phase
    with every player resolved and the engine's `IsGameOver()` true.
  The existing `Player.SanityCheck` (`Logic.cs:1512`) is not used: it treats wall cells as `0..1`,
  but they hold `-1..4`.
- `IsFinished` is persisted (it guards against scoring twice, §3.4) and cross-checked as above.
- `Format = 1`. A future change to the engine's fields bumps it and adds a migration.
- The RNG is **not** persisted: a restored game gets a fresh `Random`. Future draws after a restart
  are not reproducible, which is fine because nothing depends on them being so.
- MCTS trees are never persisted; bots rebuild their tree each turn (§5.3).

Tests: round trip (`FromSnapshot(ToSnapshot(g))` structurally equal to `g`) after **every move** of
greedy full games for 2, 3 and 4 players (this covers both phases, bag refills from discard, and
final scoring); corrupt snapshots (wrong shape, tile count off by one, two colours on one line,
duplicate wall colour, two FIRST markers) are rejected.

### 3.4 Strict move checks and the floor-column bug

- `IsValid(int[] colIdx, int[] colors)` treats every column `>= 5` as the floor (`Logic.cs:1277`),
  but `Player.UpdateRound` only handles exactly `5` (`Logic.cs:1614`); column 6 writes outside the
  grid. Fix: columns must be `-1`, `0..4` or `5`; anything else is invalid.
- `IsValid(int factoryIdx, int color, int row)` indexes arrays before range checks (`Logic.cs:1297`).
  Fix: return false for out-of-range `factoryIdx`, `color`, `row` first.
- `IsValid(Move)` (`Logic.cs:1328`) picks the phase from the move, not from the game. Fix: a take move
  is invalid during the wall phase and vice versa.
- Add `Game.IsFinished` (set when the final `Play` scores the game) and make `Play` throw if called
  when finished, so `Player.UpdateGame()` end bonuses (`Logic.cs:1650`) can never run twice.
  `Play`'s return value means "a new round was dealt", not success; the server ignores it.

The server never constructs moves from client-supplied counts, colours of lines, player indices or
first-player flags. It builds them with the correcting constructor `Move(Move, Game)`
(`Logic.cs:1437`) from the minimal request in §4.4, then calls `IsValid`.

Tests: column 6, negative indices, wrong-phase moves and moves after the end are all rejected; a
regular game played to the end by `GetGreedyMove` scores once.

The engine also allows taking **only the first-player marker** from the centre to the floor
(`IsValid` accepts colour FIRST at the centre, `Logic.cs:1304`; `GetPossibleActions` adds it whenever
the marker is in the centre, `Logic.cs:1077`). R5 keeps the engine's rules, so the web game keeps
this move (§4.4, §5.4).

### 3.5 Legal-move hints

For the UI (§5.4) the engine exposes, for the active player, two new **deterministic,
side-effect-free** methods (no RNG, no shuffling, unlike `GetColIdxMoves()`, which shuffles with the
game RNG at `Logic.cs:134`):

- take phase: `LegalTakes()` — every `(factory, color, row)` with colour `0..4` that `IsValid`
  accepts, plus `(centre, 5, 5)` while the marker is in the centre;
- wall phase: `WallOptions()` — for each row, `null` if the line is not complete, else its colour and
  the list of allowed targets: wall columns whose cell in that row is empty and whose column does not
  already hold that colour, plus `5` (floor). Same per-row rule as `GetColIdxMoves()` (`Logic.cs:99`).

The one constraint across rows (two completed lines of the same colour cannot take the same wall
column) is not in the per-row lists. The client enforces it while selecting, and the server's
`IsValid` rejects it. Full assignments are not enumerated: five completed lines of different colours
on an empty wall would be 6^5 = 7,776 arrays.

Test: `LegalTakes()` and `WallOptions()` agree with `GetPossibleActions()`/`GetColIdxMoves()` (as sets)
across greedy full games, and calling them does not change the game's next random draw.

### 3.6 Turn order (as the engine has it; documented, not changed)

- Take phase: seats move in turn. The move that empties the factories and centre advances
  `activePlayer` first and then switches to the wall phase (`Logic.cs:1170-1177`), so the wall phase
  starts with the seat after the last taker.
- Wall phase: every seat takes one wall turn, cyclically, **including seats with no completed
  line**; their only legal move is the all-`-1` assignment, which still clears their floor (penalties
  and discard). Whoever holds the FIRST marker becomes `newRoundPlayer` when their floor clears.
- After the last wall turn: game over (end bonuses scored once, `IsFinished`), or a new round dealt
  with `newRoundPlayer` to move.

Test: a scripted game checks the first wall seat, that a seat with no completed line still gets
(and needs) a wall turn, the next round's starter, and that the game ends only after every seat
resolved its wall turn.

## 4. Server (`server/AzulServer`)

### 4.1 Access authentication

Middleware on **every request** (pages, static assets, `/api`, SSE) except `GET /api/health`:

- Reads `Cf-Access-Jwt-Assertion`. Verifies RS256 only, signature, `aud` = `AZUL_ACCESS_AUD`,
  `iss` = `https://<team domain>`, `exp` required, `nbf` if present, 60 s leeway. Identity = the
  verified `email` claim, lowercased; a token without `email` → 401.
  `Cf-Access-Authenticated-User-Email` is ignored.
- Keys come from `https://<team domain>/cdn-cgi/access/certs`, cached for 1 hour. A token whose `kid`
  is not in the cache triggers one refresh (at most one refresh per 30 s). If the refresh fails and
  the `kid` is still unknown → **503**; if cached keys verify the token, a failed background refresh
  does not fail the request. Any verification failure → **401**.
- `AZUL_ACCESS_TEAM_DOMAIN` and `AZUL_ACCESS_AUD`: both or neither, one alone is a startup error. The
  team domain is normalised (strip `https://` and a trailing `/`).
- **Decision (the skill asks "what does the origin guard?"):** all paths, not `/api` only. Access adds
  the JWT header to every proxied request, so guarding the SPA shell and assets costs nothing, and a
  misconfigured edge then exposes nothing but `/api/health`. The loopback port answers 401 to
  everything else, so the UI is only reachable locally in dev mode.
- **Dev mode** (both unset): identity comes from an `X-Dev-User` header (default `dev@localhost`).
  The server logs a warning at startup. The serve compose file requires both variables (`${VAR:?}`),
  so production cannot start in dev mode.

### 4.2 Data (SQLite, `/data/azul.db`)

```sql
games(id TEXT PRIMARY KEY, creator TEXT, status TEXT CHECK(status IN ('lobby','playing','finished')),
      num_players INT, version INT, state_json TEXT NULL, finish_reason TEXT NULL,
      created_at TEXT, updated_at TEXT)
seats(game_id TEXT REFERENCES games(id) ON DELETE CASCADE, idx INT,
      kind TEXT CHECK(kind IN ('open','human','bot')), email TEXT NULL,
      PRIMARY KEY(game_id, idx))
moves(game_id TEXT REFERENCES games(id) ON DELETE CASCADE, version INT, seat INT, actor TEXT,
      move_json TEXT, request_id TEXT NULL, request_hash TEXT NULL, result_json TEXT, at TEXT,
      PRIMARY KEY(game_id, version), UNIQUE(game_id, request_id))
schema_version(v INT)
```

- `id`: 10 random base32 characters. `state_json`: `GameSnapshot` (§3.3), null while in the lobby.
- `version` increments on every change to a game (seat changes, start, moves); SSE and clients use it.
- Seat invariants: `open` → `email` null. `human` → `email` = the person playing it. `bot` with
  `email` null → a bot from the start; `bot` with `email` set → a human seat handed to a bot, which
  that email may take back. Unique `(game_id, email)` among non-null emails (one seat per person).
- Foreign keys on (`PRAGMA foreign_keys=ON`); deleting a game deletes its seats and moves.
- Migrations: numbered SQL files applied at startup inside a transaction; `schema_version` records
  the last one. WAL mode; one connection per command.
- Backup: `docs/deploy.md` documents `sqlite3 .backup` through `docker compose exec` and a tested
  restore.

### 4.3 Game lifecycle and seat rules

1. **Create** (`lobby`): any invited person picks 2, 3 or 4 players. The creator takes seat 0; other
   seats are `open`. The creator stays the creator for the game's life, seated or not.
2. **Lobby**: anyone may claim an open seat (one seat per person per game) or release their own. The
   creator may switch a seat between `open` and `bot` (not a `human` seat: 409), and remove a person
   from a seat (it becomes `open`). Seat changes are atomic (`UPDATE … WHERE kind='open'`).
3. **Start** (creator only, seated or not): requires at least one `human` seat (else 409). Remaining
   open seats become bots, the engine deals `new Game(numPlayers)`, the starting player is seat 0 (as
   the desktop does), status `playing`.
4. **Playing**: only a `human` seat's email moves for it, on its turn. A seated person may **hand
   their seat to a bot** (`kind=bot`, email kept) and later **take it back** (`kind=human`); the
   creator may hand any human seat to a bot (a friend who left). Each of these bumps `version`, so a
   bot move computed before a take-back is discarded (409). A bot seat that was never human stays a
   bot. Nobody takes another person's seat.
5. **Forced moves**: when the active seat is human and has exactly one legal move (a wall turn with
   no completed line, or only floor targets; or only the FIRST marker left to take), the server plays
   it after the 1 s visible delay, like a bot move (§4.7).
6. **Finished**: set in the same transaction as the scoring move; `finish_reason` is `normal` or
   `stalemate` (the engine's `MAX_STEPS` check, `Logic.cs:1235`, which only fires at a round boundary;
   kept as is). Every later command except delete gets 409.
7. **Delete**: creator only, any status. One transaction removes the game, its seats and moves; open
   streams get a `deleted` event and are closed (§4.6); later requests for the id get 404.

### 4.4 API

All bodies are JSON. State-changing requests must send `Content-Type: application/json` and, when
present, an `Origin` equal to the request host; otherwise 403. No CORS headers are sent.

| Method and path | Who | Result |
| --- | --- | --- |
| `GET /api/health` | anyone | 200 `{ok:true}`; no DB access |
| `GET /api/me` | invited | `{email}` |
| `GET /api/games` | invited | lobby, playing and last 20 finished games with seat summaries |
| `POST /api/games` `{players}` | invited | 201 game; 400 unless 2–4 |
| `GET /api/games/{id}` | invited | game view (below); 404 if unknown or deleted |
| `POST /api/games/{id}/seats/{i}/claim` | invited, lobby | 200 / 409 taken or already seated |
| `POST /api/games/{id}/seats/{i}/release` | seat owner or creator, lobby | 200 |
| `POST /api/games/{id}/seats/{i}/kind` `{kind:"open"\|"bot"}` | creator, lobby, seat not human | 200 / 409 |
| `POST /api/games/{id}/start` | creator, lobby | 200 / 409 |
| `POST /api/games/{id}/seats/{i}/to-bot` | seat owner or creator, playing | 200 |
| `POST /api/games/{id}/seats/{i}/take-back` | the email kept on that bot seat, playing | 200 |
| `POST /api/games/{id}/moves` | active seat owner | see below |
| `DELETE /api/games/{id}` | creator | 204 |
| `GET /api/games/{id}/events` | invited | SSE (§4.6) |

Move request, one of:

```json
{"version": 41, "requestId": "uuid", "kind": "take", "factory": 3, "color": 1, "row": 2}
{"version": 41, "requestId": "uuid", "kind": "wall", "columns": [2, -1, 5, -1, -1]}
```

- Shape and ranges, checked before anything else (400): `factory` 0..numFactories (numFactories =
  centre); `color` 0..4, or 5 (the FIRST marker alone) only with `factory` = centre and `row` = 5;
  `row` 0..5 (5 = floor); `columns` five values in {-1, 0..4, 5}; `requestId` a UUID.
- Then, in order:
  1. **Idempotency.** If `(game, requestId)` is already in `moves`: same actor and same payload hash
     → return the stored result (even if the turn, version or status has moved on since); different
     actor or payload → 409 `{error:"request-id-reused"}`.
  2. Game exists (404) and is playing (409).
  3. `version` matches (else **409** with the current view).
  4. Caller owns the active seat (else 403).
  5. Phase matches `kind` (else 400).
  6. `Move(Move, Game)` + `IsValid` (else 400 `{error:"illegal"}`).
- Success: 200 with the new game view, and the result is stored with the move for retries.

Version is checked before ownership so that a player's own duplicate submit from two devices gets 409
(stale), not 403.

Game view (projected per viewer, §4.5):

```json
{"id": "...", "status": "playing", "version": 42, "numPlayers": 3, "creator": "a@x",
 "you": {"email": "b@y", "seat": 1},
 "seats": [{"idx":0,"kind":"human","email":"a@x"}, {"idx":1,"kind":"human","email":"b@y"},
           {"idx":2,"kind":"bot","email":null}],
 "board": {"round": 2, "phase": "take", "activeSeat": 1,
           "factories": [[0,2,1,1,0], ...], "center": [1,0,3,0,0], "centerHasFirst": true,
           "bag": [12,10,9,14,11], "discard": [0,1,0,0,2],
           "players": [{"score": 7, "lines": [[1,2], null, null, [4,4], null],
                        "wall": [[-1,0,-1,-1,-1], ...5], "floor": [3,3,4], "hasFirst": false}, ...]},
 "legal": {"takes": [[3,1,2], ...]} | {"wall": [null, {"color":2,"targets":[1,3,5]}, ...5]} | null,
 "lastMove": {"seat": 0, "kind": "take", "factory": 2, "color": 4, "row": 5, "tiles": 3} | null,
 "result": {"scores": [..], "winners": [0], "reason": "normal"} | null}
```

Projection rules from the engine (§3.3 fields):

- `factories` = `Factories[0..numFactories-1]`; `center` = the first 5 entries of
  `Factories[numFactories]`; `centerHasFirst` = its FIRST slot > 0.
- `lines[row]` = the one colour with a non-zero count as `[color, count]`, or null.
- `floor` = ordinary-colour tiles expanded from the per-colour counts in colour order (the engine keeps
  no arrival order); the FIRST marker is not in `floor` and shows only as `hasFirst`. The engine does
  not cap the floor at 7 (`Player.Play` keeps adding, `Logic.cs:1716`); the UI draws the first 7 slots
  (marker first) and a `+N` badge for the rest.
- `seats[].email` on a `bot` seat is the email that may take it back, or null.
- `legal` is present only when the viewer owns the active seat; `you.seat` is null for spectators.

### 4.5 Command processing (GameService)

- One in-memory `SemaphoreSlim` per game id. Every command takes it, loads the row, checks, applies
  to `Game.FromSnapshot(...)` (a fresh object; the stored state is never mutated in place), and in
  **one SQLite transaction** writes the new `state_json`, `version + 1`, status/finish fields and the
  `moves` row (with its result). An exception before commit leaves the stored game unchanged.
- After commit it publishes **`(gameId, version)`** to the EventHub (not a view: views differ per
  viewer) and calls `BotScheduler.Poke(gameId)`. Neither can fail the request: both are non-blocking,
  and a missed poke is repaired by the scheduler's sweep (§4.7).

### 4.6 Live updates (SSE)

- `GET /api/games/{id}/events` sends `text/event-stream` with `Cache-Control: no-cache` and
  `X-Accel-Buffering: no`, and flushes each event.
- Each stream knows its viewer's email. On a version notice it loads the committed state (cached per
  version in memory, so N viewers cause one DB read) and **projects the view for its own viewer**
  (`you`, `legal`), then sends a `state` event (`id:` = version). Views are a few KB; no diffs.
- Subscription happens before the first snapshot is read, and the first event is the current view,
  so no version can be missed in between; the client ignores events whose version is not higher than
  what it has.
- A `: ping` comment every 20 s keeps the stream inside Cloudflare's 125 s proxy read timeout
  (Cloudflare docs, error 524 page, checked 2026-10-03).
- Each subscriber has a bounded channel of version notices (capacity 1, latest wins: every event is
  a full state). Streams close when the client disconnects or after 30 minutes (the client
  reconnects); a reconnect re-runs the JWT check.
- Deleting a game sends a `deleted` event and closes its streams.

### 4.7 Bots and forced moves (BotScheduler)

The database is the source of truth for which games need a server-made move: a `playing` game whose
active seat is a `bot`, or a `human` seat with exactly one legal move (§4.3). The scheduler never
holds the only copy of that fact.

- **Work set:** an in-memory set of game ids (deduplicated, so its size is bounded by the number of
  games) plus a signal. `Poke(gameId)` adds the id and signals; it never blocks and never throws to
  the caller.
- **Sweep:** at startup and every 30 s, the scheduler queries for playing games that need a
  server-made move and pokes each one. This covers a restart (S2), a lost poke, and a worker that
  crashed.
- **Workers** (`AZUL_BOT_WORKERS`, default 2) take an id from the set, load the game, and return if
  it no longer needs a server move. For a bot seat they build a fresh
  `MCTS_Stochastic<Game, Move>(game.Clone(), 0.0f)` and grow it for `AZUL_BOT_THINK_SECONDS`
  (default 3.0). They pick `GetBestAction()`, falling back to `GetGreedyMove()` when the best child's
  win ratio is 0 (the desktop's rule, `game.py:261`). For a forced move they take the one legal move.
  Each move goes through GameService with the version read at the start; a 409 discards it.
- After a move commits, GameService pokes the game again, so the worker never waits on its own
  queue. Any exception in a worker is logged and the id is dropped; the next sweep retries it.
- The search runs on the worker's own clone and tree, so no lock is held while thinking.
- Rollouts (`Ai.cs:287`) are not cancellable mid-rollout; a rollout plays one game to the end and is
  short, so cancellation is checked between `Grow()` calls (shutdown and game deletion stop the
  search within one rollout).
- A minimum visible delay of 1 s per server-made move keeps turns followable even with a lower think
  time.

### 4.8 Health check

`AzulServer --healthcheck` makes a GET to `http://127.0.0.1:8080/api/health` and exits 0/1, so the
runtime image needs no curl.

## 5. Web client (`web/`)

### 5.1 Pages

- `/` — **Lobby**: your games first, then open lobbies to join, games in play (watch), recent
  results. "New game" with a 2/3/4 choice. Header shows `email · Sign out`
  (`/cdn-cgi/access/logout`).
- `/g/{id}` — **Game**. In `lobby` status: the seat panel (claim, release; creator: open/bot toggle,
  remove, Start). In `playing`/`finished`: the table.

### 5.2 Rendering

- SVG. Each player board is `board2.png` (900×600) as an `<image>` in a `viewBox="0 0 900 600"`,
  with tiles placed at the geometry `azul/models.py` uses (pattern lines, wall, floor, score), so it
  scales to any width. Factories use `factory.png`; tiles use the `tile_*.png` sprites.
- Sprites and the sounds used by the desktop are copied into `web/public/` at build time from
  `assets/`; the 1.5 MB fanfare wav is transcoded to a smaller mp3 during the build.

### 5.3 Layout

- **Desktop / landscape tablet (≥ 900 px wide):** factories in a ring with the centre to the left;
  your board large to the right; opponents' boards smaller below; status bar on top (round, whose
  turn, scores).
- **Phone (< 900 px):** status bar; factories and centre as a compact grid; your board full width;
  opponents as compact cards (name, score, mini wall, line fill), tap to open the full board in a
  sheet. No horizontal scrolling at 360 px width.

### 5.4 Interaction (tap, no drag)

- **Take phase, your turn:** tap a tile in a factory or the centre → all tiles of that colour there
  are highlighted and the legal destination lines (from `legal.takes`) light up; tap a line or the
  floor → a ghost preview shows where tiles land and the overflow to the floor; **Confirm** sends the
  move. Tapping the selection again or elsewhere cancels. While the FIRST marker is in the centre,
  tapping it alone offers "Take only the first-player marker" (it goes to your floor).
- **Wall phase, your turn:** completed lines are highlighted; tap a line, then one of its targets
  from `legal.wall` (a wall cell or "floor"). Once a line has a choice, the same column is removed
  from other completed lines of the same colour, so the one cross-row conflict cannot be made.
  **Confirm** when every completed line has a choice. A wall turn with only one legal move (no
  completed line, or floor only) is played by the server (§4.3) and shown as a short "Scoring your
  wall…" step.
- **Not your turn:** the table is read-only; the status bar says whose turn it is.
- **Other players' moves** arrive by SSE; the client animates `lastMove` (taken tiles move to the
  line, ~400 ms), then shows the new state.
- **Sounds:** select, invalid, your turn, bot move, round scoring, win, lose. A mute toggle is kept in
  `localStorage` (wrapped in try/catch). Audio starts after the first tap (browser autoplay rules).
- **Your turn while the tab is in the background:** the title becomes `● Your turn — Azul`.

### 5.5 Errors and session expiry

- All `fetch` calls send `X-Requested-With: XMLHttpRequest`, so an expired Access session answers
  401 instead of a redirect.
- 401 → reload the page once (a `sessionStorage` stamp prevents a loop; any success clears it), which
  sends the browser through Access login. Same pattern as catan's
  `ui/src/utils/accessSession.ts`.
- 409 on a move → apply the returned view, show "The board changed", keep no stale selection.
- 400 on a move → shake + invalid sound; selection kept.
- SSE error → reconnect with backoff (1, 2, 5, 10 s), "Reconnecting…" banner. Before reconnecting,
  `GET /api/me`; a 401 there takes the reload path above (EventSource cannot see the status itself).
- 503 → "Server can't verify sign-ins right now, retrying" banner and retry.
- `deleted` SSE event or 404 on the game → back to the lobby with "This game was deleted"; no
  reconnect.

## 6. Deployment

Follows the user's `publishing-behind-cloudflare` skill. Values come from `.env.serve` (git-ignored);
every compose command passes `--env-file .env.serve`.

### 6.1 Image

Multi-stage `Dockerfile`: `node` stage builds `web/dist`; `dotnet/sdk:10.0` stage restores and
publishes `AzulServer` (Release); runtime `dotnet/aspnet:10.0` stage copies both, creates `/data`
owned by the image's non-root `app` user, runs as `app`, listens on 8080. `.dockerignore` excludes
notebooks, `*.pt` weights, `img/`, `bin/`, `obj/`, `node_modules/`, `web/.env*`.

### 6.2 Compose (`docker-compose.serve.yml`)

```yaml
name: azul-serve
services:
  app:
    build: .
    restart: unless-stopped
    environment:
      AZUL_ACCESS_TEAM_DOMAIN: ${AZUL_ACCESS_TEAM_DOMAIN:?set in .env.serve}
      AZUL_ACCESS_AUD: ${AZUL_ACCESS_AUD:?set in .env.serve}
      AZUL_BOT_WORKERS: ${AZUL_BOT_WORKERS:-2}
      AZUL_BOT_THINK_SECONDS: ${AZUL_BOT_THINK_SECONDS:-3}
    ports: ['127.0.0.1:${AZUL_SERVE_PORT:-29648}:8080']
    volumes: ['azul-data:/data']
    healthcheck:
      test: ['CMD', '/app/AzulServer', '--healthcheck']
      interval: 10s
      retries: 6
    logging: {driver: json-file, options: {max-size: 10m, max-file: '3'}}
  cloudflared:
    image: cloudflare/cloudflared:latest
    restart: unless-stopped
    command: tunnel --no-autoupdate run
    environment:
      TUNNEL_TOKEN: ${CLOUDFLARE_TUNNEL_TOKEN:?set in .env.serve}
    depends_on:
      app: {condition: service_healthy}
volumes: {azul-data: {}}
```

Port 29648 is free on this machine (catan uses 29647). `cloudflared` reaches the app as
`http://app:8080`; no connector is installed on the host.

### 6.3 Cloudflare side (API, not dashboard)

With the skill's `cf_publish.py` (reads `~/.config/cloudflare/api-token`):

```bash
S=~/.claude/skills/publishing-behind-cloudflare/scripts
python3 $S/cf_publish.py --hostname azul.signalwave.dev --service http://app:8080 \
  --tunnel azul --app-name Azul --rule-group players --policy-name Players --session 24h --dry-run
# then without --dry-run, with --token-out ~/.config/cloudflare/azul-tunnel-token
```

- It reuses the existing `players` rule group (4 individual emails, read via the API on 2026-10-03),
  creates the reusable Allow policy `Players` (include: rule group `players`), then the Access
  application for `azul.signalwave.dev`, then the tunnel `azul`, its route and the proxied CNAME.
  Access is created before the route.
- It prints the team domain and the AUD tag for `.env.serve`, and writes the tunnel token to the
  `--token-out` file.

**Order of the first deployment.** Compose needs the real AUD and tunnel token to start any service
(every `${VAR:?}` in the file is interpolated even when one service is started), and the dry run
prints placeholders, not a usable AUD. So:

1. `cf_publish.py --dry-run`, review, then the real run: the Access application exists before the
   route. Between this step and step 4 the route points at a connector that is not running, so an
   invited person who opens the hostname sees Cloudflare's tunnel error page; nobody else gets past
   Access.
2. Write `.env.serve` from the printed team domain, AUD, and the token file.
3. `docker compose … up -d --build --wait app`, then serve-check probes 1–2 (loopback).
4. `docker compose … up -d cloudflared`, then probe 4 (public).

**Catan switch (R1).** Catan's application (`catan.signalwave.dev`) currently has one policy,
`Friends` (2 emails). After Azul is live, replace it with the `Players` policy through the Access
apps API: read the application, dry-run print the change, then update only its `policies` list.
Before the update, the application's JSON is saved to `~/.config/cloudflare/catan-app-before.json`
for rollback; `Friends` is left in place, unused. Catan's server (`catanatron/web/access.py`) verifies
the JWT's signature, algorithm, `aud`, `iss`, expiry and email, not which policy admitted the person,
and the application (with its AUD) is updated, not recreated, so no Catan restart is needed.
Acceptance: the API shows both applications with exactly the `Players` policy and Catan's AUD
unchanged; one `players` member who was not in `Friends` reaches catan.signalwave.dev; an email
outside the group is refused by both hostnames. Rollback = restore the saved `policies` list. `~/catanatron/documentation/deploy.md`
describes the policy as "Invited → Include Emails"; updating it is a change in the catanatron repo
and gets its own branch there, done only with the user's go-ahead.

### 6.4 serve-check

`scripts/serve-check.sh` (template: `~/playhub-hub-v1/scripts/serve-check.sh`):

1. `GET http://127.0.0.1:29648/api/health` → 200.
2. `GET http://127.0.0.1:29648/api/games` and `GET http://127.0.0.1:29648/` (no JWT) → 401 each.
3. (No machine-to-machine endpoints; probe omitted.)
4. `GET https://azul.signalwave.dev/` (no cookies) → a redirect whose `Location` host is the team
   domain and whose path starts with `/cdn-cgi/access/login`.

### 6.5 docs/deploy.md

Setup with `cf_publish.py`, `.env.serve` contents, `make serve` / `serve-check` / `serve-logs`,
inviting someone (add the email to the `players` rule group — this also admits them to Catan),
backup/restore, redeploy (`git pull && make serve`). Dashboard steps appear only as links to the
Cloudflare docs pages listed in the skill's reference, with the verification date.

## 7. Testing

| Level | What |
| --- | --- |
| Engine (xUnit) | §3 tests: isolation across threads, clone, snapshot round trip after every move of greedy full games (2/3/4 players), corrupt snapshots, input range and phase checks, column 6, FIRST-alone take, single end scoring, turn order (§3.6), hint methods vs. existing generators and RNG untouched. |
| Server (xUnit + `WebApplicationFactory`) | **JWT:** valid, wrong aud/iss, expired, wrong alg, no email, unknown kid then refresh, JWKS down → 503 (a test RSA key and a fake JWKS handler); static paths guarded too. **Lobby:** rules and permissions for every endpoint, creator unseated, kind on a human seat → 409, start with no human → 409. **Moves:** stale version 409, not your turn 403, illegal 400, FIRST-alone take, the same player's two devices submitting at once (different requestIds, same version) → one 200 and one 409. **Idempotency:** a retry after the turn moved on and after the game finished returns the stored 200; a reused requestId with another payload or actor → 409. **Seats in play:** take-back while a bot is thinking → the bot's move is discarded and the human moves. **Scheduler:** forced wall turns are auto-played; a worker exception, a poke that throws, and an injected failure right after commit (before publish/poke) are all repaired by the sweep; with workers busy and many games needing moves, every game keeps progressing (no deadlock). **Restart:** a fresh app on the same DB file resumes a pending bot turn. **SSE:** first event is the snapshot; one move seen at once by the mover, the next player, a spectator and the owner of a handed-to-bot seat, each with their own `you`/`legal`; a slow subscriber gets the latest version only; `deleted` closes the stream; heartbeat present. **CSRF:** non-JSON or foreign `Origin` → 403. **Delete:** rows gone, later GET 404. |
| Web (Vitest) | Selection state machine for both phases against `legal` hints, including the cross-row same-colour rule and the FIRST-alone take; floor `+N` overflow; layout breakpoints. |
| E2E (Playwright, dev mode) | Two browser contexts as two `X-Dev-User` identities plus one bot (think time 0.2 s) play a 3-player game to the end, at a phone viewport (360 px, no horizontal scroll) and a desktop viewport. A server restart mid-game: the page reconnects and the game continues. Game deleted while open → back to lobby. |
| Crash (serve stack) | `docker kill -s KILL` of the app container during a bot turn and right after a human move; after `up`, the game resumes from the last committed version and the bot turn completes (S2). |
| Load (`scripts/bot-load.sh`) | Creates 4 four-player games, starts each and hands the creator's seat to a bot (all seats bots), samples `GET /api/games/{id}` latency and CPU for 2 minutes, and records each game's version over time. Acceptance: S4, and every game advances at least once every 15 s. |
| Desktop smoke | In the SDK container with Python + pythonnet, import `azul/logic_wrapper.py` and play greedy moves to the end of a game (no display needed). |
| Deploy | `make serve-check` (§6.4); the Catan switch acceptance (§6.3); manual: a full game played from two phones. |

## 8. Milestones

1. Engine (§3) with its tests; `AzulBench`; desktop path update and smoke test.
2. Server (§4) with its tests.
3. Web client (§5) with unit and e2e tests.
4. Serve stack (§6.1–6.2, 6.4), load measurement (S4), `docs/deploy.md`.
5. Cloudflare publish (§6.3), then the Catan policy switch.

## 9. Review log

Codex adversarial review of this spec, 2026-10-03 (commit `7d06aa6`). All 14 findings were checked
against the source and accepted:

| # | Finding | Resolution |
| --- | --- | --- |
| 1 | API excluded the engine's "take only the FIRST marker" move | §3.4, §3.5, §4.4, §5.4 |
| 2 | Bounded bot queue could strand or deadlock turns | DB-driven work set + 30 s sweep, §4.7 |
| 3 | Idempotent retry checked after turn/version checks | Idempotency first, bound to actor + payload, §4.4 |
| 4 | One broadcast view cannot carry per-viewer `you`/`legal` | Version notices + per-subscriber projection, §4.5–4.6 |
| 5 | First deploy circular: compose needs AUD/token before Cloudflare setup | Explicit deploy order, §6.3 |
| 6 | Snapshot lacked `IsFinished` and player-state validity rules | §3.3 |
| 7 | `TicTacToe.cs:191` and `Test.cs` still use DeepCopy | §3.2 |
| 8 | DTO projection and uncapped floor unspecified | Projection rules, `+N` badge, §4.4 |
| 9 | Wall turn order and empty wall turns unspecified | §3.6; forced moves auto-played, §4.3 |
| 10 | `GetColIdxMoves` shuffles with the RNG; full assignments up to 7,776 | Deterministic per-row `WallOptions()`, §3.5 |
| 11 | Origin guard scope was an unasked decision | All paths guarded, A6 and §4.1 (user to confirm) |
| 12 | Seat ownership vs. controller kind | Seat invariants, creator rules, §4.2–4.3 |
| 13 | Deletion contract for live clients | §4.3, §4.6, §5.5 |
| 14 | Tests did not prove the success criteria | §7 rewritten; crash and load progress checks |
