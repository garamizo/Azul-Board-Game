# Azul reports finished games to Playhub

Date: 2026-10-04. Status: design, awaiting review. Repo: `~/Azul-Board-Game`, branch
`feat/hub-results` (worktree `~/Azul-Board-Game-hub-results`).

Azul is the second game on Playhub (`play.signalwave.dev`). The hub's design is
`~/playhub/docs/superpowers/specs/2026-10-03-hub-v1-design.md`, cited below as **hub §n**.
This spec implements the hub's results contract (hub §5) without changing it.

Companion specs:
- Catan's (`~/catanatron`, same file name). It solves the same problem in Python, and this
  spec mirrors its delivery design.
- The playhub CLI (`~/playhub`, `2026-10-04-games-cli-design.md`), which registers Azul and
  writes `AZUL_HUB_KEY`.

Paths are relative to the repo root. `server/` means `server/AzulServer/`.

## 1. Intent

**What the owner asked for.** Every game the owner builds feeds one hub, which holds
accounts, match history and OpenSkill leaderboards that include bots. Azul's web version
(`docs/superpowers/specs/2026-10-03-azul-web-design.md`) listed ratings as out of scope
(spec:43). This spec brings them in:

- finished games are reported to the hub, reliably;
- the MCTS bot gets a key that changes whenever its play can;
- Azul links back to the hub.

**Decisions taken in brainstorming (2026-10-04).**

| Question | Decision |
| --- | --- |
| Which games are reported | **Every finished game with a human.** Games outside the rated rule are sent with `rated: false` and a reason, so they still show in history. Azul cannot start a game without a human (`GameService.Start`), so in practice this means every finished game. |
| Delivery | **An outbox table plus an in-process sender.** The report is written in the same SQLite transaction as the finishing move. A `BackgroundService` sends it, with backoff. |
| Player names | **The email's local part**, which is what Azul shows today (`web/src/lib/names.ts`). The hub keeps an existing account's name (hub §5.3), so this only names new accounts and `name_at_play`. |
| Back link | **Yes.** "Playhub" and "Leaderboard" links in the header. |

**Rated rule** (Azul's own; the hub adds its universal checks, hub §5.2):

- the game ended normally: `finish_reason = "normal"`, meaning a wall row was completed. A
  1000-step stalemate does not count;
- **and no bot played a move for a seat that belongs to a person**, through `ToBot`.

All player counts (2-4) are rated, on one leaderboard. Azul has a single rule set (the
free-placement wall, azul-web spec R5), so the variant carries only the player count.

**Assumptions** (challenge any of them):

- The published stack always runs behind Cloudflare Access, so every human seat has an
  email. Dev mode's `X-Dev-User` emails are reported if a dev hub is configured, and look
  like any other email to the hub.
- Games started before this ships are never reported. They have no `started_at` and no
  captured bot key.
- One brain (`MctsBrain`) plays every bot seat, and its parameters come from the
  environment at startup (`AzulOptions`). A bot's identity is therefore per process, and
  captured per game at start.

**Success.**

- Two invited people and the MCTS bot play a 3-player game on `azul.signalwave.dev`. Within
  a minute of the last wall turn, the match is on `play.signalwave.dev` with rating changes,
  and the bot is on the Azul leaderboard.
- If the hub is down, the report arrives after it returns, untouched by anyone.
- If the server dies a millisecond after the finishing move commits, the report is still in
  the database.

## 2. Architecture

```
GameService.Mutate (per-game lock, one SQLite transaction)
  decide → Commit(next)   [next.Status == finished && g.Status != finished]
     → stamp finished_at; build the report (pure); INSERT hub_reports (pending)
     → tx.Commit()  → AfterCommit: … + hubSender.Wake()

HubSender : BackgroundService (only when AZUL_HUB_URL and AZUL_HUB_KEY are set)
  loop: reconcile → claim due rows → POST <AZUL_HUB_URL>/api/v1/results
        → sent | pending + backoff | failed → wait(wake, 60 s)

docker network "signalwave" (external): app ── http://playhub:3000 ──> playhub
```

New code lives in `server/Hub/`:

| File | Does | Depends on |
| --- | --- | --- |
| `Hub/HubReport.cs` | `HubReport.Build(GameRecord g, Game final, ISet<int> seatsBotPlayedForAPerson, HubOptions o) -> JsonObject`: the body, ranks and rated rule. Pure. | `AzulLibrary`, `Projection` |
| `Hub/BotIdentity.cs` | `BotIdentity.For(AzulOptions) -> (Key, Name)` | `AzulLibrary` assembly |
| `Hub/HubOutbox.cs` | the table's SQL: `Queue(c, tx, gameId, body)`, `Reconcile`, `ClaimDue`, `Mark*` | `Db` |
| `Hub/HubSender.cs` | `BackgroundService`; `SendOne(row)`; maps statuses to outcomes | `HubOutbox`, `HttpClient` |
| `Hub/HubCommands.cs` | `AzulServer hub status|retry|ping` command-line modes | `HubOutbox`, `HubSender` |

## 3. Data model (`Db.Migrations` entry 2, appended)

**`games`** gets four columns:
- `started_at TEXT`
- `finished_at TEXT`
- `hub_tracked INTEGER NOT NULL DEFAULT 0`
- `bot_key TEXT`

`Start` sets `started_at = Now()`, `hub_tracked = 1`, and `bot_key` to the current
`BotIdentity` key, which is stored even when the table has no bot seat, because `ToBot` can
add one later. `Mutate` sets `finished_at` on the commit that finishes the game. Existing
rows keep `hub_tracked = 0`.

**`hub_reports`** is a new table:

```sql
CREATE TABLE hub_reports(
  game_id TEXT PRIMARY KEY,            -- no FK: deleting a game must not delete its report
  body TEXT NOT NULL,                  -- the JSON exactly as sent, frozen at queue time
  status TEXT NOT NULL CHECK(status IN ('pending','sent','failed')),
  attempts INTEGER NOT NULL DEFAULT 0,
  next_attempt_at TEXT NOT NULL,
  last_status INTEGER, last_error TEXT,
  created_at TEXT NOT NULL, sent_at TEXT, alerted_at TEXT);
CREATE INDEX hub_reports_due ON hub_reports(status, next_attempt_at);
```

`GameRecord` and `GameStore` gain the four `games` columns. All timestamps use the existing
`Now()` "O" format, in UTC.

**Deleting a game.** The creator may delete a game, finished or not (`GameService.Delete`).
A queued report is not deleted with it: the match happened and stays on the hub. Its
`replay_url` then answers 404, which is acceptable.

## 4. The report (`Hub/HubReport.cs`)

The report is built inside `Mutate`, under the per-game lock, before the transaction
begins, from the `next` record and its final `Game`. It is serialised with `Json.Options`
(camelCase is irrelevant here, because the keys are written explicitly in snake_case) and
frozen in `hub_reports.body`.

- **`external_id`**: the game id, 10 random base32 characters
  (`GameService.NewId`). With 50 random bits, a collision after a volume reset is
  negligible.
- **`variant`**: `"free-wall-{n}p"`, for example `free-wall-3p`.
- **`rated` / `unrated_reason`**: the first check that fails.
  1. `finish_reason != "normal"` → `"stalemate"`.
  2. A bot moved for a person's seat → `"a bot played for a person"`. This is checked as:
     any `moves` row with `actor = 'bot'` whose seat currently has an email, or had one
     when the move was made. A `ToBot` seat keeps its email (`GameService.ToBot`), and
     `TakeBack` restores the same email, so "the seat has an email now" covers both cases.
     A seat that was a bot from the start never has an email. The query is
     `SELECT DISTINCT seat FROM moves WHERE game_id = ? AND actor = 'bot'`, intersected with
     the seats that have an email. It runs inside `decide`, on the same connection.
- **`started_at` / `finished_at`**: the stored strings, re-rendered as UTC
  `yyyy-MM-ddTHH:mm:ss.fffZ`.
- **`replay_url`**: `"{PublicOrigin}/g/{id}"` (`AZUL_PUBLIC_ORIGIN`). It is null when
  `PublicOrigin` is unset.
- **`players`**: one entry per seat, in index order.
  - **`seat`**: `"P1"`…`"P4"` (index + 1). Azul seats have no colours.
  - **A seat with an email**, whether its kind is `human` or a `bot` handed over by `ToBot`:
    - `kind: "human"`;
    - `email`;
    - `name`: the email's local part, truncated to 40 code points. If the local part is
      empty, the name is `"Player"`.

    A handed-over seat is reported under its owner. Rule 2 has already made such a match
    unrated.
  - **A bot seat without an email**: `kind: "bot"`, `bot_key` from `games.bot_key`, and
    `name: "Azul MCTS"`.
  - **`score`**: the final `player.score`, after end-game bonuses (`Player.UpdateGame`).
  - **`rank`**: by score, highest first. Ties are broken by the official rule: the player
    with more complete horizontal wall rows ranks higher. Players still tied share a rank.
    The ranking is competition-style: 1, 2, 2, 4.
    - This is the hub's ranking only. Azul's own result view (`Projection.Result`) keeps
      listing every top scorer as a winner; changing that is out of scope.
- **Guards.** `Queue` is skipped with a logged warning when either holds:
  - `hub_tracked = 0`;
  - a bot seat without an email exists and `games.bot_key` is null.

## 5. Bot identity (`Hub/BotIdentity.cs`)

Hub §5.1 says a bot's key must change whenever its effective behaviour can change.

- **The key** is `mcts@<first 12 hex of sha-256>`, taken over canonical JSON.
- **What the JSON covers:**
  - `brain`: `"MctsBrain"`.
  - `thinkSeconds`: `AzulOptions.BotThinkSeconds`.
  - `rolloutCap`: 300000, the constant `MctsBrain` stops at. It becomes a named constant
    shared with `BotScheduler.cs`.
  - `exploration`: the constant `Ai.cs` uses, sqrt(2), written as a string.
  - `brainRevision`: an integer constant next to `MctsBrain`. It is bumped whenever the
    brain's wrapper logic changes, such as the greedy fallback. A comment on the constant
    says so.
  - `engine`: the `ModuleVersionId` (MVID) of the `AzulLibrary` assembly.
    - With deterministic builds, which are the SDK default, the MVID is a hash of the
      compiled assembly. Any change to the engine's rules or AI changes it, and nothing
      else does.
    - The Docker build adds `-p:ContinuousIntegrationBuild=true`, which normalises source
      paths. The same source then gives the same MVID on any machine, and a rebuild alone
      does not change the key. The implementation plan verifies this by building the image
      twice and comparing keys.
- **What is deliberately not covered.**
  - `AZUL_BOT_WORKERS` and machine speed. They change how many rollouts fit in the think
    time, but they are a property of the host, not of the bot. This is accepted, and noted
    in `docs/deploy.md`.
  - `AZUL_MIN_MOVE_DELAY_SECONDS`, which is pacing only.
- **When it is computed.** Once at startup, then stamped into `games.bot_key` by `Start`.
  A redeploy mid-game keeps the key the game started with, which is the capture rule of
  hub §5.1.
- **`GreedyBrain`** is used only by tests, and is never reported.
- **Retiring old keys.** After a deploy that changes the key, retire the old bot from
  `~/playhub`: `npx tsx scripts/hub.ts --serve bots list --game azul`, then
  `bots retire azul <old key>`.

## 6. Delivery

### 6.1 Queueing, atomic with the finish

**Building the report.** In `Mutate`, when `commit.Next.Status == Finished` and
`g.Status != Finished`, the report is built in a `try`.

- If building throws, the error is logged and the move commits without a report.
  Reconcile picks the game up later, as below.
- The build happens before `BeginTransaction`, so a throw cannot leave a transaction half
  done.

**Writing it.** Inside the transaction, three statements run together:

1. `GameStore.Update`, with `finished_at` set;
2. `GameStore.InsertMove`;
3. `HubOutbox.Queue`: an `INSERT OR IGNORE` of a `pending` row with `next_attempt_at = now`.

`AfterCommit` then calls `hubSender.Wake()` (a no-op when the sender is off).

**Paths that can finish a game.** Every one goes through `Mutate`:

- a human move (`Move` → `Apply`);
- a bot or forced move (`ApplyServerMove`).

### 6.2 Reconcile

At the start of every sender cycle:

```sql
SELECT id FROM games
WHERE status = 'finished' AND hub_tracked = 1
  AND id NOT IN (SELECT game_id FROM hub_reports)
```

For each such game, the sender loads it under its per-game lock (`GameService` exposes
`QueueMissingReport(id)`), then builds and queues the report.

- The only way a game reaches this state is a failed build in 6.1. Reconcile retries it
  once per cycle, and it keeps failing until the bug is fixed.
- Each failure logs an error.

### 6.3 Sending

**When the sender runs.** `HubSender` is registered with `AddHostedService` in `Program.cs`.
It does nothing unless both `AZUL_HUB_URL` and `AZUL_HUB_KEY` are set. If only one is set,
it logs a warning once.

**Configuration.** A new `HubOptions` record holds `Url`, `Key` and `PublicUrl`, parsed in
`AzulOptions.FromEnvironment`'s style.

**One cycle:**

1. reconcile;
2. claim due rows one at a time: `UPDATE hub_reports SET next_attempt_at = now + 5 min
   WHERE game_id = ? AND status = 'pending' AND next_attempt_at <= now`, which takes a
   lease;
3. send;
4. wait on a `SemaphoreSlim` or channel wake, with a 60-second timeout.

**The request.**

- A single `HttpClient` with a 10-second timeout posts the stored body to
  `AZUL_HUB_URL.TrimEnd('/') + "/api/v1/results"`.
- The body is sent as `StringContent(body, UTF8, "application/json")`, with
  `Authorization: Bearer <AZUL_HUB_KEY>`.

**Outcomes** (hub §5.3), the same table as Catan's:

| Answer | Row becomes | Log |
| --- | --- | --- |
| 2xx | `sent` | information |
| 409, 422 | `failed` (permanent), body ≤ 500 chars in `last_error` | **error** |
| 401 | `pending` + backoff | **error**: the key is wrong or rotated; set `AZUL_HUB_KEY` and restart |
| 5xx, network error, timeout | `pending` + backoff | warning |
| any other status | `pending` + backoff | **error** (a configuration fault: URL, content type) |

**Backoff and alerts.**

- The retry delay is `min(60 s × 2^(attempts-1), 1 h)`, plus up to 10% jitter.
- A row pending for over an hour logs one error, then at most one an hour (`alerted_at`).

**Shutdown.** `ExecuteAsync` honours the stopping token. An interrupted send leaves its
lease to expire, and the row is retried.

### 6.4 Operator commands

These follow the `--healthcheck` precedent in `Program.cs`. Each runs the binary in a mode
that opens the SQLite database directly (WAL allows it while the server runs), prints one
JSON document and exits.

- **`/app/AzulServer hub status`** prints counts by status, the 20 newest failed rows, and
  the age of the oldest pending row.
- **`/app/AzulServer hub retry <game_id>|--all-failed [--rebuild]`** puts rows back to
  `pending`. With `--rebuild`, it deletes them instead, and reconcile rebuilds the bodies.
- **`/app/AzulServer hub ping`** posts `{}` to the configured hub. It exits 0 on 422, which
  means the hub was reached, the key accepted and the content type passed. Any other answer
  exits non-zero, with the status. The aspnet image has no curl, so `serve-check` uses this.

**Make targets:** `hub-status` and `hub-retry GAME=<id>`. Both run
`docker compose ... exec app /app/AzulServer hub ...` against the serve stack.

## 7. Configuration and deployment

| Variable | Default | Meaning |
| --- | --- | --- |
| `AZUL_HUB_URL` | empty (off) | The serve compose sets `http://playhub:3000`. |
| `AZUL_HUB_KEY` | empty (off) | The game key. Server-only; the web bundle never sees it. Vite in `web/` reads only `VITE_`-prefixed variables. |
| `AZUL_HUB_PUBLIC_URL` | empty | The hub's public origin, for the header links. The serve compose sets `https://play.signalwave.dev`. |

**`docker-compose.serve.yml`:**
- `app` joins the external network `signalwave`, which the playhub serve stack declares
  (hub §8), in addition to its default network.
- `app`'s environment gets:
  - `AZUL_HUB_URL: http://playhub:3000`
  - `AZUL_HUB_KEY: ${AZUL_HUB_KEY:-}`
  - `AZUL_HUB_PUBLIC_URL: https://play.signalwave.dev`

**`Makefile` `serve`** creates the `signalwave` network if it is missing, as playhub's does.

**`.env.serve.example`** documents `AZUL_HUB_KEY`, which the playhub CLI writes:
`npx tsx scripts/hub.ts --serve games issue-key azul --key-out ../Azul-Board-Game/.env.serve --env-var AZUL_HUB_KEY`.

**`scripts/serve-check.sh`.** Step 3, now "no machine-to-machine endpoints", becomes
`docker compose ... exec app /app/AzulServer hub ping`. It is skipped with a notice when
`AZUL_HUB_KEY` is empty.

**`Dockerfile`:** `-p:ContinuousIntegrationBuild=true` on the publish step (section 5).

**`docs/deploy.md`** gets a "Playhub" section covering:
- registering the game and writing the key;
- `make serve`;
- `make hub-status`;
- retiring superseded bot keys.

## 8. Links to the hub

- **`GET /api/me`** returns `{ email, hubUrl }`, where `hubUrl` is `AZUL_HUB_PUBLIC_URL` or
  null.
- **The header** (`web/src/App.svelte`) shows "Playhub" (→ `hubUrl`) and "Leaderboard"
  (→ `${hubUrl}/games/azul`) beside the email and Sign out, when `hubUrl` is set.
- **On a phone-width screen** the two links stay on the header line: the email already
  truncates (`.who.truncate`), and the links do not.

## 9. Failure handling summary

| Failure | Effect |
| --- | --- |
| Building the report throws | The move commits anyway and an error is logged. Reconcile retries every cycle and logs each failure. |
| The process dies after commit | The report is already in `hub_reports`, written in the same transaction. |
| Hub down, network missing, or 5xx | Pending, with backoff. A stuck alert after an hour. |
| 401 | Pending, with an error log on each attempt. |
| 409/422 | `failed`. `hub status` shows it, and `hub retry --rebuild` rebuilds the body after a fix. |
| The game is deleted before its report is sent | The report is still sent, and its replay link answers 404. |

## 10. Testing

`server/AzulServer.Tests` uses xUnit and `WebApplicationFactory`, and `web/` uses Vitest.

- **`HubReportTests`** (pure, on constructed `GameRecord`s and finished `Game`s):
  - the variant;
  - stalemate is unrated;
  - a bot move on a `ToBot` seat is unrated, including after `TakeBack`;
  - ranks: tie broken by complete rows, a shared rank after that, competition gaps;
  - a handed-over seat is reported as human;
  - names: the local part, truncated to 40, falling back to "Player";
  - timestamps end in `Z` with milliseconds;
  - the body passes the hub's limits;
  - `replay_url` is null when `PublicOrigin` is unset.
- **`BotIdentityTests`:**
  - the key is stable for the same options;
  - changing the think seconds, the rollout cap or the brain revision changes it;
  - the key matches the hub's `bot_key` regex.
- **`HubOutboxTests`** (through the API with `GreedyBrain`, as existing tests do):
  - a game finished by a human move queues one row;
  - a game finished by a bot move queues one row;
  - the row commits atomically with the move (an injected fault after `tx.Commit` leaves
    the row present);
  - a build exception leaves the move committed and the row missing, and reconcile then
    queues it;
  - a pre-existing game (`hub_tracked = 0`) queues nothing;
  - `Delete` leaves the report in place.
- **`HubSenderTests`** (fake `HttpMessageHandler`):
  - every outcome row in 6.3;
  - backoff, and the lease;
  - the stored body is sent byte-for-byte, with `Content-Type: application/json` and the
    bearer header;
  - the sender stays off without configuration;
  - the stuck alert is logged once.
- **`HubCommandsTests`:** `status`, `retry` and `ping` (the last against the fake
  handler).
- **`MigrationTests`:** a database at schema 1 upgrades, and existing games get
  `hub_tracked = 0`.
- **Web (`App.test.ts` or the header's test):** the links show only with `hubUrl`.
- **Manual end-to-end, recorded in the plan:**
  1. Run the playhub dev stack (`make up` in `~/playhub`).
  2. Issue a dev key into Azul's dev environment with the playhub CLI.
  3. Finish a 2-player game against the bot through `make dev-server`.
  4. See it rated on the hub, with `mcts@...` on the leaderboard.

## 11. Out of scope

- Display names in Azul.
- Changing Azul's own winner display, which lists every tied top scorer.
- Standard fixed-wall rules, which would be a new variant.
- Reporting abandoned games.
- Showing ratings inside Azul.
