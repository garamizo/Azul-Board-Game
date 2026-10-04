# Azul Reports Finished Games to Playhub: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every finished Azul game is reported to Playhub (`POST /api/v1/results`) reliably. The MCTS bot carries a fingerprinted key. The header links back to the hub.

**Architecture:**
- **Reporting.** The report body is built (pure function `HubReport.Build`) inside `GameService.Mutate`. It is written to a new `hub_reports` outbox table in the same SQLite transaction as the finishing move.
- **Sending.** A `BackgroundService` (`HubSender`) reconciles missing reports, claims due rows with a lease, and POSTs them with backoff, following hub §5.3 per status.
- **Operations.** Operator commands run as `/app/AzulServer hub ...`.
- **Links.** `/api/me` gains `hubUrl` for the header links.

**Tech Stack:** .NET 10 (ASP.NET Core minimal APIs, Microsoft.Data.Sqlite, System.Text.Json.Nodes); xUnit + WebApplicationFactory + `Microsoft.Extensions.TimeProvider.Testing`; Svelte 5 + Vitest + @testing-library/svelte; Docker Compose.

**Spec:** `docs/superpowers/specs/2026-10-04-hub-results-design.md` (cited as **spec §n**). The hub contract it implements: `~/playhub/docs/superpowers/specs/2026-10-03-hub-v1-design.md` section 5 (cited as **hub §5.x**). The hub's validator is `~/playhub/src/lib/server/ingest/schema.ts`.

**Depends on:** the playhub CLI branch (worktree `~/playhub-games-cli`, branch `feat/games-cli`, implemented first). Its `results validate` command is the required contract check in Task 11. Its `games put` and `games issue-key` commands provide the dev key for the end-to-end run.

## Global Constraints

**Repo and tooling**
- Work only in the worktree `~/Azul-Board-Game-hub-results` (branch `feat/hub-results`). Never touch `~/Azul-Board-Game` (main).
- Every .NET command runs through `make` in the SDK container (no SDK on the host):
  - `make test`, or `make test FILTER="FullyQualifiedName~<name>"`;
  - `make build`.
- Web commands: `cd web && npm ci` once, then `make web-test` (svelte-check + vitest).
- `make e2e-server-start` binds the fixed port **127.0.0.1:5081**. Two Azul worktrees running e2e at once collide. Check `docker ps | grep azul-e2e` first.

**Commits**
- Every commit message is the subject given in the step, then a blank line and exactly:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01C7vgx4t8sAGYwtCwUhtDmP
  ```

**Configuration**
- New environment variables, all optional, empty meaning off: `AZUL_HUB_URL`, `AZUL_HUB_KEY`, `AZUL_HUB_PUBLIC_URL`.
  - The sender runs only when both `AZUL_HUB_URL` and `AZUL_HUB_KEY` are set.
  - Exactly one of them set → one warning at startup.

**Report body (hub §5.1)**
- Keys are written explicitly in snake_case: `external_id`, `variant`, `rated`, `unrated_reason`, `started_at`, `finished_at`, `replay_url`, `players[{seat, kind, email | bot_key, name, rank, score}]`.
- `external_id` = the game id.
- `variant` = `"free-wall-{n}p"`.
- `seat` = `"P1"`…`"P4"`.
- Bot `name` = `"Azul MCTS"`.
- `bot_key` = `mcts@<first 12 lowercase hex of sha-256>`, matching `^[a-z0-9_.:@-]{1,64}$`.
- Timestamps: UTC `yyyy-MM-ddTHH:mm:ss.fffZ`, truncated (not rounded) to milliseconds.
- Unrated reasons, verbatim: `"stalemate"` (when `finish_reason != "normal"`), then `"a bot played for a person"`.
- Ranks: score descending, then more complete horizontal wall rows. Players still tied share a rank (competition ranking 1, 2, 2, 4).
- Email: trimmed with JavaScript's whitespace set and lowercased. Kept only if it matches Zod v4's `z.email()` pattern, copied verbatim (anchors adapted to .NET `\z`):
  `^(?:[A-Za-z0-9_'+\-]+\.)*[A-Za-z0-9_'+\-]*[A-Za-z0-9_+-]@(?:[A-Za-z0-9][A-Za-z0-9\-]*\.)+[A-Za-z]{2,}$`.
  Anything else becomes `null`.
- Human `name`: the email's local part, NUL removed, JS-trimmed, truncated to 40 code points, then trimmed again. `"Player"` if empty.

**Sending (hub §5.3)**
- `POST {AZUL_HUB_URL}/api/v1/results` with `Content-Type: application/json` and `Authorization: Bearer {AZUL_HUB_KEY}`. The body is the stored text, byte for byte. Timeout 10 s.
- Outcomes:
  - 2xx → `sent`;
  - 409/422 → `failed` (permanent);
  - 401 → pending, with an error log;
  - 5xx, network error or timeout → pending, with a warning;
  - any other status → pending, with an error log.
- Backoff `min(60 s × 2^(attempts−1), 1 h)` × (1 + up to 10% jitter).
- Lease: 5 min.
- Idle wait: 60 s.
- Stuck alert: one error when a row has been pending more than 1 h, then at most one per hour.

**Storage**
- The schema is append-only: the new entry is `Db.Migrations[1]`, and `schema_version` becomes 2.
- `hub_reports` has no foreign key: deleting a game never deletes its report.
- All stored timestamps use `GameService`'s existing `"O"` UTC strings. They compare lexicographically.

## Review Focus

These are the five conditions the spec implies that the per-feature tests would not catch, most likely first. Each has a test in the task named in brackets.

1. **A server restart with a backlog.** Rows already pending when the app starts must be sent on the first cycle, without waiting for a new finish. [Task 7: `PendingRowsAreSentAfterAStart`]
2. **The hub answers 2xx with an empty or non-JSON body**, for example through a proxy. The row must still become `sent`; only the status decides the outcome. [Task 7: `AnyTwoHundredIsSentWhateverTheBody`]
3. **Sub-millisecond timestamps.** `"O"` strings carry 7 fractional digits. They must be truncated, not rounded: rounding `18:41:52.9999999` would move the instant into the next second, and could put `started_at` after `finished_at`. [Task 4: `TimestampsAreTruncatedToMilliseconds`]
4. **A game deleted after its report is queued but before it is sent.** The report must still be sent. Only the replay link 404s. [Task 7: `ReportOfADeletedGameIsStillSent`]
5. **A bot hand-over on the very last turn.** The finishing move is not yet in `moves` when the report is built, so it must be unioned in. A person whose seat the bot finished for must make the match unrated. [Task 4: `BotPlayedSeatsIncludesThePendingMove`; Task 6: `ABotFinishingForAPersonMakesItUnrated`, from a deterministic penultimate state with no stored bot move, so it fails if `commit.Move` is left out]

---

## File Structure

| File | Status | Responsibility |
| --- | --- | --- |
| `server/AzulServer/Hub/HubOptions.cs` | create | `AZUL_HUB_*` settings record |
| `server/AzulServer/Hub/HubContract.cs` | create | email and name normalisation to the hub's Zod rules |
| `server/AzulServer/Hub/BotIdentity.cs` | create | bot key fingerprint |
| `server/AzulServer/Hub/HubReport.cs` | create | pure body builder + `BotPlayedSeats` query |
| `server/AzulServer/Hub/HubOutbox.cs` | create | all `hub_reports` SQL, reconcile, rebuild |
| `server/AzulServer/Hub/HubSignal.cs` | create | wake-up signal shared by `GameService` and `HubSender` |
| `server/AzulServer/Hub/HubSender.cs` | create | `BackgroundService`: claim, send, classify, backoff, alerts |
| `server/AzulServer/Hub/HubCommands.cs` | create | `AzulServer hub status\|retry\|ping\|bot-key` |
| `server/AzulServer/AzulOptions.cs` | modify | `Hub` property, parsed from the environment |
| `server/AzulServer/Data/Db.cs` | modify | migration 2; `Migrations` made `internal` |
| `server/AzulServer/Data/GameStore.cs` | modify | `GameRecord` gains `StartedAt`, `FinishedAt`, `HubTracked`, `BotKey`; read/write them |
| `server/AzulServer/Hub/HubReportBuilder.cs` | create | `IHubReportBuilder` seam over `HubReport.Build` (tests inject a throwing builder) |
| `server/AzulServer/Games/GameService.cs` | modify | stamp at `Start`; build via `IHubReportBuilder` and queue in `Mutate`'s transaction; `Delete` guard |
| `server/AzulServer/Games/BotScheduler.cs` | modify | `MctsBrain.RolloutCap`, `MctsBrain.BrainRevision` |
| `server/AzulServer/Program.cs` | modify | `hub` command dispatch; DI for `BotIdentity`, `HubSignal`, `IHubReportBuilder`, `HubSender`, the `hub` HttpClient; log the bot key |
| `server/AzulServer/Api/ApiEndpoints.cs` | modify | `/api/me` returns `hubUrl` |
| `server/AzulServer.Tests/Hub/*.cs` | create | `HubTestKit`, `HubPlay` (deterministic penultimate states), `HubContractTests`, `BotIdentityTests`, `HubSchemaTests`, `HubReportTests`, `HubOutboxTests`, `HubFlowTests`, `HubSenderTests`, `HubCommandsTests`, `golden/*.json` |
| `server/AzulServer.Tests/Play.cs` | modify | `Play.ToEnd` helper |
| `server/AzulServer.Tests/StoreTests.cs` | modify | schema version 2; new columns round-trip |
| `web/src/lib/api.ts`, `web/src/App.svelte`, `web/src/App.test.ts` | modify/create | `hubUrl` and the header links |
| `web/src/components/Lobby.test.ts`, `web/src/lib/events.test.ts` | modify | mocks gain `hubUrl: null` |
| `docker-compose.serve.yml`, `Makefile`, `Dockerfile`, `scripts/serve-check.sh`, `.env.serve.example`, `docs/deploy.md` | modify | network, env, targets, image pinning, probe, docs |

**Where the implementation differs from the spec's file table**
- Reconcile and `--rebuild` do not take `GameService`'s in-process per-game lock. They run in the separate `hub` command process, or on the sender thread, inside a SQLite `BEGIN IMMEDIATE` transaction that reads the game and writes the report. A finished game only ever changes by `Delete`, and that transaction is atomic against `Delete`, which also writes. So this is equivalent to the lock, and it works from another process.
- `HubReport.Build` takes the `GameRecord`, whose `StateJson` holds the final snapshot, instead of a `Game`. It needs only scores and grids, so it never runs `Game.FromSnapshot` validation.

---

### Task 0: Worktree setup and baseline

**Files:** none.

- [ ] **Step 1: Restore and build**

Run: `cd ~/Azul-Board-Game-hub-results && make build`
Expected: `Build succeeded.`

- [ ] **Step 2: Install the web dependencies**

Run: `cd ~/Azul-Board-Game-hub-results/web && npm ci`
Expected: exit 0.

- [ ] **Step 3: Baseline tests**

Run: `cd ~/Azul-Board-Game-hub-results && make test && make web-test`
Expected: all pass. Write down the counts. If anything fails on the untouched branch, stop and report it.

---

### Task 1: Hub options and contract normalisation

**Files:**
- Create: `server/AzulServer/Hub/HubOptions.cs`
- Create: `server/AzulServer/Hub/HubContract.cs`
- Modify: `server/AzulServer/AzulOptions.cs`
- Test: `server/AzulServer.Tests/Hub/HubContractTests.cs`

**Interfaces:**
- Produces:
  - `AzulServer.Hub.HubOptions { string? Url; string? Key; string? PublicUrl; bool SenderConfigured; bool HalfConfigured }`
  - `AzulOptions.Hub` (type `HubOptions`, default `new()`)
  - `HubContract.Email(string?) -> string?`
  - `HubContract.Name(string?, string fallback) -> string`
  - `HubContract.JsTrim(string) -> string`

- [ ] **Step 1: Write the failing tests**

`server/AzulServer.Tests/Hub/HubContractTests.cs`:

```csharp
using AzulServer.Hub;

namespace AzulServer.Tests;

public class HubContractTests
{
    [Fact]
    public void HubSettingsComeFromTheEnvironment()
    {
        var env = new Dictionary<string, string>
        {
            ["AZUL_HUB_URL"] = "http://playhub:3000/",
            ["AZUL_HUB_KEY"] = "phk_x",
            ["AZUL_HUB_PUBLIC_URL"] = "https://play.example/",
        };
        var o = AzulOptions.FromEnvironment(k => env.GetValueOrDefault(k));
        Assert.Equal("http://playhub:3000", o.Hub.Url);
        Assert.Equal("phk_x", o.Hub.Key);
        Assert.Equal("https://play.example", o.Hub.PublicUrl);
        Assert.True(o.Hub.SenderConfigured);
        Assert.False(o.Hub.HalfConfigured);
    }

    [Fact]
    public void OneHalfOfTheSenderConfigIsReported()
    {
        var o = AzulOptions.FromEnvironment(k => k == "AZUL_HUB_URL" ? "http://h" : null);
        Assert.False(o.Hub.SenderConfigured);
        Assert.True(o.Hub.HalfConfigured);
        Assert.False(AzulOptions.FromEnvironment(_ => null).Hub.HalfConfigured);
    }

    [Theory]
    [InlineData("Ann@Example.com", "ann@example.com")]
    [InlineData(" a.b+c@sub.example.co ", "a.b+c@sub.example.co")]
    [InlineData("dev@localhost", null)]
    [InlineData("a@b.co\nx", null)]
    [InlineData("no-at-sign", null)]
    [InlineData(null, null)]
    public void EmailsTheHubWouldRejectBecomeNull(string? raw, string? expected) =>
        Assert.Equal(expected, HubContract.Email(raw));

    [Theory]
    [InlineData("bob", "bob")]
    [InlineData("  bob  ", "bob")]
    [InlineData("a\0b", "ab")]
    [InlineData("﻿", "Player")]
    [InlineData("", "Player")]
    [InlineData(null, "Player")]
    public void NamesFollowTheHubRules(string? raw, string expected) =>
        Assert.Equal(expected, HubContract.Name(raw, "Player"));

    [Fact]
    public void NamesAreCutAtFortyCodePoints()
    {
        Assert.Equal(new string('a', 40), HubContract.Name(new string('a', 45), "Player"));
        var emoji = new string('a', 39) + "😀😀";  // 41 code points, 43 UTF-16 units
        Assert.Equal(new string('a', 39) + "😀", HubContract.Name(emoji, "Player"));
    }

    [Fact]
    public void JsTrimRemovesWhatJavaScriptTrims() =>
        Assert.Equal("x", HubContract.JsTrim("﻿   x\t　"));
}
```

- [ ] **Step 2: Run them and check they fail**

Run: `make test FILTER="FullyQualifiedName~HubContractTests"`
Expected: a build failure, because `AzulServer.Hub` and `AzulOptions.Hub` do not exist.

- [ ] **Step 3: Implement**

`server/AzulServer/Hub/HubOptions.cs`:

```csharp
namespace AzulServer.Hub;

/// AZUL_HUB_URL / AZUL_HUB_KEY turn the sender on (both or it stays off);
/// AZUL_HUB_PUBLIC_URL is the hub's public origin for the header links.
public sealed record HubOptions
{
    public string? Url { get; init; }
    public string? Key { get; init; }
    public string? PublicUrl { get; init; }

    public bool SenderConfigured => Url is not null && Key is not null;
    public bool HalfConfigured => (Url is null) != (Key is null);
}
```

`server/AzulServer/Hub/HubContract.cs`:

```csharp
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace AzulServer.Hub;

/// The hub validates reports with Zod and JavaScript string semantics
/// (~/playhub/src/lib/server/ingest/schema.ts); these make a value pass there
/// instead of turning the whole report into a permanent 422.
public static partial class HubContract
{
    // Zod v4 z.email(): zod/v4/core/regexes.js `email`, with `$` written as
    // \z because .NET's `$` also matches before a final newline.
    [GeneratedRegex(@"^(?:[A-Za-z0-9_'+\-]+\.)*[A-Za-z0-9_'+\-]*[A-Za-z0-9_+-]@(?:[A-Za-z0-9][A-Za-z0-9\-]*\.)+[A-Za-z]{2,}\z")]
    private static partial Regex ZodEmail();

    /// The address the hub would store, or null for one it would refuse (the
    /// hub then records "unidentified player" instead of rejecting the report).
    public static string? Email(string? email)
    {
        if (email is null) return null;
        var e = JsTrim(email).ToLowerInvariant();  // hub: z.string().trim().toLowerCase()
        return ZodEmail().IsMatch(e) ? e : null;
    }

    /// 1-40 code points after a JavaScript trim, no NUL; `fallback` when empty.
    public static string Name(string? raw, string fallback)
    {
        var s = JsTrim((raw ?? "").Replace("\0", ""));
        var cut = new StringBuilder();
        foreach (var rune in s.EnumerateRunes().Take(40)) cut.Append(rune.ToString());
        s = JsTrim(cut.ToString());
        return s.Length == 0 ? fallback : s;
    }

    /// String.prototype.trim: WhiteSpace (incl. U+FEFF and every Zs) and
    /// LineTerminator; unlike .NET's Trim, not U+0085.
    public static string JsTrim(string s)
    {
        int a = 0, b = s.Length;
        while (a < b && IsJsSpace(s[a])) a++;
        while (b > a && IsJsSpace(s[b - 1])) b--;
        return s[a..b];
    }

    static bool IsJsSpace(char c) =>
        c is '\t' or '\n' or '\v' or '\f' or '\r' or ' ' or ' ' or '﻿' or ' ' or ' '
        || CharUnicodeInfo.GetUnicodeCategory(c) == UnicodeCategory.SpaceSeparator;
}
```

In `server/AzulServer/AzulOptions.cs`, add `using AzulServer.Hub;` at the top. Add the property after `SseMaxMinutes`:

```csharp
    public HubOptions Hub { get; init; } = new();
```

Then add this entry to the object initializer returned by `FromEnvironment`, after `SseMaxMinutes = ...,`:

```csharp
            Hub = new HubOptions
            {
                Url = Get("AZUL_HUB_URL")?.TrimEnd('/'),
                Key = Get("AZUL_HUB_KEY"),
                PublicUrl = Get("AZUL_HUB_PUBLIC_URL")?.TrimEnd('/'),
            },
```

- [ ] **Step 4: Run the tests and check they pass**

Run: `make test FILTER="FullyQualifiedName~HubContractTests"`
Expected: PASS (all 18 cases).

- [ ] **Step 5: Commit**

```bash
git add server/AzulServer/Hub server/AzulServer/AzulOptions.cs server/AzulServer.Tests/Hub/HubContractTests.cs
git commit -m "server: hub settings and the hub's email/name rules"   # + trailer block (Global Constraints)
```

---

### Task 2: Bot identity

**Files:**
- Create: `server/AzulServer/Hub/BotIdentity.cs`
- Modify: `server/AzulServer/Games/BotScheduler.cs` (`MctsBrain`)
- Modify: `server/AzulServer/Program.cs`
- Test: `server/AzulServer.Tests/Hub/BotIdentityTests.cs`

**Interfaces:**
- Consumes: `AzulOptions.BotThinkSeconds`.
- Produces:
  - `MctsBrain.RolloutCap` (`const int` = 300000)
  - `MctsBrain.BrainRevision` (`const int` = 1)
  - `BotIdentity(string Key, string Name)` record
  - `BotIdentity.DisplayName` = `"Azul MCTS"`
  - `BotIdentity.For(AzulOptions)`
  - `BotIdentity.For(double thinkSeconds, int rolloutCap, int brainRevision, Guid engineMvid)`
  - a DI singleton `BotIdentity`

- [ ] **Step 1: Write the failing tests**

`server/AzulServer.Tests/Hub/BotIdentityTests.cs`:

```csharp
using System.Text.RegularExpressions;
using Azul;
using AzulServer.Games;
using AzulServer.Hub;

namespace AzulServer.Tests;

public class BotIdentityTests
{
    static readonly Guid Engine = Guid.Parse("0f8fad5b-d9cb-469f-a165-70867728950e");

    [Fact]
    public void SameInputsGiveTheSameKey() =>
        Assert.Equal(BotIdentity.For(3, 300_000, 1, Engine), BotIdentity.For(3, 300_000, 1, Engine));

    public static TheoryData<BotIdentity> Changed => new()
    {
        BotIdentity.For(2.5, 300_000, 1, Engine),
        BotIdentity.For(3, 1_000, 1, Engine),
        BotIdentity.For(3, 300_000, 2, Engine),
        BotIdentity.For(3, 300_000, 1, Guid.Parse("7c9e6679-7425-40de-944b-e07fc1f90ae7")),
    };

    [Theory]
    [MemberData(nameof(Changed))]
    public void AnyInputChangesTheKey(BotIdentity other) =>
        Assert.NotEqual(BotIdentity.For(3, 300_000, 1, Engine).Key, other.Key);

    [Fact]
    public void KeyMatchesTheHubPattern()
    {
        var id = BotIdentity.For(3, 300_000, 1, Engine);
        Assert.Matches(new Regex("^mcts@[0-9a-f]{12}$"), id.Key);
        Assert.Matches(new Regex("^[a-z0-9_.:@-]{1,64}$"), id.Key);
        Assert.Equal("Azul MCTS", id.Name);
    }

    [Fact]
    public void OptionsUseTheBrainConstantsAndTheEngineAssembly() =>
        Assert.Equal(
            BotIdentity.For(3, MctsBrain.RolloutCap, MctsBrain.BrainRevision, typeof(Game).Assembly.ManifestModule.ModuleVersionId),
            BotIdentity.For(new AzulOptions { BotThinkSeconds = 3 }));

    [Fact]
    public void TheAppRegistersItsKey()
    {
        using var app = new TestApp(new AzulOptions { BotWorkers = 0, BotThinkSeconds = 1.5 });
        Assert.Equal(BotIdentity.For(app.Options), app.Service<BotIdentity>());
    }
}
```

- [ ] **Step 2: Run them and check they fail**

Run: `make test FILTER="FullyQualifiedName~BotIdentityTests"`
Expected: a build failure, because `BotIdentity` and `MctsBrain.RolloutCap` do not exist.

- [ ] **Step 3: Implement**

In `server/AzulServer/Games/BotScheduler.cs`, replace the start of `MctsBrain` and the loop condition:

```csharp
public sealed class MctsBrain(AzulOptions options) : IBotBrain
{
    /// The search stops here even with think time left.
    public const int RolloutCap = 300_000;
    /// Part of the bot's hub key (Hub/BotIdentity.cs). Bump it whenever this
    /// class's logic changes how a move is chosen (the greedy fallback, the
    /// stopping rule, the final pick), so the hub rates the changed bot as a
    /// new account.
    public const int BrainRevision = 1;

    public Move ChooseMove(Game game, CancellationToken ct)
    {
        var root = new MCTS_Stochastic<Game, Move>(game, 0.0f);  // clones the game
        var clock = Stopwatch.StartNew();
        while (!ct.IsCancellationRequested
               && clock.Elapsed.TotalSeconds < options.BotThinkSeconds
               && root.numRolls < RolloutCap)
            root.Grow();
```

The rest of the method is unchanged.

`server/AzulServer/Hub/BotIdentity.cs`:

```csharp
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Ai;
using Azul;
using AzulServer.Games;

namespace AzulServer.Hub;

/// The bot's hub account key (hub spec 5.1): it changes whenever the bot's
/// play can. `engine` is the AzulLibrary MVID: conservative (any change to the
/// compiled engine, even unrelated code, gives a new key; spec section 5).
public sealed record BotIdentity(string Key, string Name)
{
    public const string DisplayName = "Azul MCTS";

    public static BotIdentity For(AzulOptions o) =>
        For(o.BotThinkSeconds, MctsBrain.RolloutCap, MctsBrain.BrainRevision,
            typeof(Game).Assembly.ManifestModule.ModuleVersionId);

    public static BotIdentity For(double thinkSeconds, int rolloutCap, int brainRevision, Guid engineMvid)
    {
        var inputs = new SortedDictionary<string, string>(StringComparer.Ordinal)
        {
            ["brain"] = nameof(MctsBrain),
            ["brainRevision"] = brainRevision.ToString(CultureInfo.InvariantCulture),
            ["engine"] = engineMvid.ToString("N"),
            ["exploration"] = MCTS_Stochastic<Game, Move>.c.ToString("R", CultureInfo.InvariantCulture),
            ["rolloutCap"] = rolloutCap.ToString(CultureInfo.InvariantCulture),
            ["thinkSeconds"] = thinkSeconds.ToString("R", CultureInfo.InvariantCulture),
        };
        var hash = SHA256.HashData(Encoding.UTF8.GetBytes(JsonSerializer.Serialize(inputs)));
        return new BotIdentity($"mcts@{Convert.ToHexStringLower(hash)[..12]}", DisplayName);
    }
}
```

In `server/AzulServer/Program.cs`:
- add `using AzulServer.Hub;`;
- after `builder.Services.AddSingleton<IBotBrain, MctsBrain>();`, add
  ```csharp
  builder.Services.AddSingleton(sp => BotIdentity.For(sp.GetRequiredService<AzulOptions>()));
  ```
- after `app.Services.GetRequiredService<Db>().Migrate();`, add
  ```csharp
  app.Logger.LogInformation("bot key {Key}", app.Services.GetRequiredService<BotIdentity>().Key);
  ```

- [ ] **Step 4: Run the tests and check they pass**

Run: `make test FILTER="FullyQualifiedName~BotIdentityTests|FullyQualifiedName~BotTests"`
Expected: PASS. `BotTests` is included to prove `MctsBrain` still plays.

- [ ] **Step 5: Commit**

```bash
git add server/AzulServer/Hub/BotIdentity.cs server/AzulServer/Games/BotScheduler.cs server/AzulServer/Program.cs server/AzulServer.Tests/Hub/BotIdentityTests.cs
git commit -m "server: fingerprinted bot key for the hub"   # + trailer block
```

---

### Task 3: Schema 2 (game hub columns and the outbox table)

**Files:**
- Modify: `server/AzulServer/Data/Db.cs`
- Modify: `server/AzulServer/Data/GameStore.cs`
- Modify: `server/AzulServer.Tests/StoreTests.cs:27-34` (the expected schema version)
- Test: `server/AzulServer.Tests/Hub/HubSchemaTests.cs`

**Interfaces:**
- Produces:
  - `GameRecord.StartedAt` (`string?`)
  - `GameRecord.FinishedAt` (`string?`)
  - `GameRecord.HubTracked` (`bool`)
  - `GameRecord.BotKey` (`string?`)

  All four are `init` properties declared in the record body, so every existing positional constructor call still compiles, and `with` copies them.
- Also produces: `Db.Migrations` (now `internal static`); the table `hub_reports`, including `lease_id` (the claim that may record an outcome; spec 6.3, lease ownership).

- [ ] **Step 1: Write the failing tests**

`server/AzulServer.Tests/Hub/HubSchemaTests.cs`:

```csharp
using AzulServer.Data;
using Microsoft.Data.Sqlite;

namespace AzulServer.Tests;

public sealed class HubSchemaTests : IDisposable
{
    readonly string dir = Directory.CreateTempSubdirectory("azul-schema-").FullName;
    public void Dispose() => SqliteConnection.ClearAllPools();

    static void Exec(SqliteConnection c, string sql)
    {
        using var cmd = c.CreateCommand();
        cmd.CommandText = sql;
        cmd.ExecuteNonQuery();
    }

    static object? Scalar(SqliteConnection c, string sql)
    {
        using var cmd = c.CreateCommand();
        cmd.CommandText = sql;
        return cmd.ExecuteScalar();
    }

    [Fact]
    public void SchemaOneUpgradesAndItsGamesAreNotTracked()
    {
        var db = new Db(new AzulOptions { DataDir = dir });
        using (var c = db.Open())
        {
            Exec(c, "CREATE TABLE schema_version(v INTEGER NOT NULL);");
            Exec(c, Db.Migrations[0]);
            Exec(c, "INSERT INTO schema_version(v) VALUES (1);");
            Exec(c, "INSERT INTO games(id, creator, status, num_players, version, created_at, updated_at) " +
                    "VALUES ('old', 'a@x.com', 'finished', 2, 9, 't', 't');");
        }
        db.Migrate();
        using var c2 = db.Open();
        Assert.Equal(2L, Scalar(c2, "SELECT MAX(v) FROM schema_version"));
        Assert.Equal(0L, Scalar(c2, "SELECT COUNT(*) FROM hub_reports"));
        var g = GameStore.Load(c2, "old")!;
        Assert.False(g.HubTracked);
        Assert.Null(g.StartedAt);
        Assert.Null(g.FinishedAt);
        Assert.Null(g.BotKey);
    }

    [Fact]
    public void HubColumnsRoundTripThroughInsertAndUpdate()
    {
        var db = new Db(new AzulOptions { DataDir = dir });
        db.Migrate();
        var g = new GameRecord("g1", "a@x.com", "playing", 2, 2, "{}", null, "c", "u",
            [new SeatRecord(0, "human", "a@x.com"), new SeatRecord(1, "bot", null)])
        { StartedAt = "s", HubTracked = true, BotKey = "mcts@0123456789ab" };
        using var c = db.Open();
        using (var tx = c.BeginTransaction()) { GameStore.Insert(c, tx, g); tx.Commit(); }
        var loaded = GameStore.Load(c, "g1")!;
        Assert.Equal(("s", (string?)null, true, "mcts@0123456789ab"),
            (loaded.StartedAt, loaded.FinishedAt, loaded.HubTracked, loaded.BotKey));
        using (var tx = c.BeginTransaction())
        {
            GameStore.Update(c, tx, loaded with { Version = 3, Status = "finished", FinishedAt = "f" }, 2);
            tx.Commit();
        }
        Assert.Equal("f", GameStore.Load(c, "g1")!.FinishedAt);
    }

    [Fact]
    public void HubReportsHasNoForeignKeyToGames()
    {
        var db = new Db(new AzulOptions { DataDir = dir });
        db.Migrate();
        using var c = db.Open();
        Exec(c, "INSERT INTO hub_reports(game_id, body, status, next_attempt_at, created_at) VALUES ('gone', '{}', 'pending', 'n', 'n');");
        Assert.Equal(1L, Scalar(c, "SELECT COUNT(*) FROM hub_reports"));
    }
}
```

In `server/AzulServer.Tests/StoreTests.cs`, in `MigrateTwiceIsHarmless`, change `Assert.Equal(1L, ...)` to `Assert.Equal(2L, ...)`.

- [ ] **Step 2: Run them and check they fail**

Run: `make test FILTER="FullyQualifiedName~HubSchemaTests|FullyQualifiedName~StoreTests"`
Expected: a build failure, because `GameRecord.HubTracked` does not exist and `Db.Migrations` is inaccessible.

- [ ] **Step 3: Implement**

In `server/AzulServer/Data/Db.cs`, change `static readonly string[] Migrations =` to `internal static readonly string[] Migrations =`. Append a second entry after the first raw string, before `];`:

```csharp
        """
        ALTER TABLE games ADD COLUMN started_at TEXT;
        ALTER TABLE games ADD COLUMN finished_at TEXT;
        ALTER TABLE games ADD COLUMN hub_tracked INTEGER NOT NULL DEFAULT 0;
        ALTER TABLE games ADD COLUMN bot_key TEXT;
        CREATE TABLE hub_reports(
          game_id TEXT PRIMARY KEY,
          body TEXT NOT NULL,
          status TEXT NOT NULL CHECK(status IN ('pending','sent','failed')),
          attempts INTEGER NOT NULL DEFAULT 0,
          next_attempt_at TEXT NOT NULL,
          last_status INTEGER,
          last_error TEXT,
          created_at TEXT NOT NULL,
          sent_at TEXT,
          alerted_at TEXT,
          lease_id TEXT);
        CREATE INDEX hub_reports_due ON hub_reports(status, next_attempt_at);
        """,
```

Add a comment above it: `// 2: hub reporting (docs/superpowers/specs/2026-10-04-hub-results-design.md). No FK on hub_reports: a deleted game's report is still sent.`

In `server/AzulServer/Data/GameStore.cs`:

1. Keep `GameRecord`'s positional parameters exactly as they are, and give it a body:

```csharp
public sealed record GameRecord(
    string Id, string Creator, string Status, int NumPlayers, long Version,
    string? StateJson, string? FinishReason, string CreatedAt, string UpdatedAt,
    IReadOnlyList<SeatRecord> Seats)
{
    /// Hub reporting (schema 2): stamped by Start and by the finishing commit.
    public string? StartedAt { get; init; }
    public string? FinishedAt { get; init; }
    public bool HubTracked { get; init; }
    public string? BotKey { get; init; }
}
```

2. Change `GameColumns` to:

```csharp
    const string GameColumns = "id, creator, status, num_players, version, state_json, finish_reason, created_at, updated_at, started_at, finished_at, hub_tracked, bot_key";
```

3. Replace `ReadGame` with:

```csharp
    static GameRecord ReadGame(SqliteDataReader r) => new(
        r.GetString(0), r.GetString(1), r.GetString(2), r.GetInt32(3), r.GetInt64(4),
        Str(r, 5), Str(r, 6), r.GetString(7), r.GetString(8), [])
    {
        StartedAt = Str(r, 9),
        FinishedAt = Str(r, 10),
        HubTracked = r.GetInt64(11) != 0,
        BotKey = Str(r, 12),
    };
```

4. In `Insert`, replace the command with:

```csharp
        using (var cmd = Command(c, tx,
            $"INSERT INTO games({GameColumns}) VALUES ($id, $creator, $status, $n, $v, $state, $finish, $created, $updated, $started, $finished, $tracked, $botkey)",
            ("$id", g.Id), ("$creator", g.Creator), ("$status", g.Status), ("$n", g.NumPlayers), ("$v", g.Version),
            ("$state", g.StateJson), ("$finish", g.FinishReason), ("$created", g.CreatedAt), ("$updated", g.UpdatedAt),
            ("$started", g.StartedAt), ("$finished", g.FinishedAt), ("$tracked", g.HubTracked ? 1 : 0), ("$botkey", g.BotKey)))
            cmd.ExecuteNonQuery();
```

5. In `Update`, replace the first command with:

```csharp
        using (var cmd = Command(c, tx,
            "UPDATE games SET status = $status, version = $v, state_json = $state, finish_reason = $finish, updated_at = $updated, " +
            "started_at = $started, finished_at = $finished, hub_tracked = $tracked, bot_key = $botkey " +
            "WHERE id = $id AND version = $expected",
            ("$status", g.Status), ("$v", g.Version), ("$state", g.StateJson), ("$finish", g.FinishReason),
            ("$updated", g.UpdatedAt), ("$started", g.StartedAt), ("$finished", g.FinishedAt),
            ("$tracked", g.HubTracked ? 1 : 0), ("$botkey", g.BotKey), ("$id", g.Id), ("$expected", expectedVersion)))
        {
            if (cmd.ExecuteNonQuery() != 1)
                throw new ConcurrencyException($"game {g.Id} is not at version {expectedVersion}");
        }
```

- [ ] **Step 4: Run all server tests**

Run: `make test`
Expected: PASS. Every caller of `GameStore` round-trips the new columns, and existing tests are unaffected.

- [ ] **Step 5: Commit**

```bash
git add server/AzulServer/Data server/AzulServer.Tests/StoreTests.cs server/AzulServer.Tests/Hub/HubSchemaTests.cs
git commit -m "server: schema 2, hub columns on games and the hub_reports outbox"   # + trailer block
```

---

### Task 4: The report body (`HubReport`)

**Files:**
- Create: `server/AzulServer/Hub/HubReport.cs`
- Create: `server/AzulServer.Tests/Hub/HubTestKit.cs`
- Test: `server/AzulServer.Tests/Hub/HubReportTests.cs`
- Create (generated by a test): `server/AzulServer.Tests/Hub/golden/*.json`

**Interfaces:**
- Consumes: `HubContract.Email/Name`, `BotIdentity.DisplayName`, the `GameRecord` hub properties, `Projection.Snapshot(GameRecord)`.
- Produces:
  - `HubBuild(string? Body, string? Skip)` record
  - `HubReport.Build(GameRecord g, ISet<int> botPlayedSeats, string? publicOrigin) -> HubBuild`
  - `HubReport.BotPlayedSeats(SqliteConnection c, SqliteTransaction? tx, string gameId, MoveRecord? pending) -> ISet<int>`
  - `HubReport.StalemateReason`, `HubReport.BotPlayedReason`
  - Test kit:
    - `HubFixtures.Finished(...) -> GameRecord`
    - `HubFixtures.NewDb() -> Db`
    - `FakeHub` (`HttpMessageHandler` + `IHttpClientFactory`, with an optional `Gate` that holds requests)
    - `ListLogger<T>`

- [ ] **Step 1: Write the test kit**

`server/AzulServer.Tests/Hub/HubTestKit.cs`:

```csharp
using System.Net;
using Azul;
using AzulServer.Data;
using AzulServer.Games;
using Microsoft.Extensions.Logging;

namespace AzulServer.Tests;

public static class HubFixtures
{
    public const string Started = "2026-10-04T18:02:11.0000000Z";
    public const string FinishedAt = "2026-10-04T18:41:52.1239999Z";

    public static Db NewDb()
    {
        var db = new Db(new AzulOptions { DataDir = Directory.CreateTempSubdirectory("azul-hub-").FullName });
        db.Migrate();
        return db;
    }

    static int[][] Grid(int fullRows) => Enumerable.Range(0, 5)
        .Select(r => Enumerable.Range(0, 5).Select(col => r < fullRows ? (r + col) % 5 : -1).ToArray())
        .ToArray();

    /// A finished, hub-tracked record. `players` overrides each seat's
    /// (score, complete wall rows); otherwise a greedy game's real result.
    public static GameRecord Finished(string id, IReadOnlyList<SeatRecord> seats,
        (int Score, int Rows)[]? players = null, string reason = "normal", int seed = 4)
    {
        var game = new Game(seats.Count, new Random(seed));
        while (!game.IsGameOver()) game.Play(game.GetGreedyMove());
        var snap = game.ToSnapshot();
        if (players is not null)
            snap = snap with
            {
                Players = snap.Players.Select((p, i) => p with { Score = players[i].Score, Grid = Grid(players[i].Rows) }).ToArray(),
            };
        return new GameRecord(id, "alice@example.com", Status.Finished, seats.Count, 99, Json.Serialize(snap), reason,
            "2026-10-04T18:00:00.0000000Z", FinishedAt, seats)
        {
            StartedAt = Started, FinishedAt = FinishedAt, HubTracked = true, BotKey = "mcts@0123456789ab",
        };
    }

    public static SeatRecord Human(int idx, string email) => new(idx, SeatKind.Human, email);
    public static SeatRecord Bot(int idx, string? owner = null) => new(idx, SeatKind.Bot, owner);
}

/// Records every request; answers with `Respond()` or throws `Throw`.
public sealed class FakeHub : HttpMessageHandler, IHttpClientFactory
{
    public readonly List<(HttpRequestMessage Request, string Body)> Requests = new();
    public Func<HttpResponseMessage> Respond = () => new HttpResponseMessage(HttpStatusCode.Created) { Content = new StringContent("{}") };
    public Exception? Throw;
    /// When set, every request is recorded and then held until the test completes it.
    public TaskCompletionSource? Gate;

    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken ct)
    {
        var body = request.Content is null ? "" : await request.Content.ReadAsStringAsync(ct);
        lock (Requests) Requests.Add((request, body));
        if (Gate is { } gate) await gate.Task.WaitAsync(ct);
        if (Throw is not null) throw Throw;
        return Respond();
    }

    public HttpClient CreateClient(string name) => new(this, disposeHandler: false);
}

public sealed class ListLogger<T> : ILogger<T>
{
    public readonly List<(LogLevel Level, string Message)> Entries = new();
    public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;
    public bool IsEnabled(LogLevel level) => true;
    public void Log<TState>(LogLevel level, EventId id, TState state, Exception? e, Func<TState, Exception?, string> format)
    {
        lock (Entries) Entries.Add((level, format(state, e)));
    }
    public int Count(LogLevel level, string contains)
    {
        lock (Entries) return Entries.Count(x => x.Level == level && x.Message.Contains(contains));
    }
}
```

- [ ] **Step 2: Write the failing tests**

`server/AzulServer.Tests/Hub/HubReportTests.cs`:

```csharp
using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;
using AzulServer.Data;
using AzulServer.Games;
using AzulServer.Hub;
using Microsoft.Data.Sqlite;
using static AzulServer.Tests.HubFixtures;

namespace AzulServer.Tests;

public sealed class HubReportTests : IDisposable
{
    public void Dispose() => SqliteConnection.ClearAllPools();

    static readonly HashSet<int> None = [];
    static readonly SeatRecord[] Three = [Human(0, "alice@example.com"), Human(1, "bob@example.com"), Bot(2)];

    static JsonObject Body(GameRecord g, ISet<int>? botPlayed = null, string? origin = "https://azul.example")
    {
        var b = HubReport.Build(g, botPlayed ?? None, origin);
        Assert.Null(b.Skip);
        return JsonNode.Parse(b.Body!)!.AsObject();
    }

    static JsonObject Player(JsonObject body, int i) => body["players"]![i]!.AsObject();

    [Fact]
    public void RatedReportHasTheHubShape()
    {
        var b = Body(Finished("abcdefghij", Three));
        Assert.Equal("abcdefghij", (string?)b["external_id"]);
        Assert.Equal("free-wall-3p", (string?)b["variant"]);
        Assert.True((bool)b["rated"]!);
        Assert.Null(b["unrated_reason"]);
        Assert.Equal("2026-10-04T18:02:11.000Z", (string?)b["started_at"]);
        Assert.Equal("https://azul.example/g/abcdefghij", (string?)b["replay_url"]);
        Assert.Equal(["P1", "P2", "P3"], b["players"]!.AsArray().Select(p => (string)p!["seat"]!));
        Assert.Equal(("human", "alice@example.com", "alice"),
            ((string?)Player(b, 0)["kind"], (string?)Player(b, 0)["email"], (string?)Player(b, 0)["name"]));
        Assert.Equal(("bot", "mcts@0123456789ab", "Azul MCTS"),
            ((string?)Player(b, 2)["kind"], (string?)Player(b, 2)["bot_key"], (string?)Player(b, 2)["name"]));
        Assert.Null(Player(b, 2)["email"]);
        Assert.Contains(b["players"]!.AsArray(), p => (int)p!["rank"]! == 1);
    }

    [Fact]
    public void TimestampsAreTruncatedToMilliseconds()
    {
        var g = Finished("g", Three) with { FinishedAt = "2026-10-04T18:41:52.9999999Z", StartedAt = "2026-10-04T18:41:52.9990000Z" };
        var b = Body(g);
        Assert.Equal("2026-10-04T18:41:52.999Z", (string?)b["finished_at"]);
        Assert.Equal("2026-10-04T18:41:52.999Z", (string?)b["started_at"]);
    }

    [Fact]
    public void StalemateIsUnrated()
    {
        var b = Body(Finished("g", Three, reason: "stalemate"));
        Assert.False((bool)b["rated"]!);
        Assert.Equal("stalemate", (string?)b["unrated_reason"]);
    }

    [Fact]
    public void ABotMoveForAPersonMakesItUnrated()
    {
        var b = Body(Finished("g", Three), new HashSet<int> { 1 });
        Assert.False((bool)b["rated"]!);
        Assert.Equal("a bot played for a person", (string?)b["unrated_reason"]);
    }

    [Fact]
    public void TheBotSeatPlayingForItselfStaysRated() =>
        Assert.True((bool)Body(Finished("g", Three), new HashSet<int> { 2 })["rated"]!);

    [Fact]
    public void AHandedOverSeatIsReportedUnderItsOwner()
    {
        var b = Body(Finished("g", [Human(0, "alice@example.com"), Bot(1, "bob@example.com")]));
        Assert.Equal(("human", "bob@example.com"), ((string?)Player(b, 1)["kind"], (string?)Player(b, 1)["email"]));
    }

    [Fact]
    public void TiesBreakOnCompleteRowsThenShareARank()
    {
        SeatRecord[] four = [Human(0, "a@example.com"), Human(1, "b@example.com"), Bot(2), Human(3, "d@example.com")];
        var b = Body(Finished("g", four, [(40, 1), (40, 2), (30, 0), (40, 2)]));
        Assert.Equal([3, 1, 4, 1], b["players"]!.AsArray().Select(p => (int)p!["rank"]!));
        Assert.Equal([40, 40, 30, 40], b["players"]!.AsArray().Select(p => (int)p!["score"]!));
    }

    [Fact]
    public void ADevEmailIsSentAsNullWithItsLocalPartAsName()
    {
        var b = Body(Finished("g", [Human(0, "dev@localhost"), Bot(1)]));
        Assert.Null(Player(b, 0)["email"]);
        Assert.Equal("dev", (string?)Player(b, 0)["name"]);
    }

    [Fact]
    public void ALongLocalPartIsCutToForty()
    {
        var local = new string('x', 50);
        var b = Body(Finished("g", [Human(0, $"{local}@example.com"), Bot(1)]));
        Assert.Equal(new string('x', 40), (string?)Player(b, 0)["name"]);
    }

    [Fact]
    public void NoReplayUrlWithoutAPublicOrigin() =>
        Assert.Null(Body(Finished("g", Three), origin: null)["replay_url"]);

    [Fact]
    public void UntrackedOrKeylessGamesAreSkipped()
    {
        Assert.Equal("not tracked", HubReport.Build(Finished("g", Three) with { HubTracked = false }, None, null).Skip);
        Assert.Equal("no bot key", HubReport.Build(Finished("g", Three) with { BotKey = null }, None, null).Skip);
        Assert.Null(HubReport.Build(Finished("g", Three) with { BotKey = null }, None, null).Body);
    }

    [Fact]
    public void BotPlayedSeatsIncludesThePendingMove()
    {
        var db = NewDb();
        using var c = db.Open();
        using (var tx = c.BeginTransaction())
        {
            GameStore.Insert(c, tx, Finished("g", Three) with { Status = Status.Playing });
            GameStore.InsertMove(c, tx, new MoveRecord("g", 2, 1, "auto", "{}", null, null, null, "t"));
            GameStore.InsertMove(c, tx, new MoveRecord("g", 3, 2, "bot", "{}", null, null, null, "t"));
            tx.Commit();
        }
        Assert.Equal(new HashSet<int> { 2 }, HubReport.BotPlayedSeats(c, null, "g", null));
        var pending = new MoveRecord("g", 0, 1, "bot", "{}", null, null, null, "t");
        Assert.Equal(new HashSet<int> { 1, 2 }, HubReport.BotPlayedSeats(c, null, "g", pending));
        var human = new MoveRecord("g", 0, 0, "alice@example.com", "{}", null, null, null, "t");
        Assert.Equal(new HashSet<int> { 2 }, HubReport.BotPlayedSeats(c, null, "g", human));
    }

    // ---------- golden bodies: checked against the hub's own parser in Task 11 ----------

    static string GoldenDir([CallerFilePath] string here = "") => Path.Combine(Path.GetDirectoryName(here)!, "golden");

    public static TheoryData<string> Goldens => new() { "rated-3p", "stalemate", "handover", "ties-4p", "null-email", "long-name" };

    static string GoldenBody(string name) => name switch
    {
        "rated-3p" => HubReport.Build(Finished("goldrated1", Three), None, "https://azul.signalwave.dev").Body!,
        "stalemate" => HubReport.Build(Finished("goldstale1", Three, reason: "stalemate"), None, "https://azul.signalwave.dev").Body!,
        "handover" => HubReport.Build(Finished("goldhand01", [Human(0, "alice@example.com"), Bot(1, "bob@example.com")]),
            new HashSet<int> { 1 }, "https://azul.signalwave.dev").Body!,
        "ties-4p" => HubReport.Build(Finished("goldties01",
            [Human(0, "a@example.com"), Human(1, "b@example.com"), Bot(2), Human(3, "d@example.com")],
            [(40, 1), (40, 2), (30, 0), (40, 2)]), None, "https://azul.signalwave.dev").Body!,
        "null-email" => HubReport.Build(Finished("goldnull01", [Human(0, "dev@localhost"), Bot(1)]), None, null).Body!,
        "long-name" => HubReport.Build(Finished("goldlong01", [Human(0, new string('x', 50) + "@example.com"), Bot(1)]), None, null).Body!,
        _ => throw new ArgumentException(name),
    };

    /// AZUL_UPDATE_GOLDEN=1 rewrites the files; otherwise they must match.
    [Theory]
    [MemberData(nameof(Goldens))]
    public void GoldenBodiesMatch(string name)
    {
        var path = Path.Combine(GoldenDir(), name + ".json");
        var body = GoldenBody(name);
        if (Environment.GetEnvironmentVariable("AZUL_UPDATE_GOLDEN") == "1")
        {
            Directory.CreateDirectory(GoldenDir());
            File.WriteAllText(path, body + "\n");
        }
        Assert.Equal(File.ReadAllText(path).TrimEnd('\n'), body);
    }
}
```

- [ ] **Step 3: Run them and check they fail**

Run: `make test FILTER="FullyQualifiedName~HubReportTests"`
Expected: a build failure, because `HubReport` does not exist.

- [ ] **Step 4: Implement**

`server/AzulServer/Hub/HubReport.cs`:

```csharp
using System.Globalization;
using System.Text.Json.Nodes;
using AzulServer.Data;
using AzulServer.Games;
using Microsoft.Data.Sqlite;

namespace AzulServer.Hub;

/// `Body` is the JSON to send, or null with `Skip` saying why there is none.
public sealed record HubBuild(string? Body, string? Skip);

/// The hub 5.1 body for a finished game (spec section 4). Pure: everything
/// comes from the record (its StateJson is the final snapshot) and the seats a
/// bot moved for.
public static class HubReport
{
    public const string StalemateReason = "stalemate";
    public const string BotPlayedReason = "a bot played for a person";

    /// Seats a bot made a move for: every stored bot move, plus `pending`, the
    /// finishing move, which Mutate has not inserted yet when it builds.
    public static ISet<int> BotPlayedSeats(SqliteConnection c, SqliteTransaction? tx, string gameId, MoveRecord? pending)
    {
        var seats = new HashSet<int>();
        using (var cmd = c.CreateCommand())
        {
            cmd.Transaction = tx;
            cmd.CommandText = "SELECT DISTINCT seat FROM moves WHERE game_id = $g AND actor = 'bot'";
            cmd.Parameters.AddWithValue("$g", gameId);
            using var r = cmd.ExecuteReader();
            while (r.Read()) seats.Add(r.GetInt32(0));
        }
        if (pending is { Actor: "bot" }) seats.Add(pending.Seat);
        return seats;
    }

    public static HubBuild Build(GameRecord g, ISet<int> botPlayedSeats, string? publicOrigin)
    {
        if (!g.HubTracked) return new(null, "not tracked");
        if (g.Status != Status.Finished || g.StateJson is null || g.FinishedAt is null) return new(null, "not finished");
        var snap = Projection.Snapshot(g);
        var keys = snap.Players.Select(p => (p.Score, Rows: p.Grid.Count(row => row.All(v => v >= 0)))).ToArray();
        var players = new JsonArray();
        foreach (var seat in g.Seats.OrderBy(s => s.Idx))
        {
            int i = seat.Idx;
            JsonObject p;
            if (seat.Email is { } email)
                p = new JsonObject
                {
                    ["seat"] = $"P{i + 1}",
                    ["kind"] = "human",
                    ["email"] = HubContract.Email(email),
                    ["name"] = HubContract.Name(email.Split('@')[0], "Player"),
                };
            else if (seat.Kind == SeatKind.Bot && g.BotKey is { } botKey)
                p = new JsonObject
                {
                    ["seat"] = $"P{i + 1}",
                    ["kind"] = "bot",
                    ["bot_key"] = botKey,
                    ["name"] = BotIdentity.DisplayName,
                };
            else
                return new(null, seat.Kind == SeatKind.Bot ? "no bot key" : $"seat {i} is {seat.Kind}");
            p["rank"] = 1 + keys.Count(k => k.Score > keys[i].Score || (k.Score == keys[i].Score && k.Rows > keys[i].Rows));
            p["score"] = keys[i].Score;
            players.Add(p);
        }
        string? reason = g.FinishReason != "normal" ? StalemateReason
            : g.Seats.Any(s => s.Email is not null && botPlayedSeats.Contains(s.Idx)) ? BotPlayedReason
            : null;
        var body = new JsonObject
        {
            ["external_id"] = g.Id,
            ["variant"] = $"free-wall-{g.NumPlayers}p",
            ["rated"] = reason is null,
            ["unrated_reason"] = reason,
            ["started_at"] = g.StartedAt is null ? null : Utc(g.StartedAt),
            ["finished_at"] = Utc(g.FinishedAt),
            ["replay_url"] = publicOrigin is null ? null : $"{publicOrigin}/g/{g.Id}",
            ["players"] = players,
        };
        return new(body.ToJsonString(), null);
    }

    /// "O" UTC string -> yyyy-MM-ddTHH:mm:ss.fffZ. The custom format truncates
    /// (it never rounds into the next second).
    static string Utc(string stamp) =>
        DateTimeOffset.Parse(stamp, CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal)
            .UtcDateTime.ToString("yyyy-MM-dd'T'HH:mm:ss.fff'Z'", CultureInfo.InvariantCulture);
}
```

- [ ] **Step 5: Generate the golden files, then run all the report tests**

Run: `make dotnet ARGS="test Azul.slnx --filter FullyQualifiedName~GoldenBodiesMatch -e AZUL_UPDATE_GOLDEN=1"`. This writes the six files under `server/AzulServer.Tests/Hub/golden/`. Open each one and check it by eye: snake_case keys, `"P1"` seats, timestamps ending in `.123Z`.

Then run: `make test FILTER="FullyQualifiedName~HubReportTests"`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add server/AzulServer/Hub/HubReport.cs server/AzulServer.Tests/Hub
git commit -m "server: hub report body, ranks, rated rule and golden bodies"   # + trailer block
```

---

### Task 5: The outbox (`HubOutbox`)

**Files:**
- Create: `server/AzulServer/Hub/HubOutbox.cs`
- Test: `server/AzulServer.Tests/Hub/HubOutboxTests.cs`

**Interfaces:**
- Consumes: `HubReport.Build`, `HubReport.BotPlayedSeats`, `GameStore.Load`, and the `lease_id` column (Task 3).
- Produces (all static on `HubOutbox`; `now` and the other stamps are `"O"` UTC strings):
  - **Records:**
    - `HubRow(string GameId, string Body, string Status, int Attempts, string NextAttemptAt, int? LastStatus, string? LastError, string CreatedAt, string? SentAt, string? AlertedAt, string? LeaseId)`
    - `HubClaim(string Body, int Attempts, string LeaseId)`
    - `HubFailure(string GameId, int? LastStatus, string? LastError)`
    - `HubSummary(IReadOnlyDictionary<string, int> Counts, IReadOnlyList<HubFailure> Failed, string? OldestPendingCreatedAt)`
  - **Queueing:**
    - `bool Queue(SqliteConnection c, SqliteTransaction tx, string gameId, string body, string now)`. It uses `INSERT ... ON CONFLICT(game_id) DO NOTHING`, not `INSERT OR IGNORE`, so a failing trigger or constraint still aborts the statement (Task 6's atomicity test depends on this).
    - `bool Exists(SqliteConnection c, SqliteTransaction? tx, string gameId)`
    - `HubRow? Get(SqliteConnection c, SqliteTransaction? tx, string gameId)`
  - **Reconcile:**
    - `List<string> MissingIds(SqliteConnection c)`
    - `int QueueMissing(Db db, string? publicOrigin, string now, ILogger log)`
  - **Claiming:**
    - `List<string> DueIds(SqliteConnection c, string now, int limit)`
    - `HubClaim? TryClaim(SqliteConnection c, string gameId, string now, string leaseUntil)`, which sets a fresh `lease_id`
  - **Recording outcomes:** each returns `false` when the lease is stale, meaning 0 rows were updated and the outcome must be dropped. Each clears `lease_id`.
    - `bool MarkSent(SqliteConnection c, string gameId, string leaseId, int status, string now)`
    - `bool MarkFailed(SqliteConnection c, string gameId, string leaseId, int status, string? error)`
    - `bool MarkRetry(SqliteConnection c, string gameId, string leaseId, int? status, string? error, string nextAttemptAt)`
  - **Alerts:**
    - `List<string> StuckIds(SqliteConnection c, string createdBefore, string alertedBefore)`
    - `void MarkAlerted(SqliteConnection c, string gameId, string now)`
  - **Operator commands:**
    - `HubSummary Summary(SqliteConnection c)`
    - `List<string> FailedIds(SqliteConnection c)`
    - `string ResetForRetry(SqliteConnection c, string gameId, string now)`, returning `"pending"`, `"already sent"` or `"no report"`. It is one conditional `UPDATE` that also clears `lease_id`.
    - `string Rebuild(Db db, string gameId, string? publicOrigin, string now)`, returning `"rebuilt"`, `"already sent"`, `"game deleted"` or `"unbuildable: <reason>"`. It clears `lease_id` in the same statement.

- [ ] **Step 1: Write the failing tests**

`server/AzulServer.Tests/Hub/HubOutboxTests.cs`:

```csharp
using AzulServer.Data;
using AzulServer.Games;
using AzulServer.Hub;
using Microsoft.Data.Sqlite;
using Microsoft.Extensions.Logging;
using static AzulServer.Tests.HubFixtures;

namespace AzulServer.Tests;

public sealed class HubOutboxTests : IDisposable
{
    readonly Db db = NewDb();
    public void Dispose() => SqliteConnection.ClearAllPools();

    const string T0 = "2026-10-04T19:00:00.0000000Z";
    const string T1 = "2026-10-04T19:01:00.0000000Z";
    const string T5 = "2026-10-04T19:05:00.0000000Z";
    const string T9 = "2026-10-04T19:09:00.0000000Z";
    static readonly SeatRecord[] Two = [Human(0, "alice@example.com"), Bot(1)];

    void InsertGame(GameRecord g)
    {
        using var c = db.Open();
        using var tx = c.BeginTransaction();
        GameStore.Insert(c, tx, g);
        tx.Commit();
    }

    void QueueRow(string id, string body = "{\"x\":1}", string now = T0)
    {
        using var c = db.Open();
        using var tx = c.BeginTransaction();
        HubOutbox.Queue(c, tx, id, body, now);
        tx.Commit();
    }

    HubRow Row(string id)
    {
        using var c = db.Open();
        return HubOutbox.Get(c, null, id)!;
    }

    /// Claims at T0 (lease to T5) and returns the lease id.
    string Claim(string id)
    {
        using var c = db.Open();
        return HubOutbox.TryClaim(c, id, T0, T5)!.LeaseId;
    }

    [Fact]
    public void QueueIsInsertOnce()
    {
        QueueRow("g", "{\"first\":1}");
        QueueRow("g", "{\"second\":1}");
        var row = Row("g");
        Assert.Equal(("{\"first\":1}", "pending", 0, T0, (string?)null),
            (row.Body, row.Status, row.Attempts, row.NextAttemptAt, row.LeaseId));
    }

    [Fact]
    public void AClaimIsALeaseWithItsOwnId()
    {
        QueueRow("g");
        using var c = db.Open();
        Assert.Equal(["g"], HubOutbox.DueIds(c, T0, 10));
        var first = HubOutbox.TryClaim(c, "g", T0, T5)!;
        Assert.Equal(("{\"x\":1}", 0), (first.Body, first.Attempts));
        Assert.Equal(first.LeaseId, HubOutbox.Get(c, null, "g")!.LeaseId);
        Assert.Null(HubOutbox.TryClaim(c, "g", T0, T5));   // leased
        Assert.Empty(HubOutbox.DueIds(c, T1, 10));
        var second = HubOutbox.TryClaim(c, "g", T5, T9)!;   // lease expired
        Assert.NotEqual(first.LeaseId, second.LeaseId);
        Assert.False(HubOutbox.MarkSent(c, "g", first.LeaseId, 201, T5));   // the expired claim's outcome is stale
        Assert.True(HubOutbox.MarkSent(c, "g", second.LeaseId, 201, T5));
    }

    [Fact]
    public void OutcomesAreRecordedUnderTheLease()
    {
        QueueRow("a"); QueueRow("b"); QueueRow("c");
        string la = Claim("a"), lb = Claim("b"), lc = Claim("c");
        using var c = db.Open();
        Assert.True(HubOutbox.MarkSent(c, "a", la, 201, T1));
        Assert.True(HubOutbox.MarkFailed(c, "b", lb, 422, "bad body"));
        Assert.True(HubOutbox.MarkRetry(c, "c", lc, null, "connection refused", T5));
        var a = Row("a");
        Assert.Equal(("sent", 1, (int?)201, (string?)T1, (string?)null), (a.Status, a.Attempts, a.LastStatus, a.SentAt, a.LeaseId));
        var b = Row("b");
        Assert.Equal(("failed", (int?)422, (string?)"bad body", (string?)null), (b.Status, b.LastStatus, b.LastError, b.LeaseId));
        var r = Row("c");
        Assert.Equal(("pending", 1, (int?)null, T5, (string?)null), (r.Status, r.Attempts, r.LastStatus, r.NextAttemptAt, r.LeaseId));
        var summary = HubOutbox.Summary(c);
        Assert.Equal((1, 1, 1), (summary.Counts["sent"], summary.Counts["failed"], summary.Counts["pending"]));
        Assert.Equal([new HubFailure("b", 422, "bad body")], summary.Failed);
        Assert.Equal(T0, summary.OldestPendingCreatedAt);
    }

    [Fact]
    public void ARetryInvalidatesTheInFlightOutcome()
    {
        QueueRow("g");
        var lease = Claim("g");
        using var c = db.Open();
        Assert.Equal("pending", HubOutbox.ResetForRetry(c, "g", T1));
        var reset = Row("g");
        Assert.Equal(((string?)null, T1), (reset.LeaseId, reset.NextAttemptAt));
        Assert.False(HubOutbox.MarkSent(c, "g", lease, 201, T1));
        Assert.False(HubOutbox.MarkFailed(c, "g", lease, 422, "late"));
        Assert.False(HubOutbox.MarkRetry(c, "g", lease, 500, "late", T9));
        var after = Row("g");
        Assert.Equal(("pending", 0, T1), (after.Status, after.Attempts, after.NextAttemptAt));
    }

    [Fact]
    public void StuckRowsAreFoundUntilAlerted()
    {
        QueueRow("g", now: T0);
        using var c = db.Open();
        Assert.Equal(["g"], HubOutbox.StuckIds(c, T1, T1));
        HubOutbox.MarkAlerted(c, "g", T1);
        Assert.Empty(HubOutbox.StuckIds(c, T5, T0));
        Assert.Equal(["g"], HubOutbox.StuckIds(c, T5, T5));
    }

    [Fact]
    public void ReconcileQueuesFinishedTrackedGamesWithoutAReport()
    {
        InsertGame(Finished("done", Two));
        InsertGame(Finished("old", Two) with { HubTracked = false });
        InsertGame(Finished("live", Two) with { Status = Status.Playing });
        InsertGame(Finished("keyless", Two) with { BotKey = null });
        var log = new ListLogger<HubOutboxTests>();
        Assert.Equal(1, HubOutbox.QueueMissing(db, null, T0, log));
        using var c = db.Open();
        Assert.True(HubOutbox.Exists(c, null, "done"));
        Assert.False(HubOutbox.Exists(c, null, "old"));
        Assert.False(HubOutbox.Exists(c, null, "live"));
        Assert.False(HubOutbox.Exists(c, null, "keyless"));
        Assert.Equal(1, log.Count(LogLevel.Error, "keyless"));
        Assert.Equal(["keyless"], HubOutbox.MissingIds(c));
    }

    [Fact]
    public void RetryPutsUnsentRowsBack()
    {
        QueueRow("f"); QueueRow("s");
        string lf = Claim("f"), ls = Claim("s");
        using var c = db.Open();
        HubOutbox.MarkFailed(c, "f", lf, 409, "dup");
        HubOutbox.MarkSent(c, "s", ls, 200, T1);
        Assert.Equal("pending", HubOutbox.ResetForRetry(c, "f", T5));
        Assert.Equal(("pending", T5), (Row("f").Status, Row("f").NextAttemptAt));
        Assert.Equal("already sent", HubOutbox.ResetForRetry(c, "s", T5));
        Assert.Equal("sent", Row("s").Status);
        Assert.Equal("no report", HubOutbox.ResetForRetry(c, "nope", T5));
        Assert.Empty(HubOutbox.FailedIds(c));   // f is pending again
    }

    [Fact]
    public void RebuildReplacesOnlyOnSuccessAndNeverLosesAReport()
    {
        InsertGame(Finished("g", Two));
        QueueRow("g", "{\"stale\":true}");
        QueueRow("gone", "{\"keep\":true}");
        var lease = Claim("g");   // a send is in flight
        Assert.Equal("rebuilt", HubOutbox.Rebuild(db, "g", null, T5));
        Assert.Equal("game deleted", HubOutbox.Rebuild(db, "gone", null, T5));
        var g = Row("g");
        Assert.Equal(("pending", 0, T5, (string?)null), (g.Status, g.Attempts, g.NextAttemptAt, g.LeaseId));
        Assert.Contains("\"external_id\":\"g\"", g.Body);
        using (var c = db.Open()) Assert.False(HubOutbox.MarkSent(c, "g", lease, 201, T5));   // the old body's answer is dropped
        Assert.Equal("pending", Row("g").Status);
        Assert.Equal("{\"keep\":true}", Row("gone").Body);

        InsertGame(Finished("nokey", Two) with { BotKey = null });
        QueueRow("nokey", "{\"orig\":1}");
        Assert.Equal("unbuildable: no bot key", HubOutbox.Rebuild(db, "nokey", null, T5));
        Assert.Equal("{\"orig\":1}", Row("nokey").Body);

        InsertGame(Finished("done", Two));
        QueueRow("done", "{\"sent\":1}");
        using (var c = db.Open()) HubOutbox.MarkSent(c, "done", Claim("done"), 201, T1);
        Assert.Equal("already sent", HubOutbox.Rebuild(db, "done", null, T5));
        Assert.Equal("{\"sent\":1}", Row("done").Body);
    }
}
```

- [ ] **Step 2: Run them and check they fail**

Run: `make test FILTER="FullyQualifiedName~HubOutboxTests"`
Expected: a build failure, because `HubOutbox` does not exist.

- [ ] **Step 3: Implement**

`server/AzulServer/Hub/HubOutbox.cs`:

```csharp
using AzulServer.Data;
using Microsoft.Data.Sqlite;

namespace AzulServer.Hub;

public sealed record HubRow(string GameId, string Body, string Status, int Attempts, string NextAttemptAt,
    int? LastStatus, string? LastError, string CreatedAt, string? SentAt, string? AlertedAt, string? LeaseId);

public sealed record HubClaim(string Body, int Attempts, string LeaseId);

public sealed record HubFailure(string GameId, int? LastStatus, string? LastError);

public sealed record HubSummary(IReadOnlyDictionary<string, int> Counts, IReadOnlyList<HubFailure> Failed, string? OldestPendingCreatedAt);

/// Every statement on hub_reports. Timestamps are GameService's "O" UTC
/// strings, which compare correctly as text. Outcomes are written only under
/// the lease that claimed the row (spec 6.3, lease ownership): a retry or a
/// rebuild clears lease_id, so a request still in flight cannot record its
/// answer over a reset or replaced row.
public static class HubOutbox
{
    static SqliteCommand Command(SqliteConnection c, SqliteTransaction? tx, string sql, params (string Name, object? Value)[] args)
    {
        var cmd = c.CreateCommand();
        cmd.Transaction = tx;
        cmd.CommandText = sql;
        foreach (var (name, value) in args)
            cmd.Parameters.AddWithValue(name, value ?? DBNull.Value);
        return cmd;
    }

    static List<string> Ids(SqliteConnection c, SqliteTransaction? tx, string sql, params (string, object?)[] args)
    {
        using var cmd = Command(c, tx, sql, args);
        using var r = cmd.ExecuteReader();
        var ids = new List<string>();
        while (r.Read()) ids.Add(r.GetString(0));
        return ids;
    }

    static int Exec(SqliteConnection c, SqliteTransaction? tx, string sql, params (string, object?)[] args)
    {
        using var cmd = Command(c, tx, sql, args);
        return cmd.ExecuteNonQuery();
    }

    /// ON CONFLICT DO NOTHING (not INSERT OR IGNORE): only the primary key
    /// conflict is ignored; any other failure aborts the finishing commit.
    public static bool Queue(SqliteConnection c, SqliteTransaction tx, string gameId, string body, string now) =>
        Exec(c, tx, "INSERT INTO hub_reports(game_id, body, status, attempts, next_attempt_at, created_at) " +
                    "VALUES ($g, $b, 'pending', 0, $now, $now) ON CONFLICT(game_id) DO NOTHING",
            ("$g", gameId), ("$b", body), ("$now", now)) == 1;

    public static bool Exists(SqliteConnection c, SqliteTransaction? tx, string gameId)
    {
        using var cmd = Command(c, tx, "SELECT 1 FROM hub_reports WHERE game_id = $g", ("$g", gameId));
        return cmd.ExecuteScalar() is not null;
    }

    public static HubRow? Get(SqliteConnection c, SqliteTransaction? tx, string gameId)
    {
        using var cmd = Command(c, tx,
            "SELECT game_id, body, status, attempts, next_attempt_at, last_status, last_error, created_at, sent_at, alerted_at, lease_id " +
            "FROM hub_reports WHERE game_id = $g", ("$g", gameId));
        using var r = cmd.ExecuteReader();
        if (!r.Read()) return null;
        string? S(int i) => r.IsDBNull(i) ? null : r.GetString(i);
        return new HubRow(r.GetString(0), r.GetString(1), r.GetString(2), r.GetInt32(3), r.GetString(4),
            r.IsDBNull(5) ? null : r.GetInt32(5), S(6), r.GetString(7), S(8), S(9), S(10));
    }

    public static List<string> MissingIds(SqliteConnection c) => Ids(c, null,
        "SELECT id FROM games WHERE status = 'finished' AND hub_tracked = 1 " +
        "AND id NOT IN (SELECT game_id FROM hub_reports) ORDER BY id");

    /// Reconcile (spec 6.2): build and queue each finished tracked game that
    /// has no report. One write transaction per game reads it and writes the
    /// row, so a concurrent Delete cannot interleave.
    public static int QueueMissing(Db db, string? publicOrigin, string now, ILogger log)
    {
        List<string> ids;
        using (var c = db.Open()) ids = MissingIds(c);
        int queued = 0;
        foreach (var id in ids)
        {
            try
            {
                using var c = db.Open();
                using var tx = c.BeginTransaction();
                var g = GameStore.Load(c, id, tx);
                if (g is null || Exists(c, tx, id)) continue;
                var built = HubReport.Build(g, HubReport.BotPlayedSeats(c, tx, id, null), publicOrigin);
                if (built.Body is null)
                {
                    log.LogError("hub report for game {Game} cannot be built: {Reason}", id, built.Skip);
                    continue;
                }
                if (Queue(c, tx, id, built.Body, now)) queued++;
                tx.Commit();
            }
            catch (Exception e)
            {
                log.LogError(e, "hub report for game {Game} could not be queued", id);
            }
        }
        return queued;
    }

    public static List<string> DueIds(SqliteConnection c, string now, int limit) => Ids(c, null,
        "SELECT game_id FROM hub_reports WHERE status = 'pending' AND next_attempt_at <= $now " +
        "ORDER BY next_attempt_at LIMIT $n", ("$now", now), ("$n", limit));

    /// Takes a lease: the row is not due again before `leaseUntil`, and only
    /// the returned LeaseId may record this attempt's outcome.
    public static HubClaim? TryClaim(SqliteConnection c, string gameId, string now, string leaseUntil)
    {
        var leaseId = Guid.NewGuid().ToString("N");
        using var tx = c.BeginTransaction();
        if (Exec(c, tx, "UPDATE hub_reports SET next_attempt_at = $until, lease_id = $lease " +
                        "WHERE game_id = $g AND status = 'pending' AND next_attempt_at <= $now",
                ("$until", leaseUntil), ("$lease", leaseId), ("$g", gameId), ("$now", now)) != 1)
            return null;
        var row = Get(c, tx, gameId)!;
        tx.Commit();
        return new HubClaim(row.Body, row.Attempts, leaseId);
    }

    public static bool MarkSent(SqliteConnection c, string gameId, string leaseId, int status, string now) =>
        Exec(c, null, "UPDATE hub_reports SET status = 'sent', attempts = attempts + 1, last_status = $s, " +
                      "last_error = NULL, sent_at = $now, lease_id = NULL WHERE game_id = $g AND lease_id = $l",
            ("$s", status), ("$now", now), ("$g", gameId), ("$l", leaseId)) == 1;

    public static bool MarkFailed(SqliteConnection c, string gameId, string leaseId, int status, string? error) =>
        Exec(c, null, "UPDATE hub_reports SET status = 'failed', attempts = attempts + 1, last_status = $s, " +
                      "last_error = $e, lease_id = NULL WHERE game_id = $g AND lease_id = $l",
            ("$s", status), ("$e", error), ("$g", gameId), ("$l", leaseId)) == 1;

    public static bool MarkRetry(SqliteConnection c, string gameId, string leaseId, int? status, string? error, string nextAttemptAt) =>
        Exec(c, null, "UPDATE hub_reports SET attempts = attempts + 1, last_status = $s, last_error = $e, " +
                      "next_attempt_at = $next, lease_id = NULL WHERE game_id = $g AND lease_id = $l",
            ("$s", status), ("$e", error), ("$next", nextAttemptAt), ("$g", gameId), ("$l", leaseId)) == 1;

    public static List<string> StuckIds(SqliteConnection c, string createdBefore, string alertedBefore) => Ids(c, null,
        "SELECT game_id FROM hub_reports WHERE status = 'pending' AND created_at < $created " +
        "AND (alerted_at IS NULL OR alerted_at < $alerted) ORDER BY created_at",
        ("$created", createdBefore), ("$alerted", alertedBefore));

    public static void MarkAlerted(SqliteConnection c, string gameId, string now) =>
        Exec(c, null, "UPDATE hub_reports SET alerted_at = $now WHERE game_id = $g", ("$now", now), ("$g", gameId));

    public static HubSummary Summary(SqliteConnection c)
    {
        var counts = new Dictionary<string, int> { ["pending"] = 0, ["sent"] = 0, ["failed"] = 0 };
        using (var cmd = Command(c, null, "SELECT status, COUNT(*) FROM hub_reports GROUP BY status"))
        using (var r = cmd.ExecuteReader())
            while (r.Read()) counts[r.GetString(0)] = r.GetInt32(1);
        var failed = new List<HubFailure>();
        using (var cmd = Command(c, null, "SELECT game_id, last_status, last_error FROM hub_reports " +
                                          "WHERE status = 'failed' ORDER BY created_at DESC LIMIT 20"))
        using (var r = cmd.ExecuteReader())
            while (r.Read())
                failed.Add(new HubFailure(r.GetString(0), r.IsDBNull(1) ? null : r.GetInt32(1), r.IsDBNull(2) ? null : r.GetString(2)));
        using var oldest = Command(c, null, "SELECT MIN(created_at) FROM hub_reports WHERE status = 'pending'");
        return new HubSummary(counts, failed, oldest.ExecuteScalar() as string);
    }

    public static List<string> FailedIds(SqliteConnection c) =>
        Ids(c, null, "SELECT game_id FROM hub_reports WHERE status = 'failed' ORDER BY created_at");

    /// One conditional statement (no read-then-update); clearing lease_id
    /// drops the answer of any send still in flight.
    public static string ResetForRetry(SqliteConnection c, string gameId, string now)
    {
        if (Exec(c, null, "UPDATE hub_reports SET status = 'pending', lease_id = NULL, next_attempt_at = $now " +
                          "WHERE game_id = $g AND status <> 'sent'", ("$now", now), ("$g", gameId)) == 1)
            return "pending";
        return Exists(c, null, gameId) ? "already sent" : "no report";   // only to say why nothing changed
    }

    /// Rebuild from the stored game and replace the row only when the build
    /// succeeds; a sent row and a report whose game is gone are never touched
    /// (spec 6.4). The replacing statement clears lease_id.
    public static string Rebuild(Db db, string gameId, string? publicOrigin, string now)
    {
        using var c = db.Open();
        using var tx = c.BeginTransaction();
        var g = GameStore.Load(c, gameId, tx);
        if (g is null) return "game deleted";
        var built = HubReport.Build(g, HubReport.BotPlayedSeats(c, tx, gameId, null), publicOrigin);
        if (built.Body is null) return $"unbuildable: {built.Skip}";
        int changed = Exec(c, tx,
            "INSERT INTO hub_reports(game_id, body, status, attempts, next_attempt_at, created_at) " +
            "VALUES ($g, $b, 'pending', 0, $now, $now) " +
            "ON CONFLICT(game_id) DO UPDATE SET body = excluded.body, status = 'pending', attempts = 0, " +
            "next_attempt_at = excluded.next_attempt_at, last_status = NULL, last_error = NULL, sent_at = NULL, " +
            "alerted_at = NULL, lease_id = NULL WHERE hub_reports.status <> 'sent'",
            ("$g", gameId), ("$b", built.Body), ("$now", now));
        if (changed == 0) return "already sent";
        tx.Commit();
        return "rebuilt";
    }
}
```

`ILogger` comes from `Microsoft.Extensions.Logging`, which is an implicit using in the Web SDK. If the build complains, add `using Microsoft.Extensions.Logging;`.

- [ ] **Step 4: Run the tests and check they pass**

Run: `make test FILTER="FullyQualifiedName~HubOutboxTests"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add server/AzulServer/Hub/HubOutbox.cs server/AzulServer.Tests/Hub/HubOutboxTests.cs
git commit -m "server: hub_reports outbox with lease-owned outcomes, reconcile and rebuild"   # + trailer block
```

---

### Task 6: Queue the report atomically with the finishing move

**Files:**
- Create: `server/AzulServer/Hub/HubSignal.cs`
- Create: `server/AzulServer/Hub/HubReportBuilder.cs`
- Modify: `server/AzulServer/Games/GameService.cs` (constructor, `Start`, `Mutate`, `Delete`)
- Modify: `server/AzulServer/Program.cs` (register `HubSignal`, `IHubReportBuilder`)
- Modify: `server/AzulServer.Tests/Play.cs` (add `ToEnd`)
- Create: `server/AzulServer.Tests/Hub/HubPlay.cs` (deterministic penultimate states)
- Test: `server/AzulServer.Tests/Hub/HubFlowTests.cs`

**Interfaces:**
- Consumes: `HubReport.Build`, `HubReport.BotPlayedSeats`, `HubOutbox.Queue`, `HubOutbox.Exists`, `HubOutbox.QueueMissing`, `BotIdentity` (DI), `AzulOptions.PublicOrigin`.
- Produces:
  - `HubSignal.Wake()` and `HubSignal.WaitAsync(TimeSpan, CancellationToken) -> Task<bool>`
  - `IHubReportBuilder.Build(GameRecord, ISet<int>, string?) -> HubBuild` and the default `HubReportBuilder`. This is the test seam for a real build exception.
  - `Delete` answers 409 `hub-report-pending`
  - `Play.ToEnd(HttpClient, string id, int seconds = 120) -> Task<GameView>`
  - Test helpers (used again in Task 7):
    - `HubPlay.Penultimate(int players, int seed) -> (GameSnapshot State, Move Finishing, int Seat)`
    - `HubPlay.Insert(Db, string id, SeatRecord[] seats, GameSnapshot state, string creator, bool tracked = true)`, which inserts at version 5
    - `HubPlay.Request(long version, Move m) -> MoveRequest`
    - `HubPlay.Body(long version, Move m) -> object`

- [ ] **Step 1: Write the helpers and the failing tests**

Add to `server/AzulServer.Tests/Play.cs`, inside `Play`:

```csharp
    /// Plays `c`'s turns with any legal move until the game is over (bots on).
    public static async Task<GameView> ToEnd(HttpClient c, string id, int seconds = 120)
    {
        var until = DateTime.UtcNow.AddSeconds(seconds);
        while (true)
        {
            var v = await WaitFor(c, id, x => x.Status == Status.Finished || x.Legal is not null, seconds);
            if (v.Status == Status.Finished) return v;
            if (DateTime.UtcNow > until) throw new TimeoutException($"game {id} did not finish");
            await c.Post($"/api/games/{id}/moves", AnyLegal(v));  // 409 when a forced move landed first
        }
    }
```

`server/AzulServer.Tests/Hub/HubPlay.cs`:

```csharp
using Azul;
using AzulServer.Api;
using AzulServer.Data;
using AzulServer.Games;

namespace AzulServer.Tests;

/// Deterministic end-of-game states: the hub tests need to choose who makes
/// the finishing move and how (person, bot), which a live game cannot.
public static class HubPlay
{
    /// The state one greedy move before a normal finish (a full wall row),
    /// the move, and the seat that makes it. Tries seeds from `seed` up, so a
    /// seed whose greedy game stalemates is skipped deterministically.
    public static (GameSnapshot State, Move Finishing, int Seat) Penultimate(int players, int seed)
    {
        for (int s = seed; s < seed + 50; s++)
        {
            var game = new Game(players, new Random(s));
            for (int step = 0; step < 2000 && !game.IsFinished; step++)
            {
                var state = game.ToSnapshot();
                var move = game.GetGreedyMove();
                var probe = Game.FromSnapshot(state, new Random(s));
                probe.Play(new Move(move, probe));   // a copy: the returned move is never one that was played
                if (probe.IsFinished)
                {
                    bool normal = probe.players.Any(p => Enumerable.Range(0, 5).Any(r => Enumerable.Range(0, 5).All(c => p.grid[r, c] >= 0)));
                    if (!normal) break;   // stalemate: next seed
                    return (state, new Move(move, Game.FromSnapshot(state)), state.ActivePlayer);
                }
                game.Play(move);
            }
        }
        throw new InvalidOperationException($"no normal finish for seeds {seed}..{seed + 49}");
    }

    /// A playing, hub-tracked game at version 5 with no stored moves.
    public static void Insert(Db db, string id, SeatRecord[] seats, GameSnapshot state, string creator, bool tracked = true)
    {
        var g = new GameRecord(id, creator, Status.Playing, seats.Length, 5, Json.Serialize(state), null,
            "2026-10-04T18:00:00.0000000Z", "2026-10-04T18:00:00.0000000Z", seats)
        {
            StartedAt = "2026-10-04T18:00:01.0000000Z",
            HubTracked = tracked,
            BotKey = "mcts@0123456789ab",
        };
        using var c = db.Open();
        using var tx = c.BeginTransaction();
        GameStore.Insert(c, tx, g);
        tx.Commit();
    }

    static bool IsTake(Move m) => m.colIdx[0] == Move.NOT_SET;

    public static MoveRequest Request(long version, Move m) => IsTake(m)
        ? new MoveRequest(version, Guid.NewGuid().ToString(), "take", m.factoryIdx, m.color, m.row, null)
        : new MoveRequest(version, Guid.NewGuid().ToString(), "wall", null, null, null, (int[])m.colIdx.Clone());

    /// The same move as an HTTP body for POST /api/games/{id}/moves.
    public static object Body(long version, Move m) => IsTake(m)
        ? new { version, requestId = Guid.NewGuid().ToString(), kind = "take", factory = m.factoryIdx, color = m.color, row = m.row }
        : new { version, requestId = Guid.NewGuid().ToString(), kind = "wall", columns = (int[])m.colIdx.Clone() };
}
```

`server/AzulServer.Tests/Hub/HubFlowTests.cs`:

```csharp
using System.Net;
using System.Text.Json.Nodes;
using AzulServer.Data;
using AzulServer.Games;
using AzulServer.Hub;
using Microsoft.Data.Sqlite;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using static AzulServer.Tests.HubFixtures;

namespace AzulServer.Tests;

public class HubFlowTests
{
    const string Ann = "ann@example.com", Cy = "cy@example.com";

    static AzulOptions Fast() => new() { BotWorkers = 1, MinMoveDelaySeconds = 0, SweepSeconds = 0.2 };

    /// SQL NULL comes back as null (not DBNull.Value), so Assert.Null means it.
    static object? Scalar(TestApp app, string sql)
    {
        using var c = app.Service<Db>().Open();
        using var cmd = c.CreateCommand();
        cmd.CommandText = sql;
        var v = cmd.ExecuteScalar();
        return v is DBNull ? null : v;
    }

    static void Exec(TestApp app, string sql)
    {
        using var c = app.Service<Db>().Open();
        using var cmd = c.CreateCommand();
        cmd.CommandText = sql;
        cmd.ExecuteNonQuery();
    }

    static JsonObject Report(TestApp app, string id)
    {
        using var c = app.Service<Db>().Open();
        return JsonNode.Parse(HubOutbox.Get(c, null, id)!.Body)!.AsObject();
    }

    /// `finisher` holds `seat`; every other seat is `other(i)`.
    static SeatRecord[] Seats(int players, int seat, SeatRecord finisher, Func<int, SeatRecord> other) =>
        Enumerable.Range(0, players).Select(i => i == seat ? finisher : other(i)).ToArray();

    [Fact]
    public async Task StartStampsTheHubFields()
    {
        using var app = new TestApp();
        var alice = app.Client("alice@example.com");
        var id = await Play.Started(2, alice);
        using var c = app.Service<Db>().Open();
        var g = GameStore.Load(c, id)!;
        Assert.True(g.HubTracked);
        Assert.NotNull(g.StartedAt);
        Assert.Equal(app.Service<BotIdentity>().Key, g.BotKey);
        Assert.Null(g.FinishedAt);
    }

    [Fact]
    public async Task ALiveGameQueuesExactlyOneReport()
    {
        using var app = new TestApp(Fast());
        var alice = app.Client("alice@example.com");
        var id = await Play.Started(2, alice);
        await Play.ToEnd(alice, id);
        Assert.Equal(1L, Scalar(app, $"SELECT COUNT(*) FROM hub_reports WHERE game_id = '{id}'"));
        Assert.NotNull(Scalar(app, $"SELECT finished_at FROM games WHERE id = '{id}'"));
        var r = Report(app, id);
        Assert.Equal(id, (string?)r["external_id"]);
        Assert.Equal("free-wall-2p", (string?)r["variant"]);
        Assert.Equal("alice@example.com", (string?)r["players"]![0]!["email"]);
        Assert.Equal(app.Service<BotIdentity>().Key, (string?)r["players"]![1]!["bot_key"]);
    }

    [Fact]
    public async Task ABotFinishingForAPersonMakesItUnrated()
    {
        // The person hands over just before the last move, so no bot move of
        // theirs is stored: only the pending finishing move says a bot played
        // for them. Leaving commit.Move out of BotPlayedSeats fails this test.
        using var app = new TestApp();   // bots off: the test plays the bot's move itself
        var (state, finishing, seat) = HubPlay.Penultimate(2, seed: 7);
        HubPlay.Insert(app.Service<Db>(), "botfin", Seats(2, seat, Human(seat, Ann), i => Human(i, Cy)), state, creator: Cy);
        Assert.Equal(HttpStatusCode.OK, (await app.Client(Ann).Post($"/api/games/botfin/seats/{seat}/to-bot")).StatusCode);
        Assert.Equal(0L, Scalar(app, "SELECT COUNT(*) FROM moves WHERE game_id = 'botfin'"));
        var res = await app.Service<GameService>().ApplyServerMove("botfin", 6, finishing, "bot");
        Assert.Equal(200, res.Status);
        Assert.Equal("finished", Scalar(app, "SELECT status FROM games WHERE id = 'botfin'"));
        var r = Report(app, "botfin");
        Assert.False((bool)r["rated"]!);
        Assert.Equal(HubReport.BotPlayedReason, (string?)r["unrated_reason"]);
        Assert.Equal(("human", Ann), ((string?)r["players"]![seat]!["kind"], (string?)r["players"]![seat]!["email"]));
    }

    [Fact]
    public async Task APersonFinishingStaysRated()
    {
        using var app = new TestApp();
        var (state, finishing, seat) = HubPlay.Penultimate(2, seed: 7);
        HubPlay.Insert(app.Service<Db>(), "clean", Seats(2, seat, Human(seat, Ann), i => Human(i, Cy)), state, creator: Ann);
        var res = await app.Service<GameService>().Move("clean", Ann, HubPlay.Request(5, finishing));
        Assert.Equal(200, res.Status);
        Assert.True((bool)Report(app, "clean")["rated"]!);
    }

    [Fact]
    public async Task APersonFinishingAfterABotPlayedAnotherPersonsSeatIsUnrated()
    {
        using var app = new TestApp();
        var (state, finishing, seat) = HubPlay.Penultimate(2, seed: 7);
        int other = 1 - seat;
        HubPlay.Insert(app.Service<Db>(), "mixed", Seats(2, seat, Human(seat, Ann), i => Bot(i, Cy)), state, creator: Ann);
        Exec(app, $"INSERT INTO moves(game_id, version, seat, actor, move_json, at) VALUES ('mixed', 3, {other}, 'bot', '{{}}', 't')");
        var res = await app.Service<GameService>().Move("mixed", Ann, HubPlay.Request(5, finishing));
        Assert.Equal(200, res.Status);
        var r = Report(app, "mixed");
        Assert.False((bool)r["rated"]!);
        Assert.Equal(HubReport.BotPlayedReason, (string?)r["unrated_reason"]);
    }

    [Fact]
    public async Task AnUntrackedGameQueuesNothing()
    {
        using var app = new TestApp();
        var (state, finishing, seat) = HubPlay.Penultimate(2, seed: 7);
        HubPlay.Insert(app.Service<Db>(), "old", Seats(2, seat, Human(seat, Ann), i => Bot(i)), state, creator: Ann, tracked: false);
        Assert.Equal(200, (await app.Service<GameService>().Move("old", Ann, HubPlay.Request(5, finishing))).Status);
        Assert.Equal("finished", Scalar(app, "SELECT status FROM games WHERE id = 'old'"));
        Assert.Equal(0L, Scalar(app, "SELECT COUNT(*) FROM hub_reports"));
        Assert.Equal(HttpStatusCode.NoContent, (await app.Client(Ann).DeleteAsync("/api/games/old")).StatusCode);
    }

    [Fact]
    public async Task AFailedOutboxInsertRollsBackTheFinishingMove()
    {
        using var app = new TestApp();
        var (state, finishing, seat) = HubPlay.Penultimate(2, seed: 7);
        HubPlay.Insert(app.Service<Db>(), "atomic", Seats(2, seat, Human(seat, Ann), i => Bot(i)), state, creator: Ann);
        // The failure is the outbox INSERT itself, inside the finishing transaction.
        Exec(app, "CREATE TRIGGER refuse_reports BEFORE INSERT ON hub_reports BEGIN SELECT RAISE(ABORT, 'simulated outbox failure'); END;");
        var games = app.Service<GameService>();
        await Assert.ThrowsAsync<SqliteException>(() => games.Move("atomic", Ann, HubPlay.Request(5, finishing)));
        Assert.Equal((5L, "playing", 0L, 0L), ((long)Scalar(app, "SELECT version FROM games WHERE id = 'atomic'")!,
            (string)Scalar(app, "SELECT status FROM games WHERE id = 'atomic'")!,
            (long)Scalar(app, "SELECT COUNT(*) FROM moves WHERE game_id = 'atomic'")!,
            (long)Scalar(app, "SELECT COUNT(*) FROM hub_reports")!));
        Assert.Null(Scalar(app, "SELECT finished_at FROM games WHERE id = 'atomic'"));

        Exec(app, "DROP TRIGGER refuse_reports");
        Assert.Equal(200, (await games.Move("atomic", Ann, HubPlay.Request(5, finishing))).Status);
        Assert.Equal((6L, "finished", 1L, 1L), ((long)Scalar(app, "SELECT version FROM games WHERE id = 'atomic'")!,
            (string)Scalar(app, "SELECT status FROM games WHERE id = 'atomic'")!,
            (long)Scalar(app, "SELECT COUNT(*) FROM moves WHERE game_id = 'atomic'")!,
            (long)Scalar(app, "SELECT COUNT(*) FROM hub_reports WHERE game_id = 'atomic'")!));
    }

    sealed class ThrowingBuilder : IHubReportBuilder
    {
        public HubBuild Build(GameRecord g, ISet<int> botPlayedSeats, string? publicOrigin) =>
            throw new InvalidOperationException("simulated report bug");
    }

    [Fact]
    public async Task ABuildExceptionCommitsTheMoveAndReconcileRecovers()
    {
        var log = new ListLogger<GameService>();
        using var app = new TestApp(null, s =>
        {
            s.AddSingleton<IHubReportBuilder, ThrowingBuilder>();
            s.AddSingleton<ILogger<GameService>>(log);
        });
        var (state, finishing, seat) = HubPlay.Penultimate(2, seed: 7);
        HubPlay.Insert(app.Service<Db>(), "bug", Seats(2, seat, Human(seat, Ann), i => Bot(i)), state, creator: Ann);
        Assert.Equal(200, (await app.Service<GameService>().Move("bug", Ann, HubPlay.Request(5, finishing))).Status);
        Assert.Equal("finished", Scalar(app, "SELECT status FROM games WHERE id = 'bug'"));
        Assert.Equal(0L, Scalar(app, "SELECT COUNT(*) FROM hub_reports"));
        Assert.Equal(1, log.Count(LogLevel.Error, "could not be built"));

        var refused = await app.Client(Ann).DeleteAsync("/api/games/bug");
        Assert.Equal(HttpStatusCode.Conflict, refused.StatusCode);
        Assert.Contains("hub-report-pending", await refused.Content.ReadAsStringAsync());

        // Reconcile uses the real builder (the bug "fixed").
        Assert.Equal(1, HubOutbox.QueueMissing(app.Service<Db>(), null, DateTime.UtcNow.ToString("O"), NullLogger.Instance));
        Assert.Equal(HttpStatusCode.NoContent, (await app.Client(Ann).DeleteAsync("/api/games/bug")).StatusCode);
        Assert.Equal(1L, Scalar(app, "SELECT COUNT(*) FROM hub_reports WHERE game_id = 'bug'"));
    }

    [Fact]
    public async Task ADeletedLobbyGameNeedsNoReport()
    {
        using var app = new TestApp();
        var alice = app.Client("alice@example.com");
        var g = await Play.Create(alice, 2);
        Assert.Equal(HttpStatusCode.NoContent, (await alice.DeleteAsync($"/api/games/{g.Id}")).StatusCode);
    }
}
```

`ListLogger<T>` is from Task 4's kit, and `ApiResult.Status` is the existing record property. If `HubPlay.Penultimate(2, 7)` ever lands on a forced take (only the floor is legal), `GameService.Move` still accepts it: step 4 checks only the seat and email.

- [ ] **Step 2: Run them and check they fail**

Run: `make test FILTER="FullyQualifiedName~HubFlowTests"`
Expected: build errors (no `IHubReportBuilder`), or failures (no report row, `BotKey` null).

- [ ] **Step 3: Implement**

`server/AzulServer/Hub/HubSignal.cs`:

```csharp
namespace AzulServer.Hub;

/// "There may be something to send": GameService wakes the sender after
/// queueing a report, so delivery does not wait for the 60 s idle cycle.
public sealed class HubSignal
{
    readonly SemaphoreSlim signal = new(0, 1);

    public void Wake()
    {
        try { signal.Release(); }
        catch (SemaphoreFullException) { }  // already awake
    }

    public Task<bool> WaitAsync(TimeSpan timeout, CancellationToken ct) => signal.WaitAsync(timeout, ct);
}
```

`server/AzulServer/Hub/HubReportBuilder.cs`:

```csharp
using AzulServer.Data;

namespace AzulServer.Hub;

/// GameService builds reports through this, so a test can make the build
/// throw for real (spec 10); production uses HubReport.Build unchanged.
public interface IHubReportBuilder
{
    HubBuild Build(GameRecord g, ISet<int> botPlayedSeats, string? publicOrigin);
}

public sealed class HubReportBuilder : IHubReportBuilder
{
    public HubBuild Build(GameRecord g, ISet<int> botPlayedSeats, string? publicOrigin) =>
        HubReport.Build(g, botPlayedSeats, publicOrigin);
}
```

In `server/AzulServer/Games/GameService.cs`:

1. Add `using AzulServer.Hub;`. `IFaultInjector` is unchanged.
2. Change the constructor to:

```csharp
public sealed class GameService(Db db, EventHub hub, ServerMoveQueue queue, IFaultInjector faults,
    TimeProvider time, ILogger<GameService> log, AzulOptions options, BotIdentity bot, HubSignal hubSignal,
    IHubReportBuilder reports)
```

3. In `Start`, replace the returned `Commit` with:

```csharp
        return new Commit(g with
        {
            Status = Status.Playing,
            StateJson = Json.Serialize(game.ToSnapshot()),
            Seats = g.Seats.Select(s => s.Kind == SeatKind.Open ? s with { Kind = SeatKind.Bot } : s).ToList(),
            // Hub reporting (spec 3): only games started from here on are reported,
            // under the bot key this process plays with.
            StartedAt = Now(),
            HubTracked = true,
            BotKey = bot.Key,
        });
```

4. In `Delete`, right after `if (g.Creator != viewer) return ApiResult.Error(403, "creator-only");`, add:

```csharp
            // A finished game's rows are the only inputs to its report until
            // the report exists (spec 6.1).
            if (g.Status == Status.Finished && g.HubTracked && !HubOutbox.Exists(c, null, id))
                return ApiResult.Error(409, "hub-report-pending");
```

5. In `Mutate`, replace the whole `case Commit commit:` block with:

```csharp
                case Commit commit:
                    var next = commit.Next with { Version = g.Version + 1, UpdatedAt = Now() };
                    bool finishing = next.Status == Status.Finished && g.Status != Status.Finished;
                    if (finishing) next = next with { FinishedAt = next.UpdatedAt };
                    var move = commit.Move is null ? null : commit.Move with { Version = next.Version };
                    var last = move ?? GameStore.LastMove(c, id);
                    var view = Projection.Project(next, last, viewer);
                    if (move is not null) move = move with { ResultJson = Json.Serialize(view) };
                    // Built before the transaction: a build failure is logged and the
                    // move still commits; reconcile retries it (spec 6.1).
                    string? report = finishing ? TryBuildReport(c, next, move) : null;
                    using (var tx = c.BeginTransaction())
                    {
                        GameStore.Update(c, tx, next, g.Version);
                        if (move is not null) GameStore.InsertMove(c, tx, move);
                        // Same transaction: the finish and its report commit together or not at all.
                        if (report is not null) HubOutbox.Queue(c, tx, id, report, next.UpdatedAt);
                        tx.Commit();
                    }
                    AfterCommit(id, next.Version);
                    if (report is not null) hubSignal.Wake();
                    return ApiResult.Ok(view);
```

6. Add this method below `AfterCommit`:

```csharp
    string? TryBuildReport(SqliteConnection c, GameRecord next, MoveRecord? pending)
    {
        try
        {
            var built = reports.Build(next, HubReport.BotPlayedSeats(c, null, next.Id, pending), options.PublicOrigin);
            if (built.Body is null && next.HubTracked)
                log.LogError("no hub report for game {Game}: {Reason}", next.Id, built.Skip);
            return built.Body;
        }
        catch (Exception e)
        {
            log.LogError(e, "hub report for game {Game} could not be built", next.Id);
            return null;
        }
    }
```

In `server/AzulServer/Program.cs`, after `builder.Services.AddSingleton<IFaultInjector, NoFaults>();`, add:

```csharp
builder.Services.AddSingleton<HubSignal>();
builder.Services.AddSingleton<IHubReportBuilder, HubReportBuilder>();
```

- [ ] **Step 4: Run all server tests**

Run: `make test`
Expected: PASS. If `AFailedOutboxInsertRollsBackTheFinishingMove` sees no exception, check that `HubOutbox.Queue` uses `ON CONFLICT(game_id) DO NOTHING` and not `INSERT OR IGNORE`.

- [ ] **Step 5: Commit**

```bash
git add server/AzulServer server/AzulServer.Tests
git commit -m "server: queue the hub report in the finishing commit; Delete waits for it"   # + trailer block
```

---

### Task 7: The sender (`HubSender`)

**Files:**
- Create: `server/AzulServer/Hub/HubSender.cs`
- Modify: `server/AzulServer/Program.cs`
- Test: `server/AzulServer.Tests/Hub/HubSenderTests.cs`

**Interfaces:**
- Consumes: everything from `HubOutbox` (outcome writes take the claim's `LeaseId` and return `false` when it is stale), `HubSignal`, `AzulOptions.Hub`, `AzulOptions.PublicOrigin`, and `HubPlay` from Task 6.
- Produces:
  - `HubSender.HttpName` = `"hub"`
  - `HubSender.Outcome { Sent, Failed, RetryKey, RetryTransient, RetryConfig }`
  - `HubSender.Classify(int? status) -> Outcome`
  - `HubSender.Backoff(int attempts, double jitter01) -> TimeSpan`
  - `HubSender.RunCycleAsync(CancellationToken) -> Task<int>`, the number sent
  - a hosted service registration

- [ ] **Step 1: Write the failing tests**

`server/AzulServer.Tests/Hub/HubSenderTests.cs`:

```csharp
using System.Net;
using AzulServer.Data;
using AzulServer.Games;
using AzulServer.Hub;
using Microsoft.Data.Sqlite;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Time.Testing;
using static AzulServer.Tests.HubFixtures;

namespace AzulServer.Tests;

public sealed class HubSenderTests : IDisposable
{
    readonly Db db = NewDb();
    readonly FakeHub hubHttp = new();
    readonly FakeTimeProvider time = new(DateTimeOffset.Parse("2026-10-04T19:00:00Z"));
    readonly ListLogger<HubSender> log = new();
    public void Dispose() => SqliteConnection.ClearAllPools();

    static readonly HubOptions Configured = new() { Url = "http://hub.test", Key = "phk_test" };

    HubSender Sender(HubOptions? hub = null) =>
        new(db, new AzulOptions { Hub = hub ?? Configured }, new HubSignal(), hubHttp, time, log);

    string Now => time.GetUtcNow().UtcDateTime.ToString("O");

    void QueueRow(string id, string body = "{\"external_id\":\"x\",\"é\":\"ü\"}")
    {
        using var c = db.Open();
        using var tx = c.BeginTransaction();
        HubOutbox.Queue(c, tx, id, body, Now);
        tx.Commit();
    }

    HubRow Row(string id)
    {
        using var c = db.Open();
        return HubOutbox.Get(c, null, id)!;
    }

    [Theory]
    [InlineData(200, HubSender.Outcome.Sent)]
    [InlineData(201, HubSender.Outcome.Sent)]
    [InlineData(409, HubSender.Outcome.Failed)]
    [InlineData(422, HubSender.Outcome.Failed)]
    [InlineData(401, HubSender.Outcome.RetryKey)]
    [InlineData(500, HubSender.Outcome.RetryTransient)]
    [InlineData(503, HubSender.Outcome.RetryTransient)]
    [InlineData(null, HubSender.Outcome.RetryTransient)]
    [InlineData(400, HubSender.Outcome.RetryConfig)]
    [InlineData(403, HubSender.Outcome.RetryConfig)]
    [InlineData(404, HubSender.Outcome.RetryConfig)]
    [InlineData(413, HubSender.Outcome.RetryConfig)]
    public void StatusesMapToTheHubContract(int? status, HubSender.Outcome expected) =>
        Assert.Equal(expected, HubSender.Classify(status));

    [Theory]
    [InlineData(1, 60)]
    [InlineData(2, 120)]
    [InlineData(6, 1920)]
    [InlineData(7, 3600)]
    [InlineData(30, 3600)]
    public void BackoffDoublesToAnHourPlusTenPercentJitter(int attempts, int baseSeconds)
    {
        Assert.Equal(TimeSpan.FromSeconds(baseSeconds), HubSender.Backoff(attempts, 0));
        Assert.Equal(TimeSpan.FromSeconds(baseSeconds * 1.1), HubSender.Backoff(attempts, 1));
    }

    [Fact]
    public async Task TheStoredBodyIsSentByteForByteWithBothHeaders()
    {
        QueueRow("g");
        Assert.Equal(1, await Sender().RunCycleAsync(default));
        var (req, body) = Assert.Single(hubHttp.Requests);
        Assert.Equal(HttpMethod.Post, req.Method);
        Assert.Equal("http://hub.test/api/v1/results", req.RequestUri!.ToString());
        Assert.Equal("{\"external_id\":\"x\",\"é\":\"ü\"}", body);
        Assert.Equal("application/json", req.Content!.Headers.ContentType!.MediaType);
        Assert.Equal(("Bearer", "phk_test"), (req.Headers.Authorization!.Scheme, req.Headers.Authorization.Parameter));
        Assert.Equal(("sent", 1, 201), (Row("g").Status, Row("g").Attempts, Row("g").LastStatus));
    }

    [Fact]
    public async Task AnyTwoHundredIsSentWhateverTheBody()
    {
        hubHttp.Respond = () => new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent("<html>proxy</html>") };
        QueueRow("g");
        await Sender().RunCycleAsync(default);
        Assert.Equal("sent", Row("g").Status);
    }

    [Fact]
    public async Task UnprocessableIsPermanentWithItsBodyKept()
    {
        hubHttp.Respond = () => new HttpResponseMessage(HttpStatusCode.UnprocessableEntity)
            { Content = new StringContent("{\"error\":\"players: must be 1-12\"}" + new string('x', 600)) };
        QueueRow("g");
        await Sender().RunCycleAsync(default);
        var row = Row("g");
        Assert.Equal(("failed", 422), (row.Status, row.LastStatus));
        Assert.StartsWith("{\"error\":\"players", row.LastError);
        Assert.Equal(500, row.LastError!.Length);
        Assert.Equal(1, log.Count(LogLevel.Error, "permanently"));
        await Sender().RunCycleAsync(default);
        Assert.Single(hubHttp.Requests);   // never resent
    }

    [Fact]
    public async Task AWrongKeyStaysPendingWithBackoffAndAnError()
    {
        hubHttp.Respond = () => new HttpResponseMessage(HttpStatusCode.Unauthorized);
        QueueRow("g");
        await Sender().RunCycleAsync(default);
        var row = Row("g");
        Assert.Equal(("pending", 1, 401), (row.Status, row.Attempts, row.LastStatus));
        var next = DateTimeOffset.Parse(row.NextAttemptAt) - time.GetUtcNow();
        Assert.InRange(next.TotalSeconds, 60, 66);
        Assert.Equal(1, log.Count(LogLevel.Error, "AZUL_HUB_KEY"));
        await Sender().RunCycleAsync(default);
        Assert.Single(hubHttp.Requests);   // not due yet
    }

    [Fact]
    public async Task ANetworkErrorStaysPendingWithAWarning()
    {
        hubHttp.Throw = new HttpRequestException("connection refused");
        QueueRow("g");
        await Sender().RunCycleAsync(default);
        var row = Row("g");
        Assert.Equal(("pending", (int?)null, "connection refused"), (row.Status, row.LastStatus, row.LastError));
        Assert.Equal(1, log.Count(LogLevel.Warning, "unavailable"));
    }

    [Fact]
    public async Task AConfigFaultStaysPendingWithAnError()
    {
        hubHttp.Respond = () => new HttpResponseMessage(HttpStatusCode.NotFound);
        QueueRow("g");
        await Sender().RunCycleAsync(default);
        Assert.Equal(("pending", 404), (Row("g").Status, Row("g").LastStatus));
        Assert.Equal(1, log.Count(LogLevel.Error, "AZUL_HUB_URL"));
    }

    [Fact]
    public async Task AStuckRowAlertsOnceAnHour()
    {
        hubHttp.Respond = () => new HttpResponseMessage(HttpStatusCode.ServiceUnavailable);
        QueueRow("g");
        var sender = Sender();
        await sender.RunCycleAsync(default);
        time.Advance(TimeSpan.FromHours(2));
        await sender.RunCycleAsync(default);
        time.Advance(TimeSpan.FromMinutes(10));
        await sender.RunCycleAsync(default);
        Assert.Equal(1, log.Count(LogLevel.Error, "over an hour"));
        time.Advance(TimeSpan.FromHours(1));
        await sender.RunCycleAsync(default);
        Assert.Equal(2, log.Count(LogLevel.Error, "over an hour"));
    }

    [Fact]
    public async Task ReconcileRunsAtTheStartOfACycle()
    {
        using (var c = db.Open())
        using (var tx = c.BeginTransaction())
        {
            GameStore.Insert(c, tx, Finished("lost", [Human(0, "alice@example.com"), Bot(1)]));
            tx.Commit();
        }
        Assert.Equal(1, await Sender().RunCycleAsync(default));
        Assert.Contains("\"external_id\":\"lost\"", Assert.Single(hubHttp.Requests).Body);
    }

    [Fact]
    public async Task ReportOfADeletedGameIsStillSent()
    {
        using (var c = db.Open())
        using (var tx = c.BeginTransaction())
        {
            GameStore.Insert(c, tx, Finished("del", [Human(0, "alice@example.com"), Bot(1)]));
            HubOutbox.Queue(c, tx, "del", "{\"external_id\":\"del\"}", Now);
            GameStore.Delete(c, tx, "del");
            tx.Commit();
        }
        await Sender().RunCycleAsync(default);
        Assert.Equal("sent", Row("del").Status);
    }

    [Fact]
    public async Task WithoutBothSettingsTheSenderStaysOff()
    {
        QueueRow("g");
        var off = Sender(new HubOptions { Url = "http://hub.test" });
        await off.StartAsync(default);
        await off.ExecuteTask!;
        Assert.Empty(hubHttp.Requests);
        Assert.Equal(1, log.Count(LogLevel.Warning, "both AZUL_HUB_URL and AZUL_HUB_KEY"));
    }

    [Fact]
    public async Task ADatabaseErrorDoesNotStopTheLoop()
    {
        var signal = new HubSignal();
        var sender = new HubSender(db, new AzulOptions { Hub = Configured }, signal, hubHttp, TimeProvider.System, log);
        using (var c = db.Open()) { using var cmd = c.CreateCommand(); cmd.CommandText = "ALTER TABLE hub_reports RENAME TO hub_reports_away"; cmd.ExecuteNonQuery(); }
        await sender.StartAsync(default);
        await WaitUntil(() => log.Count(LogLevel.Error, "cycle failed") >= 1);
        using (var c = db.Open()) { using var cmd = c.CreateCommand(); cmd.CommandText = "ALTER TABLE hub_reports_away RENAME TO hub_reports"; cmd.ExecuteNonQuery(); }
        QueueRow("g");
        signal.Wake();
        await WaitUntil(() => hubHttp.Requests.Count == 1);
        await sender.StopAsync(default);
    }

    static async Task WaitUntil(Func<bool> done, int seconds = 10)
    {
        var until = DateTime.UtcNow.AddSeconds(seconds);
        while (!done())
        {
            if (DateTime.UtcNow > until) throw new TimeoutException();
            await Task.Delay(20);
        }
    }

    [Fact]
    public async Task ARebuildDuringASendWinsOverTheOldOutcome()
    {
        using (var c = db.Open())
        using (var tx = c.BeginTransaction())
        {
            GameStore.Insert(c, tx, Finished("g", [Human(0, "alice@example.com"), Bot(1)]));
            HubOutbox.Queue(c, tx, "g", "{\"old\":1}", Now);
            tx.Commit();
        }
        hubHttp.Gate = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var sender = Sender();
        var cycle = sender.RunCycleAsync(default);
        await WaitUntil(() => hubHttp.Requests.Count == 1);   // the old body is in flight
        Assert.Equal("{\"old\":1}", hubHttp.Requests[0].Body);
        Assert.Equal("rebuilt", HubOutbox.Rebuild(db, "g", null, Now));
        hubHttp.Gate.SetResult();   // the hub now answers 201 to the OLD body
        await cycle;
        var row = Row("g");
        Assert.Equal(("pending", (string?)null), (row.Status, row.LeaseId));
        Assert.Contains("\"external_id\":\"g\"", row.Body);
        Assert.Equal(1, log.Count(LogLevel.Information, "stale hub outcome"));

        hubHttp.Gate = null;
        Assert.Equal(1, await sender.RunCycleAsync(default));   // the new body goes out fresh
        Assert.Equal(2, hubHttp.Requests.Count);
        Assert.Contains("\"external_id\":\"g\"", hubHttp.Requests[1].Body);
        Assert.Equal("sent", Row("g").Status);
    }

    // ---------- through the real app ----------

    [Fact]
    public async Task ASenderThatCannotReachItsDatabaseDoesNotStopPlay()
    {
        var senderLog = new ListLogger<HubSender>();
        var fake = new FakeHub();
        var notADirectory = Path.GetTempFileName();   // every Db.Open() below it throws
        using var app = new TestApp(new AzulOptions { BotWorkers = 0, Hub = Configured }, s =>
        {
            s.AddHttpClient(HubSender.HttpName).ConfigurePrimaryHttpMessageHandler(() => fake);
            s.Remove(s.Single(d => d.ServiceType == typeof(IHostedService) && d.ImplementationType == typeof(HubSender)));
            s.AddHostedService(sp => new HubSender(new Db(new AzulOptions { DataDir = notADirectory }),
                sp.GetRequiredService<AzulOptions>(), sp.GetRequiredService<HubSignal>(),
                sp.GetRequiredService<IHttpClientFactory>(), TimeProvider.System, senderLog));
        });
        const string ann = "ann@example.com";
        var (state, finishing, seat) = HubPlay.Penultimate(2, seed: 7);
        var seats = Enumerable.Range(0, 2).Select(i => i == seat ? Human(i, ann) : Bot(i)).ToArray();
        HubPlay.Insert(app.Service<Db>(), "play", seats, state, creator: ann);
        var client = app.Client(ann);

        var res = await client.Post("/api/games/play/moves", HubPlay.Body(5, finishing));
        Assert.Equal(HttpStatusCode.OK, res.StatusCode);
        await WaitUntil(() => senderLog.Count(LogLevel.Error, "cycle failed") >= 1);
        Assert.Equal(Status.Finished, (await Play.Get(client, "play")).Status);
        Assert.Equal(HttpStatusCode.OK, (await client.GetAsync("/api/health")).StatusCode);
        Assert.Equal(HttpStatusCode.OK, (await client.GetAsync("/api/games")).StatusCode);
        using var c = app.Service<Db>().Open();
        Assert.Equal("pending", HubOutbox.Get(c, null, "play")!.Status);   // queued by the move, waiting
        Assert.Empty(fake.Requests);
    }

    static TestApp Wired(FakeHub fake, string? dataDir = null) => new(
        new AzulOptions { BotWorkers = 1, MinMoveDelaySeconds = 0, SweepSeconds = 0.2, Hub = Configured },
        s => s.AddHttpClient(HubSender.HttpName).ConfigurePrimaryHttpMessageHandler(() => fake),
        dataDir);

    [Fact]
    public async Task AFinishedGameReachesTheHub()
    {
        var fake = new FakeHub();
        using var app = Wired(fake);
        var alice = app.Client("alice@example.com");
        var id = await Play.Started(2, alice);
        await Play.ToEnd(alice, id);
        await WaitUntil(() => fake.Requests.Count == 1);
        Assert.Contains($"\"external_id\":\"{id}\"", fake.Requests[0].Body);
    }

    [Fact]
    public async Task PendingRowsAreSentAfterAStart()
    {
        var dir = Directory.CreateTempSubdirectory("azul-restart-").FullName;
        var backlog = new Db(new AzulOptions { DataDir = dir });
        backlog.Migrate();
        using (var c = backlog.Open())
        using (var tx = c.BeginTransaction())
        {
            HubOutbox.Queue(c, tx, "waiting", "{\"external_id\":\"waiting\"}", DateTime.UtcNow.ToString("O"));
            tx.Commit();
        }
        var fake = new FakeHub();
        using var app = Wired(fake, dir);
        _ = app.Client();   // starts the host and its hosted services
        await WaitUntil(() => fake.Requests.Count == 1);
    }
}
```

- [ ] **Step 2: Run them and check they fail**

Run: `make test FILTER="FullyQualifiedName~HubSenderTests"`
Expected: a build failure, because `HubSender` does not exist.

- [ ] **Step 3: Implement**

`server/AzulServer/Hub/HubSender.cs`:

```csharp
using System.Net.Http.Headers;
using System.Text;
using AzulServer.Data;

namespace AzulServer.Hub;

/// Delivers hub_reports (spec 6.3). Nothing escapes ExecuteAsync: a
/// BackgroundService exception would stop the host and gameplay with it.
public sealed class HubSender(Db db, AzulOptions options, HubSignal signal, IHttpClientFactory http,
    TimeProvider time, ILogger<HubSender> log) : BackgroundService
{
    public const string HttpName = "hub";
    static readonly TimeSpan Lease = TimeSpan.FromMinutes(5);
    static readonly TimeSpan Idle = TimeSpan.FromSeconds(60);
    static readonly TimeSpan Stuck = TimeSpan.FromHours(1);

    public enum Outcome { Sent, Failed, RetryKey, RetryTransient, RetryConfig }

    /// hub 5.3: any 2xx delivered; 409/422 permanent; 401 keep and alert; 5xx
    /// and network errors keep; anything else is a configuration fault, kept.
    public static Outcome Classify(int? status) => status switch
    {
        >= 200 and < 300 => Outcome.Sent,
        409 or 422 => Outcome.Failed,
        401 => Outcome.RetryKey,
        null or >= 500 => Outcome.RetryTransient,
        _ => Outcome.RetryConfig,
    };

    public static TimeSpan Backoff(int attempts, double jitter01) =>
        TimeSpan.FromSeconds(Math.Min(60 * Math.Pow(2, attempts - 1), 3600) * (1 + 0.1 * jitter01));

    static string Stamp(DateTimeOffset t) => t.UtcDateTime.ToString("O");

    static string? Cut(string? s) => s is null ? null : s.Length <= 500 ? s : s[..500];

    protected override async Task ExecuteAsync(CancellationToken ct)
    {
        if (!options.Hub.SenderConfigured)
        {
            if (options.Hub.HalfConfigured)
                log.LogWarning("hub reporting is off: set both AZUL_HUB_URL and AZUL_HUB_KEY");
            return;
        }
        while (!ct.IsCancellationRequested)
        {
            try { await RunCycleAsync(ct); }
            catch (OperationCanceledException) when (ct.IsCancellationRequested) { return; }
            catch (Exception e) { log.LogError(e, "hub sender cycle failed"); }
            try { await signal.WaitAsync(Idle, ct); }
            catch (OperationCanceledException) { return; }
        }
    }

    /// One pass: reconcile, send every due row, alert on stuck rows.
    public async Task<int> RunCycleAsync(CancellationToken ct)
    {
        var now = time.GetUtcNow();
        HubOutbox.QueueMissing(db, options.PublicOrigin, Stamp(now), log);
        List<string> due;
        using (var c = db.Open()) due = HubOutbox.DueIds(c, Stamp(now), 50);
        int sent = 0;
        foreach (var id in due)
        {
            ct.ThrowIfCancellationRequested();
            try
            {
                if (await SendOne(id, ct)) sent++;
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested) { throw; }
            catch (Exception e)
            {
                // The lease expires and the row is retried.
                log.LogError(e, "hub report for game {Game} could not be sent", id);
            }
        }
        AlertStuck();
        return sent;
    }

    async Task<bool> SendOne(string id, CancellationToken ct)
    {
        var start = time.GetUtcNow();
        HubClaim? claim;
        using (var c = db.Open()) claim = HubOutbox.TryClaim(c, id, Stamp(start), Stamp(start + Lease));
        if (claim is null) return false;

        int? status = null;
        string? detail = null;
        try
        {
            using var req = new HttpRequestMessage(HttpMethod.Post, options.Hub.Url + "/api/v1/results")
            {
                Content = new StringContent(claim.Body, Encoding.UTF8, "application/json"),
            };
            req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", options.Hub.Key);
            using var res = await http.CreateClient(HttpName).SendAsync(req, ct);
            status = (int)res.StatusCode;
            detail = await res.Content.ReadAsStringAsync(ct);
        }
        catch (Exception e) when (e is HttpRequestException || (e is TaskCanceledException && !ct.IsCancellationRequested))
        {
            detail = e.Message;
        }

        var outcome = Classify(status);
        var after = time.GetUtcNow();
        using var c2 = db.Open();
        // Recorded only under this claim's lease: a retry or rebuild while the
        // request was in flight cleared it, and then this answer is stale.
        bool owned = outcome switch
        {
            Outcome.Sent => HubOutbox.MarkSent(c2, id, claim.LeaseId, status!.Value, Stamp(after)),
            Outcome.Failed => HubOutbox.MarkFailed(c2, id, claim.LeaseId, status!.Value, Cut(detail)),
            _ => HubOutbox.MarkRetry(c2, id, claim.LeaseId, status, Cut(detail),
                Stamp(after + Backoff(claim.Attempts + 1, Random.Shared.NextDouble()))),
        };
        if (!owned)
        {
            log.LogInformation("stale hub outcome for game {Game} dropped ({Status}): the report was reset or rebuilt meanwhile", id, status);
            return false;
        }
        switch (outcome)
        {
            case Outcome.Sent:
                log.LogInformation("hub accepted the report of game {Game} ({Status})", id, status);
                return true;
            case Outcome.Failed:
                log.LogError("hub refused the report of game {Game} permanently ({Status}): {Body}", id, status, Cut(detail));
                return false;
            default:
                if (outcome == Outcome.RetryKey)
                    log.LogError("hub rejected the game key (401) for game {Game}; reports stay pending until AZUL_HUB_KEY is fixed and the server restarted", id);
                else if (outcome == Outcome.RetryTransient)
                    log.LogWarning("hub unavailable for game {Game} ({Status}): {Detail}", id, status, Cut(detail));
                else
                    log.LogError("hub answered {Status} for game {Game}; check AZUL_HUB_URL", status, id);
                return false;
        }
    }

    void AlertStuck()
    {
        var now = time.GetUtcNow();
        using var c = db.Open();
        foreach (var id in HubOutbox.StuckIds(c, Stamp(now - Stuck), Stamp(now - Stuck)))
        {
            log.LogError("hub report for game {Game} has been pending for over an hour", id);
            HubOutbox.MarkAlerted(c, id, Stamp(now));
        }
    }
}
```

In `server/AzulServer/Program.cs`, after `builder.Services.AddHostedService<BotScheduler>();`, add:

```csharp
builder.Services.AddHttpClient(HubSender.HttpName, c => c.Timeout = TimeSpan.FromSeconds(10));
builder.Services.AddHostedService<HubSender>();
```

- [ ] **Step 4: Run all server tests**

Run: `make test`
Expected: PASS. Other tests are unaffected: without `Hub` settings the hosted sender returns at once.

- [ ] **Step 5: Commit**

```bash
git add server/AzulServer/Hub/HubSender.cs server/AzulServer/Program.cs server/AzulServer.Tests/Hub/HubSenderTests.cs
git commit -m "server: hub sender with lease, backoff, per-status outcomes and stuck alerts"   # + trailer block
```

---

### Task 8: Operator commands (`AzulServer hub ...`)

**Files:**
- Create: `server/AzulServer/Hub/HubCommands.cs`
- Modify: `server/AzulServer/Program.cs` (dispatch before `CreateBuilder`)
- Test: `server/AzulServer.Tests/Hub/HubCommandsTests.cs`

**Interfaces:**
- Consumes: `HubOutbox.Summary/FailedIds/ResetForRetry/Rebuild`, `BotIdentity.For`, `AzulOptions.FromEnvironment`.
- Produces: `HubCommands.RunAsync(string[] args, TextWriter output, Func<string, string?> env, HttpMessageHandler? handler = null) -> Task<int>`. Each subcommand prints one JSON document and returns an exit code:

| Subcommand | Output | Exit codes |
| --- | --- | --- |
| `bot-key` | `{"botKey"}` | 0 |
| `status` | `{"botKey","counts","failed","oldestPendingCreatedAt"}` | 0 |
| `retry <id>\|--all-failed [--rebuild]` | `{"results":[{"gameId","outcome"}]}` | 0 |
| `ping` | `{"ok","status"}` | 0 on 422, else 1 |
| bad usage | `{"error":"usage","usage"}` | 2 |
| any exception | `{"error":"internal","message"}` | 1 |

The `bot-key` subcommand is not in the spec. It is added so the MVID double-build check (Task 11) can read the key from a fresh image without a database.

- [ ] **Step 1: Write the failing tests**

`server/AzulServer.Tests/Hub/HubCommandsTests.cs`:

```csharp
using System.Net;
using System.Text.Json.Nodes;
using AzulServer.Data;
using AzulServer.Hub;
using Microsoft.Data.Sqlite;
using static AzulServer.Tests.HubFixtures;

namespace AzulServer.Tests;

public sealed class HubCommandsTests : IDisposable
{
    readonly string dir = Directory.CreateTempSubdirectory("azul-cmd-").FullName;
    readonly Db db;
    public HubCommandsTests()
    {
        db = new Db(new AzulOptions { DataDir = dir });
        db.Migrate();
    }
    public void Dispose() => SqliteConnection.ClearAllPools();

    Dictionary<string, string> env => new() { ["AZUL_DATA_DIR"] = dir, ["AZUL_HUB_URL"] = "http://hub.test", ["AZUL_HUB_KEY"] = "phk_test" };

    async Task<(int Code, JsonObject Out)> Run(params string[] args) => await Run(null, args);

    async Task<(int Code, JsonObject Out)> Run(HttpMessageHandler? handler, params string[] args)
    {
        var w = new StringWriter();
        var e = env;
        int code = await HubCommands.RunAsync(args, w, k => e.GetValueOrDefault(k), handler);
        return (code, JsonNode.Parse(w.ToString())!.AsObject());
    }

    void Queue(string id, string body = "{}")
    {
        using var c = db.Open();
        using var tx = c.BeginTransaction();
        HubOutbox.Queue(c, tx, id, body, "2026-10-04T19:00:00.0000000Z");
        tx.Commit();
    }

    /// Outcomes need the claim's lease (Task 5).
    void Fail(string id, int status, string error)
    {
        using var c = db.Open();
        var claim = HubOutbox.TryClaim(c, id, "2026-10-04T19:00:00.0000000Z", "2026-10-04T19:05:00.0000000Z")!;
        Assert.True(HubOutbox.MarkFailed(c, id, claim.LeaseId, status, error));
    }

    [Fact]
    public async Task BotKeyNeedsNoDatabase()
    {
        var (code, o) = await Run("bot-key");
        Assert.Equal(0, code);
        Assert.Equal(BotIdentity.For(AzulOptions.FromEnvironment(k => env.GetValueOrDefault(k))).Key, (string?)o["botKey"]);
    }

    [Fact]
    public async Task StatusCountsAndListsFailures()
    {
        Queue("a"); Queue("b");
        Fail("b", 422, "bad");
        var (code, o) = await Run("status");
        Assert.Equal(0, code);
        Assert.Equal(1, (int)o["counts"]!["pending"]!);
        Assert.Equal(1, (int)o["counts"]!["failed"]!);
        Assert.Equal("b", (string?)o["failed"]![0]!["gameId"]);
        Assert.Equal("2026-10-04T19:00:00.0000000Z", (string?)o["oldestPendingCreatedAt"]);
        Assert.NotNull(o["botKey"]);
    }

    [Fact]
    public async Task RetryOneOrAllFailed()
    {
        Queue("a"); Queue("b");
        Fail("a", 409, "x");
        Fail("b", 422, "y");
        var (code, one) = await Run("retry", "a");
        Assert.Equal(0, code);
        Assert.Equal("pending", (string?)one["results"]![0]!["outcome"]);
        var (_, all) = await Run("retry", "--all-failed");
        Assert.Equal(["b"], all["results"]!.AsArray().Select(r => (string)r!["gameId"]!));
    }

    [Fact]
    public async Task RebuildKeepsAReportWhoseGameIsGone()
    {
        using (var c = db.Open())
        using (var tx = c.BeginTransaction())
        {
            GameStore.Insert(c, tx, Finished("here", [Human(0, "alice@example.com"), Bot(1)]));
            tx.Commit();
        }
        Queue("here", "{\"old\":1}"); Queue("gone", "{\"keep\":1}");
        var (_, o) = await Run("retry", "here", "--rebuild");
        Assert.Equal("rebuilt", (string?)o["results"]![0]!["outcome"]);
        var (_, g) = await Run("retry", "gone", "--rebuild");
        Assert.Equal("game deleted", (string?)g["results"]![0]!["outcome"]);
        using var c2 = db.Open();
        Assert.Equal("{\"keep\":1}", HubOutbox.Get(c2, null, "gone")!.Body);
    }

    [Theory]
    [InlineData(422, 0)]
    [InlineData(401, 1)]
    [InlineData(500, 1)]
    public async Task PingPassesOnlyOnUnprocessable(int status, int expected)
    {
        var fake = new FakeHub { Respond = () => new HttpResponseMessage((HttpStatusCode)status) };
        var (code, o) = await Run(fake, "ping");
        Assert.Equal(expected, code);
        Assert.Equal(status, (int)o["status"]!);
        var (req, body) = Assert.Single(fake.Requests);
        Assert.Equal("{}", body);
        Assert.Equal("phk_test", req.Headers.Authorization!.Parameter);
    }

    [Fact]
    public async Task BadUsageExitsTwo()
    {
        Assert.Equal(2, (await Run("nope")).Code);
        Assert.Equal(2, (await Run("retry")).Code);
        Assert.Equal(2, (await Run()).Code);
    }
}
```

- [ ] **Step 2: Run them and check they fail**

Run: `make test FILTER="FullyQualifiedName~HubCommandsTests"`
Expected: a build failure, because `HubCommands` does not exist.

- [ ] **Step 3: Implement**

`server/AzulServer/Hub/HubCommands.cs`:

```csharp
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using AzulServer.Data;

namespace AzulServer.Hub;

/// `/app/AzulServer hub <command>` (spec 6.4): opens the SQLite database
/// directly (WAL lets it run beside the server), prints one JSON document.
public static class HubCommands
{
    const string Usage = "AzulServer hub status | bot-key | ping | retry <game_id>|--all-failed [--rebuild]";

    public static async Task<int> RunAsync(string[] args, TextWriter output, Func<string, string?> env, HttpMessageHandler? handler = null)
    {
        void Print(object value) => output.WriteLine(JsonSerializer.Serialize(value, Json.Options));
        try
        {
            var options = AzulOptions.FromEnvironment(env);
            var now = DateTime.UtcNow.ToString("O");
            switch (args)
            {
                case ["bot-key"]:
                    Print(new { botKey = BotIdentity.For(options).Key });
                    return 0;
                case ["status"]:
                {
                    using var c = new Db(options).Open();
                    var s = HubOutbox.Summary(c);
                    Print(new { botKey = BotIdentity.For(options).Key, counts = s.Counts, failed = s.Failed, oldestPendingCreatedAt = s.OldestPendingCreatedAt });
                    return 0;
                }
                case ["retry", var target, .. var rest] when rest is [] or ["--rebuild"]:
                {
                    var db = new Db(options);
                    bool rebuild = rest is ["--rebuild"];
                    List<string> ids;
                    if (target == "--all-failed") { using var c = db.Open(); ids = HubOutbox.FailedIds(c); }
                    else ids = [target];
                    var results = new List<object>();
                    foreach (var id in ids)
                    {
                        string outcome;
                        if (rebuild) outcome = HubOutbox.Rebuild(db, id, options.PublicOrigin, now);
                        else { using var c = db.Open(); outcome = HubOutbox.ResetForRetry(c, id, now); }
                        results.Add(new { gameId = id, outcome });
                    }
                    Print(new { results });
                    return 0;
                }
                case ["ping"]:
                {
                    if (!options.Hub.SenderConfigured)
                    {
                        Print(new { ok = false, status = (int?)null, message = "AZUL_HUB_URL and AZUL_HUB_KEY must both be set" });
                        return 1;
                    }
                    using var http = handler is null ? new HttpClient() : new HttpClient(handler, disposeHandler: false);
                    http.Timeout = TimeSpan.FromSeconds(10);
                    using var req = new HttpRequestMessage(HttpMethod.Post, options.Hub.Url + "/api/v1/results")
                    {
                        Content = new StringContent("{}", Encoding.UTF8, "application/json"),
                    };
                    req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", options.Hub.Key);
                    using var res = await http.SendAsync(req);
                    int status = (int)res.StatusCode;
                    // 422 = reached, key accepted, content type passed, empty report refused.
                    Print(new { ok = status == 422, status });
                    return status == 422 ? 0 : 1;
                }
                default:
                    Print(new { error = "usage", usage = Usage });
                    return 2;
            }
        }
        catch (Exception e)
        {
            Print(new { error = "internal", message = e.Message });
            return 1;
        }
    }
}
```

In `server/AzulServer/Program.cs`, directly below the `--healthcheck` early return, add:

```csharp
if (args is ["hub", ..])
    return await HubCommands.RunAsync(args[1..], Console.Out, Environment.GetEnvironmentVariable);
```

`using AzulServer.Hub;` is already there from Task 2.

- [ ] **Step 4: Run the tests and check they pass**

Run: `make test FILTER="FullyQualifiedName~HubCommandsTests"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add server/AzulServer/Hub/HubCommands.cs server/AzulServer/Program.cs server/AzulServer.Tests/Hub/HubCommandsTests.cs
git commit -m "server: AzulServer hub status|retry|ping|bot-key"   # + trailer block
```

---

### Task 9: `/api/me` hubUrl and the header links

**Files:**
- Modify: `server/AzulServer/Api/ApiEndpoints.cs:17`
- Modify: `web/src/lib/api.ts:39`
- Modify: `web/src/App.svelte`
- Modify: `web/src/components/Lobby.test.ts:16`
- Modify: `web/src/lib/events.test.ts:22,44,71`
- Test: `server/AzulServer.Tests/Hub/HubFlowTests.cs` (append)
- Test (create): `web/src/App.test.ts`

**Interfaces:**
- Consumes: `AzulOptions.Hub.PublicUrl`.
- Produces: `GET /api/me` returns `{ email, hubUrl }`; `api.me(): Promise<{ email: string; hubUrl: string | null }>`.

- [ ] **Step 1: Write the failing tests**

Append to `HubFlowTests`:

```csharp
    [Fact]
    public async Task MeNamesTheHubWhenConfigured()
    {
        using var on = new TestApp(new AzulOptions { BotWorkers = 0, Hub = new HubOptions { PublicUrl = "https://play.example" } });
        var me = JsonNode.Parse(await on.Client().GetStringAsync("/api/me"))!;
        Assert.Equal(("alice@example.com", "https://play.example"), ((string?)me["email"], (string?)me["hubUrl"]));
        using var off = new TestApp();
        Assert.Null(JsonNode.Parse(await off.Client().GetStringAsync("/api/me"))!["hubUrl"]);
    }
```

`web/src/App.test.ts`:

```ts
import { render } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import App from './App.svelte';
import { api } from './lib/api';

afterEach(() => vi.restoreAllMocks());

describe('App header', () => {
  it('links to Playhub and the Azul leaderboard when the server names the hub', async () => {
    vi.spyOn(api, 'games').mockResolvedValue([]);
    vi.spyOn(api, 'me').mockResolvedValue({ email: 'me@x', hubUrl: 'https://play.example' });
    const { findByRole } = render(App);
    expect((await findByRole('link', { name: 'Playhub' })).getAttribute('href')).toBe('https://play.example');
    expect((await findByRole('link', { name: 'Leaderboard' })).getAttribute('href')).toBe('https://play.example/games/azul');
  });

  it('shows no hub links without a hub', async () => {
    vi.spyOn(api, 'games').mockResolvedValue([]);
    vi.spyOn(api, 'me').mockResolvedValue({ email: 'me@x', hubUrl: null });
    const { findAllByText, queryByRole } = render(App);
    await findAllByText('me@x');
    expect(queryByRole('link', { name: 'Playhub' })).toBeNull();
    expect(queryByRole('link', { name: 'Leaderboard' })).toBeNull();
  });
});
```

- [ ] **Step 2: Run them and check they fail**

Run: `make test FILTER="FullyQualifiedName~MeNamesTheHub"` and `cd web && npx vitest run src/App.test.ts`
Expected: FAIL. `hubUrl` is missing, no links are rendered, and the `mockResolvedValue` object has an excess property for the old type.

- [ ] **Step 3: Implement**

`server/AzulServer/Api/ApiEndpoints.cs`. Replace the `/me` line with:

```csharp
        api.MapGet("/me", (HttpContext c, AzulOptions o) => Results.Json(new { email = c.Email(), hubUrl = o.Hub.PublicUrl }));
```

`web/src/lib/api.ts`. Replace the `me` line with:

```ts
  me: () => request<{ email: string; hubUrl: string | null }>('GET', '/api/me'),
```

`web/src/App.svelte`:
- replace the `<script>` state and the `onMount` with:

```svelte
  let email = $state<string | null>(null);
  let hubUrl = $state<string | null>(null);
  onMount(async () => {
    try {
      const me = await api.me();
      email = me.email;
      hubUrl = me.hubUrl;
    } catch { /* pages show the error */ }
  });
```

- replace the header's inner block with:

```svelte
    <a class="logo" href="/" onclick={(e) => { e.preventDefault(); navigate('/'); }}>Azul</a>
    {#if hubUrl}
      <a href={hubUrl}>Playhub</a>
      <a href={`${hubUrl}/games/azul`}>Leaderboard</a>
    {/if}
    {#if email}
      <span class="who truncate" title={email}>{email}</span>
      <a href="/cdn-cgi/access/logout">Sign out</a>
    {/if}
```

- add this to `<style>`. The links stay on the header line at phone width; the email already truncates:

```css
  .top a { white-space: nowrap; }
```

Update every successful `api.me` mock for the new type:
- `web/src/components/Lobby.test.ts:16` → `.mockResolvedValue({ email: 'me@x', hubUrl: null })`;
- `web/src/lib/events.test.ts:22`, `:44` and `:71` → `.mockResolvedValue({ email: 'a@x', hubUrl: null })`.

`:58` (`mockRejectedValue`) and `:91` (`mockReturnValue(new Promise(...))`, which takes its type from the mock) need no change.

Then confirm nothing else mocks it: `grep -rn "spyOn(api, 'me')" web/src web/e2e` must list only these six lines.

- [ ] **Step 4: Run all the tests**

Run: `make test && make web-test`
Expected: PASS. `svelte-check` reports 0 errors.

- [ ] **Step 5: Commit**

```bash
git add server/AzulServer/Api/ApiEndpoints.cs server/AzulServer.Tests/Hub/HubFlowTests.cs web/src
git commit -m "web: Playhub and leaderboard links in the header (/api/me hubUrl)"   # + trailer block
```

---

### Task 10: Deployment (network, environment, targets, image pinning, probe, docs)

**Files:**
- Modify: `docker-compose.serve.yml`
- Modify: `Makefile`
- Modify: `Dockerfile`
- Modify: `scripts/serve-check.sh`
- Modify: `.env.serve.example`
- Modify: `docs/deploy.md`

- [ ] **Step 1: Compose**

In `docker-compose.serve.yml`, under `services.app.environment`, add after `AZUL_BOT_THINK_SECONDS`:

```yaml
      # Playhub (docs/deploy.md, "Playhub"): results go to the hub on the shared
      # docker network; the key is written into .env.serve by the playhub CLI.
      AZUL_HUB_URL: http://playhub:3000
      AZUL_HUB_KEY: ${AZUL_HUB_KEY:-}
      AZUL_HUB_PUBLIC_URL: ${AZUL_HUB_PUBLIC_URL:-https://play.signalwave.dev}
```

Add `networks: [default, signalwave]` to `app`. Leave `cloudflared` alone; it stays on `default` and still reaches `http://app:8080`. At the bottom of the file, next to `volumes:`, add:

```yaml
networks:
  # Shared with playhub's serve stack (it declares the same external network).
  signalwave:
    external: true
```

- [ ] **Step 2: Makefile**

- Add `hub-status hub-retry` to `.PHONY`.
- Below `DEV := ...`, add:

```make
NETWORK := docker network inspect signalwave >/dev/null 2>&1 || docker network create signalwave
```

- Make the first recipe line of both `serve` and `serve-app` `$(NETWORK)`.
- Add after `serve-down`:

```make
# Playhub delivery (docs/deploy.md): counts, failures, the bot key.
hub-status:
	$(SERVE) exec -T app /app/AzulServer hub status

# make hub-retry GAME=<id>|--all-failed [REBUILD=1]
hub-retry:
	@test -n "$(GAME)" || (echo "GAME=<game id> or GAME=--all-failed is required" && exit 1)
	$(SERVE) exec -T app /app/AzulServer hub retry $(GAME) $(if $(REBUILD),--rebuild,)
```

- [ ] **Step 3: Pin the images and normalise paths in the Dockerfile**

Find the exact versions currently behind the floating tags:

```bash
docker pull -q mcr.microsoft.com/dotnet/sdk:10.0 && docker run --rm mcr.microsoft.com/dotnet/sdk:10.0 dotnet --version
docker pull -q mcr.microsoft.com/dotnet/aspnet:10.0 && docker run --rm --entrypoint dotnet mcr.microsoft.com/dotnet/aspnet:10.0 --list-runtimes | grep AspNetCore
```

Call the outputs `SDKV` (for example `10.0.1xx`) and `RTV` (for example `10.0.x`). Confirm both tags exist with `docker pull mcr.microsoft.com/dotnet/sdk:$SDKV` and `docker pull mcr.microsoft.com/dotnet/aspnet:$RTV`. Then in `Dockerfile`:

- `FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build` → `FROM mcr.microsoft.com/dotnet/sdk:$SDKV AS build` (the literal version);
- `FROM mcr.microsoft.com/dotnet/aspnet:10.0` → `FROM mcr.microsoft.com/dotnet/aspnet:$RTV`;
- the publish line → `RUN dotnet publish server/AzulServer/AzulServer.csproj -c Release -o /out --no-restore -p:ContinuousIntegrationBuild=true`;
- above the build stage, add the comment `# Pinned to an exact patch version, with ContinuousIntegrationBuild: the bot's hub key includes the AzulLibrary MVID, so rebuilding the same source must give the same bytes (spec section 5).`

- [ ] **Step 4: serve-check step 3**

In `scripts/serve-check.sh`:
- replace the header line `#   3. machine-to-machine endpoints without their key -> 401: there are none` with
  `#   3. the app reaches Playhub with its key (AzulServer hub ping: 422 on an empty report)`;
- replace the line `echo "note  3. no machine-to-machine endpoints in this app"` with:

```bash
if [ -n "${AZUL_HUB_KEY:-}" ]; then
  if out=$(docker compose -f docker-compose.serve.yml --env-file .env.serve exec -T app /app/AzulServer hub ping 2>&1); then
    ok "3. the app reaches Playhub and its key is accepted"
  else
    bad "3. hub ping failed: $out"
  fi
else
  echo "note  3. AZUL_HUB_KEY is empty: results are not reported to Playhub"
fi
```

- [ ] **Step 5: `.env.serve.example` and `docs/deploy.md`**

Append to `.env.serve.example`:

```
# Playhub (docs/deploy.md, "Playhub"). Written by the playhub CLI, never by hand:
#   cd ~/playhub && npx tsx scripts/hub.ts --serve games issue-key azul \
#     --key-out ../Azul-Board-Game/.env.serve --env-var AZUL_HUB_KEY
# Empty: games are not reported. AZUL_HUB_PUBLIC_URL defaults to https://play.signalwave.dev.
AZUL_HUB_KEY=
```

In `docs/deploy.md`, insert this section before `## Checking it live`:

````markdown
## Playhub

Finished games are reported to Playhub (play.signalwave.dev) over the shared docker network
`signalwave`; `make serve` creates it if playhub's stack has not.

1. Register the game and write its key (from `~/playhub`, with playhub's serve stack up):
   ```sh
   npx tsx scripts/hub.ts --serve games put azul --name Azul --play-url https://azul.signalwave.dev \
     --rated-rules "Every 2-4 player game that ends with a completed wall row and in which no seat was handed to the bot."
   npx tsx scripts/hub.ts --serve games issue-key azul --key-out ../Azul-Board-Game/.env.serve --env-var AZUL_HUB_KEY
   ```
2. `make serve`, then `make serve-check`: step 3 pings the hub with the key.
3. Delivery: `make hub-status` (counts, failures, the bot key). A report the hub refused
   (409/422) stays `failed`; after a fix, `make hub-retry GAME=<id>` resends it, or
   `make hub-retry GAME=<id> REBUILD=1` rebuilds the body first. A rotated key (401) keeps
   reports pending until the new key is in `.env.serve` and `make serve` restarts the app.
4. The bot's key (`mcts@...`) changes when the engine assembly changes (any AzulLibrary code,
   or a different compiler), when `AZUL_BOT_THINK_SECONDS` changes, or when
   `MctsBrain.BrainRevision` is bumped. After such a deploy, retire the old bot from
   `~/playhub`: `npx tsx scripts/hub.ts --serve bots list --game azul`, then
   `npx tsx scripts/hub.ts --serve bots retire azul <old key>`. Machine speed and
   `AZUL_BOT_WORKERS` change how many rollouts fit in the think time but not the key.
````

- [ ] **Step 6: Check the deploy files**

Run: `docker compose -f docker-compose.serve.yml --env-file .env.serve.example config >/dev/null` (with `AZUL_ACCESS_TEAM_DOMAIN=x AZUL_ACCESS_AUD=x CLOUDFLARE_TUNNEL_TOKEN=x` exported)
Expected: exit 0. The output of `config` lists the `signalwave` network as external.

Run: `bash -n scripts/serve-check.sh && make -n hub-status serve`
Expected: no syntax errors. The dry run shows `docker network inspect signalwave ... || docker network create signalwave`.

Run: `docker build -t azul-hub-check .`
Expected: success with the pinned images. Then run
`docker run --rm azul-hub-check hub bot-key`, which prints `{"botKey":"mcts@..."}`.

- [ ] **Step 7: Commit**

```bash
git add docker-compose.serve.yml Makefile Dockerfile scripts/serve-check.sh .env.serve.example docs/deploy.md
git commit -m "deploy: signalwave network, hub env, hub-status/hub-retry, pinned images, serve-check ping"   # + trailer block
```

---

### Task 11: Verification

**Files:** none changed, unless a check fails. A failure is fixed in the task that owns the code, and that task's tests are re-run.

- [ ] **Step 1: The full gate**

Run: `make test && make web-test`
Expected: everything passes. The test count is the Task 0 baseline plus the new tests.

Optional e2e: first check `docker ps | grep azul-e2e`, because port 5081 is shared with other Azul worktrees. Then run `make e2e-server-start && make e2e; make e2e-server-stop`.
Expected: the existing Playwright suite passes.

- [ ] **Step 2: Golden bodies against the hub's real parser**

**Required. This step blocks completion.** The playhub CLI branch (worktree `~/playhub-games-cli`, `feat/games-cli`) is implemented first. If `npx tsx scripts/hub.ts results validate --help` is not available there, stop and report the dependency as unmet. Do not hand off.

```bash
cd ~/playhub-games-cli
for f in ~/Azul-Board-Game-hub-results/server/AzulServer.Tests/Hub/golden/*.json; do
  npx tsx scripts/hub.ts results validate "$f" | head -c 200; echo "  <- $f (exit ${PIPESTATUS[0]})"
done
```

Expected: every file exits 0 with `"valid": true`. Any failure is a contract mismatch. Fix it in `HubReport` (Task 4), regenerate the golden files with `AZUL_UPDATE_GOLDEN=1`, and re-run.

- [ ] **Step 3: Two clean builds give the same bot key**

```bash
cd ~/Azul-Board-Game-hub-results
docker build --no-cache -q -t azul-mvid-a . && docker build --no-cache -q -t azul-mvid-b .
docker run --rm azul-mvid-a hub bot-key; docker run --rm azul-mvid-b hub bot-key
```

Expected: identical `botKey` values. If they differ, the build is not deterministic. Do not ship; report both keys. The spec's accepted-churn note does not cover rebuilds of the same source.

Clean up afterwards with `docker rmi azul-mvid-a azul-mvid-b azul-hub-check`.

- [ ] **Step 4: Manual end-to-end against the playhub dev stack**

Prerequisites: the playhub CLI branch exists, and `make up` is running in `~/playhub-games-cli`, which gives Postgres on 24532 and Vite on **24100**. The fixed ports mean only one playhub checkout can run this.

```bash
cd ~/playhub-games-cli
npx tsx scripts/hub.ts games put azul --name Azul --play-url http://127.0.0.1:5082
npx tsx scripts/hub.ts games issue-key azul --key-out ~/Azul-Board-Game-hub-results/.data/hub-dev.env --env-var AZUL_HUB_KEY
cd ~/Azul-Board-Game-hub-results && make e2e-publish && (cd web && npm run build)
mkdir -p .data/hub-dev-db
docker run --rm -d --name azul-hub-dev --network host --user $(id -u):$(id -g) \
  --env-file .data/hub-dev.env -e ASPNETCORE_HTTP_PORTS=5082 -e AZUL_DATA_DIR=/data -e AZUL_WEB_ROOT=/web \
  -e AZUL_HUB_URL=http://127.0.0.1:24100 -e AZUL_HUB_PUBLIC_URL=http://127.0.0.1:24100 \
  -e AZUL_PUBLIC_ORIGIN=http://127.0.0.1:5082 -e AZUL_BOT_THINK_SECONDS=0.2 -e AZUL_MIN_MOVE_DELAY_SECONDS=0.2 \
  -v $PWD/.data/e2e-server:/app:ro -v $PWD/.data/hub-dev-db:/data -v $PWD/web/dist:/web:ro \
  mcr.microsoft.com/dotnet/aspnet:10.0 dotnet /app/AzulServer.dll
```

`.data/` is gitignored: confirm with `git check-ignore .data/hub-dev.env`.

1. Open `http://127.0.0.1:5082`, and in devtools run
   `document.cookie = "azul_dev_user=ann@example.com; path=/"`, then reload.
2. Check that the header shows **Playhub** and **Leaderboard**.
3. Create a 2-player game, start it (the second seat becomes the bot), and play it to the end.
4. Within a few seconds `docker logs azul-hub-dev | grep "hub accepted"` shows the game.

The playhub dev UI at `http://127.0.0.1:24100/games/azul` should then list the match, rated, with `mcts@...` on the leaderboard. To check unrated: play a second game, hand your seat to the bot (the UI's "to bot" control) and let it finish. The match shows "unrated: a bot played for a person".

Afterwards: `docker rm -f azul-hub-dev`.

- [ ] **Step 5: Hand-off note**

Report the following:
- the test counts;
- the golden validation result: every file accepted (anything else means the work is not done);
- the two MVID keys;
- the end-to-end result;
- reminders for the deployer:
  - `make serve` needs playhub's stack, or at least the `signalwave` network;
  - write `AZUL_HUB_KEY` with the playhub CLI.

---

## Self-Review (done while writing)

**Spec coverage.** Each spec section maps to a task:

| Spec section | Task |
| --- | --- |
| §3 data model | Task 3 |
| §4 body: ids, variant, rated rule incl. the pending-move union, timestamps, replay, players, contract normalisation, ranks with tie-break, guards | Tasks 1 and 4 |
| §5 bot identity, including the MVID, pinning and CI build | Task 2 (Docker side in Task 10, double build in Task 11) |
| §6.1 atomic queueing (a failing trigger on the real INSERT); a real build exception and reconcile recovery; `Delete` refusal | Task 6 |
| §6.2 reconcile | Tasks 5 and 7 |
| §6.3 sender: outcomes, backoff, the lease and lease ownership (`lease_id`), stuck alert, containment (unit and app-level), shutdown, config warnings | Tasks 5 and 7 |
| §6.4 commands, including rebuild safety and ping | Task 8 |
| §7 configuration and deployment | Task 10 |
| §8 links | Task 9 |
| §10 tests, including the golden bodies via `results validate` (required, blocking) | Tasks 4 and 11 |
| Manual end-to-end | Task 11 |

**Deviations, recorded under File Structure:**
- reconcile and rebuild use `BEGIN IMMEDIATE` transactions instead of the in-process lock;
- `Build` takes a `GameRecord`;
- `hub bot-key` was added;
- the atomicity test forces the failure with a test-only SQLite trigger on `hub_reports`, not an `IFaultInjector` hook. The spec was amended to match.

**Placeholders.** None. The pinned image versions in Task 10 are discovered by the given commands, not invented.

**Type consistency:** the following names are the same everywhere they appear:
- `HubBuild(Body, Skip)`
- `HubClaim(Body, Attempts, LeaseId)`
- `HubOutbox.MarkSent/MarkFailed/MarkRetry(c, id, leaseId, ...) -> bool`
- `HubOutbox.Queue(c, tx, id, body, now)`
- `HubOutbox.QueueMissing(db, origin, now, log)`
- `HubReport.BotPlayedSeats(c, tx, id, pending)`
- `HubSender.HttpName`
- `HubSignal.Wake/WaitAsync`
- `IHubReportBuilder.Build(g, botPlayedSeats, publicOrigin)`
- `HubPlay.Penultimate/Insert/Request/Body`
- `BotIdentity.For(...)` and `BotIdentity.DisplayName`
- `AzulOptions.Hub.{Url,Key,PublicUrl}`
