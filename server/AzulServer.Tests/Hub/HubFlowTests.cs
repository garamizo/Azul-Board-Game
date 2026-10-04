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

    [Fact]
    public async Task MeNamesTheHubWhenConfigured()
    {
        using var on = new TestApp(new AzulOptions { BotWorkers = 0, Hub = new HubOptions { PublicUrl = "https://play.example" } });
        var me = JsonNode.Parse(await on.Client().GetStringAsync("/api/me"))!;
        Assert.Equal(("alice@example.com", "https://play.example"), ((string?)me["email"], (string?)me["hubUrl"]));
        using var off = new TestApp();
        Assert.Null(JsonNode.Parse(await off.Client().GetStringAsync("/api/me"))!["hubUrl"]);
    }
}
