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
